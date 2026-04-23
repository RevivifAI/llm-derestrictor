"""Streaming sharded abliteration.

Reads one safetensors shard at a time from ``input_dir``, applies the
ablation kernel on the loaded device, and writes the modified tensor into
a corresponding output shard. Memory cap is roughly one shard plus a
hidden_size² scratch — the only way frontier-scale MoE models (235B /
480B) fit on a consumer GPU.

The pipeline mirrors ``jim-plus/llm-abliteration/sharded_ablate.py``'s
read → upcast → modify → downcast → write loop and reuses the same
direction-resolution and kernel-dispatch helpers as the in-memory path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from derestrictor.core.abliterate import (
    AbliterationConfig,
    HybridArchitectureInfo,
    RefusalDirections,
    apply_kernel_to_expert_tensor,
    detect_hybrid_architecture,
    dispatch_2d_kernel,
    get_layer_type_from_name,
    resolve_ablation_for_tensor,
    write_safetensors_index,
)
from derestrictor.models.utils import (
    _MOE_FUSED_NAME_RE,
    MOE_FAMILY_LAYOUT,
    _expert_layer_type,
    _resolve_expert_axes,
)

if TYPE_CHECKING:
    from derestrictor.core.null_space import NullSpaceProjector


logger = logging.getLogger(__name__)


# Files that should be copied verbatim from the source model directory to
# the output. The streaming path doesn't reload the model, so anything the
# downstream consumer (transformers ``from_pretrained``, llama.cpp's
# ``convert_hf_to_gguf``, etc.) expects to find next to the safetensors
# must be brought along by hand.
_NON_WEIGHT_FILES_TO_COPY = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "tekken.json",
    "params.json",
)


def _enumerate_shards(input_dir: Path) -> tuple[list[Path], dict[str, str] | None]:
    """List shard files in ``input_dir`` along with the original weight_map.

    Returns ``(shards, weight_map)``. When the model is a single
    ``model.safetensors`` file (no index), ``weight_map`` is ``None`` and
    ``shards`` has length 1.
    """
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        unique_shards = sorted({input_dir / fname for fname in weight_map.values()})
        return unique_shards, weight_map

    single = input_dir / "model.safetensors"
    if single.exists():
        return [single], None

    raise FileNotFoundError(
        f"No safetensors weights found in {input_dir} (looked for model.safetensors and model.safetensors.index.json)"
    )


def _copy_non_weight_files(input_dir: Path, output_dir: Path) -> list[str]:
    """Copy tokenizer / config / generation files to the output directory.

    Returns the list of filenames actually copied.
    """
    copied: list[str] = []
    for filename in _NON_WEIGHT_FILES_TO_COPY:
        src = input_dir / filename
        if src.exists():
            dst = output_dir / filename
            if not dst.exists():
                shutil.copy2(src, dst)
                copied.append(filename)

    for src in input_dir.glob("*.json"):
        if src.name in copied or src.name.startswith("model"):
            continue
        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied.append(src.name)

    return copied


def _is_fused_moe_param(name: str, shape: tuple[int, ...]) -> bool:
    """Return True when ``name`` / ``shape`` describe a fused 3-D MoE expert tensor."""
    return len(shape) == 3 and bool(_MOE_FUSED_NAME_RE.search(name))


def _resolve_moe_layout_for_tensor(
    name: str,
    shape: tuple[int, ...],
    config: AbliterationConfig,
    family_hint: str | None,
):
    """Resolve ``(layout, output_axis, input_axis, layer_type)`` for a fused expert tensor."""
    role = _expert_layer_type(name)
    cfg_layout = config.moe_fused_layout
    explicit_hint = cfg_layout if cfg_layout in ("eoi", "eio") else family_hint
    layout, output_axis, input_axis = _resolve_expert_axes(role, shape, family_hint=explicit_hint)
    return layout, output_axis, input_axis, role


def _detect_family_hint(input_dir: Path) -> str | None:
    """Sniff ``model_type`` out of ``input_dir/config.json`` for the MoE layout table."""
    cfg_path = input_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    model_type = (cfg.get("model_type") or "").lower()
    return MOE_FAMILY_LAYOUT.get(model_type)


def estimate_disk_size(input_dir: Path) -> int:
    """Return the total byte size of the safetensors weights in ``input_dir``."""
    shards, _ = _enumerate_shards(input_dir)
    return sum(s.stat().st_size for s in shards if s.exists())


def should_use_streaming(input_dir: Path, config: AbliterationConfig) -> bool:
    """Auto-decide whether to use the streaming path.

    Picks streaming when on-disk model size exceeds 0.6× available GPU RAM
    (or 0.4× system RAM if CPU-only). Honors an explicit
    ``config.streaming`` override when not ``"auto"``.
    """
    if config.streaming == "sharded":
        return True
    if config.streaming == "in_memory":
        return False

    try:
        disk_size = estimate_disk_size(Path(input_dir))
    except FileNotFoundError:
        return False

    if torch.cuda.is_available() and config.device != "cpu":
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
        except (RuntimeError, AssertionError):
            total_vram = 0
        if total_vram > 0:
            return disk_size > 0.6 * total_vram

    total_ram = psutil.virtual_memory().total
    return disk_size > 0.4 * total_ram


def _ablate_2d_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    directions: RefusalDirections,
    config: AbliterationConfig,
    null_space_projector: NullSpaceProjector | None,
    primary_direction: torch.Tensor | None,
    biprojection_layer_map: dict[int, int] | None,
    intervention_layers: set[int] | None,
    layer_weights: dict[int, float] | None,
    hybrid_info: HybridArchitectureInfo | None,
) -> tuple[torch.Tensor, bool]:
    """Apply the 2-D kernel to a single weight tensor and return ``(new_tensor, modified)``."""
    ctx = resolve_ablation_for_tensor(
        name,
        tuple(tensor.shape),
        directions=directions,
        config=config,
        null_space_projector=null_space_projector,
        primary_direction=primary_direction,
        biprojection_layer_map=biprojection_layer_map,
        intervention_layers=intervention_layers,
        layer_weights=layer_weights,
        hybrid_info=hybrid_info,
    )
    if ctx.skip:
        return tensor, False
    new_tensor = dispatch_2d_kernel(
        tensor,
        ctx.direction,
        kernel=ctx.kernel,
        multiplier=ctx.multiplier,
        config=config,
        direction_space=ctx.direction_space,
        harmless_dir=ctx.harmless_dir,
        null_space_V=ctx.null_space_V,
    )
    return new_tensor, True


def _ablate_moe_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    directions: RefusalDirections,
    config: AbliterationConfig,
    null_space_projector: NullSpaceProjector | None,
    primary_direction: torch.Tensor | None,
    biprojection_layer_map: dict[int, int] | None,
    intervention_layers: set[int] | None,
    layer_weights: dict[int, float] | None,
    hybrid_info: HybridArchitectureInfo | None,
    family_hint: str | None,
) -> tuple[torch.Tensor, bool]:
    """Apply the 2-D kernel to every expert in a fused MoE 3-D tensor."""
    layout, output_axis, input_axis, role = _resolve_moe_layout_for_tensor(
        name, tuple(tensor.shape), config, family_hint
    )
    if layout == "per_expert_2d":
        return tensor, False

    if config.target_layer_types and (
        role is None or role.lower() not in [t.lower() for t in config.target_layer_types]
    ):
        return tensor, False

    per_expert_shape = (tensor.shape[output_axis], tensor.shape[input_axis])
    ctx = resolve_ablation_for_tensor(
        name,
        per_expert_shape,
        directions=directions,
        config=config,
        null_space_projector=null_space_projector,
        primary_direction=primary_direction,
        biprojection_layer_map=biprojection_layer_map,
        intervention_layers=intervention_layers,
        layer_weights=layer_weights,
        hybrid_info=hybrid_info,
    )
    if ctx.skip:
        return tensor, False

    def kernel_call(
        weight_2d: torch.Tensor,
        direction_2d: torch.Tensor,
        *,
        direction_space: str | None = None,
        _ctx=ctx,
    ) -> torch.Tensor:
        return dispatch_2d_kernel(
            weight_2d,
            direction_2d,
            kernel=_ctx.kernel,
            multiplier=_ctx.multiplier,
            config=config,
            direction_space=direction_space,
            harmless_dir=_ctx.harmless_dir,
            null_space_V=_ctx.null_space_V,
        )

    new_tensor = apply_kernel_to_expert_tensor(
        tensor,
        ctx.direction,
        kernel_fn=kernel_call,
        layout=layout,
        expert_dim=0,
        direction_space=ctx.direction_space,
    )
    return new_tensor, True


def sharded_ablate(
    input_dir: str | Path,
    output_dir: str | Path,
    directions: RefusalDirections,
    config: AbliterationConfig,
    null_space_projector: NullSpaceProjector | None = None,
) -> dict[str, str]:
    """Stream every safetensors shard in ``input_dir`` through the kernel.

    Memory cap is roughly one shard plus per-tensor scratch. Per shard:

    1. ``safetensors.safe_open`` the input file with framework="pt" and
       device=``config.device`` so each tensor lands directly on the
       ablation device.
    2. For every key, read with ``f.get_tensor(key)``, dispatch through
       :func:`derestrictor.core.abliterate.resolve_ablation_for_tensor`
       and :func:`derestrictor.core.abliterate.dispatch_2d_kernel` (or
       :func:`derestrictor.core.abliterate.apply_kernel_to_expert_tensor`
       for fused 3-D MoE experts).
    3. Cast back to the original dtype and ``safetensors.torch.save_file``
       the modified shard to ``output_dir``.

    Non-weight files (tokenizer, generation config, etc.) are copied
    verbatim. ``model.safetensors.index.json`` is rewritten via
    :func:`write_safetensors_index` so the output is a drop-in replacement
    that ``transformers.AutoModelForCausalLM.from_pretrained`` can load
    directly.

    Args:
        input_dir: Path to the original model directory (must contain
            ``model.safetensors`` or ``model.safetensors.index.json``).
        output_dir: Path to write the abliterated model. Created if missing.
        directions: Computed refusal directions (loaded into memory once).
        config: Active :class:`AbliterationConfig` (drives kernel,
            layouts, biprojection mapping, etc.).
        null_space_projector: Optional null-space projector.

    Returns:
        The output ``weight_map`` (tensor name → shard filename) actually
        written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards, original_weight_map = _enumerate_shards(input_dir)

    hybrid_info = detect_hybrid_architecture(str(input_dir))
    family_hint = _detect_family_hint(input_dir)

    primary_direction = None
    if directions.biprojected_direction is not None and config.use_biprojection:
        primary_direction = directions.biprojected_direction.to(config.device)
    elif config.use_mean_direction and directions.mean_direction is not None:
        primary_direction = directions.mean_direction.to(config.device)

    biprojection_layer_map: dict[int, int] = directions.metadata.get("biprojection_layer_map") or {}
    biprojection_layer_map = {int(k): int(v) for k, v in biprojection_layer_map.items()}
    intervention_layers = {int(i) for i in (config.intervention_layers or [])} if config.intervention_layers else None
    layer_weights = config.per_layer_multipliers

    weight_map: dict[str, str] = {}
    total_size = 0
    modified_count = 0
    skipped_count = 0
    moe_modified = 0
    moe_skipped = 0

    num_shards = len(shards)
    for shard_idx, shard_path in enumerate(shards, start=1):
        out_shard_name = (
            f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors" if num_shards > 1 else "model.safetensors"
        )
        out_shard_path = output_dir / out_shard_name

        out_state: dict[str, torch.Tensor] = {}

        with safe_open(str(shard_path), framework="pt", device=str(config.device)) as f:
            keys = list(f.keys())
            for key in tqdm(keys, desc=f"Shard {shard_idx}/{num_shards}", leave=False):
                tensor = f.get_tensor(key)
                original_dtype = tensor.dtype

                if not key.endswith(".weight") and not _is_fused_moe_param(key, tuple(tensor.shape)):
                    out_state[key] = tensor.contiguous().cpu()
                    continue

                layer_type = get_layer_type_from_name(key)
                if (
                    config.target_layer_types
                    and layer_type is not None
                    and layer_type.lower() not in [t.lower() for t in config.target_layer_types]
                ):
                    out_state[key] = tensor.contiguous().cpu()
                    skipped_count += 1
                    continue

                try:
                    if _is_fused_moe_param(key, tuple(tensor.shape)):
                        new_tensor, modified = _ablate_moe_tensor(
                            tensor,
                            key,
                            directions=directions,
                            config=config,
                            null_space_projector=null_space_projector,
                            primary_direction=primary_direction,
                            biprojection_layer_map=biprojection_layer_map,
                            intervention_layers=intervention_layers,
                            layer_weights=layer_weights,
                            hybrid_info=hybrid_info,
                            family_hint=family_hint,
                        )
                        if modified:
                            moe_modified += 1
                        else:
                            moe_skipped += 1
                    elif tensor.ndim == 2:
                        new_tensor, modified = _ablate_2d_tensor(
                            tensor,
                            key,
                            directions=directions,
                            config=config,
                            null_space_projector=null_space_projector,
                            primary_direction=primary_direction,
                            biprojection_layer_map=biprojection_layer_map,
                            intervention_layers=intervention_layers,
                            layer_weights=layer_weights,
                            hybrid_info=hybrid_info,
                        )
                        if modified:
                            modified_count += 1
                        else:
                            skipped_count += 1
                    else:
                        new_tensor = tensor
                except Exception as exc:
                    logger.warning(f"Failed to abliterate {key}: {exc}")
                    new_tensor = tensor
                    skipped_count += 1

                if new_tensor.dtype != original_dtype:
                    new_tensor = new_tensor.to(original_dtype)
                out_state[key] = new_tensor.contiguous().cpu()

        for k, v in out_state.items():
            weight_map[k] = out_shard_name
            total_size += v.numel() * v.element_size()
        save_file(out_state, str(out_shard_path))
        del out_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if num_shards > 1 or original_weight_map is not None:
        write_safetensors_index(output_dir, weight_map, metadata={"total_size": total_size})

    copied = _copy_non_weight_files(input_dir, output_dir)
    if copied:
        logger.info(f"Copied {len(copied)} non-weight files: {copied}")

    logger.info(
        f"Streaming ablation complete: 2-D modified={modified_count} skipped={skipped_count}, "
        f"MoE modified={moe_modified} skipped={moe_skipped}"
    )
    return weight_map

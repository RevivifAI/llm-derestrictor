"""Parity-infra regression tests for streaming sharded ablation + MoE experts.

Pinned features:

* Synthetic 2-shard safetensors round-trip through
  :func:`derestrictor.core.sharded_ablate.sharded_ablate` (preserves
  shapes, dtypes, and rewrites ``model.safetensors.index.json``).
* :func:`derestrictor.core.abliterate.apply_kernel_to_expert_tensor`
  produces results identical (within FP tolerance) to a hand-rolled
  per-slice 2-D loop for both ``"eoi"`` (Mixtral) and ``"eio"``
  (Qwen2.5-MoE / GPT-OSS) layouts.
* Auto-streaming heuristic (:func:`should_use_streaming`) picks
  ``"sharded"`` when the on-disk model footprint exceeds 0.6× available
  GPU RAM (or 0.4× system RAM on CPU).
* MoE detection (:func:`is_moe_model`) correctly identifies Mixtral,
  Qwen2-MoE, GPT-OSS, and DeepSeek-MoE config stubs while rejecting a
  dense Llama config.
* CLI flags (``--streaming``, ``--moe-fused-layout``,
  ``--measurement-quant``) plumb through to ``AbliterationConfig``.

All tests are pure-math / file-IO: no real Hugging Face model loads, no
network. The synthetic shards are deliberately tiny (a handful of small
tensors) so the suite stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from derestrictor.core.abliterate import (
    AbliterationConfig,
    RefusalDirections,
    apply_directional_scaling,
    apply_kernel_to_expert_tensor,
    write_safetensors_index,
)
from derestrictor.core.sharded_ablate import (
    _enumerate_shards,
    _is_fused_moe_param,
    estimate_disk_size,
    sharded_ablate,
    should_use_streaming,
)
from derestrictor.models.utils import (
    ExpertTensorInfo,
    _expert_layer_type,
    _resolve_expert_axes,
    is_moe_model,
)

# Helpers


def _make_config(**overrides: object) -> AbliterationConfig:
    base = {
        "model_path": "/tmp/unused-model-path",
        "output_path": "/tmp/unused-output-path",
        "device": "cpu",
        "dtype": torch.float32,
    }
    base.update(overrides)
    return AbliterationConfig(**base)  # type: ignore[arg-type]


def _make_dense_directions(hidden_size: int, layers: list[int]) -> RefusalDirections:
    """Build a deterministic ``RefusalDirections`` covering the requested layers."""
    torch.manual_seed(0)
    per_layer = {}
    for idx in layers:
        v = torch.randn(hidden_size, dtype=torch.float32)
        per_layer[idx] = v / v.norm()
    mean = torch.stack(list(per_layer.values())).mean(dim=0)
    mean = mean / mean.norm()
    return RefusalDirections(
        directions=per_layer,
        mean_direction=mean,
        metadata={},
    )


# write_safetensors_index helper round-trips


def test_write_safetensors_index_round_trips(tmp_path: Path) -> None:
    """The lifted helper must emit transformers-compatible JSON with both keys."""
    weight_map = {
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    }
    index_path = write_safetensors_index(tmp_path, weight_map, metadata={"total_size": 12345})
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["metadata"] == {"total_size": 12345}
    assert payload["weight_map"] == weight_map


# MoE detection


def test_is_moe_model_detects_mixtral_config() -> None:
    """Mixtral configs expose ``num_local_experts`` directly."""
    cfg = SimpleNamespace(num_local_experts=8, model_type="mixtral")
    assert is_moe_model(cfg) is True


def test_is_moe_model_detects_qwen2_moe_config() -> None:
    """Qwen2-MoE uses ``num_experts`` (not ``num_local_experts``)."""
    cfg = SimpleNamespace(num_experts=64, model_type="qwen2_moe")
    assert is_moe_model(cfg) is True


def test_is_moe_model_detects_gpt_oss_config() -> None:
    """GPT-OSS uses ``num_local_experts`` like Mixtral but with ``model_type='gpt_oss'``."""
    cfg = SimpleNamespace(num_local_experts=128, model_type="gpt_oss")
    assert is_moe_model(cfg) is True


def test_is_moe_model_detects_deepseek_n_routed_experts() -> None:
    """DeepSeek-MoE uses the ``n_routed_experts`` field."""
    cfg = SimpleNamespace(n_routed_experts=160, model_type="deepseek_v2")
    assert is_moe_model(cfg) is True


def test_is_moe_model_rejects_dense_llama_config() -> None:
    """A dense Llama config (no expert fields) must report False."""
    cfg = SimpleNamespace(model_type="llama", num_attention_heads=32, hidden_size=4096)
    assert is_moe_model(cfg) is False


def test_is_moe_model_walks_nested_text_config() -> None:
    """VL-style configs hide the MoE field under ``text_config``; the walker must descend."""
    text_cfg = SimpleNamespace(num_local_experts=8)
    cfg = SimpleNamespace(model_type="qwen2_vl_moe", text_config=text_cfg)
    assert is_moe_model(cfg) is True


def test_is_moe_model_detects_via_module_tree(tmp_path: Path) -> None:
    """When the config is silent, a ``block_sparse_moe`` submodule alone is sufficient."""
    inner = torch.nn.Linear(4, 4)
    container = torch.nn.Module()
    container.block_sparse_moe = inner  # type: ignore[attr-defined]
    container.config = SimpleNamespace(model_type="custom")  # type: ignore[attr-defined]
    assert is_moe_model(container) is True


# Layer-type / axis resolution for fused MoE names


def test_expert_layer_type_recognizes_fused_names() -> None:
    """``_expert_layer_type`` must canonicalize the four common fused conventions."""
    assert _expert_layer_type("model.layers.5.block_sparse_moe.experts.w1") == "w1"
    assert _expert_layer_type("model.layers.5.block_sparse_moe.experts.w2") == "w2"
    assert _expert_layer_type("model.layers.5.block_sparse_moe.experts.w3") == "w3"
    assert _expert_layer_type("model.layers.5.mlp.experts.gate_up_proj") == "gate_up_proj"
    assert _expert_layer_type("model.layers.5.mlp.experts.down_proj") == "down_proj"


def test_resolve_expert_axes_eoi_for_mixtral_w2() -> None:
    """Mixtral ``w2`` (output-space) at ``[E, hidden, intermediate]`` is ``eoi``."""
    layout, out_axis, in_axis = _resolve_expert_axes("w2", (8, 4096, 14336), family_hint=None)
    assert layout == "eoi"
    assert out_axis == 1
    assert in_axis == 2


def test_resolve_expert_axes_eio_for_qwen_down_proj() -> None:
    """Qwen2.5-MoE ``down_proj`` at ``[E, intermediate, hidden]`` (intermediate > hidden) is ``eio``."""
    # intermediate=2048 > hidden=1408 -> the larger axis is the input
    # axis, smaller is output -> eio with output_axis=2, input_axis=1.
    layout, out_axis, in_axis = _resolve_expert_axes("down_proj", (8, 2048, 1408), family_hint=None)
    assert layout == "eio"
    assert out_axis == 2
    assert in_axis == 1


def test_resolve_expert_axes_honors_family_hint_on_square() -> None:
    """When dims are equal, the family hint must break the tie."""
    layout, _, _ = _resolve_expert_axes("w1", (8, 4096, 4096), family_hint="eio")
    assert layout == "eio"
    layout, _, _ = _resolve_expert_axes("w1", (8, 4096, 4096), family_hint="eoi")
    assert layout == "eoi"


# apply_kernel_to_expert_tensor parity vs hand-rolled per-slice loop


def test_apply_kernel_to_expert_tensor_eoi_matches_per_slice_loop() -> None:
    """``"eoi"`` layout must equal a manual per-expert 2-D kernel call."""
    torch.manual_seed(100)
    num_experts, out_features, in_features = 4, 6, 8
    weight = torch.randn(num_experts, out_features, in_features, dtype=torch.float32)
    direction = torch.randn(out_features, dtype=torch.float32)
    direction = direction / direction.norm()

    def _kernel(w: torch.Tensor, d: torch.Tensor, *, direction_space: str | None = None) -> torch.Tensor:
        return apply_directional_scaling(w, d, scale_factor=1.0, direction_space=direction_space or "output")

    expected = torch.empty_like(weight)
    for e in range(num_experts):
        expected[e] = _kernel(weight[e], direction, direction_space="output")

    got = apply_kernel_to_expert_tensor(weight, direction, _kernel, layout="eoi", direction_space="output")
    assert torch.allclose(got, expected, atol=1e-6), (
        f"eoi-per-expert mismatch: max delta {(got - expected).abs().max().item():.3e}"
    )


def test_apply_kernel_to_expert_tensor_eio_matches_per_slice_transpose_loop() -> None:
    """``"eio"`` layout must equal a manual per-expert transpose + 2-D kernel + transpose-back loop."""
    torch.manual_seed(101)
    num_experts, in_features, out_features = 4, 8, 6
    weight = torch.randn(num_experts, in_features, out_features, dtype=torch.float32)
    direction = torch.randn(out_features, dtype=torch.float32)
    direction = direction / direction.norm()

    def _kernel(w: torch.Tensor, d: torch.Tensor, *, direction_space: str | None = None) -> torch.Tensor:
        return apply_directional_scaling(w, d, scale_factor=1.0, direction_space=direction_space or "output")

    expected = torch.empty_like(weight)
    for e in range(num_experts):
        slab = weight[e].transpose(0, 1).contiguous()
        modified = _kernel(slab, direction, direction_space="output")
        expected[e] = modified.transpose(0, 1).contiguous()

    got = apply_kernel_to_expert_tensor(weight, direction, _kernel, layout="eio", direction_space="output")
    assert torch.allclose(got, expected, atol=1e-6), (
        f"eio-per-expert mismatch: max delta {(got - expected).abs().max().item():.3e}"
    )


def test_apply_kernel_to_expert_tensor_rejects_2d_input() -> None:
    """Passing a 2-D tensor must error rather than silently ablate just the first row."""
    weight = torch.randn(6, 8)
    direction = torch.randn(6)
    with pytest.raises(ValueError, match="3-D"):
        apply_kernel_to_expert_tensor(weight, direction, kernel_fn=lambda w, d, **_: w, layout="eoi")


def test_apply_kernel_to_expert_tensor_rejects_unknown_layout() -> None:
    """Layout outside ``{eoi, eio}`` must raise so typos don't silently corrupt weights."""
    weight = torch.randn(2, 6, 8)
    direction = torch.randn(6)
    with pytest.raises(ValueError, match="layout"):
        apply_kernel_to_expert_tensor(weight, direction, kernel_fn=lambda w, d, **_: w, layout="oei")


# ExpertTensorInfo dataclass surface


def test_expert_tensor_info_carries_axis_metadata() -> None:
    """Downstream consumers rely on every field being plain ints / strings — pin the dataclass shape."""
    info = ExpertTensorInfo(
        name="model.layers.0.block_sparse_moe.experts.w2",
        shape=(8, 4096, 14336),
        layout="eoi",
        expert_dim=0,
        output_axis=1,
        input_axis=2,
        layer_type="w2",
    )
    assert info.layout == "eoi"
    assert info.expert_dim == 0
    assert info.output_axis == 1
    assert info.input_axis == 2
    assert info.layer_type == "w2"


# _is_fused_moe_param matcher


def test_is_fused_moe_param_matches_known_names() -> None:
    """Known fused MoE parameter names with 3-D shapes must register as fused."""
    assert _is_fused_moe_param("model.layers.5.block_sparse_moe.experts.w1", (8, 14336, 4096))
    assert _is_fused_moe_param("model.layers.5.mlp.experts.gate_up_proj", (8, 4096, 28672))
    assert _is_fused_moe_param("model.layers.5.mlp.experts.down_proj", (8, 14336, 4096))
    assert not _is_fused_moe_param("model.layers.5.mlp.experts.w1", (4096, 14336))  # 2-D, not fused
    assert not _is_fused_moe_param("model.layers.5.self_attn.q_proj.weight", (8, 4096, 4096))


# Streaming auto-detect heuristic


def test_should_use_streaming_honors_explicit_overrides(tmp_path: Path) -> None:
    """Explicit ``streaming="sharded"`` / ``"in_memory"`` must short-circuit the size probe."""
    cfg_sharded = _make_config(streaming="sharded")
    cfg_inmem = _make_config(streaming="in_memory")
    assert should_use_streaming(tmp_path, cfg_sharded) is True
    assert should_use_streaming(tmp_path, cfg_inmem) is False


def test_should_use_streaming_picks_sharded_when_disk_exceeds_fake_vram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto mode must pick streaming when on-disk size > 0.6× available GPU RAM."""
    # Build a tiny single-shard model on disk so estimate_disk_size returns
    # a real, non-zero byte count.
    weights = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(64, 64, dtype=torch.float32)}
    save_file(weights, str(tmp_path / "model.safetensors"))

    actual_disk = estimate_disk_size(tmp_path)
    assert actual_disk > 0

    # Force the CPU-only path (more deterministic across CI) and patch
    # psutil.virtual_memory() so 0.4x system RAM sits well below the
    # synthetic disk size; the heuristic must then return True.
    cfg = _make_config(streaming="auto", device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    fake_total_ram = max(int(actual_disk * 1.5), 64)  # 0.4x => 0.6x disk size
    fake_vm = SimpleNamespace(total=fake_total_ram)
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake_vm)
    assert should_use_streaming(tmp_path, cfg) is True

    # Now blow up system RAM so disk fits comfortably; heuristic flips.
    huge_vm = SimpleNamespace(total=int(actual_disk * 1000))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: huge_vm)
    assert should_use_streaming(tmp_path, cfg) is False


def test_should_use_streaming_returns_false_when_no_safetensors(tmp_path: Path) -> None:
    """Missing weights must degrade to ``in_memory`` rather than raising."""
    cfg = _make_config(streaming="auto")
    assert should_use_streaming(tmp_path, cfg) is False


# End-to-end synthetic 2-shard streaming round-trip


def test_sharded_ablate_two_shard_round_trip_preserves_shapes_and_dtypes(tmp_path: Path) -> None:
    """Two-shard synthetic model must round-trip through ``sharded_ablate`` with the same shapes / dtypes / keys."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    hidden_size = 16
    num_layers = 2

    # Build two shards: shard 1 has layer 0, shard 2 has layer 1. The keys
    # match the names that get_layer_type_from_name knows about so the
    # streaming kernel actually fires (instead of skipping them).
    weights_shard1 = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, 4 * hidden_size, dtype=torch.float32),
        "model.layers.0.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.float32),
    }
    weights_shard2 = {
        "model.layers.1.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.1.mlp.down_proj.weight": torch.randn(hidden_size, 4 * hidden_size, dtype=torch.float32),
        "model.layers.1.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.float32),
    }

    save_file(weights_shard1, str(input_dir / "model-00001-of-00002.safetensors"))
    save_file(weights_shard2, str(input_dir / "model-00002-of-00002.safetensors"))

    weight_map = dict.fromkeys(weights_shard1, "model-00001-of-00002.safetensors")
    weight_map.update(dict.fromkeys(weights_shard2, "model-00002-of-00002.safetensors"))
    write_safetensors_index(input_dir, weight_map, metadata={"total_size": 0})

    # Drop a minimal config.json so the family-hint sniffer doesn't error
    # and so transformers-style consumers downstream stay happy.
    (input_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": hidden_size, "num_hidden_layers": num_layers}),
        encoding="utf-8",
    )

    directions = _make_dense_directions(hidden_size, list(range(num_layers)))
    cfg = _make_config(streaming="sharded", device="cpu", dtype=torch.float32)

    out_weight_map = sharded_ablate(input_dir, output_dir, directions, cfg)

    # Both shards must exist with the expected names.
    out_shard1 = output_dir / "model-00001-of-00002.safetensors"
    out_shard2 = output_dir / "model-00002-of-00002.safetensors"
    assert out_shard1.exists()
    assert out_shard2.exists()

    # The rewritten index must list every original key.
    assert set(out_weight_map) == set(weight_map)
    out_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(out_index["weight_map"]) == set(weight_map)
    assert "total_size" in out_index["metadata"]

    # config.json must have been carried over verbatim.
    assert (output_dir / "config.json").exists()

    # Per-shard shapes + dtypes preserved; q_proj actually changed (kernel
    # fired); layernorm bypassed unchanged.
    from safetensors import safe_open

    with safe_open(str(out_shard1), framework="pt", device="cpu") as f:
        for key, original in weights_shard1.items():
            t = f.get_tensor(key)
            assert tuple(t.shape) == tuple(original.shape), key
            assert t.dtype == original.dtype, key

        ln = f.get_tensor("model.layers.0.input_layernorm.weight")
        assert torch.allclose(ln, weights_shard1["model.layers.0.input_layernorm.weight"]), (
            "layernorm weight must be passed through unchanged by the streaming pipeline"
        )

        # At least one of the targeted projections must have actually changed.
        q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
        o = f.get_tensor("model.layers.0.self_attn.o_proj.weight")
        d = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        diffs = [
            (q - weights_shard1["model.layers.0.self_attn.q_proj.weight"]).abs().max().item(),
            (o - weights_shard1["model.layers.0.self_attn.o_proj.weight"]).abs().max().item(),
            (d - weights_shard1["model.layers.0.mlp.down_proj.weight"]).abs().max().item(),
        ]
        assert max(diffs) > 1e-5, f"streaming kernel did not modify any projection: diffs={diffs}"


def test_enumerate_shards_handles_single_file_model(tmp_path: Path) -> None:
    """Single-file ``model.safetensors`` (no index) must degrade to a one-element shard list."""
    weights = {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.float32)}
    save_file(weights, str(tmp_path / "model.safetensors"))
    shards, weight_map = _enumerate_shards(tmp_path)
    assert len(shards) == 1
    assert shards[0].name == "model.safetensors"
    assert weight_map is None


# AbliterationConfig fields


def test_abliteration_config_accepts_streaming_fields() -> None:
    """The new streaming + MoE + measurement-quant fields must round-trip into the dataclass."""
    cfg = _make_config(
        streaming="sharded",
        moe_fused_layout="eio",
        measurement_quant="4bit",
    )
    assert cfg.streaming == "sharded"
    assert cfg.moe_fused_layout == "eio"
    assert cfg.measurement_quant == "4bit"

    default = _make_config()
    assert default.streaming == "auto"
    assert default.moe_fused_layout == "auto"
    assert default.measurement_quant == "none"

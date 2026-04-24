#!/usr/bin/env python3
"""Model loading utilities with Vision-Language (VL) model support.

This module provides centralized model loading that automatically detects
VL models (like Qwen3-VL) and uses the appropriate AutoModel class.
It also handles FP8 quantized models that require special loading.
"""

import builtins
import contextlib
import json
import locale
import logging
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers import BitsAndBytesConfig, FineGrainedFP8Config, Mistral3ForConditionalGeneration

# Fix Windows encoding issues - set preferred encoding for file I/O
if sys.platform == "win32":
    # Try to set UTF-8 mode for the process
    os.environ["PYTHONUTF8"] = "1"
    # Set locale to UTF-8 if possible
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_ALL, ".UTF-8")

logger = logging.getLogger(__name__)

# VL architecture patterns - maps model_type to known architecture class names.
# Both the top-level multimodal model_type and its text_config.model_type are
# listed where applicable so detect_model_type matches whichever is reported.
VL_ARCHITECTURES = {
    "qwen3_5": ["Qwen3_5ForConditionalGeneration"],
    "qwen3_5_moe": ["Qwen3_5MoeForConditionalGeneration"],
    "qwen3_vl": ["Qwen3VLForConditionalGeneration"],
    "qwen3_vl_moe": ["Qwen3VLMoeForConditionalGeneration"],
    "qwen3_omni_moe": ["Qwen3OmniMoeForConditionalGeneration"],
    "qwen2_vl": ["Qwen2VLForConditionalGeneration"],
    "qwen2_vl_moe": ["Qwen2VLMoeForConditionalGeneration"],
    "llava": ["LlavaForConditionalGeneration"],
    "llava_next": ["LlavaNextForConditionalGeneration"],
    "idefics2": ["Idefics2ForConditionalGeneration"],
    "paligemma": ["PaliGemmaForConditionalGeneration"],
    "glm4v": ["Glm4vForConditionalGeneration"],
    "mistral3": ["Mistral3ForConditionalGeneration"],
}

# Architectures that need special text-only loading even though they're multimodal.
# These use ForConditionalGeneration but can be loaded with the text-only model class.
# Note: Gemma3ForConditionalGeneration is intentionally absent because forcing the
# text-only class would drop ~400M vision encoder parameters.
#
# Qwen3.5-MoE / Qwen3-Omni-MoE are listed because their vision/audio towers are
# multi-billion-parameter modules that play no role in refusal mediation; loading
# the text-only causal LM class shrinks the measurement-phase footprint
# substantially (e.g. ~13B vision parameters skipped for Qwen3.6-35B-A3B).
FORCE_TEXT_MODEL_ARCHITECTURES: dict[str, str] = {
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniMoeForCausalLM",
}

# Patterns that indicate FP8 quantization (scale factors)
FP8_SCALE_PATTERNS = [
    r"_scale_inv$",  # Common FP8 scale inverse suffix
    r"_scale$",  # FP8 scale suffix
    r"\.fp8_scale",  # Explicit FP8 scale
]


def detect_fp8_quantization(model_path: str) -> bool:
    """Detect if a model uses FP8 quantization by checking for scale factors.

    FP8 quantized models have weight tensors with names ending in '_scale_inv'
    or '_scale'. These models require special loading to avoid meta tensor issues.

    Args:
        model_path: Path to the model directory

    Returns:
        True if FP8 quantization is detected
    """
    model_dir = Path(model_path)

    # Check safetensors index first (most common for large models)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with index_path.open(encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for key in weight_map:
                for pattern in FP8_SCALE_PATTERNS:
                    if re.search(pattern, key):
                        logger.info(f"Detected FP8 quantization (found {key})")
                        return True
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read safetensors index: {e}")

    # Check pytorch bin index as fallback
    bin_index_path = model_dir / "pytorch_model.bin.index.json"
    if bin_index_path.exists():
        try:
            with bin_index_path.open(encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            for key in weight_map:
                for pattern in FP8_SCALE_PATTERNS:
                    if re.search(pattern, key):
                        logger.info(f"Detected FP8 quantization (found {key})")
                        return True
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read pytorch bin index: {e}")

    # Check config.json for quantization_config
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as f:
                config = json.load(f)
            quant_config = config.get("quantization_config", {})
            quant_method = quant_config.get("quant_method", "")
            if "fp8" in quant_method.lower():
                logger.info(f"Detected FP8 quantization from config (method: {quant_method})")
                return True
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read config.json: {e}")

    return False


def _fix_weight_tying(model, model_path: str):
    """Fix weight tying for models loaded with FP8 dequantization.

    When loading FP8 models with dequantize=True, the weight tying between
    embed_tokens and lm_head may not be properly applied, resulting in
    lm_head.weight being randomly initialized.

    This function checks if tie_word_embeddings is enabled and manually
    copies the embed_tokens weights to lm_head if needed.

    Args:
        model: The loaded model
        model_path: Path to model directory (to read config)

    Returns:
        The model with fixed weight tying
    """
    # Check if weight tying should be enabled
    config_path = Path(model_path) / "config.json"
    tie_embeddings = False

    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as f:
                config = json.load(f)
            # Check main config and text_config for tie_word_embeddings
            tie_embeddings = config.get("tie_word_embeddings", False)
            if not tie_embeddings and "text_config" in config:
                tie_embeddings = config["text_config"].get("tie_word_embeddings", False)
        except (OSError, json.JSONDecodeError):
            pass

    if not tie_embeddings:
        return model

    # Find embed_tokens and lm_head
    embed_tokens = None
    lm_head = None

    # Try different paths for embed_tokens
    if hasattr(model, "model"):
        if hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
            embed_tokens = model.model.language_model.embed_tokens
        elif hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
            embed_tokens = model.model.model.embed_tokens

    # Find lm_head
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if embed_tokens is None or lm_head is None:
        logger.debug("Could not find embed_tokens or lm_head for weight tying fix")
        return model

    # Check if weights are already tied (same data pointer)
    if embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr():
        logger.debug("Weights already tied")
        return model

    # Check if lm_head weights look uninitialized (random) vs embed_tokens
    # If they're very different, copy embed_tokens to lm_head
    embed_norm = embed_tokens.weight.data.norm().item()
    lm_head_norm = lm_head.weight.data.norm().item()

    # If norms are very different, lm_head is likely uninitialized
    if abs(embed_norm - lm_head_norm) / max(embed_norm, lm_head_norm, 1e-8) > 0.1:
        logger.info("Fixing lm_head weight tying (copying from embed_tokens)")
        with torch.no_grad():
            lm_head.weight.data.copy_(embed_tokens.weight.data)

    return model


def detect_model_type(model_path: str) -> dict:
    """Detect model type by reading config.json.

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        dict with:
            - is_vl: bool - True if this is a vision-language model
            - model_type: str - The model_type from config
            - architectures: list - The architectures list from config
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        # Might be a HuggingFace Hub model ID - we can't detect locally
        # Let transformers handle it and we'll try AutoModelForCausalLM first
        logger.debug(f"No config.json at {model_path}, assuming standard model")
        return {"is_vl": False, "model_type": None, "architectures": []}

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])

    # Check if any architecture matches known VL patterns
    is_vl = False
    for arch in architectures:
        for vl_patterns in VL_ARCHITECTURES.values():
            if arch in vl_patterns:
                is_vl = True
                logger.info(f"Detected VL model: {arch}")
                break
        if is_vl:
            break

    # Also check model_type directly
    if not is_vl and model_type in VL_ARCHITECTURES:
        is_vl = True
        logger.info(f"Detected VL model type: {model_type}")

    # Check for FP8 quantization
    is_fp8 = detect_fp8_quantization(model_path)

    return {
        "is_vl": is_vl,
        "is_fp8": is_fp8,
        "model_type": model_type,
        "architectures": architectures,
    }


def _build_bnb_quantization_config(quantization: Literal["none", "4bit", "8bit"], dtype: torch.dtype):
    """Build a ``BitsAndBytesConfig`` for measurement-only quant loads.

    Returns ``None`` when ``quantization == "none"``.
    """
    if quantization == "none":
        return None
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Despite the misleading name, this flag governs CPU offload for
            # both 8-bit and 4-bit loads. Without it, accelerate's
            # ``device_map="auto"`` raises if any module is dispatched to CPU
            # because the quantized footprint exceeds VRAM (common for >20B
            # models on consumer GPUs).
            llm_int8_enable_fp32_cpu_offload=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    return None


def _resolve_model_local_dir(model_path: str) -> Path | None:
    """Return the local directory for *model_path*, or ``None`` if unavailable.

    * For an existing local directory, returns ``Path(model_path)``.
    * For a HuggingFace Hub model ID, returns the snapshot directory from the
      local cache if the model has been downloaded; returns ``None`` otherwise.

    Args:
        model_path: Local directory path or Hub model ID.

    Returns:
        Resolved :class:`~pathlib.Path`, or ``None``.
    """
    p = Path(model_path)
    if p.is_dir():
        return p
    try:
        cached = try_to_load_from_cache(model_path, "config.json")
        if isinstance(cached, str):
            return Path(cached).parent
    except Exception:
        pass
    return None


def _resolve_safetensors_disk_size(model_path: str) -> int:
    """Return the total size (bytes) of ``.safetensors`` shards for *model_path*.

    Handles both local directory paths and HuggingFace Hub model IDs.  For Hub
    IDs the local snapshot cache is consulted via
    :func:`huggingface_hub.try_to_load_from_cache`; returns ``0`` when no
    local snapshot exists yet (the model has never been downloaded).

    Args:
        model_path: Local directory path or Hub model ID (e.g.
            ``"meta-llama/Meta-Llama-3-8B"``).

    Returns:
        Combined byte size of all ``.safetensors`` files, or ``0`` on any
        failure.
    """
    search_root = _resolve_model_local_dir(model_path)
    if search_root is None:
        return 0
    try:
        return sum(f.stat().st_size for f in search_root.iterdir() if f.suffix == ".safetensors")
    except Exception:
        return 0


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    quantization: Literal["none", "4bit", "8bit"] = "none",
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load model and tokenizer, automatically handling VL models.

    For VL models, uses AutoModelForImageTextToText.
    For text models, uses AutoModelForCausalLM.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        device: Device to load model on ("cuda", "cpu", or "auto")
        dtype: Torch dtype for model weights
        trust_remote_code: Whether to trust remote code in model files
        quantization: Optional bitsandbytes quant for the activation
            measurement phase. ``"4bit"`` / ``"8bit"`` enable the
            corresponding ``BitsAndBytesConfig`` load; ``"none"`` (default)
            keeps the full-precision behavior. Final weight ablation should
            still happen on the unquantized weights via the streaming
            pipeline.

    Returns:
        Tuple of (model, tokenizer)
    """
    model_info = detect_model_type(model_path)
    quant_config = _build_bnb_quantization_config(quantization, dtype)
    if quant_config is not None:
        logger.info(
            f"Loading model with bitsandbytes {quantization} quantization "
            "(measurement-only path; ablation should stream the original weights)"
        )

    # Decide on the ``device_map`` we hand to ``from_pretrained``. Two cases
    # force ``device_map="auto"`` (so accelerate plans CPU/disk offload):
    #
    #   1. bnb quantization is active. The literal ``"cuda"`` placement asks
    #      accelerate to pin everything on GPU which OOMs on consumer GPUs for
    #      >20B models even at 4-bit.
    #
    #   2. The on-disk model footprint exceeds available VRAM. Without
    #      quantization the BF16 weights need genuine multi-tier offload
    #      (GPU + CPU + disk) which only ``"auto"`` plus an ``offload_folder``
    #      arranges correctly.
    #
    # Note: bnb 4-bit + CPU offload is broken in current upstream releases
    # (``quant_state.offset`` lives on a ``meta`` device and crashes inside
    # ``Linear4bit._save_to_state_dict``), so when the model is too large to
    # fit in VRAM at 4-bit we *prefer* skipping bnb entirely and using BF16
    # with native accelerate offload -- slower but actually works end-to-end.
    needs_auto_device_map = quant_config is not None
    offload_folder: str | None = None
    if device != "cpu" and torch.cuda.is_available():
        try:
            free_vram, _ = torch.cuda.mem_get_info()
        except Exception:
            free_vram = 0
        disk_size = _resolve_safetensors_disk_size(model_path)
        # 0.85 leaves headroom for activations, KV cache, and optimizer-style
        # bookkeeping accelerate keeps pinned on GPU.
        if disk_size > 0 and free_vram > 0 and disk_size > 0.85 * free_vram:
            needs_auto_device_map = True
            logger.info(
                f"Model footprint ({disk_size / 1e9:.1f} GB) exceeds 0.85x free VRAM "
                f"({free_vram / 1e9:.1f} GB); using device_map='auto' with disk offload"
            )
            # For Hub IDs use the resolved snapshot directory as the offload
            # anchor so we don't accidentally create folders relative to CWD.
            local_dir = _resolve_model_local_dir(model_path) or Path(model_path)
            offload_folder = str(local_dir.parent / "_accelerate_offload")
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
    quant_device_map = "auto" if (needs_auto_device_map and device != "cpu") else device

    # Load tokenizer (same for both VL and text models for abliteration purposes)
    logger.info(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except UnicodeDecodeError as e:
        # Windows encoding issue - try with use_fast=False as fallback
        logger.warning(f"Encoding error loading tokenizer, trying slow tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=False)
        except UnicodeDecodeError:
            # Last resort: monkey-patch open to use UTF-8 for text mode only
            original_open = builtins.open

            def utf8_open(file, mode="r", *args, **kwargs):
                # Only patch text mode opens (no 'b' in mode)
                if "b" not in mode and "encoding" not in kwargs:
                    kwargs["encoding"] = "utf-8"
                    kwargs["errors"] = "replace"
                return original_open(file, mode, *args, **kwargs)

            builtins.open = utf8_open
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            except UnicodeDecodeError as final_error:
                builtins.open = original_open
                raise RuntimeError(
                    f"Failed to load tokenizer due to encoding issues: {final_error}\n"
                    "On Windows, try running with UTF-8 mode:\n"
                    "  set PYTHONUTF8=1 && derestrictor\n"
                    "Or: python -X utf8 -m derestrictor"
                ) from final_error
            finally:
                builtins.open = original_open
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if we need to force a text-only model class
    # Some multimodal architectures (like Gemma3) have a separate text-only class
    force_text_class = None
    for arch in model_info.get("architectures", []):
        if arch in FORCE_TEXT_MODEL_ARCHITECTURES:
            force_text_class = FORCE_TEXT_MODEL_ARCHITECTURES[arch]
            logger.info(f"Detected {arch}, will use text-only class: {force_text_class}")
            break

    # Determine if we need special loading for FP8 models
    # FP8 models with scale factors cause meta tensor issues with device_map
    is_fp8 = model_info.get("is_fp8", False)
    if is_fp8:
        logger.warning(
            "FP8 quantized model detected. Loading without device_map to avoid "
            "meta tensor issues. This requires more CPU memory during loading."
        )

    # Check if this is a Mistral3 model (requires special handling for FP8)
    is_mistral3 = any("Mistral3" in arch for arch in model_info.get("architectures", []))

    # Load model with appropriate class.
    #
    # ``force_text_class`` (e.g. mapping ``Qwen3_5MoeForConditionalGeneration``
    # -> ``Qwen3_5MoeForCausalLM``) takes precedence over the VL branch so that
    # multimodal models whose vision/audio towers are irrelevant to refusal
    # mediation skip those parameters entirely during the measurement load.
    if model_info["is_vl"] and not force_text_class:
        # Mistral3 VL models require special handling
        if is_mistral3:
            logger.info("Loading Mistral3 VL model...")
            if is_fp8:
                # Mistral3 FP8 models require FineGrainedFP8Config with dequantize=True
                # We explicitly set torch_dtype to ensure consistent output dtype
                logger.info("Using FineGrainedFP8Config for Mistral3 FP8 model...")
                quant_config = FineGrainedFP8Config(dequantize=True)
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    torch_dtype=dtype,  # Ensure consistent dtype after dequantization
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
                # Fix weight tying for FP8 dequantized models
                # The lm_head.weight may not be properly tied after dequantization
                model = _fix_weight_tying(model, model_path)
                logger.info(f"Model loaded with dtype={dtype} after FP8 dequantization")
            else:
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )
        else:
            logger.info("Loading VL model with AutoModelForImageTextToText...")
            if is_fp8:
                # FP8 models: load without device_map, then move to device
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,  # Avoid meta tensors
                    trust_remote_code=trust_remote_code,
                )
                if device != "cpu":
                    logger.info(f"Moving FP8 model to {device}...")
                    model = model.to(device)
            else:
                extra_kwargs: dict = {}
                if quant_config is not None:
                    extra_kwargs["quantization_config"] = quant_config
                if offload_folder is not None:
                    extra_kwargs["offload_folder"] = offload_folder
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=quant_device_map,
                    trust_remote_code=trust_remote_code,
                    **extra_kwargs,
                )
    elif force_text_class:
        # Load with specific text-only model class to avoid multimodal issues
        logger.info(f"Loading model with specific class: {force_text_class}...")
        try:
            model_class = getattr(transformers, force_text_class, None)
            if model_class is None:
                logger.warning(f"Could not find {force_text_class}, falling back to AutoModelForCausalLM")
                model_class = AutoModelForCausalLM

            if is_fp8:
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    trust_remote_code=trust_remote_code,
                )
                if device != "cpu":
                    logger.info(f"Moving FP8 model to {device}...")
                    model = model.to(device)
            else:
                extra_kwargs = {}
                if quant_config is not None:
                    extra_kwargs["quantization_config"] = quant_config
                if offload_folder is not None:
                    extra_kwargs["offload_folder"] = offload_folder
                model = model_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=quant_device_map,
                    trust_remote_code=trust_remote_code,
                    **extra_kwargs,
                )
        except Exception as e:
            logger.warning(f"Failed to load with {force_text_class}: {e}, falling back to AutoModelForCausalLM")
            if is_fp8:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    trust_remote_code=trust_remote_code,
                )
                if device != "cpu":
                    model = model.to(device)
            else:
                extra_kwargs = {}
                if quant_config is not None:
                    extra_kwargs["quantization_config"] = quant_config
                if offload_folder is not None:
                    extra_kwargs["offload_folder"] = offload_folder
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=quant_device_map,
                    trust_remote_code=trust_remote_code,
                    **extra_kwargs,
                )
    else:
        logger.info("Loading model with AutoModelForCausalLM...")
        if is_fp8:
            if quant_config is not None:
                logger.warning("FP8 model + bitsandbytes quantization is not supported; ignoring quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=trust_remote_code,
            )
            if device != "cpu":
                logger.info(f"Moving FP8 model to {device}...")
                model = model.to(device)
        else:
            extra_kwargs = {}
            if quant_config is not None:
                extra_kwargs["quantization_config"] = quant_config
            if offload_folder is not None:
                extra_kwargs["offload_folder"] = offload_folder
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=quant_device_map,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )

    logger.info(f"Model loaded: {type(model).__name__}")
    return model, tokenizer


# MoE expert detection
#
# Mixtral, Qwen2.5-MoE, and GPT-OSS store expert weights as fused 3-D
# ``nn.Parameter`` tensors instead of per-expert ``nn.Linear`` modules.
# The default ``get_linear_layer_names`` walker silently skips them, so the
# MoE MLPs go un-ablated. These helpers expose enough metadata for the
# streaming + in-memory ablation paths to dispatch through
# :func:`derestrictor.core.abliterate.apply_kernel_to_expert_tensor`.

# Compiled patterns for fused MoE expert parameter names. Matches the three
# common conventions in the wild.
_MOE_FUSED_NAME_RE = re.compile(
    r"(?:experts\.(?:gate_up_proj|down_proj|w1|w2|w3))"
    r"|(?:block_sparse_moe\.experts\.(?:w1|w2|w3))"
    r"|(?:mlp\.experts\.(?:weight|gate_up_proj|down_proj|w1|w2|w3))",
)

# Module-tree markers that almost always indicate a MoE block.
_MOE_MODULE_MARKERS = ("block_sparse_moe", "mlp.experts", ".experts.")

# Known model-family layout hints, consumed when ``moe_fused_layout="auto"``
# can't disambiguate from shape alone (e.g. when both axes equal hidden_size).
#
# ``qwen3_5_moe`` (and its ``_text`` text-config sibling, plus ``qwen3_omni_moe``)
# store fused experts as ``nn.Parameter([E, OUT, IN])`` per the transformers
    # 5.5 ``Qwen3_5MoeExperts`` / ``Qwen3OmniMoeThinkerTextExperts`` source --
    # ``gate_up_proj`` is ``[E, 2*moe_intermediate, hidden]`` and ``down_proj``
    # is ``[E, hidden, moe_intermediate]`` -- which is the same EOI layout as
# Mixtral. Without this hint the shape-only fallback in
# :func:`_resolve_expert_axes` would mis-classify ``down_proj`` as EIO
# (because its ``in`` axis is smaller than ``out``) and ablate the wrong
# axis, silently corrupting the MLP.
MOE_FAMILY_LAYOUT: dict[str, Literal["eoi", "eio"]] = {
    "mixtral": "eoi",
    "qwen2_moe": "eio",
    "qwen3_moe": "eio",
    "qwen3_vl_moe": "eio",
    "qwen2_vl_moe": "eio",
    "qwen3_5_moe": "eoi",
    "qwen3_5_moe_text": "eoi",
    "qwen3_omni_moe": "eoi",
    "qwen3_omni_moe_text": "eoi",
    "gpt_oss": "eio",
    "gptoss": "eio",
}


@dataclass
class ExpertTensorInfo:
    """Metadata describing one fused MoE expert ``nn.Parameter``.

    Attributes:
        name: Full dotted parameter name (e.g.
            ``"model.layers.5.block_sparse_moe.experts.w2"``).
        shape: Tensor shape, ``(num_experts, dim0, dim1)``.
        layout: ``"eoi"`` for ``[E, O, I]`` (Mixtral) or ``"eio"`` for
            ``[E, I, O]`` (Qwen2.5-MoE / GPT-OSS). ``"per_expert_2d"`` is
            reserved for cases where each expert is a separate 2-D weight
            (the in-memory walker already covers that path; included here
            so callers don't have to handle ``None``).
        expert_dim: Axis carrying the expert index. Always 0 for fused
            tensors.
        output_axis: Axis (within a per-expert 2-D slice) that maps to the
            output features of the underlying linear layer.
        input_axis: Axis (within a per-expert 2-D slice) that maps to the
            input features.
        layer_type: One of the canonical role names returned by
            :func:`derestrictor.core.abliterate.get_layer_type_from_name`
            (``"w1"`` / ``"w2"`` / ``"w3"`` / ``"gate_up_proj"`` /
            ``"down_proj"``). Drives kernel direction-space inference.
    """

    name: str
    shape: tuple[int, ...]
    layout: Literal["eoi", "eio", "per_expert_2d"]
    expert_dim: int
    output_axis: int
    input_axis: int
    layer_type: str | None = None


def _moe_count_from_config(config_or_dict) -> int:
    """Pull a likely expert count from a HF config or its dict.

    Returns 0 when no MoE-style field is present.
    """
    if config_or_dict is None:
        return 0
    fields = ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")
    if isinstance(config_or_dict, dict):
        for f in fields:
            v = config_or_dict.get(f, 0) or 0
            if int(v) > 0:
                return int(v)
        text_cfg = config_or_dict.get("text_config")
        if isinstance(text_cfg, dict):
            return _moe_count_from_config(text_cfg)
        return 0
    for f in fields:
        v = getattr(config_or_dict, f, 0) or 0
        if int(v) > 0:
            return int(v)
    text_cfg = getattr(config_or_dict, "text_config", None)
    if text_cfg is not None and text_cfg is not config_or_dict:
        return _moe_count_from_config(text_cfg)
    return 0


def is_moe_model(model_or_config) -> bool:
    """Return True when the model (or config) declares MoE expert blocks.

    Detection order:

    1. ``num_local_experts`` / ``num_experts`` / ``moe_num_experts`` /
       ``n_routed_experts`` on the config (or nested ``text_config``).
    2. Module-tree walk for names matching ``block_sparse_moe``,
       ``mlp.experts``, or ``.experts.<digit>``.

    Args:
        model_or_config: Either a ``transformers`` model instance or a
            config object (or dict).

    Returns:
        True if the architecture has fused expert blocks.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    if _moe_count_from_config(cfg) > 0:
        return True

    if isinstance(model_or_config, torch.nn.Module):
        expert_idx_re = re.compile(r"\.experts\.\d+")
        for name, _ in model_or_config.named_modules():
            if any(marker in name for marker in _MOE_MODULE_MARKERS):
                return True
            if expert_idx_re.search(name):
                return True

    return False


def _expert_layer_type(name: str) -> str | None:
    """Resolve a fused expert parameter name to a canonical layer type.

    Matches the role names returned by
    :func:`derestrictor.core.abliterate.get_layer_type_from_name`.
    """
    name_l = name.lower()
    for role in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
        if role in name_l:
            return role
    if "experts.weight" in name_l:
        return None
    return None


def _resolve_expert_axes(
    role: str | None,
    shape: tuple[int, ...],
    family_hint: Literal["eoi", "eio"] | None = None,
) -> tuple[Literal["eoi", "eio"], int, int]:
    """Pick ``(layout, output_axis, input_axis)`` for a fused expert tensor.

    Conventions:

    * Mixtral (``eoi``): ``[num_experts, out_features, in_features]``,
      output_axis = 1, input_axis = 2.
    * Qwen2.5-MoE / GPT-OSS (``eio``): ``[num_experts, in_features,
      out_features]``, output_axis = 2, input_axis = 1.

    For ``down_proj`` / ``w2`` (output-space), the axis equal to
    ``hidden_size`` is the output axis. ``family_hint`` is consulted only
    when shape inspection is ambiguous (both candidate axes equal).
    """
    if len(shape) != 3:
        return ("eoi", 1, 2)
    _, d0, d1 = shape
    if family_hint:
        if family_hint == "eoi":
            return ("eoi", 1, 2)
        return ("eio", 2, 1)

    if role in ("down_proj", "w2"):
        if d0 < d1:
            return ("eoi", 1, 2)
        if d1 < d0:
            return ("eio", 2, 1)
        return ("eoi", 1, 2)

    if role in ("gate_up_proj",):
        if d0 < d1:
            return ("eoi", 1, 2)
        if d1 < d0:
            return ("eio", 2, 1)
        return ("eio", 2, 1)

    if d0 < d1:
        return ("eio", 2, 1)
    if d1 < d0:
        return ("eoi", 1, 2)
    return ("eoi", 1, 2)


def iter_expert_tensors(
    model: torch.nn.Module,
    family_hint: Literal["eoi", "eio"] | None = None,
) -> Iterator[ExpertTensorInfo]:
    """Yield :class:`ExpertTensorInfo` for every fused MoE expert parameter.

    Walks ``model.named_parameters()`` and matches against the fused-name
    regex. Only 3-D parameters are returned; per-expert 2-D weights are
    already handled by the standard linear-layer walker and are skipped
    here to avoid double-counting.

    Args:
        model: The transformers model to scan.
        family_hint: Optional ``"eoi"`` / ``"eio"`` override. When ``None``,
            the family hint is derived from ``model.config.model_type``
            via :data:`MOE_FAMILY_LAYOUT` (when present).

    Yields:
        One :class:`ExpertTensorInfo` per fused expert parameter found.
    """
    if family_hint is None:
        cfg = getattr(model, "config", None)
        model_type = (getattr(cfg, "model_type", "") or "").lower()
        family_hint = MOE_FAMILY_LAYOUT.get(model_type)

    for name, param in model.named_parameters():
        if not _MOE_FUSED_NAME_RE.search(name):
            continue
        if param.ndim != 3:
            continue
        role = _expert_layer_type(name)
        layout, output_axis, input_axis = _resolve_expert_axes(role, tuple(param.shape), family_hint=family_hint)
        yield ExpertTensorInfo(
            name=name,
            shape=tuple(param.shape),
            layout=layout,
            expert_dim=0,
            output_axis=output_axis,
            input_axis=input_axis,
            layer_type=role,
        )

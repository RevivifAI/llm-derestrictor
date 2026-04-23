"""Architecture-support tests for Qwen3.5-MoE / Qwen3.6-A3B.

Pinning the metadata for ``qwen3_5_moe`` (and the related
``qwen3_omni_moe`` family) so changes to layer-name parsing, the
non-LM-stack pass-through list, or the MoE family layout table stay
correct for the multimodal hybrid-MoE architectures used by Qwen3.6.

The fused expert layout (``[E, OUT, IN]`` = EOI) is verified directly
against the transformers ``Qwen3_5MoeExperts`` ``nn.Parameter`` shapes
in :func:`test_resolve_expert_axes_with_qwen3_5_eoi_hint`. Without that
hint :func:`derestrictor.models.utils._resolve_expert_axes` would
mis-classify ``down_proj`` (whose ``in`` axis is smaller than ``out``)
as EIO and ablate the wrong tensor axis, silently corrupting the MLP.
"""

from __future__ import annotations

import pytest

from derestrictor.core.abliterate import (
    get_layer_index_from_name,
    get_layer_type_from_name,
    is_non_lm_stack_tensor,
    is_residual_boundary_tensor,
)
from derestrictor.models.utils import (
    MOE_FAMILY_LAYOUT,
    VL_ARCHITECTURES,
    _resolve_expert_axes,
)


class TestNonLMStackDetection:
    """``is_non_lm_stack_tensor`` must isolate vision / audio / MTP weights.

    These components live outside the language-model residual stream that
    mediates refusal. Letting them fall into the ablation dispatcher would
    either re-use unrelated LM-layer directions (MTP shares ``layers.N``
    numbering) or, for projections whose shape coincidentally matches the
    LM hidden_size (e.g. Qwen3-VL ``model.visual.merger.linear_fc2.weight``
    at ``[hidden, 4*hidden]``), pass the shape-mismatch guard and get
    silently mutated.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.self_attn.k_proj.weight",
            "model.visual.blocks.0.mlp.linear_fc1.weight",
            "model.visual.merger.linear_fc2.weight",
            "model.audio_tower.encoder.layers.3.self_attn.q_proj.weight",
            "audio_tower.proj.weight",
        ],
    )
    def test_non_lm_prefixes_are_excluded(self, name: str) -> None:
        assert is_non_lm_stack_tensor(name) is True
        assert get_layer_index_from_name(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.5.mlp.experts.gate_up_proj",
            "model.language_model.layers.5.mlp.shared_expert.down_proj.weight",
            "model.layers.10.self_attn.o_proj.weight",
            "lm_head.weight",
        ],
    )
    def test_lm_stack_tensors_pass_through(self, name: str) -> None:
        assert is_non_lm_stack_tensor(name) is False


class TestResidualBoundaryDetection:
    """``is_residual_boundary_tensor`` must isolate ``embed_tokens`` / ``lm_head``.

    Both tensors are part of the LM proper but live at the residual-stream
    boundary (``[vocab, hidden]``), so the hidden-space refusal direction
    cannot be cleanly projected out without ambiguous axis selection. The
    dispatcher must skip them explicitly: otherwise ``lm_head`` reaches the
    kernel and trips the ``"Direction shape ... doesn't match weight shape
    ..."`` warning, and ``embed_tokens`` (whose ``get_layer_type_from_name``
    returns ``None``) silently falls through to the legacy input-axis
    match and gets mutated per-row — observed in the Qwen3.6-35B-A3B run
    on 2026-04-23 where ``embed_tokens.weight`` was modified while
    ``lm_head.weight`` was correctly skipped, leaving the codebase
    inconsistent across the two boundary tensors.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "lm_head.weight",
            "model.lm_head.weight",
            "model.model.lm_head.weight",
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "embed_tokens.weight",
        ],
    )
    def test_boundary_tensors_are_detected(self, name: str) -> None:
        assert is_residual_boundary_tensor(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.language_model.layers.5.mlp.experts.gate_up_proj",
            "model.language_model.layers.5.mlp.shared_expert.down_proj.weight",
            "model.layers.10.self_attn.o_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.visual.merger.linear_fc2.weight",
            "mtp.layers.0.mlp.experts.gate_up_proj",
        ],
    )
    def test_internal_layers_are_not_boundary(self, name: str) -> None:
        assert is_residual_boundary_tensor(name) is False

    def test_substring_does_not_match(self) -> None:
        # Defensive: a hypothetical fused MoE name containing the substring
        # ``embed`` must not be mis-classified. The detector matches on the
        # exact tensor-suffix after the final dot, not raw substrings.
        assert is_residual_boundary_tensor("model.language_model.layers.5.mlp.embed_router.weight") is False
        assert is_residual_boundary_tensor("model.fancy_lm_head_proxy.weight") is False


class TestLayerIndexFromQwen3_5MoeNames:
    """``get_layer_index_from_name`` must handle the ``model.language_model.layers.N`` prefix.

    The Qwen3.5/3.6 multimodal layout nests the LM under
    ``model.language_model``; the unanchored ``layers\\.(\\d+)\\.`` regex
    already finds the index, but only after the upstream non-LM-stack
    filter has confirmed it isn't an MTP/vision tensor.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 0),
            ("model.language_model.layers.5.mlp.experts.gate_up_proj", 5),
            ("model.language_model.layers.39.mlp.shared_expert.gate_proj.weight", 39),
            ("model.layers.7.self_attn.o_proj.weight", 7),
        ],
    )
    def test_lm_layer_index_extraction(self, name: str, expected: int) -> None:
        assert get_layer_index_from_name(name) == expected

    def test_mtp_layer_index_is_not_confused_with_lm(self) -> None:
        # ``mtp.layers.0.*`` would otherwise match ``layers.0.`` and claim the LM
        # layer-0 refusal direction; the non-LM filter must reject it.
        assert get_layer_index_from_name("mtp.layers.0.mlp.experts.gate_up_proj") is None


class TestLayerTypeFromQwen3_5MoeNames:
    """``get_layer_type_from_name`` must canonicalize the new naming."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.language_model.layers.5.mlp.experts.gate_up_proj", "gate_up_proj"),
            ("model.language_model.layers.5.mlp.experts.down_proj", "down_proj"),
            ("model.language_model.layers.5.mlp.shared_expert.gate_proj.weight", "gate_proj"),
            ("model.language_model.layers.5.mlp.shared_expert.up_proj.weight", "up_proj"),
            ("model.language_model.layers.5.mlp.shared_expert.down_proj.weight", "down_proj"),
            ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "in_proj_qkv"),
            ("model.language_model.layers.0.linear_attn.in_proj_z.weight", "in_proj_z"),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", "out_proj"),
        ],
    )
    def test_canonical_layer_types(self, name: str, expected: str) -> None:
        assert get_layer_type_from_name(name) == expected


class TestQwen3_5MoeArchRegistry:
    """``VL_ARCHITECTURES`` and ``MOE_FAMILY_LAYOUT`` must register the new arch.

    Without these entries the loader picks ``AutoModelForCausalLM`` for an
    image-text-to-text class (likely failing) and the MoE shape-sniffing
    fallback picks the wrong fused-expert axis (silently mis-ablating).
    """

    def test_qwen3_5_moe_is_a_known_vl_arch(self) -> None:
        assert "qwen3_5_moe" in VL_ARCHITECTURES
        assert "Qwen3_5MoeForConditionalGeneration" in VL_ARCHITECTURES["qwen3_5_moe"]

    def test_qwen3_5_moe_layout_is_eoi(self) -> None:
        assert MOE_FAMILY_LAYOUT.get("qwen3_5_moe") == "eoi"
        assert MOE_FAMILY_LAYOUT.get("qwen3_5_moe_text") == "eoi"

    def test_qwen3_omni_moe_layout_is_eoi(self) -> None:
        assert MOE_FAMILY_LAYOUT.get("qwen3_omni_moe") == "eoi"
        assert MOE_FAMILY_LAYOUT.get("qwen3_omni_moe_text") == "eoi"


class TestQwen3_5MoeExpertAxes:
    """Per-expert 2-D slice axes for the fused Qwen3.5-MoE expert tensors.

    Reflects ``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts``:

        gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden))
        down_proj    = nn.Parameter(torch.empty(num_experts, hidden, intermediate))

    With ``F.linear(state, gate_up_proj[e])`` the second axis is
    ``out_features``, so the fused layout is ``[E, OUT, IN]`` (EOI).
    """

    # Real Qwen3.6-35B-A3B sizes: hidden=2048, moe_intermediate=512.
    GATE_UP_SHAPE = (256, 1024, 2048)  # [E, 2*moe_intermediate, hidden]
    DOWN_SHAPE = (256, 2048, 512)  # [E, hidden, moe_intermediate]

    def test_resolve_expert_axes_with_qwen3_5_eoi_hint(self) -> None:
        assert _resolve_expert_axes("gate_up_proj", self.GATE_UP_SHAPE, family_hint="eoi") == ("eoi", 1, 2)
        assert _resolve_expert_axes("down_proj", self.DOWN_SHAPE, family_hint="eoi") == ("eoi", 1, 2)

    def test_shape_only_fallback_misclassifies_down_proj(self) -> None:
        # Documents the failure mode the new MOE_FAMILY_LAYOUT entry prevents:
        # without the hint, down_proj's smaller ``in`` axis pulls the heuristic
        # to EIO, which would ablate the wrong axis and corrupt the MLP.
        assert _resolve_expert_axes("down_proj", self.DOWN_SHAPE, family_hint=None) == ("eio", 2, 1)

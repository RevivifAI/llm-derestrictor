"""Phase-0 math hardening regression tests for ``derestrictor.core.abliterate``.

Each test pins one of the six issues called out in the parity-kernels plan:

* H1 — Welford accumulator dtype tracks ``use_float64_subtraction``.
* H2 — Outlier clipping is reflected in the Welford means consumed downstream.
* H3 — Per-neuron projection's else-branch decomposes per-row (per-output-neuron).
* H4 — Square-weight ambiguity resolves via an explicit ``direction_space`` override.
* H5 — Cross-layer biprojection is orthogonal to ``ĥ_L`` after the two-pass step.
* H6 — ``min_quality_threshold`` filters candidates before top-N ranking.

These tests are pure-math: they construct small synthetic tensors and avoid
loading any real model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from derestrictor.core.abliterate import (
    AbliterationConfig,
    WelfordMeanAccumulator,
    apply_per_neuron_norm_preserving_projection,
    compute_biprojected_direction,
    compute_biprojection_mapping,
    infer_direction_space,
    select_biprojection_layers,
    winsorize_activations,
)


def _make_config(**overrides: object) -> AbliterationConfig:
    base = {
        "model_path": "/tmp/unused-model-path",
        "output_path": "/tmp/unused-output-path",
    }
    base.update(overrides)
    return AbliterationConfig(**base)  # type: ignore[arg-type]


def test_h1_welford_default_dtype_is_float64() -> None:
    """The accumulator must default to float64 to honor the precision policy."""
    acc = WelfordMeanAccumulator(hidden_dim=4)
    assert acc.dtype == torch.float64
    assert acc.mean.dtype == torch.float64


def test_h1_welford_preserves_subtraction_precision() -> None:
    """A float32 Welford collapses small residuals around a large mean; float64 keeps them.

    Construct samples directly in float64 with a base of ``1e7`` and per-sample
    residuals of ``±1e-3``. Float32 has ~7 decimal digits of precision, so a
    sample like ``1e7 + 1e-3`` truncates to ``1e7`` once cast — its residual
    is gone before Welford even sees it, and the means of the two halves end
    up identical. Float64 carries roughly 15 decimal digits and preserves the
    residual, so the per-half means must differ.
    """
    samples_a = torch.full((32, 4), 1.0e7, dtype=torch.float64) + 1.0e-3
    samples_b = torch.full((32, 4), 1.0e7, dtype=torch.float64) - 1.0e-3

    acc_f32 = WelfordMeanAccumulator(hidden_dim=4, dtype=torch.float32)
    acc_f32.update(samples_a)
    mean_a_f32 = acc_f32.get_mean().clone()
    acc_f32.reset()
    acc_f32.update(samples_b)
    mean_b_f32 = acc_f32.get_mean().clone()

    acc_f64 = WelfordMeanAccumulator(hidden_dim=4, dtype=torch.float64)
    acc_f64.update(samples_a)
    mean_a_f64 = acc_f64.get_mean().clone()
    acc_f64.reset()
    acc_f64.update(samples_b)
    mean_b_f64 = acc_f64.get_mean().clone()

    assert torch.all(mean_a_f32 == mean_b_f32), (
        f"f32 unexpectedly preserved the 1e-3 residual at magnitude 1e7: delta={(mean_a_f32 - mean_b_f32)}"
    )
    delta_f64 = mean_a_f64 - mean_b_f64
    assert torch.all(delta_f64 > 0), f"f64 lost the residual signal: delta={delta_f64}"


def test_h2_winsorize_returns_input_dtype_and_clips_outliers() -> None:
    """Winsorize promotes to f64 internally but returns the input dtype with values clipped."""
    activations = torch.randn(64, 8, dtype=torch.float32)
    activations[0, 0] = 1.0e6
    clipped = winsorize_activations(activations, percentile=0.95)
    assert clipped.dtype == torch.float32
    assert clipped.abs().max().item() < 1.0e3


def test_h3_per_neuron_projection_preserves_per_output_neuron_norms_output_space() -> None:
    """The output-space branch must preserve per-row (per-output-neuron) norms.

    Pre-fix the else-branch normalized per-column and silently preserved
    per-input-neuron norms — exactly opposite to the grimjim sample this kernel
    cites. Build a non-square weight so the bug cannot hide behind a square
    coincidence, drive an output-space ablation via ``direction_space``, and
    require row norms to be invariant to within float32 noise.
    """
    torch.manual_seed(0)
    out_features, in_features = 6, 12
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    refusal = F.normalize(torch.randn(out_features, dtype=torch.float32), dim=0)

    original_row_norms = weight.norm(dim=1)
    new_weight = apply_per_neuron_norm_preserving_projection(
        weight, refusal, scale_factor=1.0, direction_space="output"
    )
    assert new_weight.shape == weight.shape
    new_row_norms = new_weight.norm(dim=1)
    assert torch.allclose(new_row_norms, original_row_norms, atol=1e-5), (
        f"row norms drifted: max delta {(new_row_norms - original_row_norms).abs().max().item():.2e}"
    )


def test_h3_per_neuron_projection_zeroes_refusal_component_input_space() -> None:
    """Input-space ablation with scale=1 must zero the refusal projection per row."""
    torch.manual_seed(1)
    out_features, in_features = 8, 10
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    refusal = F.normalize(torch.randn(in_features, dtype=torch.float32), dim=0)

    new_weight = apply_per_neuron_norm_preserving_projection(weight, refusal, scale_factor=1.0, direction_space="input")
    residual = new_weight @ refusal
    assert residual.abs().max().item() < 1e-5


def test_h4_infer_direction_space_for_known_layer_types() -> None:
    """Known residual-writing and residual-reading layers must classify correctly."""
    assert infer_direction_space("model.layers.10.self_attn.o_proj.weight") == "output"
    assert infer_direction_space("model.layers.10.mlp.down_proj.weight") == "output"
    assert infer_direction_space("lm_head.weight") == "output"
    assert infer_direction_space("model.layers.10.self_attn.q_proj.weight") == "input"
    assert infer_direction_space("model.layers.10.mlp.up_proj.weight") == "input"
    assert infer_direction_space("model.embed_tokens.weight") is None


def test_h4_square_weight_with_direction_space_override_picks_output_axis() -> None:
    """For square weights both axes match the direction; the override must win.

    When ``hidden_size == num_heads * head_dim`` (Llama-style attention), an
    ``o_proj`` weight is square and refusal of length ``hidden_size`` matches
    both ``in_features`` and ``out_features``. The legacy shape heuristic
    always took the input-axis branch, silently corrupting ``o_proj``
    semantics. Pinning ``direction_space="output"`` must produce a materially
    different result from ``"input"`` and must measurably reduce the
    output-space refusal projection ``refusal @ W`` (per-row renormalization
    in the per-neuron kernel keeps the reduction approximate, not exact).
    """
    torch.manual_seed(2)
    dim = 8
    weight = torch.randn(dim, dim, dtype=torch.float32)
    refusal = F.normalize(torch.randn(dim, dtype=torch.float32), dim=0)

    out_branch = apply_per_neuron_norm_preserving_projection(
        weight, refusal, scale_factor=1.0, direction_space="output"
    )
    in_branch = apply_per_neuron_norm_preserving_projection(weight, refusal, scale_factor=1.0, direction_space="input")
    assert not torch.allclose(out_branch, in_branch, atol=1e-5), "override had no effect"

    output_residual_before = (refusal @ weight).norm().item()
    output_residual_after = (refusal @ out_branch).norm().item()
    assert output_residual_after < 0.5 * output_residual_before, (
        f"output-space ablation did not reduce refusal column projection: "
        f"before={output_residual_before:.3f} after={output_residual_after:.3f}"
    )

    input_residual_after = (in_branch @ refusal).norm().item()
    assert input_residual_after < 1e-4, (
        f"input-space override should have driven row projection to zero, got {input_residual_after:.2e}"
    )


def test_h5_compute_biprojected_direction_is_orthogonal_to_harmless() -> None:
    """After the two-pass step the biprojected direction must be orthogonal to ĥ_L."""
    torch.manual_seed(3)
    d = 16
    h_L = torch.randn(d)
    r_M = torch.randn(d)
    r_bi = compute_biprojected_direction(r_M, h_L, two_pass=True)
    h_L_hat = F.normalize(h_L, dim=0)
    dot = (r_bi @ h_L_hat).item()
    assert abs(dot) < 1e-6, f"biprojected direction not orthogonal to ĥ_L (dot={dot:.2e})"
    assert abs(r_bi.norm().item() - 1.0) < 1e-6


def test_h5_biprojection_zeroes_known_parallel_component() -> None:
    """A direction with a known parallel component must end up purely in the orthogonal one."""
    d = 8
    e1 = torch.zeros(d)
    e1[0] = 1.0
    e2 = torch.zeros(d)
    e2[1] = 1.0
    r_M = 0.7 * e1 + 0.3 * e2  # 0.7 along ĥ_L, 0.3 orthogonal
    r_bi = compute_biprojected_direction(r_M, harmless_mean=e1, two_pass=True)
    assert abs(r_bi[0].item()) < 1e-6
    assert abs(abs(r_bi[1].item()) - 1.0) < 1e-6


def test_h5_biprojection_mapping_nearest_picks_closest_measurement_layer() -> None:
    """The ``"nearest"`` policy must minimize ``|L − M|`` per intervention layer."""
    intervention = [0, 1, 2, 5, 8, 9, 10]
    measurement = [3, 7]
    mapping = compute_biprojection_mapping(intervention, measurement, "nearest")
    assert mapping == {0: 3, 1: 3, 2: 3, 5: 3, 8: 7, 9: 7, 10: 7}


def test_h5_biprojection_mapping_single_collapses_to_top_ranked() -> None:
    """The ``"single"`` policy must always pick the first measurement layer."""
    mapping = compute_biprojection_mapping([0, 5, 10], [7, 3], "single")
    assert mapping == {0: 7, 5: 7, 10: 7}


def test_h5_biprojection_mapping_explicit_dict_round_trips() -> None:
    """An explicit ``L→M`` dict must round-trip after dropping unmapped layers."""
    explicit = {1: 7, 5: 3, 99: 7}  # 99 not in intervention; dropped
    mapping = compute_biprojection_mapping([0, 1, 5], [3, 7], explicit)
    assert mapping == {1: 7, 5: 3}


def test_h6_min_quality_threshold_drops_low_quality_measurement_candidates() -> None:
    """Layers below ``min_quality_threshold`` must not survive into measurement."""
    quality_scores = {
        0: {
            "quality": 0.05,
            "snr": 0.1,
            "cos_sim": 0.5,
            "refusal_norm": 1.0,
            "harmful_norm": 2.0,
            "harmless_norm": 2.0,
            "layer_type": "unknown",
        },
        1: {
            "quality": 0.50,
            "snr": 0.6,
            "cos_sim": 0.2,
            "refusal_norm": 1.0,
            "harmful_norm": 2.0,
            "harmless_norm": 2.0,
            "layer_type": "unknown",
        },
        2: {
            "quality": 0.90,
            "snr": 0.9,
            "cos_sim": 0.0,
            "refusal_norm": 1.0,
            "harmful_norm": 2.0,
            "harmless_norm": 2.0,
            "layer_type": "unknown",
        },
    }
    measurement, _ = select_biprojection_layers(
        quality_scores,
        num_layers=4,
        num_measurement_layers=3,
        min_quality_threshold=0.4,
    )
    assert measurement == [2, 1]


def test_h6_min_quality_threshold_default_is_a_no_op() -> None:
    """The default ``0.0`` must keep the legacy top-N selection unchanged."""
    quality_scores = {
        i: {
            "quality": float(i) / 10,
            "snr": 0.5,
            "cos_sim": 0.0,
            "refusal_norm": 1.0,
            "harmful_norm": 2.0,
            "harmless_norm": 2.0,
            "layer_type": "unknown",
        }
        for i in range(5)
    }
    measurement, _ = select_biprojection_layers(quality_scores, num_layers=5, num_measurement_layers=2)
    assert measurement == [4, 3]


def test_config_dataclass_accepts_phase_0_fields() -> None:
    """``AbliterationConfig`` must accept all Phase-0 additions without error."""
    cfg = _make_config(
        use_biprojection=True,
        biprojection_mapping="nearest",
        use_direction_ensemble=False,
        min_quality_threshold=0.1,
        use_quality_selection=True,
    )
    assert cfg.biprojection_mapping == "nearest"
    assert cfg.use_direction_ensemble is False
    assert cfg.min_quality_threshold == 0.1

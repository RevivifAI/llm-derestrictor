"""Phase-1 parity-kernel regression tests for ``derestrictor.core.abliterate``.

Pinned features:

* Householder/Rodrigues isometric rotation kernel.
* Directional scaling kernel (full ablation, identity, amplification).
* Magnitude sparsification of the refusal direction.
* Two-pass Gram-Schmidt option in ``orthogonalize_against_harmless``.
* ``ablation_kernel`` selector dispatch in ``AbliterationConfig``.

All tests are pure-math: small synthetic tensors, no model loading.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from derestrictor.core.abliterate import (
    AbliterationConfig,
    apply_directional_scaling,
    apply_householder_rotation,
    magnitude_sparsify,
    orthogonalize_against_harmless,
)


def _make_config(**overrides: object) -> AbliterationConfig:
    base = {
        "model_path": "/tmp/unused-model-path",
        "output_path": "/tmp/unused-output-path",
    }
    base.update(overrides)
    return AbliterationConfig(**base)  # type: ignore[arg-type]


# Householder rotation


def test_householder_antipodal_full_reflection_negates_refusal_output_space() -> None:
    """At ``scale = 1.0`` antipodal Householder is a true reflection.

    The operation is ``W -= 2 (W * s) (x) s`` along the projected axis, so
    ``W_new @ s = -(W @ s)``. Magnitude is preserved (sign flipped). This
    pins the upstream ``jim-plus`` semantics literally: ``scale = 1.0`` is
    ``s -> -s`` rotation, not a projection. To actually zero the refusal
    column see ``test_householder_half_reflection_zeroes_refusal_*``.
    """
    torch.manual_seed(0)
    out_features, in_features = 6, 8
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    src = F.normalize(torch.randn(out_features, dtype=torch.float32), dim=0)

    expected_negated = -(src @ weight)
    new_weight = apply_householder_rotation(weight, src, tgt_dir=None, scale_factor=1.0, direction_space="output")
    assert new_weight.shape == weight.shape
    assert torch.allclose(src @ new_weight, expected_negated, atol=1e-4)


def test_householder_half_reflection_zeroes_refusal_output_space() -> None:
    """At ``scale = 0.5`` antipodal Householder collapses to the rank-1 projection.

    ``cos_t_m1 = -2`` so ``W += 0.5 * -2 * (W * s) (x) s = W - (W * s) (x) s``,
    which zeroes the s-component exactly. This is the operating point a
    user wants when actually ablating, hence the test pins it explicitly.
    """
    torch.manual_seed(1)
    weight = torch.randn(6, 8, dtype=torch.float32)
    src = F.normalize(torch.randn(6, dtype=torch.float32), dim=0)
    new_weight = apply_householder_rotation(weight, src, tgt_dir=None, scale_factor=0.5, direction_space="output")
    residual = (src @ new_weight).norm().item()
    assert residual < 1e-4, f"half-reflection did not zero refusal: {residual:.2e}"


def test_householder_antipodal_preserves_per_input_feature_norms_output_space() -> None:
    """Output-space rotation in PyTorch [Out, In] preserves per-INPUT-feature column norms.

    Upstream operates on a transposed [In, Out] view with the projected
    axis last; "row norms" in that view are per-input-feature column norms
    in our PyTorch [Out, In] layout. The renorm clamp is then a no-op
    because reflection is exactly isometric, so this test is sensitive to
    any sign or transpose error in the kernel.
    """
    torch.manual_seed(2)
    weight = torch.randn(6, 8, dtype=torch.float32)
    src = F.normalize(torch.randn(6, dtype=torch.float32), dim=0)
    original_col_norms = weight.norm(dim=0)
    new_weight = apply_householder_rotation(weight, src, tgt_dir=None, scale_factor=1.0, direction_space="output")
    new_col_norms = new_weight.norm(dim=0)
    assert torch.allclose(new_col_norms, original_col_norms, atol=1e-5), (
        f"per-input-feature column norms drifted: max delta "
        f"{(new_col_norms - original_col_norms).abs().max().item():.2e}"
    )


def test_householder_antipodal_preserves_per_output_neuron_norms_input_space() -> None:
    """Input-space rotation in PyTorch [Out, In] preserves per-OUTPUT-neuron row norms.

    No transpose happens for input-space; "rows" are per-output-neuron in
    PyTorch convention so direct ``norm(dim=1)`` is the right invariant.
    """
    torch.manual_seed(3)
    weight = torch.randn(5, 9, dtype=torch.float32)
    src = F.normalize(torch.randn(9, dtype=torch.float32), dim=0)
    original_row_norms = weight.norm(dim=1)
    new_weight = apply_householder_rotation(weight, src, tgt_dir=None, scale_factor=1.0, direction_space="input")
    new_row_norms = new_weight.norm(dim=1)
    assert torch.allclose(new_row_norms, original_row_norms, atol=1e-5)


def test_householder_identity_at_scale_zero() -> None:
    """``scale_factor = 0`` must be a no-op regardless of geometry."""
    torch.manual_seed(4)
    weight = torch.randn(4, 6, dtype=torch.float32)
    src = F.normalize(torch.randn(4, dtype=torch.float32), dim=0)
    new_weight = apply_householder_rotation(weight, src, scale_factor=0.0, direction_space="output")
    assert torch.allclose(new_weight, weight, atol=1e-6)


def test_householder_general_rotation_preserves_per_input_feature_norms() -> None:
    """A non-antipodal rotation between two arbitrary unit vectors stays isometric.

    Same axis convention as the antipodal case (per-input-feature column
    norms in our PyTorch [Out, In] layout) since the underlying upstream
    view is the same.
    """
    torch.manual_seed(5)
    out_features = 7
    weight = torch.randn(out_features, 5, dtype=torch.float32)
    src = F.normalize(torch.randn(out_features, dtype=torch.float32), dim=0)
    tgt = F.normalize(torch.randn(out_features, dtype=torch.float32), dim=0)

    original_col_norms = weight.norm(dim=0)
    new_weight = apply_householder_rotation(weight, src, tgt_dir=tgt, scale_factor=1.0, direction_space="output")
    new_col_norms = new_weight.norm(dim=0)
    assert torch.allclose(new_col_norms, original_col_norms, atol=1e-5)


# Directional scaling


def test_directional_scaling_full_ablation_zeroes_refusal_output_space() -> None:
    """``scale_factor = 1.0`` must zero ``W @ s`` to machine precision (double-tap)."""
    torch.manual_seed(10)
    weight = torch.randn(6, 8, dtype=torch.float32)
    s = F.normalize(torch.randn(6, dtype=torch.float32), dim=0)
    new_weight = apply_directional_scaling(weight, s, scale_factor=1.0, direction_space="output")
    residual = (s @ new_weight).norm().item()
    assert residual < 1e-5, f"refusal column projection not zeroed: {residual:.2e}"


def test_directional_scaling_identity_at_scale_zero() -> None:
    """``scale_factor = 0.0`` must round-trip the input weight (per-row renorm is a no-op)."""
    torch.manual_seed(11)
    weight = torch.randn(4, 6, dtype=torch.float32)
    s = F.normalize(torch.randn(4, dtype=torch.float32), dim=0)
    new_weight = apply_directional_scaling(weight, s, scale_factor=0.0, direction_space="output")
    assert torch.allclose(new_weight, weight, atol=1e-5)


def test_directional_scaling_amplifies_refusal_when_scale_negative() -> None:
    """``scale_factor = -1.0`` (induction / invert) must measurably increase the refusal projection.

    The kernel renormalizes per row, so the increase isn't an exact factor of
    two on the projection norm — instead each row keeps its original norm
    while the unit-vector direction tilts further toward ``s``.
    """
    torch.manual_seed(12)
    weight = torch.randn(6, 8, dtype=torch.float32)
    s = F.normalize(torch.randn(6, dtype=torch.float32), dim=0)

    before = (s @ weight).norm().item()
    new_weight = apply_directional_scaling(weight, s, scale_factor=-1.0, direction_space="output")
    after = (s @ new_weight).norm().item()
    assert after > before, f"amplification did not increase refusal projection: before={before:.3f} after={after:.3f}"


def test_directional_scaling_preserves_per_input_feature_norms_output_space() -> None:
    """The double-tap renorm clamp must restore each input-feature column norm.

    Output-space directional scaling transposes to [In, Out] (matching the
    upstream view), so the per-row renorm there corresponds to per-INPUT-
    feature column norms in our PyTorch [Out, In] layout.
    """
    torch.manual_seed(13)
    weight = torch.randn(8, 5, dtype=torch.float32)
    s = F.normalize(torch.randn(8, dtype=torch.float32), dim=0)
    original_norms = weight.norm(dim=0)
    new_weight = apply_directional_scaling(weight, s, scale_factor=1.0, direction_space="output")
    assert torch.allclose(new_weight.norm(dim=0), original_norms, atol=1e-5)


def test_directional_scaling_preserves_per_output_neuron_norms_input_space() -> None:
    """For input-space refusal the kernel preserves per-output-neuron row norms.

    No transpose happens for input-space ablation; the renorm therefore
    targets PyTorch row norms directly. Pinning both axes prevents future
    refactors from accidentally collapsing the two cases.
    """
    torch.manual_seed(14)
    weight = torch.randn(8, 5, dtype=torch.float32)
    s = F.normalize(torch.randn(5, dtype=torch.float32), dim=0)
    original_norms = weight.norm(dim=1)
    new_weight = apply_directional_scaling(weight, s, scale_factor=1.0, direction_space="input")
    assert torch.allclose(new_weight.norm(dim=1), original_norms, atol=1e-5)


# Magnitude sparsification


def test_magnitude_sparsify_keeps_exact_count_at_half() -> None:
    """``fraction = 0.5`` must keep exactly ``floor(numel / 2)`` non-zero entries."""
    torch.manual_seed(20)
    t = torch.randn(64)
    out = magnitude_sparsify(t, 0.5)
    assert int((out != 0).sum().item()) == 32


def test_magnitude_sparsify_keeps_largest_magnitudes() -> None:
    """The retained entries must be exactly the top-k by absolute magnitude."""
    t = torch.tensor([0.1, -5.0, 0.2, 3.0, -0.05, 4.0])
    out = magnitude_sparsify(t, 0.5)  # keep top 3
    assert out.tolist() == [0.0, -5.0, 0.0, 3.0, 0.0, 4.0]


def test_magnitude_sparsify_zero_fraction_returns_zeros() -> None:
    """A ``fraction <= 0`` must return an all-zero tensor of the same shape."""
    t = torch.randn(10)
    out = magnitude_sparsify(t, 0.0)
    assert out.shape == t.shape
    assert torch.all(out == 0)


def test_magnitude_sparsify_full_fraction_is_identity() -> None:
    """``fraction >= 1.0`` must return the input unchanged."""
    t = torch.randn(10)
    out = magnitude_sparsify(t, 1.0)
    assert torch.equal(out, t)


# Two-pass Gram-Schmidt


def test_two_pass_orthogonalization_smaller_residual_than_single_pass() -> None:
    """Two-pass must leave a strictly smaller (or equal) residual ``r·h_hat`` than one-pass.

    Construct a deliberately ill-conditioned pair where harmful and harmless
    means are nearly parallel (cosine similarity ~ 1). One pass of the
    subtraction leaves a residual proportional to the float epsilon of the
    near-cancellation; a second pass kills it.
    """
    torch.manual_seed(30)
    d = 32
    base = torch.randn(d, dtype=torch.float32)
    base = F.normalize(base, dim=0)
    perturb = 1e-7 * torch.randn(d, dtype=torch.float32)
    harmful = base * 1.0 + perturb
    harmless = base * 1.0
    refusal = harmful - harmless

    h_hat = F.normalize(harmless, dim=0)

    one = orthogonalize_against_harmless(refusal, harmless, two_pass=False)
    two = orthogonalize_against_harmless(refusal, harmless, two_pass=True)

    res_one = abs((one @ h_hat).item())
    res_two = abs((two @ h_hat).item())
    assert res_two <= res_one
    assert res_two < 1e-7


# Config dispatch


def test_config_accepts_phase_1_fields_and_defaults_preserve_legacy_behavior() -> None:
    """The new fields must accept Phase-1 values and keep legacy defaults at the prior values."""
    cfg = _make_config(
        ablation_kernel="householder",
        invert_ablation=True,
        direction_sparsity=0.01,
        per_layer_sparsity={35: 0.001, 36: 0.001},
        two_pass_orthogonalization=False,
        token_position="second_generated",
    )
    assert cfg.ablation_kernel == "householder"
    assert cfg.invert_ablation is True
    assert cfg.direction_sparsity == 0.01
    assert cfg.per_layer_sparsity == {35: 0.001, 36: 0.001}
    assert cfg.two_pass_orthogonalization is False
    assert cfg.token_position == "second_generated"

    default_cfg = _make_config()
    assert default_cfg.ablation_kernel is None
    assert default_cfg.invert_ablation is False
    assert default_cfg.direction_sparsity == 0.0
    assert default_cfg.per_layer_sparsity is None
    assert default_cfg.two_pass_orthogonalization is True
    assert default_cfg.token_position == "last"

"""Pure-math regression tests for the heretic benign parity set (WEB-1042).

Covers the flexible per-component ablation weight kernel
(:class:`WeightDistribution`) and the continuous residual-direction
interpolation added in :mod:`derestrictor.core.heretic_kernel`, plus their
wiring into :func:`derestrictor.core.abliterate.resolve_ablation_for_tensor`.

No model or GPU is required: every case uses small synthetic tensors.
"""

import pytest
import torch

from derestrictor.core.abliterate import (
    AbliterationConfig,
    RefusalDirections,
    resolve_ablation_for_tensor,
)
from derestrictor.core.heretic_kernel import (
    COMPONENT_ATTENTION,
    COMPONENT_MLP,
    WeightDistribution,
    component_for_layer_type,
    interpolate_direction,
    resolve_component_weight,
)


def test_weight_distribution_peak_and_linear_decay():
    dist = WeightDistribution(max_weight=1.0, max_weight_position=4.0, min_weight=0.2, min_weight_distance=4.0)

    assert dist.weight_at(4) == pytest.approx(1.0)
    assert dist.weight_at(6) == pytest.approx(0.6)
    assert dist.weight_at(2) == pytest.approx(0.6)
    assert dist.weight_at(8) == pytest.approx(0.2)
    assert dist.weight_at(0) == pytest.approx(0.2)


def test_weight_distribution_skips_beyond_distance():
    dist = WeightDistribution(max_weight=1.0, max_weight_position=5.0, min_weight=0.0, min_weight_distance=2.0)

    assert dist.weight_at(5) == pytest.approx(1.0)
    assert dist.weight_at(7) == pytest.approx(0.0)
    assert dist.weight_at(3) == pytest.approx(0.0)
    assert dist.weight_at(8) == pytest.approx(0.0)
    assert dist.weight_at(0) == pytest.approx(0.0)


def test_weight_distribution_zero_max_weight_disables_component():
    dist = WeightDistribution(max_weight=0.0, max_weight_position=3.0, min_weight=0.0, min_weight_distance=5.0)

    assert dist.weight_at(3) == pytest.approx(0.0)
    assert dist.weight_at(1) == pytest.approx(0.0)


def test_weight_distribution_nonpositive_distance_is_inert():
    dist = WeightDistribution(max_weight=1.0, max_weight_position=0.0, min_weight=1.0, min_weight_distance=0.0)

    assert dist.weight_at(0) == pytest.approx(0.0)
    assert dist.weight_at(5) == pytest.approx(0.0)


def test_component_for_layer_type_maps_known_types():
    assert component_for_layer_type("o_proj") == COMPONENT_ATTENTION
    assert component_for_layer_type("out_proj") == COMPONENT_ATTENTION
    assert component_for_layer_type("down_proj") == COMPONENT_MLP
    assert component_for_layer_type("w2") == COMPONENT_MLP
    assert component_for_layer_type("fc2") == COMPONENT_MLP
    assert component_for_layer_type("output_linear") == COMPONENT_MLP


def test_component_for_layer_type_rejects_non_abliterable():
    assert component_for_layer_type(None) is None
    assert component_for_layer_type("q_proj") is None
    assert component_for_layer_type("gate_proj") is None


def test_resolve_component_weight_returns_none_when_not_configured():
    distributions = {COMPONENT_ATTENTION: WeightDistribution(max_weight=1.0, min_weight_distance=10.0)}

    assert resolve_component_weight(distributions, "o_proj", None) is None
    assert resolve_component_weight(distributions, "q_proj", 0) is None
    assert resolve_component_weight(distributions, "down_proj", 0) is None
    assert resolve_component_weight(distributions, "o_proj", 0) == pytest.approx(1.0)


def test_interpolate_direction_returns_exact_layers():
    d0 = torch.tensor([1.0, 0.0])
    d1 = torch.tensor([0.0, 1.0])
    directions = {0: d0, 1: d1}

    assert torch.equal(interpolate_direction(directions, 0.0), d0)
    assert torch.equal(interpolate_direction(directions, 1.0), d1)


def test_interpolate_direction_midpoint_is_normalized_lerp():
    directions = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}

    midpoint = interpolate_direction(directions, 0.5)

    assert torch.allclose(midpoint, torch.tensor([2.0**-0.5, 2.0**-0.5]), atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(midpoint), torch.tensor(1.0), atol=1e-6)


def test_interpolate_direction_clamps_out_of_range():
    d0 = torch.tensor([1.0, 0.0])
    d1 = torch.tensor([0.0, 1.0])
    directions = {0: d0, 1: d1}

    assert torch.equal(interpolate_direction(directions, -3.0), d0)
    assert torch.equal(interpolate_direction(directions, 9.0), d1)


def test_interpolate_direction_handles_layer_gaps():
    d2 = torch.tensor([1.0, 0.0])
    d5 = torch.tensor([0.0, 1.0])
    directions = {2: d2, 5: d5}

    assert torch.equal(interpolate_direction(directions, 2.0), d2)
    assert torch.equal(interpolate_direction(directions, 5.0), d5)
    assert torch.allclose(
        interpolate_direction(directions, 3.5),
        torch.tensor([2.0**-0.5, 2.0**-0.5]),
        atol=1e-6,
    )


def test_interpolate_direction_empty_raises():
    with pytest.raises(ValueError):
        interpolate_direction({}, 0.5)


def test_config_heretic_fields_default_off():
    config = AbliterationConfig(model_path="m", output_path="o")

    assert config.heretic_weight_distributions is None
    assert config.heretic_direction_index is None


def _heretic_config(**overrides) -> AbliterationConfig:
    base = {
        "model_path": "m",
        "output_path": "o",
        "device": "cpu",
        "dtype": torch.float32,
        "heretic_weight_distributions": {
            COMPONENT_ATTENTION: WeightDistribution(
                max_weight=1.0, max_weight_position=0.0, min_weight=0.0, min_weight_distance=1.0
            ),
            COMPONENT_MLP: WeightDistribution(
                max_weight=0.0, max_weight_position=0.0, min_weight=0.0, min_weight_distance=1.0
            ),
        },
    }
    base.update(overrides)
    return AbliterationConfig(**base)


def _directions() -> RefusalDirections:
    direction = torch.tensor([1.0, 0.0, 0.0])
    return RefusalDirections(directions={0: direction}, mean_direction=direction.clone())


def test_resolve_applies_attention_heretic_weight():
    ctx = resolve_ablation_for_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        (3, 3),
        directions=_directions(),
        config=_heretic_config(),
    )

    assert not ctx.skip
    assert ctx.multiplier == pytest.approx(1.0)


def test_resolve_skips_mlp_when_heretic_weight_is_zero():
    ctx = resolve_ablation_for_tensor(
        "model.layers.0.mlp.down_proj.weight",
        (3, 3),
        directions=_directions(),
        config=_heretic_config(),
    )

    assert ctx.skip
    assert ctx.multiplier == pytest.approx(0.0)

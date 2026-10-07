"""Heretic-style parametric ablation kernel and residual-direction interpolation.

This module implements the benign algorithmic capability set of
`p-e-w/heretic <https://github.com/p-e-w/heretic>`_:

* A flexible per-component ablation weight distribution over transformer
  layers, defined by ``max_weight``, ``max_weight_position``, ``min_weight``
  and ``min_weight_distance`` (see heretic's ``modifiers/abliteration.py``).
* A continuous ``direction_index`` that linearly interpolates between the
  residual directions of the two nearest layers and renormalizes the result.

Heretic's abuse-oriented extras are intentionally not implemented here: this
module never sources directions from a genuinely harmful prompt set, and its
arbitrary-rank behaviour steering (ARA) is out of scope. See the WEB-1042
capability inventory for the full parity split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Canonical heretic component keys. Heretic ablates the attention
# out-projection and the MLP down-projection.
COMPONENT_ATTENTION = "attn.o_proj"
COMPONENT_MLP = "mlp.down_proj"

# Layer-type substrings, as returned by
# :func:`derestrictor.core.abliterate.get_layer_type_from_name`, mapped to the
# heretic component they belong to.
_ATTENTION_LAYER_TYPES = frozenset({"o_proj", "out_proj"})
_MLP_LAYER_TYPES = frozenset({"down_proj", "w2", "fc2", "output_linear"})


@dataclass(frozen=True)
class WeightDistribution:
    """Heretic's per-component ablation weight shape over layers.

    Mirrors ``heretic.modifiers.abliteration.WeightDistribution``. The weight
    for a layer is ``max_weight`` at ``max_weight_position`` and decays
    linearly to ``min_weight`` at distance ``min_weight_distance``. Layers
    farther than ``min_weight_distance`` are skipped (weight ``0.0``).

    Attributes:
        max_weight: Peak ablation weight. ``0.0`` disables the component.
        max_weight_position: Layer index (float) of the peak weight.
        min_weight: Ablation weight at exactly ``min_weight_distance``.
        min_weight_distance: Distance in layers over which the weight decays
            from ``max_weight`` to ``min_weight``.
    """

    max_weight: float = 1.0
    max_weight_position: float = 0.0
    min_weight: float = 0.0
    min_weight_distance: float = 1.0

    def weight_at(self, layer_index: int) -> float:
        """Return the ablation weight for ``layer_index``.

        Args:
            layer_index: Transformer layer index.

        Returns:
            The interpolated weight, or ``0.0`` when the layer is farther than
            ``min_weight_distance`` from ``max_weight_position``.
        """
        distance = abs(float(layer_index) - self.max_weight_position)
        if self.min_weight_distance <= 0.0 or distance > self.min_weight_distance:
            return 0.0
        fraction = distance / self.min_weight_distance
        return self.max_weight + fraction * (self.min_weight - self.max_weight)


def component_for_layer_type(layer_type: str | None) -> str | None:
    """Map a derestrictor layer type to a heretic component key.

    Args:
        layer_type: Sublayer name from
            :func:`derestrictor.core.abliterate.get_layer_type_from_name`.

    Returns:
        ``"attn.o_proj"``, ``"mlp.down_proj"``, or ``None`` when the layer
        type is not an abliterable heretic component.
    """
    if layer_type is None:
        return None
    normalized = layer_type.lower()
    if normalized in _ATTENTION_LAYER_TYPES:
        return COMPONENT_ATTENTION
    if normalized in _MLP_LAYER_TYPES:
        return COMPONENT_MLP
    return None


def resolve_component_weight(
    distributions: dict[str, WeightDistribution],
    layer_type: str | None,
    layer_index: int | None,
) -> float | None:
    """Resolve the heretic weight for one tensor.

    Args:
        distributions: Per-component weight distributions.
        layer_type: Sublayer name of the tensor.
        layer_index: Layer index of the tensor, or ``None`` when unknown.

    Returns:
        The weight multiplier, or ``None`` when the tensor's component has no
        configured distribution so the caller leaves the multiplier unchanged.
    """
    if layer_index is None:
        return None
    component = component_for_layer_type(layer_type)
    if component is None:
        return None
    distribution = distributions.get(component)
    if distribution is None:
        return None
    return distribution.weight_at(layer_index)


def interpolate_direction(
    directions: dict[int, torch.Tensor],
    index: float,
) -> torch.Tensor:
    """Interpolate a residual direction at a continuous layer index.

    Mirrors heretic's float ``direction_index``: the two nearest available
    layer directions are linearly interpolated and the result is L2
    normalized. Out-of-range indices clamp to the nearest endpoint.

    Args:
        directions: Mapping of layer index to a residual direction tensor.
        index: Continuous layer position.

    Returns:
        The normalized interpolated direction, in the dtype of the bracketing
        tensors.

    Raises:
        ValueError: If ``directions`` is empty.
    """
    if not directions:
        raise ValueError("Cannot interpolate a direction from an empty mapping")

    layers = sorted(directions)
    if index <= layers[0]:
        return directions[layers[0]]
    if index >= layers[-1]:
        return directions[layers[-1]]

    lower = max(layer for layer in layers if layer <= index)
    upper = min(layer for layer in layers if layer >= index)
    if lower == upper:
        return directions[lower]

    fraction = (index - lower) / (upper - lower)
    lower_vec = directions[lower]
    upper_vec = directions[upper]
    mixed = torch.lerp(lower_vec.float(), upper_vec.float(), fraction)
    return F.normalize(mixed, p=2, dim=0).to(lower_vec.dtype)

"""Smoke tests for the reorganised ``derestrictor`` package.

These tests are deliberately light:

* they verify that the new src-layout imports wire up end-to-end, and
* they exercise one pure-math invariant from the core algorithms.

They must NOT load real Hugging Face models or touch the network.
"""

from __future__ import annotations

import importlib

import pytest
import torch


def test_package_import() -> None:
    """The top-level package must be importable and carry a version string."""
    pkg = importlib.import_module("derestrictor")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__  # non-empty


@pytest.mark.parametrize(
    "module",
    [
        "derestrictor.core.abliterate",
        "derestrictor.core.null_space",
        "derestrictor.core.feature_surgery",
        "derestrictor.core.kl_monitor",
        "derestrictor.cli.main",
        "derestrictor.cli.components",
        "derestrictor.cli.feature_surgery",
        "derestrictor.config.manager",
        "derestrictor.data.loader",
        "derestrictor.data.harm_filter",
        "derestrictor.models.utils",
        "derestrictor.eval.detector",
        "derestrictor.eval.scanner",
        "derestrictor.eval.calibration",
        "derestrictor.eval.compare",
        "derestrictor.eval.harness",
        "derestrictor.export.gguf",
        "derestrictor.sae.loader",
        "derestrictor.scripts.abliteration_search",
        "derestrictor.scripts.batch_quantize",
        "derestrictor.scripts.build_dataset",
        "derestrictor.scripts.upload_dataset",
        "derestrictor.scripts.convert_to_bf16",
    ],
)
def test_submodule_imports(module: str) -> None:
    """Every first-party submodule in the new layout must import cleanly."""
    importlib.import_module(module)


def test_orthogonal_projection_is_idempotent() -> None:
    """Projecting W onto the hyperplane orthogonal to a unit direction d is idempotent.

    Given a unit vector ``d`` and a weight matrix ``W``, the projection
    ``P(W) = W - (W @ d) ⊗ d`` removes the component of each row aligned with
    ``d``. Applying ``P`` twice must equal applying it once, and every row of
    ``P(W)`` must be orthogonal to ``d``.
    """
    torch.manual_seed(0)
    d_in, d_out = 16, 8
    d = torch.randn(d_in)
    d = d / d.norm()
    W = torch.randn(d_out, d_in)

    def project(weights: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        coeffs = weights @ direction  # [d_out]
        return weights - torch.outer(coeffs, direction)

    W1 = project(W, d)
    W2 = project(W1, d)

    assert torch.allclose(W1, W2, atol=1e-6), "projection should be idempotent"

    residual = W1 @ d
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6), (
        "every row of the projected matrix must be orthogonal to d"
    )

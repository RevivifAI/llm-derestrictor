"""Shared pytest fixtures and markers for the derestrictor test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - torch is a required dep, guard is defensive
    torch = None  # type: ignore[assignment]


@pytest.fixture
def hf_offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force Hugging Face libraries into offline mode for the duration of a test.

    Prevents tests from accidentally reaching out to the Hub. The env vars are
    restored automatically by ``monkeypatch`` at teardown.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked ``requires_cuda`` when no CUDA device is available."""
    _ = config
    cuda_available = torch is not None and torch.cuda.is_available()

    if cuda_available or os.environ.get("DERESTRICTOR_FORCE_CUDA_TESTS") == "1":
        return

    skip_cuda = pytest.mark.skip(reason="CUDA device not available")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)

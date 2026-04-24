"""Tests for ``ClassifierRefusalDetector`` and the auto-fallback factory.

These tests monkeypatch ``transformers.pipeline`` (referenced via
``derestrictor.eval.detector.pipeline``) so no real classifier is
downloaded. They pin:

* Classifier input is the generated continuation, not the input prompt.
* ``ClassifierUnavailableError`` propagates on load failure.
* :func:`create_refusal_detector` falls back to
  :class:`LogLikelihoodRefusalDetector` when the classifier is
  unavailable.
* Basic plumbing of ``max_new_tokens`` / ``threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from derestrictor.eval import detector as detector_mod
from derestrictor.eval.detector import (
    ClassifierDetectorConfig,
    ClassifierRefusalDetector,
    ClassifierUnavailableError,
    LogLikelihoodRefusalDetector,
    create_refusal_detector,
)


class _Batch(dict):
    """Dict subclass with ``input_ids`` attribute and a no-op ``to()``."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device: str) -> _Batch:
        return self


@dataclass
class _FakeTokenizer:
    pad_token: str | None = "<pad>"
    eos_token: str = "<eos>"
    padding_side: str = "right"
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat_template(self) -> str:
        return "no-think-here"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return f"<|prompt|>{messages[-1]['content']}"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) % 97 for c in text[:4]] or [1]

    @property
    def pad_token_id(self) -> int:
        return 0

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> Any:
        del return_tensors, padding, truncation
        max_len = max(len(t) for t in texts)
        ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, t in enumerate(texts):
            for j, c in enumerate(t):
                ids[i, j] = ord(c) % 97

        return _Batch(ids)

    def batch_decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return [f"GEN_{int(row.sum().item())}" for row in ids]


class _FakeModel:
    """Generation-capable model stub."""

    def __init__(self, new_token_ids: list[int] | None = None) -> None:
        self.device = "cpu"
        self.new_token_ids = new_token_ids or [7, 8, 9]

    def generate(self, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        batch = input_ids.shape[0]
        max_new = kwargs["max_new_tokens"]
        new = torch.tensor(self.new_token_ids[:max_new], dtype=torch.long).repeat(batch, 1)
        return torch.cat([input_ids, new], dim=1)


class _FakePipeline:
    """Stand-in for a text-classification pipeline.

    Returns per-input ``[{"label": "REJECTION", "score": r}, {"label": "NORMAL", "score": 1-r}]``
    where ``r`` is pulled from ``scores`` by input index, or a fixed
    default.
    """

    def __init__(self, scores: list[float] | None = None, default: float = 0.9) -> None:
        self.scores = scores
        self.default = default
        self.inputs: list[list[str]] = []
        self.batch_sizes: list[int] = []

    def __call__(self, texts: list[str], *, batch_size: int = 1) -> list[list[dict[str, Any]]]:
        self.inputs.append(list(texts))
        self.batch_sizes.append(batch_size)
        out: list[list[dict[str, Any]]] = []
        for i, _ in enumerate(texts):
            r = self.scores[i] if self.scores is not None and i < len(self.scores) else self.default
            out.append([
                {"label": "REJECTION", "score": r},
                {"label": "NORMAL", "score": 1.0 - r},
            ])
        return out


def test_classifier_sees_generated_text_not_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classifier is fed the model's generated continuation, not the prompt."""
    fake_pipe = _FakePipeline(default=0.95)

    def _fake_pipeline_factory(*_args: Any, **_kwargs: Any) -> _FakePipeline:
        return fake_pipe

    monkeypatch.setattr(detector_mod, "pipeline", _fake_pipeline_factory)

    det = ClassifierRefusalDetector(_FakeModel(), _FakeTokenizer())
    results = det.detect_refusal_with_scores(["are you ok?"])

    assert len(results) == 1
    is_refusal, score = results[0]
    assert is_refusal is True
    assert score == pytest.approx(0.95)
    assert fake_pipe.inputs, "classifier was never invoked"
    assert all(text.startswith("GEN_") for batch in fake_pipe.inputs for text in batch), (
        f"classifier got non-generated text: {fake_pipe.inputs!r}"
    )


def test_threshold_controls_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scores below threshold yield non-refusal."""
    fake_pipe = _FakePipeline(scores=[0.2, 0.8])
    monkeypatch.setattr(detector_mod, "pipeline", lambda *_a, **_kw: fake_pipe)

    det = ClassifierRefusalDetector(
        _FakeModel(),
        _FakeTokenizer(),
        ClassifierDetectorConfig(threshold=0.5, batch_size=2),
    )

    results = det.detect_refusal_with_scores(["a", "b"])

    assert [r for r, _ in results] == [False, True]


def test_classifier_unavailable_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load failures surface as ClassifierUnavailableError."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("offline / not cached")

    monkeypatch.setattr(detector_mod, "pipeline", _boom)

    det = ClassifierRefusalDetector(_FakeModel(), _FakeTokenizer())
    with pytest.raises(ClassifierUnavailableError):
        det.ensure_classifier_loaded()


def test_create_refusal_detector_falls_back_to_loglik(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory returns a log-likelihood detector when the classifier won't load."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no network")

    monkeypatch.setattr(detector_mod, "pipeline", _boom)

    det = create_refusal_detector(_FakeModel(), _FakeTokenizer())

    assert isinstance(det, LogLikelihoodRefusalDetector)


def test_create_refusal_detector_uses_classifier_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory returns the classifier detector on successful load."""
    fake_pipe = _FakePipeline()
    monkeypatch.setattr(detector_mod, "pipeline", lambda *_a, **_kw: fake_pipe)

    det = create_refusal_detector(_FakeModel(), _FakeTokenizer())

    assert isinstance(det, ClassifierRefusalDetector)


def test_create_refusal_detector_prefer_classifier_false_skips_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """prefer_classifier=False returns log-likelihood detector without probing classifier."""
    calls = {"n": 0}

    def _track(*_args: Any, **_kwargs: Any) -> Any:
        calls["n"] += 1
        return _FakePipeline()

    monkeypatch.setattr(detector_mod, "pipeline", _track)

    det = create_refusal_detector(_FakeModel(), _FakeTokenizer(), prefer_classifier=False)

    assert isinstance(det, LogLikelihoodRefusalDetector)
    assert calls["n"] == 0

"""Tests for thinking-mode handling in ``LogLikelihoodRefusalDetector``.

These tests use fake tokenizer/model stand-ins so they stay on CPU, do
not touch the network, and do not load any real HuggingFace weights.
They pin the detector's behavior on three axes:

* Chat-template ``enable_thinking`` handling (passed / not-passed /
  legacy-tokenizer signature mismatch).
* Inclusion of ``<think>\\n``-prefixed anchors for reasoning templates.
* Graceful fallback when the tokenizer lacks the reasoning-template
  surface entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from derestrictor.eval.detector import (
    DEFAULT_THINKING_REFUSAL_ANCHORS,
    LogLikelihoodRefusalDetector,
    RefusalDetectorConfig,
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
    """Minimal tokenizer with a configurable chat-template surface."""

    supports_enable_thinking: bool = True
    template_has_think_tag: bool = True
    pad_token: str | None = "<pad>"
    eos_token: str = "<eos>"
    padding_side: str = "right"
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat_template(self) -> str:
        return "{{<think>\n}}" if self.template_has_think_tag else "{{user}}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        if "enable_thinking" in kwargs and not self.supports_enable_thinking:
            raise TypeError("apply_chat_template() got unexpected kwarg 'enable_thinking'")
        self.calls.append({
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        })
        enable_thinking = kwargs.get("enable_thinking", True)
        marker = "THINK_ON" if enable_thinking else "THINK_OFF"
        content = messages[-1]["content"]
        return f"[{marker}|gen={add_generation_prompt}] {content}\n"

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


class _FakeModel:
    """Model stub returning random-but-deterministic logits on CPU."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.device = "cpu"

    def __call__(self, **kwargs: Any) -> Any:
        input_ids = kwargs["input_ids"]
        batch, seq = input_ids.shape
        generator = torch.Generator().manual_seed(int(input_ids.sum().item()) % 2**31)
        logits = torch.randn(batch, seq, self.vocab_size, generator=generator)
        return SimpleNamespace(logits=logits)


def test_format_prompt_disables_thinking_when_supported() -> None:
    """With reasoning tokenizer + default config, enable_thinking=False is passed."""
    tok = _FakeTokenizer(supports_enable_thinking=True, template_has_think_tag=True)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    out = det.format_prompt("hello")

    assert "THINK_OFF" in out
    assert tok.calls[-1]["enable_thinking"] is False


def test_format_prompt_opt_out_of_disable_thinking() -> None:
    """Setting disable_thinking=False preserves the template's default."""
    tok = _FakeTokenizer(supports_enable_thinking=True, template_has_think_tag=True)
    config = RefusalDetectorConfig(disable_thinking=False)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok, config)

    out = det.format_prompt("hello")

    assert "THINK_ON" in out
    assert "enable_thinking" not in tok.calls[-1]


def test_format_prompt_legacy_tokenizer_falls_back_cleanly() -> None:
    """Tokenizer that rejects enable_thinking falls back to plain template."""
    tok = _FakeTokenizer(supports_enable_thinking=False, template_has_think_tag=False)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    out = det.format_prompt("hello")

    assert "THINK_ON" in out
    assert all("enable_thinking" not in call for call in tok.calls)


def test_reasoning_template_unions_thinking_anchors() -> None:
    """Reasoning templates widen the scored anchor set with <think> anchors."""
    tok = _FakeTokenizer(supports_enable_thinking=True, template_has_think_tag=True)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    anchors = det._effective_anchors()

    for t in DEFAULT_THINKING_REFUSAL_ANCHORS:
        assert t in anchors


def test_non_reasoning_template_only_scores_base_anchors() -> None:
    """Non-reasoning tokenizers score only the base anchors."""
    tok = _FakeTokenizer(supports_enable_thinking=False, template_has_think_tag=False)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    anchors = det._effective_anchors()

    assert all(not a.startswith("<think>") for a in anchors)


def test_template_string_only_still_treated_as_reasoning() -> None:
    """Legacy tokenizer whose raw template contains <think> is reasoning-like."""
    tok = _FakeTokenizer(supports_enable_thinking=False, template_has_think_tag=True)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    anchors = det._effective_anchors()

    for t in DEFAULT_THINKING_REFUSAL_ANCHORS:
        assert t in anchors


@pytest.mark.parametrize("supports", [True, False])
def test_detect_refusal_with_scores_runs_end_to_end(supports: bool) -> None:
    """Detector smoke-runs on both reasoning and non-reasoning tokenizers."""
    tok = _FakeTokenizer(supports_enable_thinking=supports, template_has_think_tag=supports)
    det = LogLikelihoodRefusalDetector(_FakeModel(), tok)

    results = det.detect_refusal_with_scores(["prompt A", "prompt B"])

    assert len(results) == 2
    assert all(isinstance(r, bool) and isinstance(s, float) for r, s in results)

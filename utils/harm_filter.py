"""Harm filter: drops harmful prompts that mention or target children.

Applied unconditionally to the ``harmful`` split of RevivifAI/derestriction
before sampling. Runs two stages:

1. Broad keyword regex covering child-adjacent terms (word-boundary aware).
2. Semantic cosine similarity against child-harm anchor prompts, using
   ``sentence-transformers/all-MiniLM-L6-v2``.

The filter is intentionally aggressive — we err heavily on the side of
dropping borderline prompts rather than letting a child-adjacent prompt slip
into the dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_KEYWORD_PATTERNS: list[str] = [
    r"child(?:ren)?",
    r"\bkid\b",
    r"\bkids\b",
    r"\bminor\b",
    r"\bminors\b",
    r"underage",
    r"toddler",
    r"toddlers",
    r"infant",
    r"infants",
    r"\bbaby\b",
    r"\bbabies\b",
    r"preschool",
    r"kindergarten",
    r"elementary[- ]school",
    r"\bpupil\b",
    r"\bpupils\b",
    r"\bteen\b",
    r"\bteens\b",
    r"teenager",
    r"teenagers",
    r"\byouth\b",
    r"\byouths\b",
    r"juvenile",
    r"juveniles",
    r"adolescent",
    r"adolescents",
    r"\bstudent\b",
    r"\bstudents\b",
    r"schoolchild(?:ren)?",
    r"\bboy\b",
    r"\bboys\b",
    r"\bgirl\b",
    r"\bgirls\b",
    r"\bson\b",
    r"\bsons\b",
    r"\bdaughter\b",
    r"\bdaughters\b",
    r"pedophil(?:e|es|ia|ic)",
    r"paedophil(?:e|es|ia|ic)",
    r"\bCSAM\b",
    r"grooming",
    r"pre[- ]teen",
    r"pre[- ]teens",
]

_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS), re.IGNORECASE)


_SEMANTIC_ANCHORS: list[str] = [
    "harming or endangering a child",
    "sexual content involving minors",
    "targeting schoolchildren or students",
    "exploiting a teenager or young person",
    "abducting a child",
    "grooming a minor online",
    "teaching a child to do something dangerous or illegal",
]


@dataclass
class FilterDrop:
    """Record of a prompt dropped by the filter, for the audit log."""

    prompt: str
    source: str
    stage: str  # "keyword" or "semantic"
    trigger: str  # matching keyword or top anchor text
    score: float  # 1.0 for keyword hits, cosine sim for semantic hits


class HarmFilter:
    """Callable filter that returns True to KEEP a prompt, False to drop."""

    def __init__(
        self,
        semantic_threshold: float = 0.55,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.embedder_name = embedder_name
        self._embedder = None
        self._anchor_emb: Optional[np.ndarray] = None
        self.drops: list[FilterDrop] = []
        self.semantic_scores: list[float] = []

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedder_name)
        anchors = self._embedder.encode(
            _SEMANTIC_ANCHORS, normalize_embeddings=True, convert_to_numpy=True
        )
        self._anchor_emb = anchors  # (n_anchors, dim)

    def _keyword_hit(self, text: str) -> Optional[str]:
        m = _KEYWORD_RE.search(text)
        return m.group(0) if m else None

    def filter(self, rows: Iterable[dict]) -> list[dict]:
        """Filter a list of {'Prompt': str, 'Source': str} rows.

        Drops are recorded on self.drops for later audit logging.
        """
        rows = list(rows)
        kept: list[dict] = []
        needs_semantic: list[tuple[int, dict]] = []

        for row in rows:
            prompt = row["Prompt"]
            hit = self._keyword_hit(prompt)
            if hit is not None:
                self.drops.append(
                    FilterDrop(
                        prompt=prompt,
                        source=row["Source"],
                        stage="keyword",
                        trigger=hit,
                        score=1.0,
                    )
                )
                continue
            needs_semantic.append((len(kept), row))
            kept.append(row)

        if not kept:
            return kept

        self._load_embedder()
        prompts = [r["Prompt"] for _, r in needs_semantic]
        emb = self._embedder.encode(
            prompts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64,
            show_progress_bar=True,
        )
        sims = emb @ self._anchor_emb.T  # (n, n_anchors)
        max_sims = sims.max(axis=1)
        max_idx = sims.argmax(axis=1)
        self.semantic_scores.extend(max_sims.tolist())

        keep_mask = [True] * len(kept)
        for (kept_idx, row), top_sim, anchor_idx in zip(needs_semantic, max_sims, max_idx):
            if top_sim >= self.semantic_threshold:
                keep_mask[kept_idx] = False
                self.drops.append(
                    FilterDrop(
                        prompt=row["Prompt"],
                        source=row["Source"],
                        stage="semantic",
                        trigger=_SEMANTIC_ANCHORS[int(anchor_idx)],
                        score=float(top_sim),
                    )
                )

        return [r for r, keep in zip(kept, keep_mask) if keep]

    def write_audit_log(self, drops_path: Path, histogram_path: Path) -> None:
        """Write per-drop JSONL and a semantic-score histogram for calibration."""
        drops_path.parent.mkdir(parents=True, exist_ok=True)
        with drops_path.open("w", encoding="utf-8") as f:
            for d in self.drops:
                f.write(
                    json.dumps(
                        {
                            "prompt": d.prompt,
                            "source": d.source,
                            "stage": d.stage,
                            "trigger": d.trigger,
                            "score": d.score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if self.semantic_scores:
            scores = np.asarray(self.semantic_scores, dtype=float)
            bin_edges = np.linspace(0.0, 1.0, 21)
            counts, _ = np.histogram(scores, bins=bin_edges)
            histogram = {
                "threshold": self.semantic_threshold,
                "n_scored": int(scores.size),
                "mean": float(scores.mean()),
                "max": float(scores.max()),
                "min": float(scores.min()),
                "bins": [
                    {"lo": float(lo), "hi": float(hi), "count": int(c)}
                    for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], counts)
                ],
            }
            histogram_path.parent.mkdir(parents=True, exist_ok=True)
            histogram_path.write_text(
                json.dumps(histogram, indent=2, ensure_ascii=False), encoding="utf-8"
            )

"""Harm filter: drops unwanted prompts from the harmful split before sampling.

Applied unconditionally to the ``harmful`` split of RevivifAI/derestriction
before sampling. Runs two stages per category:

1. Broad keyword regex (word-boundary aware).
2. Semantic cosine similarity against category anchor prompts, using
   ``sentence-transformers/all-MiniLM-L6-v2``.

Two categories are filtered:

- ``child``  — prompts that mention, target, or involve children, teenagers,
  or students. Child-adjacent prompts are always unacceptable for an
  abliteration corpus.
- ``sexual`` — prompts that request graphic sexual acts, pornography, erotica,
  or otherwise sexual depictions. Refusal directions need not be computed
  from sexually explicit material, and keeping it out shrinks the dataset's
  attack surface.

The filter is intentionally aggressive — we err heavily on the side of
dropping borderline prompts rather than letting child- or sex-adjacent
prompts slip into the dataset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CHILD_KEYWORD_PATTERNS: list[str] = [
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

_CHILD_SEMANTIC_ANCHORS: list[str] = [
    "harming or endangering a child",
    "sexual content involving minors",
    "targeting schoolchildren or students",
    "exploiting a teenager or young person",
    "abducting a child",
    "grooming a minor online",
    "teaching a child to do something dangerous or illegal",
]


_SEXUAL_KEYWORD_PATTERNS: list[str] = [
    r"pornograph(?:y|ic|ically|er|ers)",
    r"\bporn\b",
    r"\berotica?\b",
    r"\berotically\b",
    r"\bsmut(?:ty)?\b",
    r"\bhentai\b",
    r"\blolicon\b",
    r"\bnude\b",
    r"\bnudity\b",
    r"\bnaked\b",
    r"masturbat(?:e|es|ed|ing|ion|ory)",
    r"\borgasm(?:s|ic|ed|ing)?\b",
    r"\bejaculat(?:e|es|ed|ing|ion)\b",
    r"\bclimax(?:ed|ing|es)?\b",
    r"\bgenital(?:s|ia)?\b",
    r"\bpenis(?:es)?\b",
    r"\bvagina(?:l|s)?\b",
    r"\bvulva(?:s)?\b",
    r"\bclitoris\b",
    r"\btesticl(?:e|es)\b",
    r"\bscrotum\b",
    r"\bnipples?\b",
    r"\bbreasts?\b",
    r"\bbutthole\b",
    r"\banus\b",
    r"\banal\s+(?:sex|intercourse|penetration)\b",
    r"\bintercourse\b",
    r"\bcoitus\b",
    r"\bcopulat(?:e|es|ed|ing|ion)\b",
    r"\bfornicat(?:e|es|ed|ing|ion)\b",
    r"\bfellatio\b",
    r"\bcunnilingus\b",
    r"\banilingus\b",
    r"\bblow[-\s]?job(?:s)?\b",
    r"\bhand[-\s]?job(?:s)?\b",
    r"\bdeepthroat(?:ing|ed)?\b",
    r"\brimming\b",
    r"\brimjob(?:s)?\b",
    r"\bcum(?:ming|med|shot|shots)?\b",
    r"\bsemen\b",
    r"\bsperm\b",
    r"\bBDSM\b",
    r"\bbondage\b",
    r"\bfetish(?:es|istic|ism)?\b",
    r"\bkink(?:y|s)?\b",
    r"\bsex[-\s]?toy(?:s)?\b",
    r"\bdildo(?:s)?\b",
    r"\bvibrator(?:s)?\b",
    r"\bbutt[-\s]?plug(?:s)?\b",
    r"\bsex\s+scene(?:s)?\b",
    r"\bsex(?:ually)?\s+explicit\b",
    r"\bsexually\s+graphic\b",
    r"\bgraphic\s+sex(?:ual)?\b",
    r"\bsex\s+act(?:s)?\b",
    r"\bsexual\s+act(?:s)?\b",
    r"\bsexual\s+encounter(?:s)?\b",
    r"\bsexual\s+fantasy\b",
    r"\bsexual\s+fantasies\b",
    r"\bsexual\s+role[-\s]?play(?:ing)?\b",
    r"\berotic\s+role[-\s]?play(?:ing)?\b",
    r"\bhave\s+sex\b",
    r"\bhaving\s+sex\b",
    r"\bmake\s+love\b",
    r"\bmaking\s+love\b",
    r"\bget\s+laid\b",
    r"\bfuck(?:ed|ing|s|er|ers)?\b",
    r"\brap(?:e|ed|es|ing|ist|ists)\b",
    r"\bmolest(?:er|ers|ed|ing|ation)\b",
    r"\bincest(?:uous)?\b",
    r"\bbestial(?:ity)?\b",
    r"\bzoophil(?:ia|e|es|ic)\b",
    r"\bnecrophil(?:ia|e|es|ic)\b",
    r"\bprostitut(?:e|es|ion|ing)\b",
    r"\bsex[-\s]?work(?:er|ers|ing)\b",
    r"\bescort\s+service(?:s)?\b",
    r"\bstrip(?:per|pers|tease|club|clubs)\b",
    r"\bcamgirl(?:s)?\b",
    r"\bOnlyFans\b",
]

_SEXUAL_SEMANTIC_ANCHORS: list[str] = [
    "write pornographic or erotic fiction",
    "describe a graphic sex scene in explicit detail",
    "produce sexually explicit material",
    "generate adult pornography",
    "describe two people having sex",
    "write a detailed depiction of sexual acts",
    "create non-consensual sexual content",
    "narrate a rape scene",
    "roleplay an explicit sexual encounter",
]


# Category registry: each entry is (category_name, compiled_keyword_regex, anchor_texts).
_CATEGORIES: tuple[tuple[str, list[str], list[str]], ...] = (
    ("child", _CHILD_KEYWORD_PATTERNS, _CHILD_SEMANTIC_ANCHORS),
    ("sexual", _SEXUAL_KEYWORD_PATTERNS, _SEXUAL_SEMANTIC_ANCHORS),
)


@dataclass
class FilterDrop:
    """Record of a prompt dropped by the filter, for the audit log."""

    prompt: str
    source: str
    stage: str  # "keyword" or "semantic"
    category: str  # which filter category fired: "child" or "sexual"
    trigger: str  # matching keyword or top anchor text
    score: float  # 1.0 for keyword hits, cosine sim for semantic hits


class HarmFilter:
    """Callable filter that returns True to KEEP a prompt, False to drop.

    Runs the child-harm filter and the sexual-content filter back to back.
    Each category has its own keyword regex and semantic anchors but shares
    the same embedder for efficiency.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.55,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.embedder_name = embedder_name
        self._embedder = None
        # Compile one keyword regex and encode anchors per category.
        self._keyword_res: dict[str, re.Pattern[str]] = {
            name: re.compile("|".join(patterns), re.IGNORECASE) for name, patterns, _ in _CATEGORIES
        }
        self._anchor_texts: dict[str, list[str]] = {name: anchors for name, _, anchors in _CATEGORIES}
        self._anchor_embs: dict[str, np.ndarray] = {}
        self.drops: list[FilterDrop] = []
        self.semantic_scores: list[float] = []

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedder_name)
        for name, anchors in self._anchor_texts.items():
            self._anchor_embs[name] = self._embedder.encode(
                anchors,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

    @staticmethod
    def _keyword_hit(text: str, regex: re.Pattern[str]) -> str | None:
        m = regex.search(text)
        return m.group(0) if m else None

    def _filter_one_category(self, rows: list[dict], category: str) -> list[dict]:
        """Filter ``rows`` through a single category (keyword + semantic)."""
        regex = self._keyword_res[category]
        kept: list[dict] = []
        needs_semantic: list[tuple[int, dict]] = []

        for row in rows:
            prompt = row["Prompt"]
            hit = self._keyword_hit(prompt, regex)
            if hit is not None:
                self.drops.append(
                    FilterDrop(
                        prompt=prompt,
                        source=row["Source"],
                        stage="keyword",
                        category=category,
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
        anchor_emb = self._anchor_embs[category]
        prompts = [r["Prompt"] for _, r in needs_semantic]
        emb = self._embedder.encode(
            prompts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True,
        )
        sims = emb @ anchor_emb.T  # (n, n_anchors)
        max_sims = sims.max(axis=1)
        max_idx = sims.argmax(axis=1)
        self.semantic_scores.extend(max_sims.tolist())

        keep_mask = [True] * len(kept)
        anchor_texts = self._anchor_texts[category]
        for (kept_idx, row), top_sim, anchor_idx in zip(needs_semantic, max_sims, max_idx, strict=False):
            if top_sim >= self.semantic_threshold:
                keep_mask[kept_idx] = False
                self.drops.append(
                    FilterDrop(
                        prompt=row["Prompt"],
                        source=row["Source"],
                        stage="semantic",
                        category=category,
                        trigger=anchor_texts[int(anchor_idx)],
                        score=float(top_sim),
                    )
                )

        return [r for r, keep in zip(kept, keep_mask, strict=False) if keep]

    def filter(self, rows: Iterable[dict]) -> list[dict]:
        """Filter a list of {'Prompt': str, 'Source': str} rows.

        Applies every configured category in order. Drops are recorded on
        ``self.drops`` for later audit logging.
        """
        kept = list(rows)
        for category, _, _ in _CATEGORIES:
            kept = self._filter_one_category(kept, category)
        return kept

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
                            "category": d.category,
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
                    for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], counts, strict=False)
                ],
            }
            histogram_path.parent.mkdir(parents=True, exist_ok=True)
            histogram_path.write_text(json.dumps(histogram, indent=2, ensure_ascii=False), encoding="utf-8")

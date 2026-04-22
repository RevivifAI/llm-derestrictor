"""Build the RevivifAI/derestriction dataset from public HuggingFace sources.

Pulls harmful / harmless / preservation prompts from well-known abliteration and
capability-evaluation datasets, normalizes every row to ``{Prompt, Source}``,
dedupes within and across splits, applies the children filter to the harmful
split, and writes three parquet files to ``build/derestriction/``.

Run with::

    uv run python -m derestrictor.scripts.build_dataset

Environment variables:
    HF_HOME, HF_DATASETS_CACHE    override the HuggingFace cache location
    HF_TOKEN                      optional, raises rate limits
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()

# src/derestrictor/scripts/build_dataset.py → repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = PROJECT_ROOT / "build" / "derestriction"
AUDIT_DROPS_PATH = PROJECT_ROOT / "build" / "harm_filter_drops.jsonl"
AUDIT_HIST_PATH = PROJECT_ROOT / "build" / "harm_filter_histogram.json"

TARGET_PER_SPLIT = 10_000
SEED = 20260422

_PUNCT_STRIP = string.punctuation + " \t\n\r"


# Normalization / dedup helpers


def normalize_prompt(text: str) -> str:
    """Return a user-facing prompt: single-spaced, trimmed, reasonable length."""
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def dedup_key(text: str) -> str:
    """Normalization used only for dedup equality."""
    return re.sub(r"\s+", " ", text).strip().lower().strip(_PUNCT_STRIP)


def is_ascii_english(text: str) -> bool:
    """Heuristic: keep only ASCII-heavy strings to avoid non-English prompts."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.95


# Source loaders


@dataclass
class SourceSpec:
    """Specification for a single upstream dataset source feeding the build."""

    name: str  # value written into the Source column
    dataset_id: str
    config: str | None
    split: str
    column: str
    filter_fn: Callable[[dict], bool] | None = None
    cap: int | None = None  # hard cap on raw rows taken from this source


def _iter_rows(spec: SourceSpec) -> list[dict]:
    print(f"  Loading {spec.dataset_id} cfg={spec.config} split={spec.split} …", flush=True)
    ds = load_dataset(spec.dataset_id, spec.config, split=spec.split)
    rows: list[dict] = []
    for row in ds:
        if spec.filter_fn is not None and not spec.filter_fn(row):
            continue
        prompt = normalize_prompt(row.get(spec.column, ""))
        if not prompt or len(prompt) < 8 or len(prompt) > 2000:
            continue
        if not is_ascii_english(prompt):
            continue
        rows.append({"Prompt": prompt, "Source": spec.name})
        if spec.cap is not None and len(rows) >= spec.cap:
            break
    print(f"    kept {len(rows):>6d} from {spec.dataset_id}", flush=True)
    return rows


HARMFUL_SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="LLM-LAT/harmful-dataset",
        dataset_id="LLM-LAT/harmful-dataset",
        config=None,
        split="train",
        column="prompt",
    ),
    SourceSpec(
        name="Undi95/orthogonal-activation-steering-TOXIC",
        dataset_id="Undi95/orthogonal-activation-steering-TOXIC",
        config=None,
        split="test",
        column="goal",
    ),
    SourceSpec(
        name="mlabonne/harmful_behaviors",
        dataset_id="mlabonne/harmful_behaviors",
        config=None,
        split="train+test",
        column="text",
    ),
    SourceSpec(
        name="wangzhang/abliterix-datasets",
        dataset_id="wangzhang/abliterix-datasets",
        config="harmful_1000",
        split="train",
        column="prompt",
        filter_fn=lambda r: r.get("language") == "en",
    ),
    # Chinese Sensitive Topics from the Multiverse Computing refusal-evaluation
    # benchmark. These prompts are treated as "harmful" here only in the sense
    # that China-aligned instruct models currently *refuse* them, so they feed
    # the refusal-direction computation. They are not harmful per se.
    SourceSpec(
        name="MultiverseComputingCAI/llm-refusal-evaluation:ccp_sensitive",
        dataset_id="MultiverseComputingCAI/llm-refusal-evaluation",
        config=None,
        split="ccp_sensitive",
        column="prompt",
    ),
    SourceSpec(
        name="MultiverseComputingCAI/llm-refusal-evaluation:deccp_censored",
        dataset_id="MultiverseComputingCAI/llm-refusal-evaluation",
        config=None,
        split="deccp_censored",
        column="prompt",
    ),
]


def _alpaca_no_input(row: dict) -> bool:
    return (row.get("input") or "").strip() == ""


HARMLESS_SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="tatsu-lab/alpaca",
        dataset_id="tatsu-lab/alpaca",
        config=None,
        split="train",
        column="instruction",
        filter_fn=_alpaca_no_input,
    ),
    SourceSpec(
        name="mlabonne/harmless_alpaca",
        dataset_id="mlabonne/harmless_alpaca",
        config=None,
        split="train",
        column="text",
    ),
    SourceSpec(
        name="yahma/alpaca-cleaned",
        dataset_id="yahma/alpaca-cleaned",
        config=None,
        split="train",
        column="instruction",
        filter_fn=_alpaca_no_input,
    ),
]


# Preservation is budgeted so the mix stays balanced across capabilities.
PRESERVATION_BUDGETS: dict[str, int] = {
    "cais/mmlu": 3000,
    "openai/gsm8k": 2000,
    "allenai/ai2_arc": 2000,
    "HuggingFaceH4/MATH-500": 500,
    "openai/openai_humaneval": 164,
    "tatsu-lab/alpaca": 10_000,  # top-up source
}

PRESERVATION_SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="cais/mmlu",
        dataset_id="cais/mmlu",
        config="all",
        split="test",
        column="question",
    ),
    SourceSpec(
        name="openai/gsm8k",
        dataset_id="openai/gsm8k",
        config="main",
        split="train",
        column="question",
    ),
    SourceSpec(
        name="allenai/ai2_arc",
        dataset_id="allenai/ai2_arc",
        config="ARC-Challenge",
        split="train+test+validation",
        column="question",
    ),
    SourceSpec(
        name="allenai/ai2_arc",
        dataset_id="allenai/ai2_arc",
        config="ARC-Easy",
        split="train+test+validation",
        column="question",
    ),
    SourceSpec(
        name="HuggingFaceH4/MATH-500",
        dataset_id="HuggingFaceH4/MATH-500",
        config=None,
        split="test",
        column="problem",
    ),
    SourceSpec(
        name="openai/openai_humaneval",
        dataset_id="openai/openai_humaneval",
        config=None,
        split="test",
        column="prompt",
    ),
    SourceSpec(
        name="tatsu-lab/alpaca",
        dataset_id="tatsu-lab/alpaca",
        config=None,
        split="train",
        column="instruction",
        filter_fn=_alpaca_no_input,
    ),
]


# Dedup / sampling


def dedup(rows: list[dict], seen: set[str]) -> list[dict]:
    """Return ``rows`` without entries whose dedup key is already in ``seen``.

    Updates ``seen`` in place with the keys of the emitted rows.
    """
    out: list[dict] = []
    for r in rows:
        k = dedup_key(r["Prompt"])
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def sample_to_target(
    rows: list[dict],
    target: int,
    rng: random.Random,
    split_name: str,
    budgets: dict[str, int] | None = None,
) -> list[dict]:
    """Sample exactly `target` rows, honoring per-source budgets when given.

    Budgets are applied first (so the final mix is balanced); any shortfall is
    then filled from the combined leftover pool. Fails loudly if the pool is
    smaller than `target`.
    """
    if len(rows) < target:
        raise RuntimeError(
            f"{split_name}: pool has {len(rows)} rows, need {target}. Add more sources or loosen filters."
        )

    if not budgets:
        rng.shuffle(rows)
        return rows[:target]

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["Source"], []).append(r)
    for lst in by_source.values():
        rng.shuffle(lst)

    picked: list[dict] = []
    leftover: list[dict] = []
    for source, lst in by_source.items():
        budget = budgets.get(source, 0)
        take = min(budget, len(lst))
        picked.extend(lst[:take])
        leftover.extend(lst[take:])

    if len(picked) > target:
        rng.shuffle(picked)
        picked = picked[:target]
    elif len(picked) < target:
        rng.shuffle(leftover)
        picked.extend(leftover[: target - len(picked)])

    rng.shuffle(picked)
    return picked


# Main build


def build() -> dict[str, Dataset]:
    """Build the full derestriction dataset and write splits to ``BUILD_DIR``."""
    rng = random.Random(SEED)  # noqa: S311 - non-cryptographic sampling seeded for reproducibility
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- harmful ---------------------------------------------------------
    print("\n[harmful] gathering pool …", flush=True)
    harmful_raw: list[dict] = []
    for spec in HARMFUL_SOURCES:
        harmful_raw.extend(_iter_rows(spec))
    print(f"  raw total: {len(harmful_raw)}")

    from derestrictor.data.harm_filter import HarmFilter

    hf = HarmFilter()
    harmful_filtered = hf.filter(harmful_raw)
    hf.write_audit_log(AUDIT_DROPS_PATH, AUDIT_HIST_PATH)
    print(
        f"  harm filter: kept {len(harmful_filtered)}, dropped {len(harmful_raw) - len(harmful_filtered)}"
        f" (keyword: {sum(1 for d in hf.drops if d.stage == 'keyword')}, "
        f"semantic: {sum(1 for d in hf.drops if d.stage == 'semantic')})"
    )

    seen_keys: set[str] = set()
    harmful_dedup = dedup(harmful_filtered, seen_keys)
    print(f"  after dedup: {len(harmful_dedup)}")
    harmful_rows = sample_to_target(harmful_dedup, TARGET_PER_SPLIT, rng, "harmful")

    # ---- harmless --------------------------------------------------------
    print("\n[harmless] gathering pool …", flush=True)
    harmless_raw: list[dict] = []
    for spec in HARMLESS_SOURCES:
        harmless_raw.extend(_iter_rows(spec))
    print(f"  raw total: {len(harmless_raw)}")

    harmless_dedup = dedup(harmless_raw, seen_keys)
    print(f"  after dedup (cross-split): {len(harmless_dedup)}")
    harmless_rows = sample_to_target(harmless_dedup, TARGET_PER_SPLIT, rng, "harmless")

    # ---- preservation ----------------------------------------------------
    print("\n[preservation] gathering pool …", flush=True)
    preservation_raw: list[dict] = []
    for spec in PRESERVATION_SOURCES:
        preservation_raw.extend(_iter_rows(spec))
    print(f"  raw total: {len(preservation_raw)}")

    preservation_dedup = dedup(preservation_raw, seen_keys)
    print(f"  after dedup (cross-split): {len(preservation_dedup)}")
    preservation_rows = sample_to_target(
        preservation_dedup,
        TARGET_PER_SPLIT,
        rng,
        "preservation",
        budgets=PRESERVATION_BUDGETS,
    )

    # ---- materialize -----------------------------------------------------
    splits: dict[str, Dataset] = {}
    for name, rows in (
        ("harmful", harmful_rows),
        ("harmless", harmless_rows),
        ("preservation", preservation_rows),
    ):
        ds = Dataset.from_list(rows)
        parquet_path = BUILD_DIR / f"{name}-00000-of-00001.parquet"
        ds.to_parquet(parquet_path)
        splits[name] = ds
        src_counts = Counter(r["Source"] for r in rows)
        print(f"\n[{name}] wrote {parquet_path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
        for src, n in src_counts.most_common():
            print(f"    {n:>5d}  {src}")

    summary = {
        "seed": SEED,
        "target_per_split": TARGET_PER_SPLIT,
        "splits": {
            name: {
                "rows": len(rows),
                "sources": Counter(r["Source"] for r in rows),
            }
            for name, rows in (
                ("harmful", harmful_rows),
                ("harmless", harmless_rows),
                ("preservation", preservation_rows),
            )
        },
    }
    (BUILD_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=dict), encoding="utf-8")

    return splits


def main() -> int:
    """Command-line entry point for building the derestriction dataset."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())

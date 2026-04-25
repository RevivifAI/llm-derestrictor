"""Build the RevivifAI/derestriction dataset from public HuggingFace sources.

Three splits are produced:

- ``restrict`` - prompts the model **should** refuse (child harm, targeted
  violence, weapons of mass destruction, graphic sexual content,
  self-harm encouragement). Populated by harm-filter hits moved out of
  ``derestrict``, WildGuardMix prompts whose ``subcategory`` maps to a
  restricted bucket, plus any safety violations flagged in ``allow``
  sources.
- ``derestrict`` - prompts the model should learn to **answer**. Includes
  AdvBench / JailbreakBench misuse-style behaviors (filtered to strip the
  genuinely harmful ones), the Chinese-sensitive-topic prompts from
  ``MultiverseComputingCAI/llm-refusal-evaluation``, and all of
  WildGuardMix whose ``subcategory`` is not routed to restrict (benign
  over-refusal prompts, cyberattack, fraud, copyright, privacy, etc.).
- ``allow`` - benign instructions (Alpaca, JailbreakBench benign) plus
  capability-preservation prompts (MMLU, GSM8K, ARC, MATH-500, HumanEval).

Every upstream source is the **original** publication - no AdvBench
repackagings (``LLM-LAT/harmful-dataset``, ``mlabonne/harmful_behaviors``,
``Undi95/orthogonal-activation-steering-TOXIC``,
``wangzhang/abliterix-datasets``) and no Alpaca forks
(``mlabonne/harmless_alpaca``, ``yahma/alpaca-cleaned``).

The entire upstream is copied - no per-split row caps - so the output
parquet faithfully reflects each source.

Run with::

    uv run python -m derestrictor.scripts.build_dataset

Environment variables:
    HF_HOME, HF_DATASETS_CACHE    override the HuggingFace cache location
    HF_TOKEN                      required for gated sources
                                  (``allenai/wildguardmix``); also raises
                                  anonymous rate limits for open datasets
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from dotenv import load_dotenv

from derestrictor.data.harm_filter import HarmFilter

load_dotenv()

# src/derestrictor/scripts/build_dataset.py → repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = PROJECT_ROOT / "build" / "derestriction"
AUDIT_DROPS_PATH = PROJECT_ROOT / "build" / "harm_filter_drops.jsonl"
AUDIT_HIST_PATH = PROJECT_ROOT / "build" / "harm_filter_histogram.json"
ALLOW_FLAG_PATH = PROJECT_ROOT / "build" / "allow_split_safety_flags.jsonl"

_PUNCT_STRIP = string.punctuation + " \t\n\r"

# Fixed schema used for every parquet written by the build.
# ``Restriction_Category`` is only populated for rows in the ``restrict``
# split; it is null elsewhere.
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("Prompt", pa.string(), nullable=False),
        pa.field("Source", pa.string(), nullable=False),
        pa.field("Restriction_Category", pa.string(), nullable=True),
    ]
)


# Normalization / dedup helpers


def normalize_prompt(text: str) -> str:
    """Return a user-facing prompt: single-spaced, trimmed, reasonable length."""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
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


def _iter_rows(spec: SourceSpec) -> list[dict]:
    print(f"  Loading {spec.dataset_id} cfg={spec.config} split={spec.split} …", flush=True)
    ds = load_dataset(
        spec.dataset_id,
        spec.config,
        split=spec.split,
        token=os.environ.get("HF_TOKEN"),
    )
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
    print(f"    kept {len(rows):>6d} from {spec.dataset_id}", flush=True)
    return rows


def _alpaca_no_input(row: dict) -> bool:
    return (row.get("input") or "").strip() == ""


# ---- derestrict sources ----------------------------------------------------
# AdvBench (Zou et al. 2023, arXiv:2307.15043) is the canonical 520-row
# set of misuse behaviors. JailbreakBench (Chao et al. 2024,
# arXiv:2404.01318) layers 100 curated misuse behaviors + 100 benign
# parallels on top. The Multiverse Computing refusal benchmark contributes
# Chinese Sensitive Topics (CCP Sensitive + DeCCP) - non-harmful prompts
# that China-aligned instruct models refuse, so they anchor the refusal
# direction usefully.
DERESTRICT_SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="walledai/AdvBench",
        dataset_id="walledai/AdvBench",
        config=None,
        split="train",
        column="prompt",
    ),
    SourceSpec(
        name="JailbreakBench/JBB-Behaviors:harmful",
        dataset_id="JailbreakBench/JBB-Behaviors",
        config="behaviors",
        split="harmful",
        column="Goal",
    ),
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


# ---- allow sources ---------------------------------------------------------
# Alpaca (Taori et al. 2023) and JailbreakBench/benign provide benign
# instruction-style prompts; the remaining sources are canonical capability
# benchmarks used for null-space capability preservation.
ALLOW_SOURCES: list[SourceSpec] = [
    SourceSpec(
        name="tatsu-lab/alpaca",
        dataset_id="tatsu-lab/alpaca",
        config=None,
        split="train",
        column="instruction",
        filter_fn=_alpaca_no_input,
    ),
    SourceSpec(
        name="JailbreakBench/JBB-Behaviors:benign",
        dataset_id="JailbreakBench/JBB-Behaviors",
        config="behaviors",
        split="benign",
        column="Goal",
    ),
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
        name="allenai/ai2_arc:Challenge",
        dataset_id="allenai/ai2_arc",
        config="ARC-Challenge",
        split="train+test+validation",
        column="question",
    ),
    SourceSpec(
        name="allenai/ai2_arc:Easy",
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
]


# ---- WildGuardMix (gated) --------------------------------------------------
# ``allenai/wildguardmix`` is a 92k-prompt safety moderation corpus with a
# ``subcategory`` column giving finer delineation than AdvBench. Access is
# gated behind an AI2 Responsible Use acceptance, so an HF_TOKEN with that
# approval is required; if the token is missing or not authorized, the
# loader logs a warning and skips the dataset rather than aborting the
# build. Subcategory values observed in ``wildguardtrain`` (also used by
# ``wildguardtest``):
#
#   benign
#   others
#   social_stereotypes_and_unfair_discrimination
#   disseminating_false_or_misleading_information_encouraging_disinformation_campaigns
#   sensitive_information_organization_government
#   toxic_language_hate_speech
#   violence_and_physical_harm
#   private_information_individual
#   defamation_encouraging_unethical_or_unsafe_actions
#   fraud_assisting_illegal_activities
#   sexual_content
#   mental_health_over-reliance_crisis
#   copyright_violations
#   cyberattack
#   causing_material_harm_by_disseminating_misinformation
#
# The subcategory labels are *noisy* - inspection shows the same prompt
# occasionally tagged into several buckets because the adversarial prompt
# templates are reused across risk themes. The ``HarmFilter`` pass later
# in the pipeline acts as a belt-and-suspenders safety net: any WildGuardMix
# prompt we route to ``derestrict`` that still trips the filter keywords
# (child harm, WMD, self-harm, targeted violence, sexual deviancy) is
# moved to ``restrict``.
WILDGUARDMIX_DATASET_ID = "allenai/wildguardmix"
WILDGUARDMIX_CONFIGS: tuple[tuple[str, str], ...] = (
    ("wildguardtrain", "train"),
    ("wildguardtest", "test"),
)

# Subcategories routed *directly* to restrict without passing through
# derestrict first. Chosen to match the user-facing policy:
# genuinely-harmful prompts (targeted violence, graphic sexual content,
# self-harm / pro-ana, materially-harmful misinformation) stay refused.
_WILDGUARDMIX_RESTRICT_SUBCATEGORIES: frozenset[str] = frozenset(
    {
        "violence_and_physical_harm",
        "sexual_content",
        "mental_health_over-reliance_crisis",
        "causing_material_harm_by_disseminating_misinformation",
    }
)


@dataclass
class _WildGuardMixRows:
    """Partition of WildGuardMix prompts after subcategory routing."""

    derestrict: list[dict]
    restrict: list[dict]
    skipped: int


def _iter_wildguardmix(token: str | None) -> _WildGuardMixRows:
    """Load and partition ``allenai/wildguardmix`` by ``subcategory``.

    Args:
        token: Hugging Face token with accepted AI2 Responsible Use terms.
            ``None`` (or a token without access) triggers a soft-skip - the
            returned partition is empty and a warning is printed so the rest
            of the build can proceed on open datasets only.

    Returns:
        A :class:`_WildGuardMixRows` whose ``derestrict`` and ``restrict``
        lists each carry ``{"Prompt", "Source", "Restriction_Category"?}``
        dicts. ``skipped`` counts rows rejected by the normaliser
        (non-ASCII, empty, or length-bounded) plus rows with ``None``
        ``prompt_harm_label`` that we can't confidently route.
    """
    if not token:
        print(
            f"  [{WILDGUARDMIX_DATASET_ID}] HF_TOKEN not set - skipping the gated "
            f"WildGuardMix source. Set HF_TOKEN (with AI2 Responsible Use "
            f"acceptance) to include it.",
            flush=True,
        )
        return _WildGuardMixRows(derestrict=[], restrict=[], skipped=0)

    seen_prompts: set[str] = set()
    derestrict: list[dict] = []
    restrict: list[dict] = []
    skipped = 0

    for config, split in WILDGUARDMIX_CONFIGS:
        print(
            f"  Loading {WILDGUARDMIX_DATASET_ID} cfg={config} split={split} …",
            flush=True,
        )
        try:
            ds = load_dataset(
                WILDGUARDMIX_DATASET_ID,
                config,
                split=split,
                token=token,
            )
        except Exception as exc:  # pragma: no cover - network / auth dependent
            print(
                f"    WARNING: failed to load {WILDGUARDMIX_DATASET_ID}/{config}: "
                f"{type(exc).__name__}: {exc}. Skipping.",
                flush=True,
            )
            continue

        kept_d = 0
        kept_r = 0
        for row in ds:
            prompt_raw = row.get("prompt")
            subcategory = row.get("subcategory") or ""
            harm_label = row.get("prompt_harm_label")

            prompt = normalize_prompt(prompt_raw)
            if not prompt or len(prompt) < 8 or len(prompt) > 2000:
                skipped += 1
                continue
            if not is_ascii_english(prompt):
                skipped += 1
                continue

            key = dedup_key(prompt)
            if key in seen_prompts:
                continue
            seen_prompts.add(key)

            source = f"{WILDGUARDMIX_DATASET_ID}:{config}:{subcategory or 'unknown'}"
            # Route by subcategory first; fall back to prompt_harm_label if
            # the subcategory is missing / unknown so we don't silently drop
            # the row.
            if subcategory in _WILDGUARDMIX_RESTRICT_SUBCATEGORIES:
                restrict.append(
                    {
                        "Prompt": prompt,
                        "Source": source,
                        "Restriction_Category": f"wildguardmix:{subcategory}",
                    }
                )
                kept_r += 1
            elif subcategory:
                # Every other labelled bucket - including ``benign``,
                # ``others``, ``cyberattack``, ``fraud_assisting_illegal_
                # activities``, ``copyright_violations``, the privacy and
                # defamation buckets, hate-speech and stereotype buckets,
                # and the political-disinformation bucket - is a prompt
                # the user wants unlocked. The HarmFilter pass afterwards
                # will still relocate any that trip its keyword / semantic
                # stages (e.g. a prompt about minors slipping under
                # ``cyberattack``).
                derestrict.append({"Prompt": prompt, "Source": source})
                kept_d += 1
            elif harm_label == "harmful":
                # No subcategory but GPT-4 / annotators flagged the prompt
                # as harmful - conservatively send to derestrict and let
                # the HarmFilter have the final word.
                derestrict.append({"Prompt": prompt, "Source": source})
                kept_d += 1
            else:
                # No subcategory and no harm label ⇒ not enough signal.
                skipped += 1

        print(
            f"    kept {kept_d:>6d} → derestrict, {kept_r:>6d} → restrict from {WILDGUARDMIX_DATASET_ID}/{config}",
            flush=True,
        )

    return _WildGuardMixRows(derestrict=derestrict, restrict=restrict, skipped=skipped)


# Dedup


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


def dedup_within_split(rows: list[dict]) -> list[dict]:
    """Drop duplicates inside a single split before the cross-split pass.

    Returns a fresh list; does not touch the input. First occurrence wins,
    which preserves the priority order of the callers (e.g. WildGuardMix
    restrict-routed rows come before HarmFilter-moved rows).
    """
    local_seen: set[str] = set()
    return dedup(rows, local_seen)


def _rows_to_parquet(rows: list[dict], path: Path) -> None:
    """Write ``rows`` as a Snappy-compressed parquet matching ``PARQUET_SCHEMA``."""
    prompts = [r["Prompt"] for r in rows]
    sources = [r["Source"] for r in rows]
    restrictions = [r.get("Restriction_Category") for r in rows]
    table = pa.Table.from_pydict(
        {
            "Prompt": prompts,
            "Source": sources,
            "Restriction_Category": restrictions,
        },
        schema=PARQUET_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def _write_safety_flags(flags: list[dict]) -> None:
    """Persist allow-split safety violations so they can be reviewed."""
    ALLOW_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALLOW_FLAG_PATH.open("w", encoding="utf-8") as f:
        for row in flags:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Main build


def build() -> dict[str, list[dict]]:
    """Build the full derestriction dataset and write splits to ``BUILD_DIR``."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- derestrict ------------------------------------------------------
    print("\n[derestrict] gathering pool …", flush=True)
    derestrict_raw: list[dict] = []
    for spec in DERESTRICT_SOURCES:
        derestrict_raw.extend(_iter_rows(spec))
    print(f"  raw total from open sources: {len(derestrict_raw)}")

    # ---- WildGuardMix (gated) --------------------------------------------
    print("\n[wildguardmix] gathering gated pool …", flush=True)
    wg = _iter_wildguardmix(os.environ.get("HF_TOKEN"))
    derestrict_raw.extend(wg.derestrict)
    wildguardmix_restrict_rows = wg.restrict
    print(
        f"  wildguardmix → derestrict: {len(wg.derestrict)}, "
        f"direct-to-restrict: {len(wildguardmix_restrict_rows)}, "
        f"skipped: {wg.skipped}"
    )
    print(f"  raw derestrict total (all sources): {len(derestrict_raw)}")

    # ---- allow -----------------------------------------------------------
    print("\n[allow] gathering pool …", flush=True)
    allow_raw: list[dict] = []
    for spec in ALLOW_SOURCES:
        allow_raw.extend(_iter_rows(spec))
    print(f"  raw total: {len(allow_raw)}")

    # ---- filter ---------------------------------------------------------
    # The single HarmFilter instance accumulates drops across both pools so
    # the audit log captures everything that landed in restrict, no matter
    # which pool it came from.
    hf = HarmFilter()

    print("\n[filter] routing genuinely-harmful prompts out of derestrict …", flush=True)
    derestrict_result = hf.filter(derestrict_raw)
    print(f"  derestrict: kept {len(derestrict_result.kept)}, moved-to-restrict {len(derestrict_result.restricted)}")

    print("\n[filter] auditing allow pool for safety violations …", flush=True)
    allow_result = hf.filter(allow_raw)
    allow_violations = allow_result.restricted
    if allow_violations:
        print(
            f"  WARNING: {len(allow_violations)} prompts in the allow pool matched the "
            f"harm filter. They have been moved to restrict and logged to "
            f"{ALLOW_FLAG_PATH.relative_to(PROJECT_ROOT)}.",
            flush=True,
        )
        per_source = Counter(r["Source"] for r in allow_violations)
        per_category = Counter(r["Restriction_Category"] for r in allow_violations)
        for src, n in per_source.most_common():
            print(f"    {n:>5d}  source   {src}")
        for cat, n in per_category.most_common():
            print(f"    {n:>5d}  category {cat}")
        _write_safety_flags(allow_violations)
    else:
        print("  allow pool is clean.", flush=True)

    hf.write_audit_log(AUDIT_DROPS_PATH, AUDIT_HIST_PATH)
    total_restricted = len(wildguardmix_restrict_rows) + len(derestrict_result.restricted) + len(allow_violations)
    print(
        f"\n[filter] audit: kept-derestrict={len(derestrict_result.kept)}, "
        f"restrict={total_restricted} "
        f"(wildguardmix-direct={len(wildguardmix_restrict_rows)}, "
        f"filter-moved={len(derestrict_result.restricted)}, "
        f"allow-violations={len(allow_violations)}), "
        f"keyword={sum(1 for d in hf.drops if d.stage == 'keyword')}, "
        f"semantic={sum(1 for d in hf.drops if d.stage == 'semantic')}"
    )

    # ---- dedup ----------------------------------------------------------
    # Two-phase dedup so both within-split and cross-split drops are
    # visible in the audit log:
    #   (1) Within each split, collapse case-insensitive duplicates
    #       coming from different source files (e.g. AdvBench and
    #       JailbreakBench sharing a behavior, or WildGuardMix having a
    #       prompt tagged into two subcategories).
    #   (2) Cross-split, enforce priority ``restrict → derestrict → allow``
    #       so an ambiguous prompt lands in the more conservative bucket
    #       and never appears in more than one split.
    #
    # Inside restrict, WildGuardMix-routed rows come first so their
    # fine-grained ``Restriction_Category`` wins over a coarser HarmFilter
    # tag on the same prompt.
    print("\n[dedup] pass 1 - within-split …", flush=True)
    restrict_raw = wildguardmix_restrict_rows + derestrict_result.restricted + allow_violations
    restrict_unique = dedup_within_split(restrict_raw)
    derestrict_unique = dedup_within_split(derestrict_result.kept)
    allow_unique = dedup_within_split(allow_result.kept)
    print(
        f"  restrict:   {len(restrict_raw):>6d} → {len(restrict_unique):>6d} "
        f"(dropped {len(restrict_raw) - len(restrict_unique):>5d})"
    )
    print(
        f"  derestrict: {len(derestrict_result.kept):>6d} → {len(derestrict_unique):>6d} "
        f"(dropped {len(derestrict_result.kept) - len(derestrict_unique):>5d})"
    )
    print(
        f"  allow:      {len(allow_result.kept):>6d} → {len(allow_unique):>6d} "
        f"(dropped {len(allow_result.kept) - len(allow_unique):>5d})"
    )

    print("\n[dedup] pass 2 - cross-split (priority restrict → derestrict → allow) …", flush=True)
    seen: set[str] = set()
    restrict_rows = dedup(restrict_unique, seen)
    derestrict_rows = dedup(derestrict_unique, seen)
    allow_rows = dedup(allow_unique, seen)
    print(
        f"  restrict:   {len(restrict_unique):>6d} → {len(restrict_rows):>6d} "
        f"(dropped {len(restrict_unique) - len(restrict_rows):>5d})"
    )
    print(
        f"  derestrict: {len(derestrict_unique):>6d} → {len(derestrict_rows):>6d} "
        f"(dropped {len(derestrict_unique) - len(derestrict_rows):>5d})"
    )
    print(
        f"  allow:      {len(allow_unique):>6d} → {len(allow_rows):>6d} "
        f"(dropped {len(allow_unique) - len(allow_rows):>5d})"
    )

    # ---- materialize -----------------------------------------------------
    splits: dict[str, list[dict]] = {
        "restrict": restrict_rows,
        "derestrict": derestrict_rows,
        "allow": allow_rows,
    }
    for name, rows in splits.items():
        parquet_path = BUILD_DIR / f"{name}-00000-of-00001.parquet"
        _rows_to_parquet(rows, parquet_path)
        src_counts = Counter(r["Source"] for r in rows)
        print(f"\n[{name}] wrote {parquet_path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")
        for src, n in src_counts.most_common():
            print(f"    {n:>5d}  {src}")
        if name == "restrict":
            cat_counts = Counter(r.get("Restriction_Category") or "unknown" for r in rows)
            for cat, n in cat_counts.most_common():
                print(f"    {n:>5d}  category={cat}")

    summary = {
        "splits": {
            name: {
                "rows": len(rows),
                "sources": Counter(r["Source"] for r in rows),
                "restriction_categories": (
                    Counter(r.get("Restriction_Category") or "unknown" for r in rows) if name == "restrict" else None
                ),
            }
            for name, rows in splits.items()
        },
        "allow_safety_violations": len(allow_violations),
        "wildguardmix": {
            "derestrict_prompts": len(wg.derestrict),
            "restrict_prompts": len(wildguardmix_restrict_rows),
            "skipped_prompts": wg.skipped,
            "restrict_subcategories": sorted(_WILDGUARDMIX_RESTRICT_SUBCATEGORIES),
        },
        "dedup": {
            "restrict": {
                "raw": len(restrict_raw),
                "after_within_split": len(restrict_unique),
                "after_cross_split": len(restrict_rows),
            },
            "derestrict": {
                "raw": len(derestrict_result.kept),
                "after_within_split": len(derestrict_unique),
                "after_cross_split": len(derestrict_rows),
            },
            "allow": {
                "raw": len(allow_result.kept),
                "after_within_split": len(allow_unique),
                "after_cross_split": len(allow_rows),
            },
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

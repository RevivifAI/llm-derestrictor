"""Upload build/derestriction/*.parquet to the HuggingFace Hub.

Expects ``HF_TOKEN`` in ``.env`` (or the environment) with write access to the
``RevivifAI`` organization. Creates the ``RevivifAI/derestriction`` dataset
repo if it does not yet exist, uploads all three parquet files under ``data/``,
and commits a dataset card README.

Run with::

    uv run python -m derestrictor.scripts.upload_dataset
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file

load_dotenv()

# src/derestrictor/scripts/upload_dataset.py → repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = PROJECT_ROOT / "build" / "derestriction"
REPO_ID = "RevivifAI/derestriction"
SPLITS: tuple[str, ...] = ("restrict", "derestrict", "allow")

# Parquet filename prefixes from the pre-2.0.0 split taxonomy. The 2.0.0
# release replaces these with the new ``restrict/derestrict/allow`` splits,
# and the upload pass deletes any surviving legacy files on the Hub so
# ``load_dataset`` doesn't silently return stale rows.
LEGACY_SPLIT_PREFIXES: tuple[str, ...] = ("harmful", "harmless", "preservation")


def _render_readme() -> str:
    summary_path = BUILD_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def _src_table(src_counts: dict[str, int]) -> str:
        lines = ["| Source | Rows |", "|---|---:|"]
        for src, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{src}` | {n:,} |")
        return "\n".join(lines)

    def _cat_table(cat_counts: dict[str, int] | None) -> str:
        if not cat_counts:
            return ""
        lines = ["", "| Restriction category | Rows |", "|---|---:|"]
        for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{cat}` | {n:,} |")
        return "\n".join(lines)

    restrict_block = (
        _src_table(summary["splits"]["restrict"]["sources"])
        + "\n"
        + _cat_table(summary["splits"]["restrict"].get("restriction_categories"))
    )
    derestrict_block = _src_table(summary["splits"]["derestrict"]["sources"])
    allow_block = _src_table(summary["splits"]["allow"]["sources"])
    allow_flag_count = summary.get("allow_safety_violations", 0)
    wg_info = summary.get("wildguardmix") or {}
    wg_derestrict = wg_info.get("derestrict_prompts", 0)
    wg_restrict = wg_info.get("restrict_prompts", 0)
    wg_restrict_subs = wg_info.get("restrict_subcategories") or []
    wg_subs_list = ", ".join(f"`{s}`" for s in wg_restrict_subs) or "_(none)_"

    return f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - abliteration
  - refusal
  - activation-steering
  - alignment
  - red-team
pretty_name: Derestriction
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: restrict
        path: data/restrict-*.parquet
      - split: derestrict
        path: data/derestrict-*.parquet
      - split: allow
        path: data/allow-*.parquet
---

# Derestriction

Three aligned prompt splits used to compute refusal directions, apply
orthogonal-projection abliteration, and preserve downstream capabilities
when uncensoring language models. Assembled from **original** public
HuggingFace datasets — no AdvBench repackagings, no Alpaca forks — by
[RevivifAI](https://huggingface.co/RevivifAI) for the
[llm-derestrictor](https://github.com/RevivifAI/llm-derestrictor) toolkit.

## Splits

| Split | Purpose |
|---|---|
| `restrict` | Prompts the model **should** refuse after derestriction. Populated by harm-filter hits (child harm, targeted violence, weapons of mass destruction, graphic sexual content, self-harm encouragement) pulled out of `derestrict` plus any safety violations flagged in `allow` sources. Carries a `Restriction_Category` column identifying which filter category fired. |
| `derestrict` | Prompts the model should **learn to answer**. AdvBench / JailbreakBench misuse-style behaviors stripped of their genuinely-harmful subset, plus non-harmful but routinely-refused prompts (Chinese Sensitive Topics). These drive the refusal-direction computation. |
| `allow` | Benign instructions (Alpaca, JailbreakBench benign) plus capability-preservation prompts (MMLU, GSM8K, ARC, MATH-500, HumanEval). Used as the baseline anchor and for null-space capability preservation. |

## Safety notice

> The `derestrict` split contains prompts that aligned language models
> currently refuse; it exists for **alignment / abliteration / red-team
> research only** — do **not** fine-tune models toward these prompts.
> Five categories are partitioned out into `restrict` via a two-stage
> keyword + semantic filter before sampling:
>
> 1. Child-adjacent content (minors, students, grooming).
> 2. Graphic sexual acts, pornography, erotica.
> 3. Targeted violence (assassination, murder, torture, terrorism).
> 4. Weapons of mass destruction (bombs, bio/chem/nuke/radiological).
> 5. Self-harm encouragement (suicide methods, pro-ana).
>
> The `allow` pool is audited against the same filter; any hits
> ({allow_flag_count} in this build) are moved to `restrict` and logged.

## WildGuardMix integration

[`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix)
is a gated (AI2 Responsible Use) corpus with finer-grained
`subcategory` labels than AdvBench. Every prompt is routed by its
`subcategory`:

- Routed directly to `restrict` (genuinely harmful): {wg_subs_list}.
- Routed to `derestrict`: every other WildGuardMix `subcategory`,
  including `benign`, `others`, `cyberattack`, `fraud_assisting_illegal_activities`,
  `copyright_violations`, `sensitive_information_organization_government`,
  `private_information_individual`, `toxic_language_hate_speech`,
  `social_stereotypes_and_unfair_discrimination`, and
  `disseminating_false_or_misleading_information_encouraging_disinformation_campaigns`.

The two-stage harm filter then runs as a belt-and-suspenders pass over
the WildGuardMix-derestrict rows — any that still trip the keyword /
semantic detectors are moved to `restrict`. This build contributed
{wg_derestrict:,} prompts to `derestrict` and {wg_restrict:,} directly to
`restrict` from WildGuardMix (before cross-split dedup).

## Schema

Every row in every split has three columns:

| Column                 | Type            | Description                                             |
|------------------------|-----------------|---------------------------------------------------------|
| `Prompt`               | string          | A single instruction prompt, normalized whitespace.     |
| `Source`               | string          | HuggingFace dataset id the prompt came from.            |
| `Restriction_Category` | string \\| null  | Filter category that fired (`restrict` only; null elsewhere). |

Rows are deduplicated case-insensitively both within and across splits —
no prompt appears in more than one split. Priority during dedup is
`restrict` → `derestrict` → `allow`, so ambiguous rows prefer the more
conservative bucket.

### restrict

{restrict_block}

### derestrict

{derestrict_block}

### allow

{allow_block}

## Loading

```python
from datasets import load_dataset

restrict = load_dataset("RevivifAI/derestriction", split="restrict")
derestrict = load_dataset("RevivifAI/derestriction", split="derestrict")
allow = load_dataset("RevivifAI/derestriction", split="allow")

print(derestrict[0])
# {{'Prompt': '...', 'Source': 'walledai/AdvBench', 'Restriction_Category': None}}
```

## Intended use

- Computing refusal directions for abliteration / orthogonal-activation-steering research using `derestrict` vs `allow`.
- Evaluating refusal-rate and over-refusal-rate under controlled conditions.
- Capturing preservation activations for null-space-constrained weight edits
  (see [AlphaEdit](https://arxiv.org/abs/2410.02355)) using `allow`.
- Keeping `restrict` prompts **refused** after abliteration — they are a
  safety floor, not a training target.

**Not** intended for capability training or reinforcement fine-tuning toward
the `derestrict` or `restrict` prompts. Redistribute under MIT with this
safety notice intact.

## Reproduction

```bash
git clone https://github.com/RevivifAI/llm-derestrictor
cd llm-derestrictor
uv sync --group dev
uv run python -m derestrictor.scripts.build_dataset
uv run python -m derestrictor.scripts.upload_dataset
```

## Citation

```
@misc{{derestriction2026,
  author       = {{RevivifAI}},
  title        = {{Derestriction: Restrict/derestrict/allow prompt splits for abliteration research}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/RevivifAI/derestriction}}}}
}}
```

## Sources and licenses

Prompts are drawn verbatim from upstream datasets and retain their original
licensing. The composite dataset is released under MIT to match
[llm-derestrictor](https://github.com/RevivifAI/llm-derestrictor). Attribution
lives in the `Source` column of every row.
"""


def main() -> int:
    """Command-line entry point for uploading the derestriction dataset to the Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment (.env)", file=sys.stderr)
        return 1

    files = {f"data/{name}-00000-of-00001.parquet": BUILD_DIR / f"{name}-00000-of-00001.parquet" for name in SPLITS}
    for local in files.values():
        if not local.exists():
            print(f"ERROR: missing local file {local} — run build_derestriction_dataset first", file=sys.stderr)
            return 1

    print(f"Creating (or reusing) dataset repo {REPO_ID} …", flush=True)
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    readme_text = _render_readme()
    readme_path = BUILD_DIR / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")

    print("Uploading README.md …", flush=True)
    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card",
    )

    for remote, local in files.items():
        print(f"Uploading {remote} …", flush=True)
        upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload {remote}",
        )

    # Remove legacy pre-2.0.0 parquet files from the Hub so readers that
    # called ``load_dataset("RevivifAI/derestriction", split="harmful")``
    # fail loudly instead of silently loading stale data.
    api = HfApi(token=token)
    try:
        remote_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"WARNING: could not list repo files: {exc}", file=sys.stderr)
        remote_files = []
    legacy_files = [
        f
        for f in remote_files
        if f.startswith("data/")
        and f.endswith(".parquet")
        and any(Path(f).name.startswith(prefix + "-") for prefix in LEGACY_SPLIT_PREFIXES)
    ]
    for remote in legacy_files:
        print(f"Deleting legacy file {remote} …", flush=True)
        try:
            api.delete_file(
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Remove legacy split file {remote} (2.0.0 taxonomy)",
            )
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"WARNING: failed to delete {remote}: {exc}", file=sys.stderr)

    print(f"\nDone. View at https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

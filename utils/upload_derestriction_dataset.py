"""Upload build/derestriction/*.parquet to the HuggingFace Hub.

Expects ``HF_TOKEN`` in ``.env`` (or the environment) with write access to the
``RevivifAI`` organization. Creates the ``RevivifAI/derestriction`` dataset
repo if it does not yet exist, uploads all three parquet files under ``data/``,
and commits a dataset card README.

Run with::

    uv run python -m utils.upload_derestriction_dataset
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import create_repo, upload_file

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build" / "derestriction"
REPO_ID = "RevivifAI/derestriction"


def _render_readme() -> str:
    summary_path = BUILD_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def _src_table(src_counts: dict[str, int]) -> str:
        lines = ["| Source | Rows |", "|---|---:|"]
        for src, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{src}` | {n:,} |")
        return "\n".join(lines)

    harmful = _src_table(summary["splits"]["harmful"]["sources"])
    harmless = _src_table(summary["splits"]["harmless"]["sources"])
    preservation = _src_table(summary["splits"]["preservation"]["sources"])

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
      - split: harmful
        path: data/harmful-*.parquet
      - split: harmless
        path: data/harmless-*.parquet
      - split: preservation
        path: data/preservation-*.parquet
---

# Derestriction

Three aligned 10,000-row splits used to compute refusal directions, apply
orthogonal-projection abliteration, and preserve downstream capabilities
when uncensoring language models. Assembled from public HuggingFace datasets
by [RevivifAI](https://huggingface.co/RevivifAI) for the
[llm-derestrictor](https://github.com/RevivifAI/llm-derestrictor) toolkit.

## Safety notice

> The `harmful` split contains prompts designed to elicit genuinely harmful
> content. It exists for **alignment / abliteration / red-team research only**
> — do **not** fine-tune models toward these prompts. Prompts that mention,
> target, or involve children (including teenagers and students) have been
> removed via a two-stage keyword + semantic filter before sampling.

## Schema

Every row in every split has exactly two columns:

| Column   | Type   | Description                                         |
|----------|--------|-----------------------------------------------------|
| `Prompt` | string | A single instruction prompt, normalized whitespace. |
| `Source` | string | HuggingFace dataset id the prompt came from.        |

## Splits

All three splits contain exactly **10,000** rows, deduplicated case-insensitively
both within and across splits (no prompt appears in more than one split).

### harmful — AdvBench-style refusal-target prompts

{harmful}

Filter: prompts are passed through a two-stage harm filter before sampling —
a keyword regex for child-related terms plus a semantic cosine-similarity
pass against child-harm anchor prompts using
`sentence-transformers/all-MiniLM-L6-v2` at threshold 0.55. See
[`utils/harm_filter.py`](https://github.com/RevivifAI/llm-derestrictor/blob/main/utils/harm_filter.py)
for the full keyword list and anchors.

### harmless — benign instruction prompts (Alpaca-style)

{harmless}

### preservation — capability-preservation prompts (math, code, reasoning, QA)

{preservation}

## Loading

```python
from datasets import load_dataset

harmful = load_dataset("RevivifAI/derestriction", split="harmful")
harmless = load_dataset("RevivifAI/derestriction", split="harmless")
preservation = load_dataset("RevivifAI/derestriction", split="preservation")

print(harmful[0])
# {{'Prompt': '...', 'Source': 'LLM-LAT/harmful-dataset'}}
```

## Intended use

- Computing refusal directions for abliteration / orthogonal-activation-steering research.
- Evaluating refusal-rate and over-refusal-rate under controlled conditions.
- Capturing preservation activations for null-space-constrained weight edits
  (see [AlphaEdit](https://arxiv.org/abs/2410.02355)).

**Not** intended for capability training or reinforcement fine-tuning toward
the harmful prompts. Redistribute under MIT with this safety notice intact.

## Reproduction

```bash
git clone https://github.com/RevivifAI/llm-derestrictor
cd llm-derestrictor
uv sync --group dev
uv run python -m utils.build_derestriction_dataset
uv run python -m utils.upload_derestriction_dataset
```

Seed: `{summary["seed"]}`.

## Citation

```
@misc{{derestriction2026,
  author       = {{RevivifAI}},
  title        = {{Derestriction: Aligned harmful/harmless/preservation prompt splits for abliteration research}},
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

    files = {
        f"data/{name}-00000-of-00001.parquet": BUILD_DIR / f"{name}-00000-of-00001.parquet"
        for name in ("harmful", "harmless", "preservation")
    }
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

    print(f"\nDone. View at https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

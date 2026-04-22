"""Thin loader for the RevivifAI/derestriction dataset on HuggingFace.

Every prompt used by this project (harmful / harmless / preservation) is
sourced exclusively from ``RevivifAI/derestriction``. Call :func:`load_split`
to get a list of prompt strings for a split. The dataset caches under
``~/.cache/huggingface/datasets`` on first use.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv()


DATASET_ID = "RevivifAI/derestriction"
VALID_SPLITS: tuple[str, ...] = ("harmful", "harmless", "preservation")


def _validate_split(split: str) -> str:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {VALID_SPLITS}."
        )
    return split


@lru_cache(maxsize=8)
def _load_dataset_cached(split: str, revision: Optional[str]):
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    return load_dataset(DATASET_ID, split=split, revision=revision, token=token)


def load_split(
    split: str,
    n: Optional[int] = None,
    seed: int = 0,
    revision: Optional[str] = None,
) -> list[str]:
    """Return the ``Prompt`` column of a split as a list of strings.

    Args:
        split: One of ``"harmful"``, ``"harmless"``, ``"preservation"``.
        n: If given and less than split size, random-sample ``n`` prompts.
        seed: RNG seed for the sampling.
        revision: Optional Hub revision / git SHA to pin.
    """
    _validate_split(split)
    ds = _load_dataset_cached(split, revision)
    prompts = list(ds["Prompt"])
    if n is not None and n < len(prompts):
        rng = random.Random(seed)
        prompts = rng.sample(prompts, n)
    return prompts


def refresh_cache(splits: Iterable[str] = VALID_SPLITS) -> None:
    """Force-redownload the dataset from the Hub, bypassing the local cache."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    _load_dataset_cached.cache_clear()
    for split in splits:
        _validate_split(split)
        load_dataset(
            DATASET_ID,
            split=split,
            token=token,
            download_mode="force_redownload",
        )


def split_size(split: str, revision: Optional[str] = None) -> int:
    _validate_split(split)
    return len(_load_dataset_cached(split, revision))

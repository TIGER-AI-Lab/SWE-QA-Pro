"""HuggingFace Hub benchmark loader."""

from __future__ import annotations

from typing import Iterable, Optional

from datasets import load_dataset

BENCH_REPO_ID = "TIGER-Lab/SWE-QA-Pro-Bench"


def load_benchmark(split: str = "test", limit: Optional[int] = None):
    ds = load_dataset(BENCH_REPO_ID, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def iter_records(ds) -> Iterable[dict]:
    for row in ds:
        yield dict(row)

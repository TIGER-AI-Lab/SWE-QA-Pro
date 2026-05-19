"""Aggregate average scores across one or more scored jsonl files.

Usage:
    python -m sweqapro.scoring.stats out/foo.scored.jsonl out/bar.scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCORE_KEYS = (
    "correctness",
    "completeness",
    "clarity",
    "relevance",
    "reasoning",
    "total_score",
)


def _aggregate(paths: list[Path]) -> dict:
    sums = defaultdict(float)
    counts = defaultdict(int)
    for p in paths:
        if not p.exists():
            print(f"[stats] missing file: {p}", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k in SCORE_KEYS:
                    if k in obj:
                        sums[k] += obj[k]
                        counts[k] += 1
    return {k: (sums[k] / counts[k] if counts[k] else None, counts[k]) for k in SCORE_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SWE-QA-Pro judge scores")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    result = _aggregate(args.files)
    print(f"Files processed: {len(args.files)}")
    print("-" * 40)
    for k in SCORE_KEYS:
        avg, count = result[k]
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"
        print(f"{k:12s} | count={count:5d} | avg={avg_str}")


if __name__ == "__main__":
    main()

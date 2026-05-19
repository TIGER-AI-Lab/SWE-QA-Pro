#!/usr/bin/env python3
"""
Unified judge runner. Scores a benchmark output JSONL using either the OpenAI
or DeepSeek chat completions API as the judge backend.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sweqapro.scoring.judge import evaluate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SWE-QA-Pro benchmark outputs")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge", choices=("openai", "deepseek"), default="openai")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="low",
        help="OpenAI judge only: reasoning.effort passed to /v1/responses (default: low).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    summary = evaluate_jsonl(
        input_path=args.input,
        output_path=args.output,
        judge_provider=args.judge,
        judge_model=args.judge_model,
        max_workers=args.workers,
        resume=not args.no_resume,
        reasoning_effort=args.reasoning_effort,
    )
    print(
        f"[run_judge] scored={summary['scored']} "
        f"avg_total={summary['average_total_score']} "
        f"judge={summary['judge_provider']}/{summary['judge_model']} "
        f"output={summary['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Direct-answer benchmark runner (no tools, single-turn).

For vllm-local models this script owns a single subprocess-managed local vLLM
server whose lifetime is bound to this one invocation. The `agent_vllm` block
of the model spec is ignored because direct mode never emits tool calls.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sweqapro import registry
from sweqapro.data import load_benchmark, iter_records
from sweqapro.direct import run_direct
from sweqapro.vllm_server import VLLMServer


@contextmanager
def _server_context(spec):
    if spec.provider == "vllm-local":
        with VLLMServer(
            spec.model_id,
            with_tools=False,
            vllm=spec.vllm,
        ) as server:
            yield server.base_url
        return
    yield None


def run(
    model_name: str,
    output: Path,
    split: str,
    limit: Optional[int],
    resume: bool,
) -> None:
    spec = registry.resolve(model_name)
    print(
        f"[run_direct] model={model_name} provider={spec.provider} output={output}",
        flush=True,
    )

    ds = load_benchmark(split=split, limit=limit)
    total = len(ds)
    print(f"[run_direct] total={total}", flush=True)

    with _server_context(spec) as base_url:
        llm = registry.build_llm(spec, base_url=base_url, with_tools=False)
        count = run_direct(
            llm,
            iter_records(ds),
            output,
            resume=resume,
            total=total,
            desc=f"direct:{model_name}",
        )
        print(f"[run_direct] wrote {count} new records to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct-answer SWE-QA-Pro benchmark")
    parser.add_argument("--model", required=True, help=f"One of: {', '.join(registry.list_models())}")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(
        model_name=args.model,
        output=args.output,
        split=args.split,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

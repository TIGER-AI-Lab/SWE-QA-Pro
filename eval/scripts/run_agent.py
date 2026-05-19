#!/usr/bin/env python3
"""
Agent-mode benchmark runner.

Loads the SWE-QA-Pro-Bench from HuggingFace Hub, dispatches per model provider,
and for vllm-local models spawns a single subprocess-managed local vLLM server
whose lifetime is bound to this one invocation.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Set

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sweqapro import registry
from sweqapro.agent import ToolCallingAgent
from sweqapro.data import load_benchmark
from sweqapro.vllm_server import VLLMServer

API_PROVIDERS = {"openai", "anthropic", "gemini"}
VLLM_WORKERS_CAP = 4


def _load_done_queries(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("agent_result") and obj.get("question"):
                done.add(obj["question"])
    return done


def _resolve_repo_path(repo_root: Path, repo_name: str) -> Path:
    return repo_root / repo_name.split("/")[-1]


def _run_one(agent: ToolCallingAgent, item: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    question = item["question"]
    repo_name = item["repo"]
    repo_path = _resolve_repo_path(repo_root, repo_name)
    if not repo_path.exists():
        return {**item, "agent_result": None, "error": f"Repo not found: {repo_path}"}

    result = agent.query(question=question, repo_path=str(repo_path))
    result_to_save = dict(result)
    result_to_save.pop("trajectory", None)
    return {**item, "agent_result": result_to_save}


def _resolve_workers(requested: int, provider: str) -> int:
    if provider in API_PROVIDERS:
        return max(1, requested)
    if requested > VLLM_WORKERS_CAP:
        print(
            f"[run_agent] --workers={requested} is above the vllm-local cap "
            f"({VLLM_WORKERS_CAP}); clamping to {VLLM_WORKERS_CAP}. "
            f"A single vLLM server is shared across workers.",
            flush=True,
        )
        return VLLM_WORKERS_CAP
    return max(1, requested)


@contextmanager
def _server_context(spec, with_tools: bool):
    if spec.provider == "vllm-local":
        with VLLMServer(
            spec.model_id,
            with_tools=with_tools,
            vllm=spec.vllm,
            agent_vllm=spec.agent_vllm,
        ) as server:
            yield server.base_url
        return
    yield None


def run(
    model_name: str,
    output: Path,
    repo_root: Path,
    split: str,
    limit: Optional[int],
    workers: int,
    resume: bool,
) -> None:
    spec = registry.resolve(model_name)
    effective_workers = _resolve_workers(workers, spec.provider)
    print(
        f"[run_agent] model={model_name} provider={spec.provider} "
        f"workers={effective_workers} output={output}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_queries(output) if resume else set()

    ds = load_benchmark(split=split, limit=limit)
    pending = [dict(ds[i]) for i in range(len(ds)) if ds[i]["question"] not in done]
    print(
        f"[run_agent] total={len(ds)} done={len(done)} pending={len(pending)}",
        flush=True,
    )

    with _server_context(spec, with_tools=True) as base_url:
        llm = registry.build_llm(spec, base_url=base_url, with_tools=True)
        llm_no_tools = registry.build_llm(spec, base_url=base_url, with_tools=False)
        agent = ToolCallingAgent(
            llm=llm,
            llm_no_tools=llm_no_tools,
            provider=("openai" if spec.provider == "vllm-local" else spec.provider),
            model_label=model_name,
            max_iterations=spec.max_iterations,
            history_window=spec.history_window,
            max_context_length=spec.max_context_length,
            context_warning_threshold=spec.context_warning_threshold,
        )

        write_lock = threading.Lock()
        with output.open("a", encoding="utf-8") as fout:
            if effective_workers <= 1:
                for item in tqdm(pending, desc=f"agent:{model_name}"):
                    try:
                        obj = _run_one(agent, item, repo_root)
                    except Exception as e:
                        obj = {**item, "agent_result": None, "error": str(e)}
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()
            else:
                with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                    futures = {ex.submit(_run_one, agent, item, repo_root): item for item in pending}
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"agent:{model_name}"):
                        item = futures[fut]
                        try:
                            obj = fut.result()
                        except Exception as e:
                            obj = {**item, "agent_result": None, "error": str(e)}
                        with write_lock:
                            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            fout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-mode SWE-QA-Pro benchmark")
    parser.add_argument("--model", required=True, help=f"One of: {', '.join(registry.list_models())}")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("./repos"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(
        model_name=args.model,
        output=args.output,
        repo_root=args.repo_root,
        split=args.split,
        limit=args.limit,
        workers=args.workers,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

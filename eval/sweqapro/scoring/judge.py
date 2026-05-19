"""Unified LLM judge backed by either OpenAI or DeepSeek (OpenAI-compatible)."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from openai import OpenAI
from tqdm import tqdm

from ..config import PROMPTS_DIR

JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_prompt.txt"

SCORE_KEYS = ("correctness", "completeness", "clarity", "relevance", "reasoning")

DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-5-2025-08-07",
    "deepseek": "deepseek-chat",
}

DEFAULT_REASONING_EFFORT = "low"


def _load_prompt_template() -> str:
    with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _make_client(provider: str) -> OpenAI:
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    else:
        raise ValueError(f"Unknown judge provider: {provider}")
    if not api_key:
        raise RuntimeError(f"Missing API key for judge provider '{provider}'")
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_candidate(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    agent = record.get("agent_result") or {}
    if isinstance(agent, dict):
        ans = (agent.get("answer") or "").strip()
        if ans and ans != "No answer found":
            return {"answer": ans, "source": "agent"}
    direct = (record.get("direct_answer") or "").strip()
    if direct and direct != "No answer found":
        return {"answer": direct, "source": "direct"}
    return None


def _parse_scores(text: str) -> Optional[Dict[str, int]]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    for k in SCORE_KEYS:
        if k not in data or not isinstance(data[k], int) or not (1 <= data[k] <= 10):
            return None
    return {k: int(data[k]) for k in SCORE_KEYS}


def _score_one(
    provider: str,
    client: OpenAI,
    model: str,
    template: str,
    question: str,
    reference: str,
    candidate: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Optional[Dict[str, int]]:
    prompt = template.format(question=question, reference=reference, candidate=candidate)
    try:
        if provider == "openai":
            # GPT-5 / o-series use the /v1/responses endpoint with explicit
            # reasoning and verbosity controls. `temperature` is not accepted
            # on reasoning models — `reasoning.effort` controls latency/cost.
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": reasoning_effort},
                text={"verbosity": "low"},
            )
            content = (getattr(resp, "output_text", None) or "").strip()
        else:
            # DeepSeek (and other OpenAI-compatible providers) use /v1/chat/completions.
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[judge] API error: {e}", flush=True)
        return None
    return _parse_scores(content)


def _load_scored(path: Path) -> Set[str]:
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
            q = obj.get("question")
            if q:
                done.add(q)
    return done


def _process(
    record: Dict[str, Any],
    provider: str,
    client: OpenAI,
    model: str,
    template: str,
    reasoning_effort: str,
) -> Optional[Dict[str, Any]]:
    question = record.get("question") or ""
    reference = record.get("answer") or ""
    if not reference:
        return None
    cand = _get_candidate(record)
    if cand is None:
        return None
    scores = _score_one(
        provider, client, model, template, question, reference, cand["answer"],
        reasoning_effort=reasoning_effort,
    )
    if scores is None:
        return None
    return {
        "repo": record.get("repo", ""),
        "commit_id": record.get("commit_id", ""),
        "cluster": record.get("cluster", ""),
        "qa_type": record.get("qa_type", ""),
        "question": question,
        "candidate_answer": cand["answer"],
        "answer_source": cand["source"],
        "reference_answer": reference,
        **scores,
        "total_score": sum(scores.values()),
    }


def evaluate_jsonl(
    input_path: Path,
    output_path: Path,
    judge_provider: str = "openai",
    judge_model: Optional[str] = None,
    max_workers: int = 4,
    resume: bool = True,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    judge_model = judge_model or DEFAULT_JUDGE_MODELS.get(judge_provider)
    if not judge_model:
        raise ValueError(f"No default judge model for provider '{judge_provider}'")

    template = _load_prompt_template()
    client = _make_client(judge_provider)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_scored(output_path) if resume else set()

    all_records = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_records.append(rec)

    pending = [r for r in all_records if r.get("question") not in done]
    print(
        f"[run_judge] judge={judge_provider}/{judge_model} "
        f"total={len(all_records)} resumed={len(done)} pending={len(pending)} "
        f"workers={max_workers}",
        flush=True,
    )

    mode = "a" if resume and output_path.exists() else "w"
    total = 0
    sum_score = 0
    failed = 0

    pbar_desc = f"judge:{judge_provider}"
    with output_path.open(mode, encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process, rec, judge_provider, client, judge_model, template, reasoning_effort
                ): rec
                for rec in pending
            }
            with tqdm(total=len(pending), desc=pbar_desc) as pbar:
                running_avg = 0.0
                for fut in concurrent.futures.as_completed(futures):
                    src_rec = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        failed += 1
                        tqdm.write(f"[judge] exception on '{src_rec.get('question','')[:60]}...': {e}")
                        pbar.update(1)
                        continue

                    if result is None:
                        failed += 1
                        tqdm.write(f"[judge] no-score for '{src_rec.get('question','')[:60]}...' (parse fail or missing reference)")
                        pbar.update(1)
                        continue

                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    total += 1
                    sum_score += result["total_score"]
                    running_avg = sum_score / total

                    tqdm.write(
                        f"[{result['answer_source']:6s}] {result['total_score']:>2}/50 "
                        f"(c={result['correctness']} cp={result['completeness']} "
                        f"r={result['relevance']} cl={result['clarity']} rs={result['reasoning']}) "
                        f"{result['question'][:70]}..."
                    )
                    pbar.set_postfix(avg=f"{running_avg:.2f}", failed=failed)
                    pbar.update(1)

    avg = (sum_score / total) if total else 0.0
    print(
        f"[run_judge] done. scored={total} failed={failed} avg_total_score={avg:.4f} -> {output_path}",
        flush=True,
    )
    return {
        "scored": total,
        "average_total_score": round(avg, 4),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "output": str(output_path),
    }

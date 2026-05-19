"""Direct-mode runner: single-turn answer without any tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

DIRECT_SYSTEM_PROMPT = (
    "You are a senior software engineer answering questions about open-source "
    "repositories. Answer based on your prior knowledge of the named repository. "
    "Be precise, complete, and cite specific files/symbols when relevant. Output "
    "your final answer inside a <finish>...</finish> block."
)


def _load_done(path: Path) -> Set[str]:
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
            if obj.get("direct_answer") and obj.get("question"):
                done.add(obj["question"])
    return done


def _build_user(record: dict) -> str:
    repo = record.get("repo", "")
    question = record.get("question", "")
    return (
        f"Repository: {repo}\n"
        f"Question: {question}\n\n"
        "Answer the question using your knowledge of this repository. "
        "Wrap the final answer in <finish>...</finish>."
    )


def run_direct(
    llm,
    records: Iterable[dict],
    output: Path,
    resume: bool = True,
    total: Optional[int] = None,
    desc: str = "direct",
) -> int:
    """Run direct-mode benchmark.

    ``total`` is used for the progress bar; if not provided, we try ``len(records)``
    (works for lists / HF Datasets) and otherwise fall back to an untotaled bar.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(output) if resume else set()
    mode = "a" if resume and output.exists() else "w"

    if total is None:
        try:
            total = len(records)  # type: ignore[arg-type]
        except TypeError:
            total = None

    written = 0
    with output.open(mode, encoding="utf-8") as fout:
        for rec in tqdm(records, total=total, desc=desc):
            q = rec.get("question")
            if q in done:
                continue
            messages = [
                SystemMessage(content=DIRECT_SYSTEM_PROMPT),
                HumanMessage(content=_build_user(rec)),
            ]
            try:
                resp = llm.invoke(messages)
                content = resp.content if isinstance(resp.content, str) else str(resp.content)
            except Exception as e:
                obj = {**rec, "direct_answer": None, "error": str(e)}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            obj = {**rec, "direct_answer": content}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
    return written

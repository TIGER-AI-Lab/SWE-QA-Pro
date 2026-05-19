"""Unified LangGraph tool-calling agent for SWE-QA-Pro.

A single `ToolCallingAgent` class works across OpenAI / Anthropic / Gemini /
DeepSeek / vLLM-local backends. The `llm` (with tools bound) and `llm_no_tools`
(for the forced-finish turn) are constructed by `sweqapro.registry.build_llm`
and injected here; this module knows nothing about provider classes.

The loop, retry behavior, force-finish, degenerate-output detection, and history
management mirror SWE-QA-Pro-dev's `base.py` + `toolcall_agent.py`, collapsed
into one provider-agnostic class.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import PROMPTS_DIR
from .history import ConversationHistory
from .tools import execute_readonly_command, semantic_search, view_codebase


_DEGENERATE_REPEAT_RE = re.compile(r"(.{5,80})\1{8,}", re.DOTALL)
_TOOL_LEAK_MARKERS = [
    "toolmessage(",
    "tool_calls=",
    "additional_kwargs=",
    "response_metadata=",
    "file view:",
    "view_codebase",
    "semantic_search(",
    "execute_readonly_command",
]
_FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)


class AgentState(TypedDict):
    trajectory: List[Dict[str, Any]]
    question: str
    repo_path: str
    current_step: int
    final_answer: str
    tool_calls: List[Dict[str, Any]]
    history_manager: ConversationHistory
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    preamble: List[AnyMessage]
    tool_stats: Dict[str, Any]


def _build_messages(state: AgentState, turn_human: Optional[HumanMessage] = None):
    msgs: List[AnyMessage] = []
    msgs.extend(state["preamble"])
    msgs.extend(state["history_manager"].flatten())
    if turn_human is not None:
        msgs.append(turn_human)
    return msgs


class ToolCallingAgent:
    """Single agent class that supports any LangChain chat model with tools bound."""

    def __init__(
        self,
        llm,
        llm_no_tools,
        provider: str,
        model_label: str,
        *,
        max_iterations: int = 10,
        history_window: int = 10,
        max_context_length: int = 32768,
        context_warning_threshold: float = 0.825,
        system_prompt_path: Optional[str] = None,
        quiet: bool = False,
    ):
        self.llm = llm
        self.llm_no_tools = llm_no_tools
        self.provider = provider
        self.model_label = model_label
        self.max_iterations = max_iterations
        self.history_window = history_window
        self.max_context_length = max_context_length
        self.context_warning_threshold = context_warning_threshold
        self.console = Console(markup=False, quiet=quiet)
        self._system_prompt_template = self._load_system_prompt(system_prompt_path)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ prompts

    @staticmethod
    def _load_system_prompt(path: Optional[str]) -> str:
        target = path or str(PROMPTS_DIR / "agent_system_prompt.txt")
        with open(target, "r", encoding="utf-8") as f:
            return f.read()

    def _system_prompt(self, repo_path: str) -> str:
        return self._system_prompt_template.replace("{repo_path}", repo_path)

    @staticmethod
    def _initial_human_prompt(repo_path: str, question: str) -> str:
        return (
            f"Repository Path: {repo_path}\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "- Please analyze the codebase to answer this question.\n"
            "- Provide a step-by-step explanation before calling any tools.\n"
            "- Follow this workflow:\n"
            "  1) Inspect the repository structure\n"
            "  2) Search for relevant files and symbols\n"
            "  3) Examine specific implementations\n"
            "  4) Cross-validate your findings\n"
            "  5) Provide a complete answer with evidence inside a <finish> block\n"
        )

    # ---------------------------------------------------------------- tool exec

    @staticmethod
    def _run_tool(call: Dict[str, Any], repo_path: str) -> str:
        name = call.get("name")
        args = call.get("args", {}) or {}
        try:
            if name == "semantic_search":
                result = semantic_search(
                    term=args["term"],
                    path=args.get("path", repo_path),
                    python_only=args.get("python_only", False),
                    max_files=args.get("max_files", 100),
                    include_lines=args.get("include_lines", False),
                )
                return json.dumps(result)
            if name == "view_codebase":
                return view_codebase(
                    path=args["path"],
                    view_range=args.get("view_range"),
                    concise=args.get("concise", False),
                    python_only=args.get("python_only", False),
                )
            if name == "execute_readonly_command":
                return execute_readonly_command(args["command"])
            return f"ERROR: Unknown tool {name}"
        except Exception as e:
            return f"ERROR: Tool failed: {e}"

    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in getattr(response, "tool_calls", None) or []:
            out.append({"name": tc["name"], "args": tc.get("args") or {}, "id": tc.get("id")})
        return out

    # -------------------------------------------------------------- token / guard

    @staticmethod
    def _extract_token_usage(response: Any) -> Optional[Dict[str, int]]:
        """Return token usage as {prompt_tokens, completion_tokens, total_tokens}.

        Sources, in priority order:
          1. ``response.usage_metadata`` — LangChain 0.3+ unified field
             ({input_tokens, output_tokens, total_tokens}) populated by every
             provider that supports usage reporting.
          2. ``response.response_metadata["token_usage"]`` — OpenAI's legacy
             shape ({prompt_tokens, completion_tokens, total_tokens}).
          3. ``response.response_metadata["usage"]`` — Anthropic / Gemini
             legacy shape; key names vary, so we coalesce.
        """
        def _norm(raw: Dict[str, Any]) -> Dict[str, int]:
            prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0
            completion = raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0
            total = raw.get("total_tokens", 0) or 0
            if not total:
                total = prompt + completion
            return {
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
                "total_tokens": int(total),
            }

        unified = getattr(response, "usage_metadata", None)
        if unified:
            return _norm(unified)

        meta = getattr(response, "response_metadata", None) or {}
        raw = meta.get("token_usage") or meta.get("usage")
        if raw:
            return _norm(raw)
        return None

    def _context_warning(self, token_usage: Optional[Dict[str, int]]) -> Tuple[bool, str]:
        if not token_usage:
            return False, ""
        total = token_usage.get("total_tokens", 0)
        if total == 0:
            return False, ""
        if total / self.max_context_length >= self.context_warning_threshold:
            return True, f"context limit ({total}/{self.max_context_length})"
        return False, ""

    @staticmethod
    def _is_degenerate(text: str) -> bool:
        if not text or len(text.strip()) < 100:
            return False
        lower = text.lower()
        if any(m in lower for m in _TOOL_LEAK_MARKERS):
            return True
        if lower.count("</think>") + lower.count("<think>") >= 8:
            stripped = re.sub(r"</?think>", "", lower)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if len(stripped) < 200:
                return True
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 30:
            cnt = Counter(lines)
            most_common_line, freq = cnt.most_common(1)[0]
            unique_ratio = len(cnt) / len(lines)
            is_importish = most_common_line.startswith(("from ", "import ")) or " import " in most_common_line
            if (unique_ratio < 0.25 and freq >= 10) or (is_importish and freq >= 8):
                return True
        if _DEGENERATE_REPEAT_RE.search(re.sub(r"\s+", " ", text)):
            return True
        return False

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        m = _FINISH_RE.search(text or "")
        return m.group(1).strip() if m else None

    @staticmethod
    def _content_text(content) -> str:
        """Flatten a LangChain response.content into a plain text string.

        OpenAI / Gemini return a plain string. Anthropic (and Bedrock-Claude)
        return a list of content blocks like
            [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
        — we keep only the textual portion; tool_use blocks are surfaced
        separately via ``response.tool_calls`` and don't belong in the text body.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", "") or "")
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _parse_retry_after(error_text: str, default: float = 60.0) -> float:
        """Best-effort parse of how long to wait after a 429.

        OpenAI's 429 body usually contains 'Please try again in 1.234s' (or '12ms').
        Anthropic / vLLM may include 'retry-after: <seconds>' in headers stringified
        into the exception. Fall back to ``default`` when nothing is found.
        """
        m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)\b", error_text, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            return v / 1000.0 if m.group(2).lower() == "ms" else v
        m = re.search(r"retry[-_ ]after[\":\s]+(\d+(?:\.\d+)?)", error_text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return default

    # ------------------------------------------------------------------ logging

    def _log_query_start(self, question: str, repo_path: str) -> None:
        self.console.print(Panel(
            f"Question: {question}\n"
            f"Repository: {repo_path}\n"
            f"Model: {self.model_label} ({self.provider})\n"
            f"Max Iterations: {self.max_iterations}\n"
            f"Max Context Length: {self.max_context_length}",
            title="New Query Started",
            border_style="blue",
            box=box.DOUBLE,
        ))

    def _log_step_start(self, step: int) -> None:
        self.console.print(Panel(
            f"Step {step + 1} Starting",
            title="Agent Step",
            border_style="blue",
            box=box.ROUNDED,
        ))

    def _log_llm_call(self, step: int, prompt_chars: int) -> None:
        t = Table(title=f"LLM Call - Step {step + 1}", box=box.ROUNDED)
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("Model", self.model_label)
        t.add_row("Provider", self.provider)
        t.add_row("Prompt Length", f"{prompt_chars:,} chars")
        self.console.print(t)

    def _log_llm_response(self, content: str, token_usage: Optional[Dict[str, int]]) -> None:
        if token_usage:
            u = Table(title="Token Usage (Current Call)", box=box.SIMPLE)
            u.add_column("Type", style="cyan")
            u.add_column("Value", style="green")
            u.add_row("Input Tokens", str(token_usage.get("prompt_tokens", "N/A")))
            u.add_row("Output Tokens", str(token_usage.get("completion_tokens", "N/A")))
            u.add_row("Total Tokens", str(token_usage.get("total_tokens", "N/A")))
            total = token_usage.get("total_tokens", 0) or 0
            if total > 0:
                ratio = total / self.max_context_length
                color = "red" if ratio >= 0.9 else ("yellow" if ratio >= self.context_warning_threshold else "green")
                u.add_row("Context Usage", f"[{color}]{total}/{self.max_context_length} ({ratio*100:.1f}%)[/{color}]")
            self.console.print(u)

        body = content or "(empty)"
        self.console.print(Panel(
            Syntax(body, "markdown", theme="github-dark"),
            title="LLM Response",
            border_style="green",
            box=box.ROUNDED,
        ))

    def _log_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        t = Table(title=f"Tool Call: {tool_name}", box=box.ROUNDED)
        t.add_column("Parameter", style="cyan")
        t.add_column("Value", style="yellow")
        for k, v in args.items():
            sv = str(v)
            if len(sv) > 100:
                sv = sv[:100] + "..."
            t.add_row(k, sv)
        self.console.print(t)

    def _log_tool_result(self, tool_name: str, result: str, dt: float) -> None:
        preview = result[:300] + "..." if len(result) > 300 else result
        self.console.print(Panel(
            f"Execution Time: {dt:.2f}s\n\nResult Preview:\n{preview}",
            title=f"Tool Result: {tool_name}",
            border_style="cyan",
            box=box.ROUNDED,
        ))

    def _log_final_answer(self, answer: str) -> None:
        self.console.print(Panel(
            Markdown(answer),
            title="Final Answer",
            border_style="green",
            box=box.DOUBLE,
        ))

    def _log_force_finish(self, reason: str, step: int) -> None:
        self.console.print(Panel(
            f"Step {step + 1}: forcing answer generation\nReason: {reason}",
            title="Force Finish",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    def _log_query_complete(
        self,
        total_time: float,
        steps: int,
        tool_calls_count: int,
        stop_reason: str,
        status: str,
        token_usage: Dict[str, int],
    ) -> None:
        t = Table(title="Query Completed", box=box.ROUNDED)
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("Total Time", f"{total_time:.2f}s")
        t.add_row("Total Steps", f"{steps}/{self.max_iterations}")
        t.add_row("Total Tool Calls", str(tool_calls_count))
        t.add_row("Input Tokens", f"{token_usage.get('prompt_tokens', 0):,}")
        t.add_row("Output Tokens", f"{token_usage.get('completion_tokens', 0):,}")
        t.add_row("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
        t.add_row("Stop Reason", stop_reason)
        t.add_row("Status", status)
        self.console.print(t)
        self.console.print()

    def _log_retry(self, attempt: int, error: str) -> None:
        self.console.print(Panel(
            f"Attempt {attempt + 1} failed: {error[:200]}\nRetrying with fresh history window.",
            title="Retry",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    # -------------------------------------------------------------------- graph

    def _should_force_finish(
        self, step: int, token_usage: Optional[Dict[str, int]]
    ) -> Tuple[bool, str]:
        if step >= self.max_iterations - 1:
            return True, "MAX TURNS REACHED"
        warn, reason = self._context_warning(token_usage)
        if warn:
            return True, reason
        return False, ""

    def _force_finish(self, state: AgentState, reason: str) -> AgentState:
        question = state["question"]
        repo_path = state["repo_path"]
        history_manager = state["history_manager"]
        step = state["current_step"]

        self._log_force_finish(reason, step)

        prompt_text = (
            f"--- {reason.upper()} ---\n"
            "This is the final allowed turn. You MUST NOT call any tools.\n"
            "Provide your best possible final answer now inside a <finish>...</finish> block.\n\n"
            f"Question: {question}\nRepository path: {repo_path}\n\n"
            f"Chat history and tool results:\n{history_manager.flatten()}\n"
        )
        force_messages = [
            SystemMessage(content="You are an expert code analyst. Provide a final answer based on collected information."),
            HumanMessage(content=prompt_text),
        ]
        response = self.llm_no_tools.invoke(force_messages)
        token_usage = self._extract_token_usage(response) or {}
        content = self._content_text(response.content)

        if self._is_degenerate(content) or self._extract_final_answer(content) is None:
            # Retry with the full history preamble
            force_human = HumanMessage(content=(
                f"--- {reason.upper()} ---\n"
                "Final allowed turn. NO tool calls. Output a <finish>...</finish> block now."
            ))
            response = self.llm_no_tools.invoke(_build_messages(state, turn_human=force_human))
            token_usage = self._extract_token_usage(response) or {}
            content = self._content_text(response.content)
            if self._is_degenerate(content):
                raise RuntimeError("Degenerate output detected (forced_final)")
            if self._extract_final_answer(content) is None:
                raise RuntimeError("No <finish> block in forced final")

        if "<tool_call>" in content or getattr(response, "tool_calls", None):
            raise RuntimeError("Force generation produced tool_call")

        final_answer = self._extract_final_answer(content) or content.strip()
        state["final_answer"] = final_answer
        state["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
        state["completion_tokens"] = token_usage.get("completion_tokens", 0)
        state["total_tokens"] = token_usage.get("total_tokens", 0)
        state["current_step"] = step + 1
        history_manager.add_interaction([AIMessage(content=content)])
        self._log_llm_response(content, token_usage)
        self._log_final_answer(final_answer)
        return state

    def _agent_step(self, state: AgentState) -> AgentState:
        step = state.get("current_step", 0)
        history_manager = state["history_manager"]
        token_usage = {
            "prompt_tokens": state["prompt_tokens"],
            "completion_tokens": state["completion_tokens"],
            "total_tokens": state["total_tokens"],
        }
        force, reason = self._should_force_finish(step, token_usage)
        if force:
            return self._force_finish(state, reason)

        self._log_step_start(step)
        messages = _build_messages(state)
        prompt_chars = sum(len(str(m.content)) for m in messages)
        self._log_llm_call(step, prompt_chars)
        response = self.llm.invoke(messages)
        token_usage = self._extract_token_usage(response) or {}
        content = self._content_text(response.content)
        self._log_llm_response(content, token_usage)

        final_answer = self._extract_final_answer(content)
        if final_answer:
            state["final_answer"] = final_answer
            state["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
            state["completion_tokens"] = token_usage.get("completion_tokens", 0)
            state["total_tokens"] = token_usage.get("total_tokens", 0)
            state["current_step"] = step + 1
            history_manager.add_interaction([AIMessage(content=content)])
            self._log_final_answer(final_answer)
            return state

        tool_calls_data = self._extract_tool_calls(response)

        if (
            not tool_calls_data
            and re.search(
                r"\b(semantic_search|view_codebase|execute_readonly_command)\b",
                content.lower(),
            )
        ):
            raise RuntimeError("Tool mentioned but not called (degenerate planning loop)")

        step_messages: List[AnyMessage] = [
            AIMessage(content=content, tool_calls=getattr(response, "tool_calls", None) or [])
        ]

        formatted: List[Dict[str, Any]] = []
        for tc in tool_calls_data:
            tool_name = tc["name"]
            tool_args = tc["args"]
            self._log_tool_call(tool_name, tool_args)
            t0 = time.time()
            result = self._run_tool({"name": tool_name, "args": tool_args}, state["repo_path"])
            dt = time.time() - t0
            self._log_tool_result(tool_name, result, dt)

            state["tool_stats"]["counts"][tool_name] = (
                state["tool_stats"]["counts"].get(tool_name, 0) + 1
            )
            state["tool_stats"]["records"].append({
                "step": step,
                "tool": tool_name,
                "args": tool_args,
                "execution_time": dt,
            })
            formatted.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "execution_time": dt,
            })
            step_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        if tool_calls_data:
            history_manager.add_interaction(step_messages)
            state["trajectory"].append({
                "step": step,
                "response": content,
                "tool_calls": formatted,
                "token_usage": token_usage,
            })
            state["tool_calls"] = state.get("tool_calls", []) + formatted
        else:
            # No tools, no finish — record assistant turn to preserve history.
            history_manager.add_interaction([AIMessage(content=content)])

        state["current_step"] = step + 1
        state["history_manager"] = history_manager
        state["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
        state["completion_tokens"] = token_usage.get("completion_tokens", 0)
        state["total_tokens"] = token_usage.get("total_tokens", 0)
        return state

    def _build_graph(self):
        def should_continue(state: AgentState) -> str:
            if state.get("final_answer"):
                return "end"
            if state.get("current_step", 0) > self.max_iterations - 1:
                return "end"
            return "continue"

        wf = StateGraph(AgentState)
        wf.add_node("agent", self._agent_step)
        wf.add_edge(START, "agent")
        wf.add_conditional_edges("agent", should_continue, {"continue": "agent", "end": END})
        return wf.compile()

    # ----------------------------------------------------------------- public

    def query(self, question: str, repo_path: str) -> Dict[str, Any]:
        self._log_query_start(question, repo_path)
        preamble = [
            SystemMessage(content=self._system_prompt(repo_path)),
            HumanMessage(content=self._initial_human_prompt(repo_path, question)),
        ]

        # Up to 7 attempts on token/degenerate errors, sliding history window each retry.
        retry_token_keywords = [
            "context length", "max tokens", "token limit", "too many tokens",
            "input too long", "context window", "rate limit reached",
            "tool_call", "repetitive/looping content", "no finish found",
            "tool mentioned but not called", "no <finish>",
        ]
        attempts = 7

        for attempt in range(attempts):
            try:
                initial_state = AgentState(
                    question=question,
                    repo_path=repo_path,
                    current_step=0,
                    final_answer="",
                    tool_calls=[],
                    history_manager=ConversationHistory(self.history_window),
                    trajectory=[],
                    total_tokens=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    preamble=preamble,
                    tool_stats={"counts": {}, "records": []},
                )
                t0 = time.time()
                final_state = self.graph.invoke(
                    initial_state,
                    config={"recursion_limit": self.max_iterations + 1},
                )
                elapsed = time.time() - t0

                answer = final_state.get("final_answer") or "No answer found"
                steps = final_state.get("current_step", 0)
                if answer != "No answer found":
                    stop_reason = (
                        f"Forced after max iterations ({self.max_iterations})"
                        if steps >= self.max_iterations
                        else "Natural completion"
                    )
                    status = "success"
                else:
                    stop_reason = "Unknown"
                    status = "unknown"

                token_usage_out = {
                    "prompt_tokens": final_state.get("prompt_tokens", 0),
                    "completion_tokens": final_state.get("completion_tokens", 0),
                    "total_tokens": final_state.get("total_tokens", 0),
                }
                self._log_query_complete(
                    total_time=elapsed,
                    steps=steps,
                    tool_calls_count=len(final_state.get("tool_calls", [])),
                    stop_reason=stop_reason,
                    status=status,
                    token_usage=token_usage_out,
                )

                return {
                    "query": question,
                    "code_base_dir": repo_path,
                    "answer": answer,
                    "status": status,
                    "stop_reason": stop_reason,
                    "steps_completed": steps,
                    "max_iterations": self.max_iterations,
                    "trajectory": final_state["trajectory"],
                    "retry_attempts": attempt,
                    "token_usage": {
                        "prompt_tokens": final_state.get("prompt_tokens", 0),
                        "completion_tokens": final_state.get("completion_tokens", 0),
                        "total_tokens": final_state.get("total_tokens", 0),
                    },
                    "total_time": elapsed,
                    "tool_usage": {
                        "counts": final_state["tool_stats"]["counts"],
                        "records": final_state["tool_stats"]["records"],
                    },
                    "provider": self.provider,
                    "model": self.model_label,
                }

            except Exception as e:
                msg = str(e).lower()
                if any(kw in msg for kw in retry_token_keywords) and attempt < attempts - 1:
                    # 429 rate-limit: sleep until the bucket refills before retrying,
                    # otherwise we just blast the provider again and re-fail.
                    if "rate limit" in msg or "429" in msg:
                        delay = self._parse_retry_after(str(e))
                        self._log_retry(attempt, f"{e} (sleeping {delay:.1f}s before retry)")
                        time.sleep(delay)
                    else:
                        self._log_retry(attempt, str(e))
                    continue
                return {
                    "query": question,
                    "code_base_dir": repo_path,
                    "answer": f"Error: {e}",
                    "status": "failed",
                    "stop_reason": f"Error after {attempt + 1} attempts",
                    "steps_completed": 0,
                    "max_iterations": self.max_iterations,
                    "trajectory": [],
                    "retry_attempts": attempt,
                    "error": str(e),
                    "provider": self.provider,
                    "model": self.model_label,
                }

        return {
            "query": question,
            "code_base_dir": repo_path,
            "answer": "Error: all retry attempts exhausted",
            "status": "failed",
        }

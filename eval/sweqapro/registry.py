"""Model registry. Reads `configs/models.yaml` and constructs LangChain LLMs.

The registry resolves a user-facing `--model` name to a `ModelSpec` and provides
`build_llm(spec, base_url=None, with_tools=True)` which returns a LangChain Chat
model already `.bind_tools`'d (when applicable). The agent layer never imports
provider-specific classes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

from .config import MODELS_YAML
from .tool_schemas import TOOL_SCHEMAS


@dataclass
class ModelSpec:
    name: str
    provider: str
    model_id: str
    base_url_env: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_context_length: int = 32768
    max_iterations: int = 10
    history_window: int = 10
    context_warning_threshold: float = 0.825
    vllm: Optional[Dict[str, Any]] = None
    agent_vllm: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


_cache: Dict[str, Any] = {}


def _load_yaml() -> Dict[str, Any]:
    if "doc" not in _cache:
        with open(MODELS_YAML, "r", encoding="utf-8") as f:
            _cache["doc"] = yaml.safe_load(f) or {}
    return _cache["doc"]


def list_models() -> list[str]:
    return list((_load_yaml().get("models") or {}).keys())


def resolve(name: str) -> ModelSpec:
    doc = _load_yaml()
    defaults = doc.get("defaults") or {}
    models = doc.get("models") or {}
    if name not in models:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(models.keys()) or '<none>'}"
        )
    raw = models[name]
    return ModelSpec(
        name=name,
        provider=raw["provider"],
        model_id=raw["model_id"],
        base_url_env=raw.get("base_url_env"),
        api_key_env=raw.get("api_key_env"),
        temperature=float(raw.get("temperature", defaults.get("temperature", 0.0))),
        max_context_length=int(raw.get("max_context_length", defaults.get("max_context_length", 32768))),
        max_iterations=int(raw.get("max_iterations", defaults.get("max_iterations", 10))),
        history_window=int(raw.get("history_window", defaults.get("history_window", 10))),
        context_warning_threshold=float(
            raw.get("context_warning_threshold", defaults.get("context_warning_threshold", 0.825))
        ),
        vllm=raw.get("vllm"),
        agent_vllm=raw.get("agent_vllm"),
        extra={k: v for k, v in raw.items() if k not in {
            "provider", "model_id", "base_url_env", "api_key_env",
            "temperature", "max_context_length", "max_iterations",
            "history_window", "context_warning_threshold", "vllm", "agent_vllm",
        }},
    )


def _require_env(var: str, ctx: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"Environment variable {var} is required for {ctx} but is unset")
    return val


def build_llm(spec: ModelSpec, base_url: Optional[str] = None, with_tools: bool = True):
    """Return a LangChain chat model. `base_url` overrides the env-resolved URL
    (used by vllm-local where the server URL is only known at runtime)."""
    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = _require_env(spec.api_key_env or "OPENAI_API_KEY", spec.name)
        url = base_url or (spec.base_url_env and os.environ.get(spec.base_url_env)) or None
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=url,
            model=spec.model_id,
            temperature=spec.temperature,
        )
        if with_tools:
            return llm.bind_tools(TOOL_SCHEMAS, parallel_tool_calls=False)
        return llm

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = _require_env(spec.api_key_env or "ANTHROPIC_API_KEY", spec.name)
        llm = ChatAnthropic(
            api_key=api_key,
            model=spec.model_id,
            temperature=spec.temperature,
        )
        if with_tools:
            return llm.bind_tools(TOOL_SCHEMAS)
        return llm

    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = _require_env(spec.api_key_env or "GEMINI_API_KEY", spec.name)
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=spec.model_id,
            temperature=spec.temperature,
        )
        if with_tools:
            return llm.bind_tools(TOOL_SCHEMAS)
        return llm

    if spec.provider == "vllm-local":
        from langchain_openai import ChatOpenAI

        if base_url is None:
            raise RuntimeError(
                f"vllm-local model '{spec.name}' requires a base_url. "
                "Start the VLLMServer context first."
            )
        llm = ChatOpenAI(
            api_key="EMPTY",
            base_url=base_url,
            model=spec.model_id,
            temperature=spec.temperature,
        )
        if with_tools:
            return llm.bind_tools(TOOL_SCHEMAS, parallel_tool_calls=False)
        return llm

    raise ValueError(f"Unknown provider: {spec.provider}")

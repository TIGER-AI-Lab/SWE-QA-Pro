# SWE-QA-Pro Bench Evaluation

This folder contains code needed to evaluate an LLM on the [SWE-QA-Pro Bench](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench) benchmark under two settings:

- **Direct mode** — the LLM answers in a single turn from its prior knowledge of the repository, with no tools.
- **Agent mode** — the LLM iterates a LangGraph tool-calling loop with `semantic_search`, `view_codebase`, and `execute_readonly_command` against the target repository, then emits a final answer inside a `<finish>...</finish>` block.

A unified judge (OpenAI or DeepSeek) scores the candidate answers against the reference ground truth.

## Supported models

| Name (`--model`) | Provider | Notes |
| --- | --- | --- |
| `gpt-4o` | OpenAI | |
| `gpt-4.1` | OpenAI | |
| `claude-sonnet-4.5` | Anthropic (native) | via `langchain-anthropic` |
| `gemini-2.5-pro` | Google | via `langchain-google-genai` |
| `deepseek-v4` | DeepSeek (OpenAI-compatible) | |
| `qwen3-8b` | vLLM-local | subprocess-managed local vLLM server |
| `qwen3-32b` | vLLM-local | subprocess-managed local vLLM server |
| `devstral-24b` | vLLM-local | subprocess-managed local vLLM server |
| `llama-3.3-70b` | vLLM-local | subprocess-managed local vLLM server |

For the `vllm-local` models, `scripts/run_agent.py` / `scripts/run_direct.py` start a single local vLLM OpenAI-compatible server as a **child subprocess** on a free port, poll `GET /v1/models` until it returns 200, run the benchmark, and then terminate the server on exit. The vLLM server is NOT embedded in the current Python interpreter; it is an external process whose lifetime is scoped to a single runner invocation.

To add, remove, or retune a model, edit `configs/models.yaml` — no code changes required.

## Install

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# cd into the eval/ directory of this checkout
cd path/to/SWE-QA-Pro/eval

# Create virtual environment (must NOT be named `sweqapro` — that is the package directory)
uv venv --python 3.11 .venv
# Activate environment
source .venv/bin/activate
# Install dependencies
uv pip install -r requirements.txt
# Fill in the API keys you need in .env
```

You only need API keys for the providers you intend to run. For example, a pure `qwen3-8b` evaluation needs no remote keys at all; a `gpt-4o` evaluation only needs `OPENAI_API_KEY`.

## Clone the target repositories

Agent mode needs the actual source trees on disk — each benchmark item points to a `(repo, commit_id)` pair, and the agent runs its read-only tools against the checked-out copy. Use `clone_repos.sh` to clone all 26 repositories at their exact benchmark commits:

```bash
# from eval/
bash clone_repos.sh
```

This populates `./repos/<repo>` (matching the default `--repo-root`). The repository list and commit hashes are in [`repos.txt`](repos.txt) — one `<git_url> <commit_hash>` per line. Override the source list or destination via env vars if needed:

```bash
REPO_FILE=./my-repos.txt TARGET_DIR=/data/swe-qa-repos bash clone_repos.sh
```

If you put the repos somewhere else, pass `--repo-root <dir>` to the runners.

## Run

### Agent mode

```bash
# API-hosted models
python scripts/run_agent.py --model gpt-4o        --output out/gpt-4o.agent.jsonl
python scripts/run_agent.py --model gpt-4.1       --output out/gpt-4.1.agent.jsonl
python scripts/run_agent.py --model claude-sonnet-4.5 --output out/claude.agent.jsonl
python scripts/run_agent.py --model gemini-2.5-pro --output out/gemini.agent.jsonl
python scripts/run_agent.py --model deepseek-v4 --output out/deepseek.agent.jsonl

# vLLM-local models (each command owns its vLLM subprocess)
python scripts/run_agent.py --model qwen3-8b     --output out/qwen3-8b.agent.jsonl
python scripts/run_agent.py --model qwen3-32b    --output out/qwen3-32b.agent.jsonl
python scripts/run_agent.py --model devstral-24b --output out/devstral-24b.agent.jsonl
python scripts/run_agent.py --model llama-3.3-70b --output out/llama-3.3-70b.agent.jsonl
```

Common flags:

- `--repo-root <dir>` — directory containing cloned repositories (default `./repos`). Each benchmark item's `repo` is resolved to `<repo_root>/<basename(repo)>`.
- `--limit <N>` — only run the first N benchmark items.
- `--split <name>` — HuggingFace dataset split (default `test`).
- `--workers <N>` — benchmark-level parallelism. **Default `1` (serial)**. See [Parallelism](#parallelism-and-defaults) below.
- `--no-resume` — overwrite existing output instead of resuming.

### Direct mode

```bash
python scripts/run_direct.py --model gpt-4o --output out/gpt-4o.direct.jsonl
python scripts/run_direct.py --model qwen3-8b --output out/qwen3-8b.direct.jsonl
# ... same for the other models
```

Direct mode always runs with tools disabled; the `agent_vllm` block of the model spec is ignored, so for `vllm-local` models the vLLM server is started without `--enable-auto-tool-choice` / `--tool-call-parser`. **Direct mode does not currently expose a `--workers` flag — it is always serial.** This is usually fine since each question is a single LLM call.

### Judge

```bash
# OpenAI judge
python scripts/run_judge.py \
    --input out/gpt-4o.agent.jsonl \
    --output out/gpt-4o.agent.scored.jsonl \
    --judge openai

# DeepSeek judge
python scripts/run_judge.py \
    --input out/gpt-4o.agent.jsonl \
    --output out/gpt-4o.agent.scored.jsonl \
    --judge deepseek
```

The judge picks the best available candidate from each input record: it prefers `agent_result.answer` if present and non-empty, otherwise falls back to `direct_answer`. The judge runs **4 workers by default** (`--workers 4`); raise or lower to fit your judge-provider rate limit.

**Judge defaults**

| `--judge` | Default model | Endpoint | Notes |
| --- | --- | --- | --- |
| `openai` (default) | `gpt-5-2025-08-07` | `/v1/responses` | Calls with `reasoning.effort=low` and `text.verbosity=low`. Override via `--judge-model <id>` or `--reasoning-effort {minimal,low,medium,high}`. |
| `deepseek` | `deepseek-chat` | `/v1/chat/completions` | `temperature=0`; `--reasoning-effort` is ignored. |

### Parallelism and defaults

| Script | Parallelism | Default | Notes |
| --- | --- | --- | --- |
| `run_agent.py` | Per-question `ThreadPoolExecutor` | `--workers 1` (serial) | API providers (`openai` / `anthropic` / `gemini`) accept any value, subject to your provider rate limit. For `vllm-local` the value is clamped to **4** and all workers share the same vLLM server. |
| `run_direct.py` | None | always serial | Direct mode does not expose `--workers`. |
| `run_judge.py` | Per-record `ThreadPoolExecutor` | `--workers 4` | Bounded by the judge provider's rate limit, not the benchmark size. |

**Recommended values**

- `gpt-4o` / `gpt-4.1` / `claude-sonnet-4.5` / `gemini-2.5-pro` / `deepseek-v4`: `--workers 4` to `--workers 16` is typically safe on **Tier 2+** API keys. Higher values just trade latency for more 429s.
- **Low-tier OpenAI keys** (Tier 1 = 30 K TPM for GPT-4.1 / GPT-4o): keep `--workers 1`. Each agent step uses ~3–4 K tokens, so even 2–3 concurrent workers can blow the per-minute budget and produce repeated 429 retries. Upgrade your tier or stay serial.
- `vllm-local` (`qwen3-8b`, `qwen3-32b`, `devstral-24b`, `llama-3.3-70b`): `--workers 4` (the cap). The vLLM server batches requests internally, so going above 4 from the client side mostly adds queueing.

**On rate-limit (429) handling.** The agent catches 429s, parses any `Please try again in Xs` / `retry-after` hint from the provider, sleeps for that long (default 60 s if nothing parsed), and retries — up to 7 attempts per question. If your key is genuinely capped, **dropping `--workers` is faster than letting the agent burn attempts on retries**, because every retry restarts the graph from step 0 and re-pays for the system prompt + history.

**Output safety**: when `--workers > 1`, writes are protected by a `threading.Lock`, so resume works correctly. With `--workers 1` the script writes line-by-line as it goes.

### Tuning the agent loop (per-model)

Per-model defaults — number of agent iterations, history window, max context length, context-warning threshold — live in [`configs/models.yaml`](configs/models.yaml). The top-level `defaults` block applies to every model; any field set under a specific model overrides it. Example:

```yaml
defaults:
  temperature: 0.0
  max_context_length: 32768
  max_iterations: 10
  history_window: 10
  context_warning_threshold: 0.825

models:
  gpt-4o:
    provider: openai
    model_id: gpt-4o-2024-11-20
    # ... inherits all defaults

  qwen3-8b:
    provider: vllm-local
    model_id: Qwen/Qwen3-8B
    max_iterations: 15           # override just this model
    vllm:                        # passed to `vllm serve <model_id>`
      tensor_parallel_size: 1
      max_model_len: 32768
      gpu_memory_utilization: 0.90
      dtype: bfloat16
      max_num_seqs: 64
      max_num_batched_tokens: 8192
      enable_chunked_prefill: true
    agent_vllm:                  # only applied in agent mode (with_tools=True)
      enable_auto_tool_choice: true
      tool_call_parser: hermes
```

How those fields are consumed at runtime:

- `temperature`, `max_iterations`, `history_window`, `max_context_length`, `context_warning_threshold` → read by `sweqapro/registry.py` into a `ModelSpec`, then passed to `ToolCallingAgent(...)` in `scripts/run_agent.py`. The agent stops once `current_step >= max_iterations` and starts force-finishing once token usage crosses `context_warning_threshold * max_context_length`.
- `vllm` block → flag-translated and applied to every `vllm serve ...` launch by `sweqapro/vllm_server.py`.
- `agent_vllm` block → applied **only** when `with_tools=True` (i.e. by `run_agent.py`). `run_direct.py` skips it, so direct mode never launches vLLM with tool-choice parsing on.

To add a new backend, append an entry under `models:` — no code changes are needed unless the provider is something other than `openai` / `anthropic` / `gemini` / `vllm-local`, in which case extend `build_llm` in `sweqapro/registry.py`.

### Aggregate scores

```bash
python -m sweqapro.scoring.stats out/gpt-4o.agent.scored.jsonl out/claude.agent.scored.jsonl
```

## Repository layout

```
eval/
├── README.md
├── requirements.txt
├── .env                                # API keys (gitignored; fill in before running)
├── clone_repos.sh                      # Clones every repo in repos.txt at its benchmark commit
├── repos.txt                           # List of "<git_url> <commit_hash>" pairs (one per line)
│
├── configs/
│   └── models.yaml                     # Model registry: provider, model_id, vLLM/agent params
│
├── prompts/
│   ├── agent_system_prompt.txt         # System prompt for agent mode (PROCESS + OUTPUT protocols)
│   └── judge_prompt.txt                # 5-axis judge rubric template
│
├── scripts/                            # Thin CLI entry points (no logic, only argparse + dispatch)
│   ├── run_agent.py                    # Agent-mode runner
│   ├── run_direct.py                   # Direct-mode (no-tools, single-turn) runner
│   └── run_judge.py                    # Scores a runner's JSONL against ground truth
│
└── sweqapro/                           # The Python package — all logic lives here
    ├── __init__.py
    ├── config.py                       # Loads .env and exposes project paths
    ├── tool_schemas.py                 # OpenAI-style JSON schemas for the three tools (shared)
    ├── registry.py                     # YAML → ModelSpec; build_llm() factory per provider
    ├── data.py                         # HF Hub benchmark loader (load_benchmark, iter_records)
    ├── history.py                      # Sliding-window ConversationHistory
    ├── agent.py                        # ToolCallingAgent — single provider-agnostic class
    ├── direct.py                       # run_direct(): single-turn answer with no tools
    ├── vllm_server.py                  # VLLMServer context manager (subprocess + port poll)
    │
    ├── tools/                          # The three read-only inspection tools
    │   ├── __init__.py
    │   ├── semantic_search.py          # Substring search over a file or directory tree
    │   ├── view_codebase.py            # File/dir viewer with view_range and Python outline
    │   └── execute_readonly_command.py # Allowlisted bash shim (ls/grep/find/cat/...)
    │
    └── scoring/
        ├── __init__.py
        ├── judge.py                    # evaluate_jsonl() — unified OpenAI / DeepSeek judge
        └── stats.py                    # CLI: aggregate scored JSONLs into per-axis averages
```

### Key modules

- **`sweqapro/agent.py` — `ToolCallingAgent`**
  One class for every backend. The LangGraph loop is provider-agnostic: it consumes the LangChain unified `response.tool_calls` interface, runs the tools, manages the conversation history, and emits the `<finish>...</finish>` answer. Force-finish kicks in at `max_iterations` or at `context_warning_threshold * max_context_length` and reuses a tool-free LLM copy (`llm_no_tools`) to avoid recursive tool calls. A degenerate-output guard (repetitive lines, leaked tool markers, runaway `<think>` chains) triggers a retry with a fresh history window.

- **`sweqapro/registry.py` — model factory**
  `resolve(name)` returns a `ModelSpec`. `build_llm(spec, base_url=None, with_tools=True)` returns a LangChain chat model bound to the shared tool schemas. Adding a new provider is a single branch here — the agent does not change.

- **`sweqapro/vllm_server.py` — `VLLMServer`**
  Context manager that picks a free port, launches `vllm serve <model_id>` as a child subprocess in its own process group, polls `GET /v1/models` until ready, and SIGTERMs the group on exit. The `vllm` block of the model spec is always applied; the `agent_vllm` block is only applied when `with_tools=True`, so direct mode starts vLLM without `--enable-auto-tool-choice`.

- **`sweqapro/scoring/judge.py` — unified judge**
  Both `openai` and `deepseek` backends use the OpenAI Python client against different `base_url`s. Each record's candidate answer is `agent_result.answer` when present and non-empty, otherwise `direct_answer`. Scores are 5 integer axes in `[1, 10]` plus their sum.

## Notes for contributors

- The benchmark is loaded from HuggingFace Hub at runtime — there is no local JSONL copy shipped with this release.
- One agent class, multiple backends. The LangGraph loop only depends on LangChain's unified `response.tool_calls`. To add a new provider, extend `build_llm` in `registry.py` and (if needed) add an entry to `configs/models.yaml` — do not add `if provider == "..."` branches in `agent.py`.
- For OSS models served via vLLM, the `vllm` block of the model spec is always applied; `agent_vllm` is only applied in agent mode. This separation is enforced in `sweqapro/vllm_server.py`.
- **Do not name your virtualenv `sweqapro`** — `sweqapro/` is the package directory, and `uv venv --python 3.11 sweqapro` will silently overwrite it. Use `.venv` (or any other name).

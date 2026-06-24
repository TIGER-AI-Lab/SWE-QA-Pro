# Stage 2: Reinforcement Learning (RL) for SWE-QA-Pro 8B

The second stage of the SWE-QA-Pro training recipe. Starting from the SFT checkpoint, we run agentic RL (GRPO) with [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool), where the policy interacts with a repository tool server over multiple turns and is optimized against the SWE-QA-Pro reward.

## 1. Environment Installation

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd path/to/SWE-QA-Pro/train

# Create the env (a folder ./verl-tool) and activate it
uv venv --python 3.10 verl-tool
source verl-tool/bin/activate

cd RL/verl-tool
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"
uv pip install "flash-attn==2.8.3" --no-build-isolation --no-binary flash-attn
```

## 2. Training

Before running:

- Set the judge API key (required). The reward is computed by an OpenAI judge, so set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for a non-OpenAI endpoint) in `.env`.
- Set `model_name` in the script to your Stage 1 SFT checkpoint (local path or HF repo).
- Authenticate Weights & Biases: set `WANDB_API_KEY` in the script, or run `wandb login`. Leave `WANDB_API_KEY` empty to disable reporting.
- Set `n_gpus_per_node` in the script to your GPU count (defaults to 8).

```bash
# from RL/verl-tool
bash scripts/train_sweqapro_8B.sh
```

The script launches the `swe_qa_pro` tool server as a background process and then starts the verl-tool PPO/GRPO trainer against it.

#!/usr/bin/env bash
set -x
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="SWE-QA-Pro_8B_hermes"
fi

export WANDB_API_KEY=""
export WANDB_PROJECT="SWE-QA-Pro"
export WANDB_NAME=$RUN_NAME                   
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300

# Resolve paths relative to this script's location (SFT/), so the script
# works regardless of the directory it's launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="Qwen/Qwen3-8B"
OUTPUT_DIR="output/SWE-QA-Pro_8B_hermes"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Number of GPUs for distributed training. Defaults to 8 GPUs;
# adjust --nproc_per_node to match the number of GPUs on your machine.
DISTRIBUTED_ARGS="--nproc_per_node 8"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sft_qwen3_8b_hermes_$(date +%Y%m%d_%H%M%S).log"

torchrun ${DISTRIBUTED_ARGS} $(which swift) sft \
  --use_hf True \
  \
  --model $MODEL_PATH \
  --train_type full \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  \
  --dataset "$SFT_DIR/dataset/train.jsonl" \
  --split_dataset_ratio 0 \
  --dataset_num_proc 16 \
  --streaming False \
  --strict False \
  --remove_unused_columns False \
  --dataloader_num_workers 8 \
  \
  --agent_template "hermes" \
  --loss_scale hermes \
  --response_prefix "" \
  \
  --packing False \
  --max_length 32768 \
  --truncation_strategy delete \
  \
  --deepspeed zero3 \
  --gradient_checkpointing True \
  \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  \
  --num_train_epochs 3 \
  --save_strategy epoch \
  \
  --report_to wandb \
  --logging_first_step True \
  --logging_steps 1 \
  \
  --ddp_backend nccl \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

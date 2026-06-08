#!/bin/bash
# SUM + DAPO combined training for deepseek-math-7b.

RUN_NAME=deepseek-math-7b_MATH_SUM_DAPO
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"
export HF_DATASETS_CACHE="/mnt/workspace/wxc/.cache/huggingface/datasets"
export HF_HOME="/mnt/workspace/wxc/.cache/huggingface"
export WANDB_DIR="/mnt/workspace/wxc/resum/wandb/$RUN_NAME"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

mkdir -p "$WANDB_DIR"
rm -rf /mnt/workspace/wxc/resum/checkpoints/$RUN_NAME
cd /mnt/workspace/wxc/resum

GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes $NUM_GPUS \
    src/open_r1/grpo_sum.py \
    --config recipes/deepseek-math-7b-warm/grpo/config_grpo_sum_dapo.yaml \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    > /mnt/workspace/wxc/resum/run_$RUN_NAME.log 2>&1

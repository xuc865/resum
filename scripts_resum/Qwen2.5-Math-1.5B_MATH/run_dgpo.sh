#!/bin/bash

RUN_NAME=Qwen2.5-Math-1.5B_MATH-lighteval_DGPO
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"
export HF_DATASETS_CACHE="/mnt/workspace/wxc/.cache/huggingface/datasets"
export HF_HOME="/mnt/workspace/wxc/.cache/huggingface"
export WANDB_DIR="/mnt/workspace/wxc/resum/wandb/$RUN_NAME"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$WANDB_DIR"
rm -rf /mnt/workspace/wxc/resum/checkpoints/$RUN_NAME
cd /mnt/workspace/wxc/resum

GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes $NUM_GPUS \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-1.5B/grpo/config_grpo.yaml \
    --enable_dgpo True --enable_dgpo_dqw True --dgpo_dqw_temp 2.0 \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 8 \
    --max_completion_length 2048 \
    --max_steps 234 \
    --eval_steps 10 \
    > /mnt/workspace/wxc/resum/run_$RUN_NAME.log 2>&1
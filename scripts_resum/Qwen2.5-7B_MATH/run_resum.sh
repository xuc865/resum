#!/bin/bash

RUN_NAME=Qwen2.5-7B_MATH-augmented_DGPO
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"
export WANDB_DIR="/mnt/workspace/wxc/resum/wandb/$exp_name"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"
mkdir -p "$WANDB_DIR"
cd /mnt/workspace/wxc/resum
export PYTHONNOUSERSITE=1  # Prevent ~/.local packages from polluting the conda env (fixes numpy ImportError)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-7B/grpo/config_grpo_augmented.yaml \
    --enable_dgpo True --enable_dgpo_dqw True --dgpo_dqw_temp 2.0 \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    > /mnt/workspace/wxc/resum/run_$RUN_NAME.log 2>&1
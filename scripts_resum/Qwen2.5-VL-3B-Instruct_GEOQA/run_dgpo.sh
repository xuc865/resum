export PYTHONNOUSERSITE=1  # Prevent ~/.local packages from polluting the conda env (fixes numpy ImportError)
#!/bin/bash

RUN_NAME=Qwen2.5-VL-3B-Instruct_GEOQA-R1V-revised_DGPO
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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    src/open_r1/grpo_vlm.py \
    --config recipes/Qwen2.5-VL-3B-Instruct/grpo/config_grpo.yaml \
    --enable_dgpo True --enable_dgpo_dqw True --dgpo_dqw_temp 2.0 \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME
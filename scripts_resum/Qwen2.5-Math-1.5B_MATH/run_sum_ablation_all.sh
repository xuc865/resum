#!/bin/bash
# Run all SUM rollout budget ablation experiments sequentially.
# Skips b2_i1 and b4_i1 (num_generations=1 is not supported by trl GRPO).
#
# Experiments:
#   budget=2 : b2_i2  (init=2, n_leaves=0)
#   budget=4 : b4_i2  (init=2, n_leaves=1), b4_i4  (init=4, n_leaves=0)
#   budget=16: b16_i2 (init=2, n_leaves=7), b16_i4 (init=4, n_leaves=3), b16_i8 (init=8, n_leaves=1)

set -e

SCRIPT_DIR="/mnt/workspace/wxc/resum/scripts_resum/Qwen2.5-Math-1.5B_MATH"
LOG_DIR="/mnt/workspace/wxc/resum"

EXPERIMENTS=(   
    "b16_i4"
    "b16_i8"
)

echo "======================================================"
echo "SUM Budget Ablation: running ${#EXPERIMENTS[@]} experiments"
echo "======================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------------------"
    echo "Starting experiment: $EXP  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "------------------------------------------------------"
    bash "$SCRIPT_DIR/run_sum_${EXP}.sh"
    echo "Finished experiment: $EXP  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "======================================================"
echo "All experiments done! ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "======================================================"

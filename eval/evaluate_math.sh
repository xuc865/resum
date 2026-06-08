#!/bin/bash
# Usage:
#   bash eval/evaluate_math.sh <model_path> [--enrich-prompt true|false]
#
# --enrich-prompt true  : inject SUM system prompt that explicitly asks the model
#                         to periodically summarize and reflect (used as enriched baseline)
# --enrich-prompt false : use standard math system prompt (default)

export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

MODEL=$1
shift

# Parse optional --enrich-prompt flag
ENRICH_PROMPT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --enrich-prompt)
            ENRICH_PROMPT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Select system prompt based on --enrich-prompt flag
STANDARD_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."
SUM_PROMPT="Please reason step by step. As you work through the problem, periodically summarize your progress and reflect on your approach to make sure you are on the right track. Put your final answer within \\boxed{}."

if [[ "$ENRICH_PROMPT" == "true" ]]; then
    SYSTEM_PROMPT="$SUM_PROMPT"
    PROMPT_TAG="_enriched"
    echo "[evaluate_math] Using enriched SUM system prompt."
else
    SYSTEM_PROMPT="$STANDARD_PROMPT"
    PROMPT_TAG=""
    echo "[evaluate_math] Using standard system prompt."
fi

MODEL_NAME=${MODEL//\//_}
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
TIME_STAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=logs/eval/${MODEL_NAME}${PROMPT_TAG}/$TIME_STAMP

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"

TASKS="aime24 aime25 amc23 math_500 minerva olympiadbench"
for TASK in $TASKS; do
    lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
        --system-prompt "$SYSTEM_PROMPT" \
        --custom-tasks eval/custom_math_tasks.py \
        --use-chat-template \
        --output-dir $OUTPUT_DIR
done
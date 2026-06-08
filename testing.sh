cd /mnt/workspace/wxc/resum
source activate
conda activate /mnt/workspace/wxc/miniconda3/envs/train_Re2

# 标准 baseline（standard prompt）
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash eval/evaluate_math.sh /mnt/workspace/wxc/Agent/models/Qwen2.5-Math-1.5B

# 富化 prompt baseline（注入 SUM system prompt，要求模型周期性总结和反思）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   bash eval/evaluate_math.sh /mnt/workspace/wxc/Agent/models/Qwen2.5-Math-1.5B --enrich-prompt true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   bash eval/evaluate_math.sh /mnt/workspace/wxc/Agent/models/Qwen2.5-3B --enrich-prompt true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   bash eval/evaluate_math.sh /mnt/workspace/wxc/Agent/models/Qwen2.5-math-7B --enrich-prompt true


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   bash eval/evaluate_math.sh /mnt/workspace/wxc/resum/checkpoints/Qwen2.5-7B_MATH-lighteval_DGPO_ORIGINAL
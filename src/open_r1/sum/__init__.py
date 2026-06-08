"""
SUM (Summary-branching RL) module for ReSum.

Ported from verl-based implementation at:
  rl-resolving-main/verl-redo-continue/recipe/sum/

Key components:
  - sum_branching : build branch inputs from <sum>...</sum> positions
  - sum_reward    : IS-ratio based summary reward computation
  - sum_advantage : intra + cross group advantage computation
  - sum_trainer   : SumGRPOTrainer extending trl.GRPOTrainer
"""

"""
Advantage computation utilities for the SUM tree-rollout method.

NOTE: As of the natural-language-prefix redesign, the main advantage computation
is inlined directly in SumGRPOTrainer._generate_and_score_completions using
std-normalized GRPO advantages (same as trl baseline). This module is kept as
a reference implementation and may be used for future extensions.

Std-normalized GRPO advantage (per prompt group):
    advantage(i) = (r_i - mean(r)) / (std(r) + eps)

This matches trl's default advantage computation and keeps loss in a stable range.
No verl dependencies -- pure numpy/torch.
"""

from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-4  # Matches trl default (std_grouped_rewards + 1e-4)


def compute_std_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """
    Compute std-normalized advantages for one prompt's rollouts.

    Parameters
    ----------
    rewards : np.ndarray of shape (n,)
        Sequence-level reward for each rollout.

    Returns
    -------
    advantages : np.ndarray of shape (n,)
        Std-normalized advantages. Zero when all rewards are identical.
    """
    n = len(rewards)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    return ((rewards - mean_r) / (std_r + _EPS)).astype(np.float32)

def compute_std_advantages_for_batch(
    sequence_rewards: np.ndarray,
    question_uids: np.ndarray,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level std-normalized advantages for the full batch.

    Parameters
    ----------
    sequence_rewards : np.ndarray of shape (bs,)
        Scalar sequence-level reward for each rollout.
    question_uids : np.ndarray of shape (bs,)
        Prompt-level group identifier (rollouts sharing the same uid are grouped).
    response_mask : torch.Tensor of shape (bs, response_len)
        1 for valid response tokens, 0 for padding.

    Returns
    -------
    advantages : torch.Tensor of shape (bs, response_len)
        Scalar advantage broadcast to every valid response token.
    """
    bs = len(sequence_rewards)
    adv_scalars = np.zeros(bs, dtype=np.float32)

    for uid in np.unique(question_uids):
        indices = np.where(question_uids == uid)[0]
        group_rewards = sequence_rewards[indices]
        adv_scalars[indices] = compute_std_advantages_for_group(group_rewards)

    adv_tensor = torch.tensor(adv_scalars, dtype=torch.float32, device=response_mask.device)
    advantages = adv_tensor.unsqueeze(-1) * response_mask.float()
    return advantages

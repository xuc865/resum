"""
Summary reward computation for the SUM method (ReSum port).

For each <sum>...</sum> position that was branched, we compute a reward signal
based on the consistency of the two branch continuations:

  Branch A: reasoning prefix + masked sum → continuation
  Branch B: masked reasoning prefix + sum content → continuation

The reward measures how similar the first `top_k_tokens` of the two branches are,
using an importance-sampling ratio style metric (symmetric, clipped):

  For each token position t in [0, top_k_tokens):
    r_t = p_A(token_t) / p_B(token_t)
    sim_t = min(r_t, clip) + min(1/r_t, clip) - 2   ∈ [-2, 2*(clip-1)]

  summary_reward = mean over t of sim_t, normalised to [0, 1].

Ported from verl-based recipe/sum/sum_reward.py.
No verl dependencies — pure torch.
"""

from __future__ import annotations

import torch

DEFAULT_TOP_K_TOKENS = 50
DEFAULT_IS_CLIP = 5.0


def compute_summary_reward_from_logprobs(
    log_probs_a: torch.Tensor,
    log_probs_b: torch.Tensor,
    top_k: int = DEFAULT_TOP_K_TOKENS,
    is_clip: float = DEFAULT_IS_CLIP,
) -> float:
    """
    Compute the summary reward from the log-probabilities of the first `top_k`
    continuation tokens of Branch A and Branch B.

    Parameters
    ----------
    log_probs_a : Tensor of shape (seq_len_a,)
        Per-token log-probabilities of the Branch A continuation (response tokens only,
        padding excluded).
    log_probs_b : Tensor of shape (seq_len_b,)
        Same for Branch B.
    top_k : int
        Number of leading tokens to compare.
    is_clip : float
        Clipping value for the IS ratio.

    Returns
    -------
    float
        Reward in [0, 1]. Higher means more consistent (more neutral summary).
    """
    k = min(top_k, log_probs_a.shape[0], log_probs_b.shape[0])
    if k == 0:
        return 0.0

    lp_a = log_probs_a[:k].float()
    lp_b = log_probs_b[:k].float()

    # IS ratio: r = exp(log_p_A - log_p_B)
    log_ratio = lp_a - lp_b
    ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

    # Symmetric clipped similarity: min(r, C) + min(1/r, C) - 2
    sim = (
        torch.minimum(ratio, torch.full_like(ratio, is_clip))
        + torch.minimum(1.0 / ratio.clamp(min=1e-8), torch.full_like(ratio, is_clip))
        - 2.0
    )

    # Normalise to [0, 1]: shift by 2, divide by 2*clip
    reward = (sim.mean().item() + 2.0) / (2.0 * is_clip)
    return float(reward)


def compute_all_summary_rewards(
    branch_log_probs: torch.Tensor,
    branch_attention_mask: torch.Tensor,
    branch_meta: list[dict],
    top_k: int = DEFAULT_TOP_K_TOKENS,
    is_clip: float = DEFAULT_IS_CLIP,
) -> dict[tuple[int, int], float]:
    """
    Compute summary rewards for all (parent_idx, sum_span_idx) pairs.

    Parameters
    ----------
    branch_log_probs : Tensor of shape (n_branches, response_len)
        Per-token log-probabilities of the branch continuations.
    branch_attention_mask : Tensor of shape (n_branches, total_len)
        Attention mask for branch inputs (used to exclude padding from response).
    branch_meta : list[dict]
        Metadata returned by make_summary_branch_inputs().
    top_k : int
    is_clip : float

    Returns
    -------
    dict mapping (parent_idx, sum_span_idx) → float reward
    """
    response_len = branch_log_probs.shape[1]

    # Group branches by (parent_idx, sum_span_idx), keeping meta for continuation_len
    pair_to_branches: dict[tuple[int, int], dict[str, int]] = {}
    pair_to_meta: dict[tuple[int, int], dict] = {}
    for branch_row_idx, meta in enumerate(branch_meta):
        key = (meta["parent_idx"], meta["sum_span_idx"])
        if key not in pair_to_branches:
            pair_to_branches[key] = {}
            pair_to_meta[key] = meta
        pair_to_branches[key][meta["branch_type"]] = branch_row_idx

    rewards: dict[tuple[int, int], float] = {}

    for key, type_to_idx in pair_to_branches.items():
        if "A" not in type_to_idx or "B" not in type_to_idx:
            continue

        idx_a = type_to_idx["A"]
        idx_b = type_to_idx["B"]

        # Extract only the continuation log-probs (last continuation_len tokens).
        # branch_log_probs has shape (n_branches, response_len) where response_len
        # = comp_len of the original batch. The branch input is:
        #   [pad...pad | prompt | truncated_response | continuation]
        # _get_per_token_logps returns log-probs for the last `response_len` tokens,
        # which covers the tail of truncated_response + continuation.
        # We only want the continuation part, which is the last continuation_len tokens.
        continuation_len = pair_to_meta[key].get("continuation_len", response_len)
        continuation_len = min(continuation_len, response_len)

        if continuation_len == 0:
            # No continuation tokens available — skip this pair
            continue

        # Take the last continuation_len log-probs (= continuation tokens)
        lp_a_full = branch_log_probs[idx_a]  # (response_len,)
        lp_b_full = branch_log_probs[idx_b]

        # Also apply attention mask to exclude any padding within the last tokens
        resp_attn_a = branch_attention_mask[idx_a, -response_len:]
        resp_attn_b = branch_attention_mask[idx_b, -response_len:]

        # Mask out padding, then take last continuation_len valid tokens
        lp_a_valid = lp_a_full[resp_attn_a.bool()]
        lp_b_valid = lp_b_full[resp_attn_b.bool()]

        lp_a = lp_a_valid[-continuation_len:] if lp_a_valid.shape[0] >= continuation_len else lp_a_valid
        lp_b = lp_b_valid[-continuation_len:] if lp_b_valid.shape[0] >= continuation_len else lp_b_valid

        reward = compute_summary_reward_from_logprobs(lp_a, lp_b, top_k=top_k, is_clip=is_clip)
        rewards[key] = reward

    return rewards

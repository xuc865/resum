"""
Summary-branching utilities for the SUM method (ReSum port).

For each original rollout, we:
1. Find all <sum>...</sum> positions in the decoded response.
2. Randomly select up to `max_sum_branches` of them.
3. For each selected position, build two branch inputs:
   - Branch A: prompt + normal reasoning prefix + empty <sum></sum> (sum content masked)
   - Branch B: prompt + masked reasoning segment (since last </sum>) + normal <sum>content</sum>
4. Return a dict of tensors ready for model forward, together with metadata
   that maps each branch back to its parent rollout and sum position.

Ported from verl-based recipe/sum/sum_branching.py.
Key change: replaced verl.DataProto with plain torch.Tensor dicts.
"""

import random
import re
from typing import Optional

import torch

SUM_OPEN_TAG = "<sum>"
SUM_CLOSE_TAG = "</sum>"

_SUM_PATTERN = re.compile(r"<sum>(.*?)</sum>", re.DOTALL)

def _find_sum_spans(text: str) -> list[tuple[int, int, int, int]]:
    """
    Return list of (full_start, inner_start, inner_end, full_end) for each <sum>...</sum>.

    - full_start  : char index of '<' in '<sum>'
    - inner_start : char index of first char inside <sum>
    - inner_end   : char index after last char inside <sum>
    - full_end    : char index after '>' in '</sum>'
    """
    spans = []
    for match in _SUM_PATTERN.finditer(text):
        spans.append((match.start(), match.start(1), match.end(1), match.end()))
    return spans

def make_summary_branch_inputs(
    tokenizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_sum_branches: int = 4,
    max_input_len: int = 4096,
    seed: Optional[int] = None,
    decoded_responses: Optional[list[str]] = None,
    max_total_branchings: Optional[int] = None,
) -> tuple[dict, list[dict]]:
    """
    Build branch inputs for all rollouts in a generation batch.

    Parameters
    ----------
    tokenizer : transformers tokenizer
    prompt_ids : Tensor of shape (batch, prompt_len)
        Prompt token ids (left-padded).
    response_ids : Tensor of shape (batch, response_len)
        Response token ids (right-padded).
    attention_mask : Tensor of shape (batch, prompt_len + response_len)
        Full sequence attention mask.
    max_sum_branches : int
        Maximum number of <sum> positions to branch per rollout.
    max_input_len : int
        Maximum total length for branch inputs (prompt + truncated response).
    seed : int or None
    decoded_responses : list[str] or None
        Pre-decoded response strings (one per batch row). If provided, avoids
        redundant tokenizer.decode() calls. EOS stripping is still applied here.
    max_total_branchings : int or None
        Global budget cap on the total number of (parent, sum_pos) branchings
        across the entire batch. Each branching produces exactly 2 branch inputs
        (A and B). If None, no global cap is applied (only max_sum_branches
        per-rollout limit applies).

        Budget rationale: to keep total rollout count ≤ baseline + 1 branching
        per prompt on average, set this to ``batch_size // num_generations``.
        That allows at most 1 extra branching (2 branch sequences) per prompt
        on average, so total sequences = batch_size + 2*(batch_size//num_generations)
        ≈ baseline * (1 + 2/num_generations).

    Returns
    -------
    branch_inputs : dict with keys 'input_ids', 'attention_mask'
        Tensors of shape (n_branches, max_input_len), left-padded.
        Empty dict if no <sum> tags found.
    branch_meta : list[dict]
        One entry per row in branch_inputs with keys:
          - parent_idx   : index into the original batch
          - sum_span_idx : which <sum> position (0-based) in that rollout
          - branch_type  : 'A' or 'B'
          - sum_count    : total number of <sum> tags in the parent rollout
    """
    rng = random.Random(seed)
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    # Track total branchings used so far (each branching = 1 selected <sum> pos = 2 branch seqs)
    total_branchings_used: int = 0

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    input_ids_list: list[torch.Tensor] = []
    attn_mask_list: list[torch.Tensor] = []
    branch_meta: list[dict] = []

    # Set padding_side to left once for all branch inputs
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for parent_idx in range(batch_size):
            full_attn = attention_mask[parent_idx]
            prompt_attn = full_attn[:prompt_len]
            response_attn = full_attn[prompt_len:]

            unpad_prompt_ids = prompt_ids[parent_idx][prompt_attn.bool()]

            # Use pre-decoded response if available, otherwise decode now
            if decoded_responses is not None:
                response_text = decoded_responses[parent_idx]
                # Strip EOS if present (caller may not have stripped it)
                if tokenizer.eos_token and response_text.endswith(tokenizer.eos_token):
                    response_text = response_text[: -len(tokenizer.eos_token)]
            else:
                unpad_response_ids = response_ids[parent_idx][response_attn.bool()]
                response_text = tokenizer.decode(
                    unpad_response_ids.tolist(), skip_special_tokens=False
                )
                if tokenizer.eos_token and response_text.endswith(tokenizer.eos_token):
                    response_text = response_text[: -len(tokenizer.eos_token)]

            sum_spans = _find_sum_spans(response_text)
            total_sum_count = len(sum_spans)

            if total_sum_count == 0:
                continue

            # Check global branching budget before selecting positions for this rollout
            if max_total_branchings is not None and total_branchings_used >= max_total_branchings:
                continue

            # Per-rollout limit: also respect remaining global budget
            per_rollout_limit = max_sum_branches
            if max_total_branchings is not None:
                remaining = max_total_branchings - total_branchings_used
                per_rollout_limit = min(per_rollout_limit, remaining)

            selected_indices = rng.sample(
                range(total_sum_count),
                min(per_rollout_limit, total_sum_count),
            )

            prompt_text = tokenizer.decode(
                unpad_prompt_ids.tolist(), skip_special_tokens=False
            )

            for span_idx in selected_indices:
                full_start, inner_start, inner_end, full_end = sum_spans[span_idx]
                prev_end_char = sum_spans[span_idx - 1][3] if span_idx > 0 else 0

                prefix_before_sum = response_text[:full_start]
                sum_content = response_text[inner_start:inner_end]
                prefix_before_reasoning = response_text[:prev_end_char]
                # Continuation: the text after </sum> in the original rollout.
                # Both branches share the same continuation so we can compare
                # their log-probs on identical tokens (teacher-forcing style).
                continuation_text = response_text[full_end:]

                # Tokenize continuation once to know its length (used in reward)
                continuation_ids = tokenizer(
                    continuation_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                continuation_len = int(continuation_ids.shape[0])

                for branch_type in ("A", "B"):
                    if branch_type == "A":
                        truncated_response = prefix_before_sum + SUM_OPEN_TAG + SUM_CLOSE_TAG
                    else:
                        truncated_response = (
                            prefix_before_reasoning
                            + SUM_OPEN_TAG
                            + sum_content
                            + SUM_CLOSE_TAG
                        )

                    # Tokenize truncated_response to know its exact token length.
                    # This is needed in sum_reward.py to determine how many of the
                    # last `response_len` log-probs belong to the continuation vs.
                    # the truncated_response (in case the full input was truncated).
                    truncated_resp_ids = tokenizer(
                        prompt_text + truncated_response,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"][0]
                    truncated_response_len = int(truncated_resp_ids.shape[0])

                    # Append the shared continuation so teacher-forcing computes
                    # log-probs on the same tokens for both branches.
                    full_input_text = prompt_text + truncated_response + continuation_text
                    encoded = tokenizer(
                        full_input_text,
                        return_tensors="pt",
                        max_length=max_input_len,
                        truncation=True,
                        padding="max_length",
                    )
                    # Actual number of non-padding tokens in this branch input
                    actual_input_len = int(encoded["attention_mask"][0].sum().item())
                    # Actual continuation tokens = total valid tokens - truncated_response tokens
                    # (capped at continuation_len in case of truncation)
                    actual_continuation_len = min(
                        continuation_len,
                        max(0, actual_input_len - truncated_response_len),
                    )

                    input_ids_list.append(encoded["input_ids"])
                    attn_mask_list.append(encoded["attention_mask"])

                    branch_meta.append(
                        {
                            "parent_idx": parent_idx,
                            "sum_span_idx": span_idx,
                            "branch_type": branch_type,
                            "sum_count": total_sum_count,
                            "continuation_len": actual_continuation_len,
                        }
                    )

                # Each (parent_idx, span_idx) pair = 1 branching (A + B = 2 seqs)
                total_branchings_used += 1
    finally:
        # Always restore padding_side
        tokenizer.padding_side = original_padding_side

    if not input_ids_list:
        return {}, []

    branch_inputs = {
        "input_ids": torch.cat(input_ids_list, dim=0),
        "attention_mask": torch.cat(attn_mask_list, dim=0),
    }
    return branch_inputs, branch_meta

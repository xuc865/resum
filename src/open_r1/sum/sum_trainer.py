"""
SumGRPOTrainer: extends trl.GRPOTrainer with tree-rollout RL for summarization.

Training loop changes vs. GRPOTrainer
--------------------------------------
After the standard rollout generation (inside _generate_and_score_completions),
we perform a second "tree branch" generation pass:

1. For each rollout, select m truncation points (e.g. at 50% of tokens).
2. At each truncation point, inject a natural-language summary prefix
   (e.g. "\nIn summary,") -- NO closing tag required.
3. Generate k continuations from each truncated+injected prefix (tree leaves).
4. Score all leaves with the same reward function (accuracy).
5. Merge leaves into the same question_uid group as the original rollouts.
6. Compute SUM group advantages over the merged group (original + leaves).
7. Return only the original rollouts' advantages to trl's compute_loss.

Design rationale:
- Natural-language prefix avoids forcing the model to learn a specific tag format.
- The model is free to summarize and reflect in its own words after the prefix.
- Leaves always have the summary prefix, creating a natural contrast with
  original rollouts that do not.
- Advantages are std-normalized (same as trl baseline) to keep loss in range.

IMPORTANT: All SUM advantage computation is done ONCE inside
_generate_and_score_completions (which is called once per rollout batch and
whose result is buffered by trl). compute_loss simply uses the pre-computed
advantages from inputs["advantages"] -- it does NOT recompute them.
"""

from __future__ import annotations

import hashlib
import random
import re

import numpy as np
import torch

from trl import GRPOTrainer, GRPOConfig

# Natural-language summary prefixes injected at truncation points.
# Multiple candidates are used so the model is exposed to diverse phrasings
# and does not overfit to a single injected pattern.
# All candidates start with a newline and end without a closing tag --
# the model is free to summarize and reflect in its own words.
# NOTE: These prefixes are also added to _NATURAL_SUM_PATTERNS below so that
# if the model spontaneously generates one of them, it is correctly detected
# as a natural summarization event.
_SUM_PREFIXES = [
    "\nIn summary,",
    "\nTo summarize my progress so far,",
    "\nLet me summarize what I have so far.",
    "\nLet me take stock of what I have.",
    "\nLet me step back and review.",
    "\nLet me recap what I have done so far.",
    "\nPutting it all together,",
    "\nLet me consolidate my work so far.",
]

# Default prefix (used when step_seed is not available)
_SUM_PREFIX = _SUM_PREFIXES[0]

# Patterns used to detect natural summarization behavior in model outputs.
# These are phrases the model may spontaneously generate when it summarizes
# its own reasoning -- WITHOUT being prompted by our injected prefix.
# We reward this behavior MORE than our artificially injected branches.
#
# The list is intentionally broad: any phrase that signals the model is
# stepping back to review, consolidate, or reflect on its progress counts.
# Matching is case-insensitive and substring-based (no full-word boundary
# required), so "summarizing my work" and "let me re-examine" both match.
_NATURAL_SUM_PATTERNS = [
    # --- explicit summarize / summary ---
    "to summarize",
    "let me summarize",
    "i'll summarize",
    "i will summarize",
    "summarizing",
    "in summary",
    "a quick summary",
    "brief summary",
    "quick recap",
    "let me give a summary",
    # --- recap / review ---
    "to recap",
    "let me recap",
    "to review",
    "let me review",
    "reviewing",
    "let me re-examine",
    "re-examining",
    "let me revisit",
    "revisiting",
    "let me re-check",
    "let me double-check",
    "double-checking",
    "let me go back",
    "going back to",
    # --- progress check / so far ---
    # NOTE: bare "so far" is intentionally excluded -- it is extremely common in
    # math reasoning ("so far we have x = 3") and would match almost everything.
    # We keep only the more explicit meta-commentary forms.
    "what i have so far",
    "what we have so far",
    "so far i have",
    "so far, i have",
    "so far we have",
    "so far, we have",
    "up to now",
    "thus far",
    "at this point, i have",
    "at this point, we have",
    "let me take stock",
    "taking stock of",
    "let me consolidate",
    "consolidating",
    # --- conclusion (only explicit meta-commentary, NOT bare math connectives) ---
    # NOTE: "therefore", "thus", "hence" are intentionally excluded -- they appear
    # in almost every math solution and would match nearly all rollouts, destroying
    # the discriminative signal we want.
    "in conclusion",
    "to conclude",
    "let me conclude",
    "concluding",
    "putting it all together",
    "combining the above",
    "combining these results",
    "combining everything",
    "taking stock",
    "to wrap up",
    "wrapping up",
    # --- reflection / check ---
    "reflecting on",
    "let me reflect",
    "let me think about",
    "let me reconsider",
    "reconsidering",
    "stepping back",
    "let me step back",
    "let me pause",
    "wait, let me",
    "actually, let me",
    "let me verify",
    "verifying",
    "let me check",
    "checking my work",
    "let me confirm",
    "confirming",
    # --- rethink / new approach ---
    # Phrases indicating the model is abandoning the current path and restarting
    # with a cleaner perspective -- a strong form of meta-cognitive reflection.
    "let me try a different approach",
    "let me try another approach",
    "let me try again",
    "let me start over",
    "let me restart",
    "let me redo",
    "let me rethink",
    "rethinking",
    "let me approach this differently",
    "a different way to think about",
    "alternatively,",
    "on second thought",
    # --- wait / hmm (R1-style reflection tokens) ---
    # Short hesitation markers that signal the model is pausing to reflect.
    # These are intentionally short but only matched when followed by context
    # that implies reflection (substring match is sufficient here).
    "wait,",
    "wait.",
    "hmm,",
    "hmm.",
    "hold on,",
    "hold on.",
    "actually,",
    # --- overall / in total ---
    # NOTE: bare "overall", "in total", "in all" are excluded -- too common in
    # math solutions. Only keep explicit meta-commentary forms.
    "to put it all together",
    "to put it another way",
    "to rephrase",
    "in other words, let me",
]


# Pre-compiled regex for natural SUM detection.
# Joining all patterns with | and compiling once is dramatically faster than
# calling str.find() 150 times per rollout. re.search returns the leftmost
# match, so we use it directly instead of scanning for earliest position.
_NATURAL_SUM_REGEX = re.compile(
    "|".join(re.escape(p) for p in _NATURAL_SUM_PATTERNS),
    re.IGNORECASE,
)


def _find_natural_sum_char_pos(text: str) -> int:
    """
    Find the character position of the first natural summarization phrase in text.

    Returns the start position of the phrase, or -1 if none found.
    Uses a pre-compiled regex union for O(n) scanning instead of O(n*P) where
    P is the number of patterns (~150). This is ~10-50x faster per call.

    NOTE: No exclusion of _SUM_PREFIX is needed here. This function is called
    on the ORIGINAL rollout completions (output of the base model generation),
    which are produced BEFORE any prefix injection. The injected _SUM_PREFIX
    only appears in leaf-node prompts, never in the original completions.
    Therefore any match here is genuinely a natural summarization phrase.
    """
    match = _NATURAL_SUM_REGEX.search(text)
    return match.start() if match else -1


def _prompt_ids_to_uid(prompt_ids_row: torch.Tensor, pad_id: int) -> str:
    """
    Derive a stable string uid from a single row of prompt token ids.
    Used to group rollouts that share the same prompt for SUM advantage computation.
    We strip leading pad tokens so that left-padded prompts with the same content
    map to the same uid regardless of padding length.

    Uses numpy tobytes() instead of str(list) for ~10x faster serialization on
    long prompt sequences.
    """
    tokens_np = prompt_ids_row.cpu().numpy()
    # Find first non-pad token index
    non_pad = np.where(tokens_np != pad_id)[0]
    start = int(non_pad[0]) if len(non_pad) > 0 else 0
    key_bytes = tokens_np[start:].tobytes()
    return hashlib.md5(key_bytes).hexdigest()


class SumGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer augmented with tree-rollout summarization mechanism.

    Extra GRPOConfig fields (passed via training_args):
      logps_chunk_size        : int   = 4       chunk size for logits computation

      # Tree-rollout hyperparameters
      sum_tree_inject         : bool  = True    enable tree-branch generation
      sum_tree_n_leaves       : int   = 2       continuations per truncation point
      sum_tree_trunc_ratios   : str   = "0.5"   truncation ratios (comma-sep)
      sum_tree_n_per_rollout  : int   = 1       truncation points per rollout
      sum_tree_max_new_tokens : int   = 256     max new tokens for leaf generation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args: GRPOConfig = self.args
        self.logps_chunk_size: int = int(getattr(training_args, "logps_chunk_size", 4))

        # Tree-rollout hyperparameters
        self.sum_tree_inject: bool = bool(getattr(training_args, "sum_tree_inject", True))
        self.sum_tree_n_leaves: int = int(getattr(training_args, "sum_tree_n_leaves", 2))
        self.sum_tree_n_per_rollout: int = int(
            getattr(training_args, "sum_tree_n_per_rollout", 1)
        )
        self.sum_tree_max_new_tokens: int = int(
            getattr(training_args, "sum_tree_max_new_tokens", 256)
        )
        # Parse comma-separated truncation ratios
        raw_ratios = getattr(training_args, "sum_tree_trunc_ratios", "0.5")
        if isinstance(raw_ratios, str):
            self.sum_tree_trunc_ratios: list[float] = [
                float(r.strip()) for r in raw_ratios.split(",") if r.strip()
            ]
        else:
            self.sum_tree_trunc_ratios = list(raw_ratios)

        # Natural summarization reward hyperparameters.
        # When the model spontaneously produces a summarization phrase (e.g.
        # "to summarize", "in summary"), we split from that point and generate
        # leaves -- just like the injected branches, but with a higher reward
        # weight so that natural summarization is rewarded MORE than artificial.
        self.natural_sum_inject: bool = bool(
            getattr(training_args, "natural_sum_inject", True)
        )
        self.natural_sum_leaf_weight: float = float(
            getattr(training_args, "natural_sum_leaf_weight", 2.0)
        )
        self.natural_sum_bonus: float = float(
            getattr(training_args, "natural_sum_bonus", 0.3)
        )

    # ------------------------------------------------------------------
    # Tree-branch generation (unified single-pass)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_all_sum_branches(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        question_uids: np.ndarray,
        reward_funcs: list,
        reward_weights: list,
        inputs: list,
        step_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unified single-pass branch generation: detect natural SUM and generate injected
        branches for the rest, all in ONE vLLM generate call.

        Replaces the old two-method design (_generate_natural_sum_branches then
        _generate_tree_branches) that required two separate vLLM calls. One vLLM
        call eliminates the fixed scheduling / KV-cache warm-up overhead (~50% speedup
        on the branching phase).

        Additionally:
        - Batch-decode all completions in a single tokenizer.batch_decode call.
        - Use char-ratio approximation instead of re-encoding prefix_text to find
          the token-level truncation point (avoids one tokenizer.encode per rollout).
        - Score all leaves in a single batch call (_score_leaf_completions_batch).

        Returns
        -------
        natural_leaf_rewards  : np.ndarray[float32] shape (n_nat_leaves,)
        natural_leaf_uids     : np.ndarray[object]  shape (n_nat_leaves,)
        injected_leaf_rewards : np.ndarray[float32] shape (n_inj_leaves,)
        injected_leaf_uids    : np.ndarray[object]  shape (n_inj_leaves,)
        natural_sum_mask      : np.ndarray[bool]    shape (bs,)  -- True for rollouts with natural SUM
        injected_sum_mask     : np.ndarray[bool]    shape (bs,)  -- True for rollouts with injected branch
        """
        tokenizer = self.processing_class
        device = prompt_ids.device
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id

        rng = random.Random(step_seed)
        bs = prompt_ids.shape[0]
        n_leaves = self.sum_tree_n_leaves
        min_remaining = max(8, self.sum_tree_max_new_tokens // 4)

        # Randomly select injected summary prefix (varies per step for diversity)
        selected_prefix = rng.choice(_SUM_PREFIXES)
        prefix_ids = tokenizer.encode(selected_prefix, add_special_tokens=False)
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)

        natural_sum_mask = np.zeros(bs, dtype=bool)
        # injected_sum_mask[i] = True iff rollout i actually received an injected branch.
        # This is set precisely when a job with is_natural=False is created for rollout i,
        # avoiding the ambiguity of uid-based reverse lookup used in the old approach.
        injected_sum_mask = np.zeros(bs, dtype=bool)

        vllm_engine = getattr(self, "llm", None)
        use_vllm = vllm_engine is not None
        if not use_vllm:
            unwrapped_model = self.accelerator.unwrap_model(self.model)

        # -- Phase 1: batch-decode all completions then detect natural SUM ----
        # batch_decode is much faster than looping tokenizer.decode one by one.
        valid_comp_id_lists: list[list[int]] = []
        valid_comp_lens: list[int] = []
        for rollout_idx in range(bs):
            valid_mask = completion_mask[rollout_idx].bool()
            ids = completion_ids[rollout_idx][valid_mask]
            valid_comp_id_lists.append(ids.tolist())
            valid_comp_lens.append(len(ids))

        comp_texts: list[str] = tokenizer.batch_decode(
            valid_comp_id_lists, skip_special_tokens=True
        )

        # -- Phase 2: build leaf jobs (natural + injected) --------------------
        # Each natural job: (rollout_idx, leaf_prompt_ids, truncated_comp_ids, is_natural=True)
        # Each injected job: (rollout_idx, leaf_prompt_ids, truncated_comp_ids, is_natural=False)
        # We also store the "full_prefix_text" needed for reward scoring post-generation.
        #
        # Job record: (rollout_idx, leaf_prompt_ids, truncated_comp_text, is_natural)
        # truncated_comp_text is precomputed here so we don't decode again after generation.
        # Each entry: (rollout_idx: int, leaf_prompt_ids: Tensor, trunc_text: str, is_natural: bool)
        all_jobs: list[tuple] = []

        for rollout_idx in range(bs):
            valid_comp_len = valid_comp_lens[rollout_idx]
            if valid_comp_len < 10:
                continue

            valid_comp_ids_tensor = completion_ids[rollout_idx][
                completion_mask[rollout_idx].bool()
            ]
            comp_text = comp_texts[rollout_idx]

            prompt_valid_mask = prompt_ids[rollout_idx] != pad_id
            unpad_prompt_ids = prompt_ids[rollout_idx][prompt_valid_mask]

            # -- Natural SUM detection --
            nat_char_pos = _find_natural_sum_char_pos(comp_text)
            if nat_char_pos >= 0:
                # Approximate token index from char ratio instead of re-encoding prefix.
                # This is ~correct because BPE tokens have roughly uniform char length.
                # We clamp to [1, valid_comp_len-1] for safety.
                char_ratio = nat_char_pos / max(len(comp_text), 1)
                trunc_token_len = max(1, min(
                    int(valid_comp_len * char_ratio),
                    valid_comp_len - 1,
                ))
                if valid_comp_len - trunc_token_len >= min_remaining:
                    natural_sum_mask[rollout_idx] = True
                    truncated_comp_ids = valid_comp_ids_tensor[:trunc_token_len]
                    # Leaf prompt: unpad_prompt + truncated_comp (no extra prefix injected)
                    leaf_prompt_ids = torch.cat([unpad_prompt_ids, truncated_comp_ids], dim=0)
                    trunc_text = comp_text[:nat_char_pos]
                    all_jobs.append((rollout_idx, leaf_prompt_ids, trunc_text, True))
                    continue  # Natural branch takes priority; skip injected for this rollout

            # -- Injected SUM branch (only for rollouts without natural summary) --
            if not (self.sum_tree_inject and self.sum_tree_trunc_ratios):
                continue

            selected_ratios = rng.sample(
                self.sum_tree_trunc_ratios,
                min(self.sum_tree_n_per_rollout, len(self.sum_tree_trunc_ratios)),
            )
            for trunc_ratio in selected_ratios:
                trunc_len = max(1, int(valid_comp_len * trunc_ratio))
                truncated_comp_ids = valid_comp_ids_tensor[:trunc_len]
                # Build leaf prompt: unpad_prompt + truncated_comp + injected prefix
                leaf_prompt_ids = torch.cat(
                    [unpad_prompt_ids, truncated_comp_ids, prefix_tensor], dim=0
                )
                # Decode truncated completion text for reward scoring later.
                # Use char-ratio slice as approximation (avoids another tokenizer call).
                trunc_char_end = int(len(comp_text) * trunc_ratio)
                trunc_text = comp_text[:trunc_char_end]
                injected_sum_mask[rollout_idx] = True
                all_jobs.append((rollout_idx, leaf_prompt_ids, trunc_text, False))

        # -- Phase 3: single unified vLLM / HF generate call -----------------
        if not all_jobs:
            empty_f = np.array([], dtype=np.float32)
            empty_o = np.array([], dtype=object)
            return empty_f, empty_o, empty_f, empty_o, natural_sum_mask, injected_sum_mask

        if use_vllm:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                max_tokens=self.sum_tree_max_new_tokens,
                temperature=getattr(self, "temperature", 1.0),
                top_p=getattr(self, "top_p", 1.0),
            )

            batch_images = (
                inputs[0].get("image", None)
                if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict)
                else None
            )

            # Flatten: each job → n_leaves vLLM prompts
            all_vllm_inputs: list = []
            for rollout_idx, leaf_prompt_ids, _, _ in all_jobs:
                prompt_text = tokenizer.decode(
                    leaf_prompt_ids.tolist(), skip_special_tokens=False
                )
                if batch_images is not None:
                    rollout_image = batch_images[rollout_idx]
                    entry = {"prompt": prompt_text, "multi_modal_data": {"image": rollout_image}}
                else:
                    entry = prompt_text
                all_vllm_inputs.extend([entry] * n_leaves)

            try:
                all_vllm_outputs = vllm_engine.generate(
                    all_vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception as generation_error:
                print(f"[SUM unified] vLLM generation error: {generation_error}", flush=True)
                # On failure: reset both masks -- no leaves were generated for any rollout.
                natural_sum_mask[:] = False
                injected_sum_mask[:] = False
                empty_f = np.array([], dtype=np.float32)
                empty_o = np.array([], dtype=object)
                return empty_f, empty_o, empty_f, empty_o, natural_sum_mask, injected_sum_mask

            # Collect all leaf texts for batch scoring
            leaf_texts: list[str] = []
            leaf_rollout_indices: list[int] = []
            leaf_is_natural: list[bool] = []

            for job_idx, (rollout_idx, _, trunc_text, is_natural) in enumerate(all_jobs):
                job_outputs = all_vllm_outputs[job_idx * n_leaves : (job_idx + 1) * n_leaves]
                # Batch-decode all leaf continuations in this job at once
                continuation_token_ids = [
                    list(out.outputs[0].token_ids) for out in job_outputs
                ]
                continuation_texts = tokenizer.batch_decode(
                    continuation_token_ids, skip_special_tokens=True
                )
                injected_text = "" if is_natural else selected_prefix
                for cont_text in continuation_texts:
                    full_leaf_text = trunc_text + injected_text + cont_text
                    leaf_texts.append(full_leaf_text)
                    leaf_rollout_indices.append(rollout_idx)
                    leaf_is_natural.append(is_natural)

        else:
            # HF path: generate per-job (variable prompt lengths prevent easy batching)
            leaf_texts = []
            leaf_rollout_indices = []
            leaf_is_natural = []

            for rollout_idx, leaf_prompt_ids, trunc_text, is_natural in all_jobs:
                leaf_prompt_len = leaf_prompt_ids.shape[0]
                leaf_input = leaf_prompt_ids.unsqueeze(0).expand(n_leaves, -1).contiguous()
                leaf_attn = torch.ones_like(leaf_input)
                try:
                    generated = unwrapped_model.generate(
                        input_ids=leaf_input,
                        attention_mask=leaf_attn,
                        max_new_tokens=self.sum_tree_max_new_tokens,
                        do_sample=True,
                        temperature=getattr(self, "temperature", 1.0),
                        top_p=getattr(self, "top_p", 1.0),
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
                    new_token_ids_list = generated[:, leaf_prompt_len:].tolist()
                except Exception as generation_error:
                    print(
                        f"[SUM unified] HF generation error at rollout {rollout_idx}: "
                        f"{generation_error}",
                        flush=True,
                    )
                    # Revert the corresponding mask so Step 4.1 won't falsely credit
                    # this rollout with leaves that were never generated.
                    if is_natural:
                        natural_sum_mask[rollout_idx] = False
                    else:
                        injected_sum_mask[rollout_idx] = False
                    continue

                continuation_texts = tokenizer.batch_decode(
                    new_token_ids_list, skip_special_tokens=True
                )
                injected_text = "" if is_natural else selected_prefix
                for cont_text in continuation_texts:
                    full_leaf_text = trunc_text + injected_text + cont_text
                    leaf_texts.append(full_leaf_text)
                    leaf_rollout_indices.append(rollout_idx)
                    leaf_is_natural.append(is_natural)

        # -- Phase 4: batch-score all leaves ----------------------------------
        all_leaf_rewards = self._score_leaf_completions_batch(
            leaf_texts=leaf_texts,
            rollout_indices=leaf_rollout_indices,
            inputs=inputs,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
        )

        # -- Phase 5: split into natural / injected arrays --------------------
        nat_rewards, nat_uids, inj_rewards, inj_uids = [], [], [], []
        for leaf_idx, (reward, rollout_idx, is_natural) in enumerate(
            zip(all_leaf_rewards, leaf_rollout_indices, leaf_is_natural)
        ):
            uid = question_uids[rollout_idx]
            if is_natural:
                nat_rewards.append(reward)
                nat_uids.append(uid)
            else:
                inj_rewards.append(reward)
                inj_uids.append(uid)

        return (
            np.array(nat_rewards, dtype=np.float32),
            np.array(nat_uids, dtype=object),
            np.array(inj_rewards, dtype=np.float32),
            np.array(inj_uids, dtype=object),
            natural_sum_mask,
            injected_sum_mask,
        )

    def _score_leaf_completions_batch(
        self,
        leaf_texts: list[str],
        rollout_indices: list[int],
        inputs: list,
        reward_funcs: list,
        reward_weights: list,
    ) -> list[float]:
        """
        Score all leaf completions in a single batch call per reward function.

        Instead of calling reward_func once per leaf (serial), we group all leaves
        by their source prompt (prompt_idx) and call reward_func once per prompt
        with a batch of completions. This amortizes the per-call overhead of
        reward functions that do batched inference (e.g. verifier models).

        Parameters
        ----------
        leaf_texts      : list of decoded leaf completion strings
        rollout_indices : list of rollout_idx corresponding to each leaf
        inputs          : trl raw batch (list[dict], one dict per prompt)
        reward_funcs    : list of reward functions
        reward_weights  : list of weights (same length as reward_funcs)

        Returns
        -------
        list[float] of total weighted rewards, one per leaf, in the same order.
        """
        if not leaf_texts:
            return []

        num_gen = self.args.num_generations
        n_leaves = len(leaf_texts)

        # Build conversational format expected by trl reward functions:
        # list[list[dict]] where each inner list is [{"role": "assistant", "content": "..."}]
        leaf_completions_conv = [
            [{"role": "assistant", "content": text}] for text in leaf_texts
        ]

        # Map each leaf to its source prompt_idx
        prompt_indices = [rollout_idx // num_gen for rollout_idx in rollout_indices]

        # Determine unique prompt indices and build per-leaf sample kwargs
        # We need to pass kwargs to reward_func in batch (same length as completions).
        # Since leaves from different prompts may have different kwargs, we must
        # call reward_func once per batch. We group by prompt_idx to keep batches
        # homogeneous (same sample_dict per batch), which is what reward functions expect.

        # Group leaves by prompt_idx for homogeneous batching
        from collections import defaultdict
        prompt_to_leaf_indices: dict[int, list[int]] = defaultdict(list)
        for leaf_idx, prompt_idx in enumerate(prompt_indices):
            prompt_to_leaf_indices[prompt_idx].append(leaf_idx)

        leaf_rewards = [0.0] * n_leaves

        for prompt_idx, leaf_indices in prompt_to_leaf_indices.items():
            sample_dict = inputs[prompt_idx] if prompt_idx < len(inputs) else {}
            sample_inputs = {
                key: [val] * len(leaf_indices)
                for key, val in sample_dict.items()
                if key not in ("input_ids", "attention_mask", "labels")
            }
            batch_completions = [leaf_completions_conv[i] for i in leaf_indices]

            for reward_func, weight in zip(reward_funcs, reward_weights):
                try:
                    scores = reward_func(
                        completions=batch_completions,
                        **sample_inputs,
                    )
                    if scores is not None:
                        for batch_pos, leaf_idx in enumerate(leaf_indices):
                            if batch_pos < len(scores):
                                leaf_rewards[leaf_idx] += float(weight) * float(scores[batch_pos])
                except Exception as score_error:
                    print(
                        f"[SUM unified] _score_leaf_completions_batch error "
                        f"(prompt_idx={prompt_idx}): {score_error}",
                        flush=True,
                    )

        return leaf_rewards

    # ------------------------------------------------------------------
    # Core override: compute SUM advantages once after rollout generation
    # ------------------------------------------------------------------

    def _generate_and_score_completions(self, inputs):
        """
        Override _generate_and_score_completions to inject tree-rollout SUM advantages.

        trl calls this method ONCE per rollout batch and buffers the result.
        The buffered result is then sliced into mini-batches and passed to
        compute_loss multiple times (once per gradient-accumulation step and
        once per num_iterations). By computing SUM advantages here (instead of
        inside compute_loss), we guarantee:

        1. Tree branches are generated exactly once per rollout batch.
        2. question_uids are derived from prompt content (stable hash), not
           random UUIDs, so rollouts from the same prompt are always grouped
           correctly regardless of how many times compute_loss is called.
        3. The advantages stored in the buffer are the final SUM advantages;
           compute_loss uses them as-is without any recomputation.
        """
        import sys
        import time

        _is_main = (not hasattr(self, "accelerator")) or self.accelerator.is_main_process

        def _log(msg):
            if _is_main:
                step = getattr(self.state, "global_step", "?")
                ts = time.strftime("%H:%M:%S")
                print(f"[SUM step={step} {ts}] {msg}", flush=True, file=sys.stderr)

        # Run the standard trl rollout + scoring pipeline first.
        output = super()._generate_and_score_completions(inputs)

        # Only apply SUM during training (not eval).
        if not self.model.training:
            return output

        prompt_ids: torch.Tensor = output["prompt_ids"]
        completion_ids: torch.Tensor = output["completion_ids"]
        completion_mask: torch.Tensor = output["completion_mask"]
        raw_rewards: torch.Tensor = output.get("rewards")
        trl_advantages: torch.Tensor = output["advantages"]

        bs = prompt_ids.shape[0]
        device = prompt_ids.device
        pad_id = self.processing_class.pad_token_id

        comp_mask = completion_mask

        # -- Step 1: build sequence rewards from raw rewards ---------------
        if raw_rewards is not None:
            sequence_rewards = raw_rewards.cpu().numpy().astype(np.float32)
        elif trl_advantages.dim() == 1:
            sequence_rewards = trl_advantages.cpu().numpy().astype(np.float32)
        else:
            sequence_rewards = trl_advantages.sum(dim=-1).cpu().numpy().astype(np.float32)

        _log(f"bs={bs}, reward_mean={sequence_rewards.mean():.3f}")

        # -- Step 2: derive stable question_uids ---------------------------
        num_generations = self.args.num_generations
        if bs % num_generations == 0:
            n_prompts = bs // num_generations
            prompt_uids = [
                _prompt_ids_to_uid(prompt_ids[i * num_generations], pad_id=pad_id)
                for i in range(n_prompts)
            ]
            question_uids = np.repeat(
                np.array(prompt_uids, dtype=object),
                repeats=num_generations,
            )
        else:
            question_uids = np.array(
                [_prompt_ids_to_uid(prompt_ids[i], pad_id=pad_id) for i in range(bs)],
                dtype=object,
            )

        reward_funcs = getattr(self, "reward_funcs", [])
        reward_weights_list = getattr(self, "reward_weights", [1.0] * len(reward_funcs))
        global_step = getattr(self.state, "global_step", 0)

        natural_leaf_rewards = np.array([], dtype=np.float32)
        natural_leaf_uids = np.array([], dtype=object)
        injected_leaf_rewards = np.array([], dtype=np.float32)
        injected_leaf_uids = np.array([], dtype=object)
        natural_sum_mask = np.zeros(bs, dtype=bool)
        # injected_sum_mask must be initialized here so Step 4.1 can always reference it,
        # even when run_branching is False and _generate_all_sum_branches is never called.
        injected_sum_mask = np.zeros(bs, dtype=bool)

        run_branching = self.natural_sum_inject or (
            self.sum_tree_inject and len(self.sum_tree_trunc_ratios) > 0
        )
        if run_branching:
            (
                natural_leaf_rewards,
                natural_leaf_uids,
                injected_leaf_rewards,
                injected_leaf_uids,
                natural_sum_mask,
                injected_sum_mask,
            ) = self._generate_all_sum_branches(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                completion_mask=comp_mask,
                question_uids=question_uids,
                reward_funcs=reward_funcs,
                reward_weights=reward_weights_list,
                inputs=inputs,
                step_seed=global_step,
            )
            n_natural = int(natural_sum_mask.sum())
            n_inject = int(injected_sum_mask.sum())
            nat_leaf_acc = float((natural_leaf_rewards > 0.5).mean()) if len(natural_leaf_rewards) > 0 else 0.0
            inj_leaf_acc = float((injected_leaf_rewards > 0.5).mean()) if len(injected_leaf_rewards) > 0 else 0.0
            _log(
                f"branches (unified): natural={n_natural}/{bs} rollouts "
                f"({len(natural_leaf_rewards)} leaves, acc={nat_leaf_acc:.2%}), "
                f"injected={n_inject}/{bs} rollouts "
                f"({len(injected_leaf_rewards)} leaves, acc={inj_leaf_acc:.2%})"
            )

        # -- Step 4: compute SUM group advantages --------------------------
        #
        # For each prompt group, augment the original rollout rewards with the
        # task rewards measured on SUM leaf continuations. Natural-summary
        # leaves receive a larger statistical weight, and rollouts that produced
        # a natural summary receive an explicit advantage bonus.
        #
        # Only original rollouts are returned to TRL for policy-gradient loss;
        # leaves are counterfactual samples used to shape the group statistics.
        orig_adv_scalars = np.zeros(bs, dtype=np.float32)

        for uid in np.unique(question_uids):
            uid_indices = np.where(question_uids == uid)[0]
            group_rewards = [float(r) for r in sequence_rewards[uid_indices]]
            group_weights = [1.0] * len(uid_indices)

            if len(injected_leaf_rewards) > 0:
                inj_mask = injected_leaf_uids == uid
                group_rewards.extend(float(r) for r in injected_leaf_rewards[inj_mask])
                group_weights.extend([1.0] * int(inj_mask.sum()))

            if len(natural_leaf_rewards) > 0:
                nat_mask = natural_leaf_uids == uid
                group_rewards.extend(float(r) for r in natural_leaf_rewards[nat_mask])
                group_weights.extend(
                    [self.natural_sum_leaf_weight] * int(nat_mask.sum())
                )

            rewards_arr = np.array(group_rewards, dtype=np.float32)
            weights_arr = np.array(group_weights, dtype=np.float32)
            weight_sum = float(weights_arr.sum())
            if len(rewards_arr) <= 1 or weight_sum <= 0.0:
                continue

            mean_r = float((rewards_arr * weights_arr).sum() / weight_sum)
            var_r = float(
                (weights_arr * np.square(rewards_arr - mean_r)).sum() / weight_sum
            )
            std_r = float(np.sqrt(max(var_r, 0.0)))
            orig_adv_scalars[uid_indices] = (
                sequence_rewards[uid_indices] - mean_r
            ) / (std_r + 1e-4)

        if self.natural_sum_bonus != 0.0:
            orig_adv_scalars[natural_sum_mask] += self.natural_sum_bonus

        # Broadcast scalar advantage to every valid response token: shape (bs, comp_len)
        adv_tensor = torch.tensor(orig_adv_scalars, dtype=torch.float32, device=device)
        new_advantages = adv_tensor.unsqueeze(-1) * comp_mask.float()

        has_sum_mask = natural_sum_mask | injected_sum_mask
        n_has_sum = int(has_sum_mask.sum())
        all_leaf_rewards_for_log = np.concatenate(
            [natural_leaf_rewards, injected_leaf_rewards]
        )
        leaf_reward_log_mean = (
            float(all_leaf_rewards_for_log.mean())
            if len(all_leaf_rewards_for_log) > 0
            else 0.0
        )
        _log(
            f"SUM advantages computed: has_sum={n_has_sum}/{bs}, "
            f"injected_leaves={len(injected_leaf_rewards)}, "
            f"natural_leaves={len(natural_leaf_rewards)}, "
            f"leaf_reward_mean={leaf_reward_log_mean:.3f}, "
            f"natural_bonus={self.natural_sum_bonus:.3f}, "
            f"natural_leaf_weight={self.natural_sum_leaf_weight:.3f}, "
            f"adv_mean={orig_adv_scalars.mean():.3f}, adv_std={orig_adv_scalars.std():.3f}. DONE."
        )

        output = dict(output)
        output["advantages"] = new_advantages
        return output

    # ------------------------------------------------------------------
    # Override _compute_loss: chunked logits to reduce peak VRAM
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        """
        Override _compute_loss to:
        1. Pass batch_size=self.logps_chunk_size to reduce peak VRAM.
        2. Handle 2D token-level advantages (bs, comp_len) from SUM.

        trl's _compute_loss uses advantages.unsqueeze(1) which assumes 1D (bs,).
        Our SUM advantages are 2D (bs, comp_len). We handle both cases:
        2D advantages are used directly; 1D advantages are unsqueeze(1)-ed to (bs, 1).
        """
        from trl.trainer.grpo_trainer import get_high_entropy_mask
        try:
            from trl.trainer.grpo_trainer import nanmin, nanmax
        except ImportError:
            def nanmin(t): return t.nanmin()
            def nanmax(t): return t.nanmax()

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=self.logps_chunk_size,  # chunked to reduce peak VRAM
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]

        # SUM advantages are 2D (bs, comp_len) for token-level reweighting.
        # trl's default path uses advantages.unsqueeze(1) which assumes 1D (bs,).
        # We handle both: 2D is used directly; 1D is unsqueeze(1)-ed to (bs, 1).
        if advantages.dim() == 2:
            per_token_advantages = advantages
        else:
            per_token_advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        log_importance_weights = log_ratio

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if getattr(self.args, "delta", None) is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * per_token_advantages
        per_token_loss2 = coef_2 * per_token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # -- Metrics logging (mirrors trl's _compute_loss) -----------------
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_mean(x: torch.Tensor) -> torch.Tensor:
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (per_token_advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (per_token_advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_mean(is_low_clipped.float())
        high_clip = masked_mean(is_high_clipped.float())
        clip_ratio = masked_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())

        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

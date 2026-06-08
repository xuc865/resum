# Copyright 2025 The HuggingFace Team / ReSum Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training entry point for the SUM (Summary-branching RL) method in ReSum.

Ported from verl-based recipe/sum/main_sum.py.

Usage:
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \\
        --num_processes 8 \\
        src/open_r1/grpo_sum.py \\
        --config recipes/Qwen2.5-7B/grpo/config_grpo_sum.yaml
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.sum.sum_trainer import SumGRPOTrainer
from trl import ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)

# ── SUM system prompt ──────────────────────────────────────────────────────────
# Injected into every training prompt so the model knows to reason step by step,
# periodically summarize progress, and reflect on its approach.
#
# Design principle: encourage natural summarization and reflection without
# requiring any specific tag format. The model is free to express summaries
# and reflections in its own words.
SUM_SYSTEM_PROMPT = (
    "Please reason step by step. "
    "As you work through the problem, periodically summarize your progress and "
    "reflect on your approach to make sure you are on the right track. "
    "Put your final answer within \\boxed{}."
)


@dataclass
class SumGRPOConfig(GRPOConfig):
    """
    Extended GRPOConfig with SUM-specific hyperparameters.

    Tree-rollout design:
    - For each rollout, select m truncation points and inject a natural-language
      summary prefix (no closing tag required).
    - Generate k leaf continuations per truncation point.
    - Merge leaves into the same question_uid group for advantage computation.
    - Advantages are std-normalized (same as trl baseline) to keep loss in range.
    """

    # Tree-rollout hyperparameters (artificially injected summary prefix)
    sum_tree_inject: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable tree-branch generation. When True, each rollout is truncated "
                "at a selected point, a natural-language summary prefix is injected, "
                "and k leaf continuations are generated per truncation point."
            )
        },
    )
    sum_tree_n_leaves: int = field(
        default=2,
        metadata={
            "help": (
                "Number of leaf continuations to generate per truncation point. "
                "Applies to both injected and natural-summary branches."
            )
        },
    )
    sum_tree_trunc_ratios: str = field(
        default="0.5",
        metadata={
            "help": (
                "Comma-separated truncation ratios for injected tree-branch generation. "
                "E.g. '0.5' means truncate at 50% of completion length."
            )
        },
    )
    sum_tree_n_per_rollout: int = field(
        default=1,
        metadata={
            "help": (
                "Number of truncation points to sample per rollout for injected branches. "
                "Points are sampled without replacement from sum_tree_trunc_ratios."
            )
        },
    )
    sum_tree_max_new_tokens: int = field(
        default=256,
        metadata={
            "help": (
                "Maximum number of new tokens to generate for each leaf continuation. "
                "Applies to both injected and natural-summary branches."
            )
        },
    )

    # Natural summarization reward hyperparameters.
    # When the model spontaneously produces a summarization phrase (e.g.
    # "to summarize", "in summary"), we split from that point and generate
    # leaves with a higher effective reward weight -- rewarding natural
    # summarization MORE than artificially injected branches.
    natural_sum_inject: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable natural summarization detection and branch generation. "
                "When True, rollouts containing natural summary phrases are detected "
                "and leaf continuations are generated from the summary point."
            )
        },
    )
    natural_sum_leaf_weight: float = field(
        default=2.0,
        metadata={
            "help": (
                "Reward weight multiplier for natural-summary leaf nodes. "
                "Natural leaves are scaled by this factor before participating in "
                "group advantage computation, so they pull the group statistics "
                "more strongly than artificially injected leaves (weight=1.0). "
                "Set > 1.0 to reward natural summarization more than injected branches."
            )
        },
    )
    natural_sum_bonus: float = field(
        default=0.3,
        metadata={
            "help": (
                "Direct advantage bonus added to rollouts that contain a natural "
                "summarization phrase. This is the strongest signal: rollouts that "
                "naturally summarize get both this bonus AND their leaves are weighted "
                "more heavily. Set to 0.0 to disable the direct bonus."
            )
        },
    )

    # Misc
    logps_chunk_size: int = field(
        default=4,
        metadata={
            "help": (
                "Chunk size for _get_per_token_logps_and_entropies. "
                "Logits are computed in chunks of this size to reduce peak VRAM."
            )
        },
    )

def main(script_args: GRPOScriptArguments, training_args: SumGRPOConfig, model_args: ModelConfig):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = get_dataset(script_args)
    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.eval_dataset_name is not None:
        logger.info(f"Loading eval dataset: {script_args.eval_dataset_name}")
        eval_dataset = datasets.load_dataset(
            script_args.eval_dataset_name, script_args.eval_dataset_config
        )

    # ── Load tokenizer and model ──────────────────────────────────────────────
    tokenizer = get_tokenizer(model_args, training_args)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # ── Reward functions ──────────────────────────────────────────────────────
    reward_funcs = get_reward_funcs(script_args)

    # ── Format dataset: inject SUM system prompt ──────────────────────────────
    def make_conversation_with_sum_prompt(
        example, prompt_column: str = script_args.dataset_prompt_column
    ):
        """Build conversation with the SUM system prompt prepended."""
        if prompt_column not in example:
            raise ValueError(
                f"Dataset Question Field Error: {prompt_column} is not supported."
            )

        prompt = [
            {"role": "system", "content": SUM_SYSTEM_PROMPT},
            {"role": "user", "content": example[prompt_column] + "\n\n**Use <sum>...</sum> tags as you reason.**"},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation_with_sum_prompt)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    if training_args.eval_strategy != "no" and eval_dataset is not None:
        eval_dataset = eval_dataset.map(make_conversation_with_sum_prompt)
        eval_dataset = eval_dataset[script_args.eval_dataset_split]
        if "messages" in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns("messages")

    # ── Initialize SumGRPOTrainer ─────────────────────────────────────────────
    if training_args.eval_strategy != "no" and eval_dataset is None:
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer = SumGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", "sum"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, SumGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

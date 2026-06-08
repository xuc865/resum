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
Training entry point for the SUM (Summary-branching RL) method applied to
Vision-Language Models (VLMs) in ReSum.

This file is the VLM counterpart of grpo_sum.py. It combines:
  - The multimodal data handling from grpo_vlm.py (image loading, filtering,
    vLLM weight patching for Qwen2.5-VL)
  - The SUM tree-rollout training logic from grpo_sum.py (SumGRPOTrainer,
    SumGRPOConfig, natural summarization detection and branching)

IMPORTANT: This file does NOT modify grpo_sum.py or grpo_vlm.py. It is a
standalone entry point that imports from both.

Usage:
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \\
        --num_processes 8 \\
        src/open_r1/grpo_sum_vlm.py \\
        --config recipes/Qwen2.5-VL-3B-Instruct/grpo/config_grpo_sum.yaml
"""

import torch
from datasets import load_dataset

from open_r1.configs import GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils.callbacks import get_callbacks
from open_r1.sum.sum_trainer import SumGRPOTrainer
from open_r1.grpo_sum import SumGRPOConfig, SUM_SYSTEM_PROMPT
from trl import ModelConfig, TrlParser, get_peft_config, get_kbit_device_map, get_quantization_config


def patch_qwen_weights_vllm():
    """
    Patch weight names of Qwen multimodal models to be consistent with
    transformers >= 4.52. Required for vLLM colocate mode to load weights
    correctly.
    See https://github.com/vllm-project/vllm/pull/19054
    """
    import vllm
    from vllm.model_executor.models.utils import WeightsMapper

    vllm.model_executor.models.ModelRegistry.models[
        "Qwen2_5_VLForConditionalGeneration"
    ].load_model_cls().hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
    vllm.model_executor.models.ModelRegistry.models[
        "Qwen2VLForConditionalGeneration"
    ].load_model_cls().hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
    print("### Patch to vllm qwen modelling applied successfully.")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, SumGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Apply vLLM weight name patch for Qwen VL models
    if "Qwen" in model_args.model_name_or_path:
        patch_qwen_weights_vllm()

    ################
    # Model init kwargs
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config)

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
            {"role": "user", "content": example[prompt_column]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation_with_sum_prompt)

    # Filter out images that are too large for efficient training
    def filter_big_images(example):
        image = example["image"]
        return image.size[0] < 1024 and image.size[1] < 1024

    dataset = dataset.filter(filter_big_images)

    # Ensure all images are in RGB format
    def convert_to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    ################
    # Reward functions
    ################
    reward_funcs = get_reward_funcs(script_args)

    ################
    # Training
    ################
    # NOTE: We pass model_name_or_path as a string (not a loaded model object)
    # so that trl's GRPOTrainer can handle the VLM processor loading internally.
    # SumGRPOTrainer inherits from GRPOTrainer and will also benefit from this.
    trainer = SumGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        # Do NOT pass processing_class here -- trl will auto-load the VLM processor
        # from the model_name_or_path, which handles both text tokens and image patches.
    )

    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

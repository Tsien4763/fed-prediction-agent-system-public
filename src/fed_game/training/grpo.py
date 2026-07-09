from __future__ import annotations

import json
import os
from pathlib import Path

from fed_game.config import RuntimeConfig, repo_path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def train_grpo(
    config: RuntimeConfig,
    *,
    train_file: str | Path,
    output_dir: str | Path,
    max_steps: int = -1,
    base_adapter_dir: str | Path | None = None,
    learning_rate: float | None = None,
    max_completion_length: int | None = None,
) -> None:
    from .common import read_chat_jsonl, require_training_deps

    require_training_deps()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from .rewards import grpo_reward_suite

    train_cfg = config.raw["training"]["grpo"]
    use_fp16 = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = {
        "torch_dtype": torch.float16 if use_fp16 else torch.float32,
        "trust_remote_code": True,
    }
    if use_fp16:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    use_new_peft_config = base_adapter_dir is None
    if base_adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(repo_path(base_adapter_dir)), is_trainable=True)
    rows = read_chat_jsonl(train_file)
    prompts = []
    for row in rows:
        messages = row["messages"]
        target = {}
        try:
            assistant = json.loads(messages[-1]["content"])
            target = assistant.get("equilibrium_strategy", {})
            target_fed = assistant.get("fed_prediction", {})
        except Exception:
            target = {}
            target_fed = {}
        user = {}
        try:
            user = json.loads(messages[1]["content"])
        except Exception:
            user = {}
        metadata = row.get("metadata", {})
        prompts.append(
            {
                "prompt": messages[:-1],
                "target_strategy": target,
                "target_fed_prediction": target_fed,
                "quarter": metadata.get("quarter") or user.get("quarter"),
                "role_id": metadata.get("role_id") or user.get("role_id"),
                "cluster_id": metadata.get("cluster_id"),
                "current_round": user.get("round_id"),
            }
        )
    dataset = Dataset.from_list(prompts)
    peft_config = LoraConfig(
        r=int(config.raw["training"]["sft"]["lora_r"]),
        lora_alpha=int(config.raw["training"]["sft"]["lora_alpha"]),
        lora_dropout=float(config.raw["training"]["sft"]["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    args = GRPOConfig(
        output_dir=str(repo_path(output_dir)),
        learning_rate=float(learning_rate if learning_rate is not None else train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["micro_batch_size"]),
        gradient_accumulation_steps=_env_int(
            "FED_GAME_GRPO_GRAD_ACCUM",
            int(train_cfg["gradient_accumulation_steps"]),
        ),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps,
        num_generations=int(train_cfg["num_generations"]),
        generation_batch_size=int(train_cfg.get("generation_batch_size", train_cfg["num_generations"])),
        max_completion_length=int(max_completion_length if max_completion_length is not None else train_cfg["max_completion_length"]),
        fp16=use_fp16,
        bf16=False,
        use_cpu=not use_fp16,
        gradient_checkpointing=True,
        logging_steps=_env_int("FED_GAME_TRAIN_LOGGING_STEPS", 10),
        save_steps=_env_int("FED_GAME_TRAIN_SAVE_STEPS", 200),
        save_total_limit=2,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[grpo_reward_suite],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config if use_new_peft_config else None,
    )
    trainer.train()
    trainer.save_model(str(repo_path(output_dir)))
    tokenizer.save_pretrained(str(repo_path(output_dir)))

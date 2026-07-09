from __future__ import annotations

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


def train_sft(
    config: RuntimeConfig,
    *,
    train_file: str | Path,
    output_dir: str | Path,
    max_steps: int = -1,
    resume_from_checkpoint: str | Path | None = None,
) -> None:
    from .common import read_chat_jsonl, render_chat, require_training_deps

    require_training_deps()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    train_cfg = config.raw["training"]["sft"]
    use_fp16 = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    rows = read_chat_jsonl(train_file)
    dataset = Dataset.from_list([{"text": render_chat(tokenizer, row["messages"])} for row in rows])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(train_cfg["max_sequence_length"]),
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir=str(repo_path(output_dir)),
        per_device_train_batch_size=int(train_cfg["micro_batch_size"]),
        gradient_accumulation_steps=_env_int(
            "FED_GAME_SFT_GRAD_ACCUM",
            int(train_cfg["gradient_accumulation_steps"]),
        ),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps,
        fp16=use_fp16,
        gradient_checkpointing=True,
        logging_steps=_env_int("FED_GAME_TRAIN_LOGGING_STEPS", 10),
        save_steps=_env_int("FED_GAME_TRAIN_SAVE_STEPS", max(1, min(40, max_steps if max_steps > 0 else 200))),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train(
        resume_from_checkpoint=str(repo_path(resume_from_checkpoint)) if resume_from_checkpoint else None
    )
    trainer.save_model(str(repo_path(output_dir)))
    tokenizer.save_pretrained(str(repo_path(output_dir)))

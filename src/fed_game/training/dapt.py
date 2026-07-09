from __future__ import annotations

from pathlib import Path

from fed_game.config import RuntimeConfig, repo_path


def train_dapt(config: RuntimeConfig, *, text_file: str | Path, output_dir: str | Path, max_steps: int = -1) -> None:
    from .common import require_training_deps

    require_training_deps()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    train_cfg = config.raw["training"]["dapt"]
    sft_cfg = config.raw["training"]["sft"]
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=int(sft_cfg["lora_r"]),
        lora_alpha=int(sft_cfg["lora_alpha"]),
        lora_dropout=float(sft_cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)
    dataset = load_dataset("text", data_files={"train": str(repo_path(text_file))})["train"]

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
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(repo_path(output_dir)))
    tokenizer.save_pretrained(str(repo_path(output_dir)))

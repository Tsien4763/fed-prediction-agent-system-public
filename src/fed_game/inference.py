from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, ensure_parent, repo_path
from .training.common import read_chat_jsonl, render_chat


def run_adapter_inference(
    config: RuntimeConfig,
    *,
    adapter_dir: str | Path,
    eval_file: str | Path,
    output_path: str | Path,
    limit: int = 8,
    max_new_tokens: int = 256,
    batch_size: int = 1,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(repo_path(adapter_dir)))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    rows = read_chat_jsonl(eval_file)[:limit]
    results = []
    valid_json = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompt_messages_batch = [row["messages"][:-1] for row in batch_rows]
        targets = [row["messages"][-1]["content"] for row in batch_rows]
        prompts = [render_chat(tokenizer, prompt_messages) for prompt_messages in prompt_messages_batch]
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        for row, prompt_messages, target, generated_ids in zip(batch_rows, prompt_messages_batch, targets, generated):
            completion_ids = generated_ids[prompt_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            parsed = extract_json_object(completion)
            if parsed is not None:
                valid_json += 1
            results.append(
                {
                    "task": row.get("task"),
                    "metadata": row.get("metadata", {}),
                    "prompt": prompt_messages,
                    "target": target,
                    "completion": completion,
                    "parsed_completion": parsed,
                }
            )

    out_path = ensure_parent(output_path)
    with out_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {
        "output_path": str(out_path),
        "samples": len(results),
        "valid_json": valid_json,
        "valid_json_rate": round(valid_json / len(results), 4) if results else 0.0,
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None

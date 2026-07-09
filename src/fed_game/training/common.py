from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fed_game.config import repo_path


def read_chat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = repo_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def require_training_deps() -> None:
    missing = []
    for module in ("torch", "transformers", "datasets", "peft", "trl"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install with `uv sync --extra train` or equivalent before training."
        )


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


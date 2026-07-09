from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "first_version.json"
SELF_PLAY_TRACE_FILENAME = "rolling_self_play_traces.jsonl"


def repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | Path) -> Path:
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def default_self_play_trace_path(config: "RuntimeConfig") -> Path:
    return config.paths["trace_dir"] / SELF_PLAY_TRACE_FILENAME


@dataclass(frozen=True)
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def base_model(self) -> str:
        return str(self.raw["model"]["base_model"])

    @property
    def max_sequence_length(self) -> int:
        return int(self.raw["model"].get("max_sequence_length", 4096))

    @property
    def paths(self) -> dict[str, Path]:
        return {key: repo_path(value) for key, value in self.raw["paths"].items()}

    @property
    def teacher_model(self) -> str:
        teacher = self.raw["teacher"]
        return os.getenv(str(teacher.get("model_env", "DEEPSEEK_MODEL")), str(teacher["default_model"]))

    @property
    def teacher_base_url(self) -> str:
        teacher = self.raw["teacher"]
        return os.getenv(str(teacher.get("base_url_env", "DEEPSEEK_BASE_URL")), str(teacher["default_base_url"]))

    @property
    def teacher_api_key(self) -> str | None:
        env_name = str(self.raw["teacher"].get("api_key_env", "DEEPSEEK_API_KEY"))
        value = os.getenv(env_name)
        return value if value else None

    @property
    def teacher_timeout(self) -> int:
        return int(self.raw["teacher"].get("timeout_seconds", 90))

    @property
    def allow_mock_teacher(self) -> bool:
        return bool(self.raw["teacher"].get("allow_mock_without_key", False))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RuntimeConfig:
    config_path = repo_path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        return RuntimeConfig(json.load(fh))

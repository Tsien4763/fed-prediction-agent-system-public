from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "data_registry.yaml"


def repo_path(*parts: str | Path) -> Path:
    path = REPO_ROOT
    for part in parts:
        path = path / part
    return path


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load the JSON-compatible YAML registry without requiring PyYAML."""
    registry_path = Path(path)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    text = registry_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{registry_path} must stay JSON-compatible YAML so the MVP can run "
            "without an external YAML parser."
        ) from exc


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_audit(state: dict[str, Any], agent_name: str, result: dict[str, Any]) -> dict[str, Any]:
    state.setdefault("audit_trail", []).append(
        {
            "agent": agent_name,
            "status": result.get("status", "ok"),
            "timestamp": utc_now(),
            "output_keys": sorted(result.keys()),
        }
    )
    return state


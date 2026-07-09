from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fed_game.config import RuntimeConfig, default_self_play_trace_path, ensure_parent
from fed_game.data_sources import read_jsonl


def build_dapt_corpus(config: RuntimeConfig, *, limit: int = 2000) -> Path:
    out_path = ensure_parent(config.paths["train_dir"] / "dapt_corpus.txt")
    rag_path = config.paths["rag_index"]
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(rag_path, limit=limit):
            text = str(row.get("text") or row.get("search_text") or "").strip()
            if not text:
                continue
            fh.write(text.replace("\r\n", "\n")[:6000])
            fh.write("\n\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"No DAPT text rows were written from {rag_path}")
    return out_path


def combine_sft_data(config: RuntimeConfig, *, limit_per_file: int | None = None) -> Path:
    train_dir = config.paths["train_dir"]
    inputs = [
        train_dir / "semantic_sft.jsonl",
        train_dir / "role_best_response_sft.jsonl",
        train_dir / "critique_traces_sft.jsonl",
        train_dir / "evidence_chain_sft.jsonl",
        train_dir / "equilibrium_distill.jsonl",
    ]
    out_path = ensure_parent(train_dir / "first_version_sft.jsonl")
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                continue
            per_file = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.setdefault("metadata", {})["source_file"] = path.name
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    per_file += 1
                    if limit_per_file is not None and per_file >= limit_per_file:
                        break
    if written == 0:
        raise RuntimeError("No SFT rows were combined. Generate teacher/self-play data first.")
    return out_path


def build_compact_equilibrium_sft(config: RuntimeConfig, *, limit: int | None = None) -> Path:
    source_path = config.paths["train_dir"] / "equilibrium_distill.jsonl"
    out_path = ensure_parent(config.paths["train_dir"] / "compact_equilibrium_sft.jsonl")
    evidence_by_quarter = _load_trace_evidence(config)
    written = 0
    with source_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            user = json.loads(row["messages"][1]["content"])
            assistant = json.loads(row["messages"][2]["content"])
            compact_user = {
                "quarter": user.get("quarter"),
                "role_id": user.get("role_id"),
                "round_id": user.get("round_id"),
                "current_strategy": _round_tree(user.get("current_strategy", {})),
                "belief": _round_tree(user.get("belief", {})),
                "policy_cost": _round_tree(user.get("policy_cost", {})),
            }
            compact_assistant = {
                "equilibrium_strategy": _round_tree(assistant.get("equilibrium_strategy", {})),
                "fed_prediction": _round_tree(assistant.get("fed_prediction", {})),
                "converged": bool(assistant.get("converged", False)),
                "evidence_chain": evidence_by_quarter.get(str(user.get("quarter")), []),
            }
            compact = {
                "task": "compact_equilibrium_distillation",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return strict JSON with keys equilibrium_strategy, fed_prediction, converged, "
                            "and evidence_chain. evidence_chain must contain exactly one ref object with source_id only. "
                            "No prose."
                        ),
                    },
                    {"role": "user", "content": json.dumps(compact_user, ensure_ascii=False, separators=(",", ":"))},
                    {"role": "assistant", "content": json.dumps(compact_assistant, ensure_ascii=False, separators=(",", ":"))},
                ],
                "metadata": row.get("metadata", {}),
            }
            out.write(json.dumps(compact, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    if written == 0:
        raise RuntimeError(f"No compact equilibrium rows written from {source_path}")
    return out_path


def _load_trace_evidence(config: RuntimeConfig) -> dict[str, list[dict[str, Any]]]:
    trace_path = default_self_play_trace_path(config)
    if not trace_path.exists():
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            trace = json.loads(line)
            quarter = str(trace.get("quarter", ""))
            if not quarter:
                continue
            refs: list[dict[str, Any]] = []
            for proposal in trace.get("proposals", []):
                if proposal.get("cluster_id") != "USA":
                    continue
                for item in proposal.get("evidence", []) or []:
                    ref = _compact_evidence_ref(item, role_id=proposal.get("role_id"))
                    if ref:
                        refs.append(ref)
                    if len(refs) >= 1:
                        break
                if len(refs) >= 1:
                    break
            for idx, item in enumerate(trace.get("evidence_chain", []) or []):
                if len(refs) >= 1:
                    break
                refs.append(
                    {
                        "source_id": f"{quarter}:trace:{idx + 1}",
                    }
                )
            result[quarter] = refs
    return result


def _compact_evidence_ref(item: dict[str, Any], *, role_id: str | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    source_id = item.get("doc_id") or item.get("source_id") or item.get("url") or item.get("title")
    if not source_id:
        return None
    return {
        "source_id": str(source_id),
    }


def _round_tree(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_tree(item) for item in value]
    return value

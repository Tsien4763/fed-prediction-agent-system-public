"""LangChain runtime that composes the public agent facades.

This module intentionally requires ``langchain-core``. Install with:

    pip install -e ".[agent]"

The heavier multi-agent game loop remains in ``fed_game.self_play``. This
runtime provides the explicit LangChain Runnable orchestration layer expected
by agent-framework evaluations.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from agents.data_perception import DataPerceptionAgent
from agents.decision_reasoning import DecisionReasoningAgent
from agents.evidence_chain import EvidenceChainAgent
from agents.multi_cluster_game import MultiClusterGameAgent
from agents.runtime_support import utc_now
from agents.semantic_extraction import SemanticExtractionAgent


AGENT_ORDER = [
    "data_perception",
    "semantic_extraction",
    "decision_reasoning",
    "multi_cluster_game",
    "evidence_chain",
]
AGENT_CLASSES = [
    DataPerceptionAgent,
    SemanticExtractionAgent,
    DecisionReasoningAgent,
    MultiClusterGameAgent,
    EvidenceChainAgent,
]


@dataclass
class AgentRuntimeState:
    """JSON-compatible runtime envelope passed through the Runnable chain."""

    event: dict[str, Any] = field(default_factory=dict)
    source_specs: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    semantic_signals: list[dict[str, Any]] = field(default_factory=list)
    decision_context: dict[str, Any] = field(default_factory=dict)
    game_context: dict[str, Any] = field(default_factory=dict)
    evidence_report: dict[str, Any] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AgentRuntimeState":
        known = {field_name for field_name in cls.__dataclass_fields__}
        payload = {key: value.get(key) for key in known if key in value}
        state = cls(**payload)
        extras = {key: item for key, item in value.items() if key not in known}
        if extras:
            state.runtime.setdefault("extra_inputs", {}).update(extras)
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_policy_forecasting_chain() -> Runnable[Any, dict[str, Any]]:
    """Build the five-agent LangChain Runnable sequence."""

    chain: Runnable[Any, dict[str, Any]] = RunnableLambda(_normalize_state).with_config(
        {"run_name": "agent_runtime.normalize_state"}
    )
    for agent_cls in AGENT_CLASSES:
        agent = agent_cls()
        chain = chain | RunnableLambda(agent.run).with_config({"run_name": f"agent_runtime.{agent.name}"})
    return chain


def invoke_policy_runtime(state: AgentRuntimeState | dict[str, Any]) -> dict[str, Any]:
    """Run one policy-forecasting orchestration pass."""

    return build_policy_forecasting_chain().invoke(state)


def _normalize_state(value: AgentRuntimeState | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, AgentRuntimeState):
        state = value.to_dict()
    elif isinstance(value, dict):
        state = AgentRuntimeState.from_mapping(value).to_dict()
    else:
        raise TypeError("LangChain policy runtime expects a dict or AgentRuntimeState.")

    state.setdefault("runtime", {})
    state["runtime"].update(
        {
            "orchestrator": "langchain_runnable_sequence",
            "agent_order": AGENT_ORDER,
            "started_at": state["runtime"].get("started_at") or utc_now(),
        }
    )
    state.setdefault("audit_trail", [])
    return state


def _demo_state() -> dict[str, Any]:
    return {
        "documents": [
            {
                "source_id": "demo_fomc",
                "url": "https://www.federalreserve.gov/demo",
                "title": "Demo FOMC statement",
                "published_at": "2024-06-12",
                "text_hash": "demo",
                "text": "Inflation remains elevated. The Committee will assess incoming data and risks.",
                "metadata": {"document_type": "fomc_statement"},
            }
        ],
        "decision_context": {
            "fed_prediction": {"hike_25bp": 0.25, "hold": 0.65, "cut_25bp": 0.10}
        },
        "game_context": {"quarter": "2024Q2"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LangChain five-agent runtime demo.")
    parser.add_argument("--input-json", type=Path, help="Optional JSON file with an AgentRuntimeState payload.")
    args = parser.parse_args(argv)

    payload = json.loads(args.input_json.read_text(encoding="utf-8")) if args.input_json else _demo_state()
    result = invoke_policy_runtime(payload)
    print(
        json.dumps(
            {
                "runtime": result["runtime"],
                "audit_trail": result["audit_trail"],
                "evidence_report": result["evidence_report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

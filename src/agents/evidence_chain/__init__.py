"""Agent 5: Evidence Chain & Risk Attribution for FOMC predictions.

Responsible for:
  - SHAP / permutation predictive attribution (which macro variables the model used)
  - VECM long-run equilibrium signal (how far from equilibrium, which direction)
  - Taylor Rule policy gap (is policy too loose or too tight)
  - LLM game theory signals (Warsh hawkish index, Warsh consistency, energy risk)
  - Identification diagnostics scaffold (event-study and placebo checks)
  - Counterfactual self-play reports under explicit scenario overrides
  - Human-readable evidence chain with predictive attribution

Entry points:
  from agents.evidence_chain import generate_report, run_identification_diagnostics, run_counterfactual

Implementation:
  Core logic in models/evidence_chain.py
  Identification diagnostics in models/identification_diagnostics.py
  Counterfactual simulation in fed_game/counterfactual.py
  Game bridge in models/game_bridge.py
  fed_game evaluation in fed_game/evaluation.py
"""
from fed_game.counterfactual import (
    CounterfactualResult,
    parse_override_assignments,
    run_counterfactual,
)
from models.identification_diagnostics import run_diagnostics as run_identification_diagnostics
from models.evidence_chain import (
    compute_shapley_attribution,
    build_evidence_chain,
    run as generate_report,
)
from datetime import datetime, timezone
from typing import Any

from agents.runtime_support import append_audit


class EvidenceChainAgent:
    """Agent boundary for evidence assembly and predictive attribution metadata."""

    name = "evidence_chain"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence_items: list[dict[str, Any]] = []
        for doc in state.get("documents", [])[:5]:
            evidence_items.append(
                {
                    "layer": "data_perception",
                    "source_id": doc.get("source_id"),
                    "url": doc.get("url"),
                    "published_at": doc.get("published_at"),
                    "title": doc.get("title"),
                    "text_hash": doc.get("text_hash"),
                }
            )
        for signal in state.get("semantic_signals", [])[:5]:
            score = signal.get("score", {})
            evidence_items.append(
                {
                    "layer": "semantic_extraction",
                    "source_id": signal.get("source_id"),
                    "hawkish_dovish_score": score.get("hawkish_dovish_score") if isinstance(score, dict) else None,
                    "method": score.get("_method") if isinstance(score, dict) else None,
                }
            )
        if state.get("game_context"):
            evidence_items.append(
                {
                    "layer": "multi_cluster_game",
                    "status": state["game_context"].get("status"),
                    "role_count": state["game_context"].get("role_count"),
                    "cluster_count": state["game_context"].get("cluster_count"),
                }
            )

        report = {
            "status": "built",
            "facade": __name__,
            "analysis_scope": "langchain_runtime_evidence",
            "evidence_items": evidence_items,
            "audit_trail_length": len(state.get("audit_trail", [])) + 1,
        }
        state["evidence_report"] = report
        state.setdefault("runtime", {})["finished_at"] = datetime.now(timezone.utc).isoformat()
        return append_audit(state, self.name, report)

__all__ = [
    "EvidenceChainAgent",
    "compute_shapley_attribution",
    "build_evidence_chain",
    "CounterfactualResult",
    "generate_report",
    "parse_override_assignments",
    "run_identification_diagnostics",
    "run_counterfactual",
]

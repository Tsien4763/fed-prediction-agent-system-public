"""Agent 2: Semantic Feature Extraction — LLM-based policy text → quantifiable vectors.

Responsible for:
  - Hawkish-dovish scoring of FOMC statements/minutes/speeches
  - Policy stance extraction (inflation concern, labor assessment, growth outlook)
  - Signal property extraction (uncertainty, commitment, strategic ambiguity)
  - Internal constraint extraction (fiscal space, institutional independence, policy turn cost)
  - Strategy card generation for RAG indexing

Entry points:
  from agents.semantic_extraction import score_document, generate_strategy_cards

Implementation:
  Core logic in models/semantic_pipeline.py (hawkish-dovish scoring)
  Strategy cards in data_engineering/build_strategy_cards.py
  RAG index in data_engineering/build_rag_index.py
  Context snapshots in data_engineering/build_context_snapshots.py
"""
from models.semantic_pipeline import score_hawkish_dovish as score_document
from data_engineering.build_strategy_cards import build_strategy_cards as generate_strategy_cards
from typing import Any

from agents.runtime_support import append_audit


class SemanticExtractionAgent:
    """Agent boundary for policy text to structured semantic features."""

    name = "semantic_extraction"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        signals = list(state.get("semantic_signals", []))
        runtime_inputs = state.get("runtime", {}).get("extra_inputs", {})
        for doc in state.get("documents", []):
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            score = score_document(
                text,
                meeting_date=doc.get("published_at") or state.get("event", {}).get("as_of"),
                api_key=runtime_inputs.get("semantic_api_key"),
                base_url=runtime_inputs.get("semantic_base_url"),
                require_llm=True,
            )
            signals.append(
                {
                    "source_id": doc.get("source_id"),
                    "url": doc.get("url"),
                    "text_hash": doc.get("text_hash"),
                    "published_at": doc.get("published_at"),
                    "score": score,
                }
            )

        state["semantic_signals"] = signals
        llm_count = sum(1 for item in signals if str(item.get("score", {}).get("_method", "")).lower().startswith("llm"))
        result = {
            "status": "scored",
            "signal_count": len(signals),
            "llm_count": llm_count,
            "tfidf_filtered_count": sum(
                1
                for item in signals
                if item.get("score", {}).get("_semantic_filter", {}).get("filter") == "tfidf_policy_context"
            ),
            "fallback_disabled": True,
            "strict_llm_required": True,
        }
        state[self.name] = result
        return append_audit(state, self.name, result)

__all__ = [
    "SemanticExtractionAgent",
    "score_document",
    "generate_strategy_cards",
]

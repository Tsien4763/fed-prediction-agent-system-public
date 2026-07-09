from __future__ import annotations

import json
from pathlib import Path

from langchain_core.runnables import Runnable

from agents.data_perception import DataPerceptionAgent
from agents.decision_reasoning import DecisionReasoningAgent
from agents.evidence_chain import EvidenceChainAgent
from agents.langchain_runtime import AGENT_CLASSES, AGENT_ORDER, build_policy_forecasting_chain, invoke_policy_runtime
from agents.multi_cluster_game import MultiClusterGameAgent
from agents.semantic_extraction import SemanticExtractionAgent
from data_engineering.source_monitor import PolicySourceSpec


def _install_fake_deepseek(monkeypatch) -> list[dict[str, object]]:
    from models import semantic_pipeline

    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            content = json.dumps(
                {
                    "hawkish_dovish_score": 0.42,
                    "inflation_concern": 0.8,
                    "labor_market_assessment": 0.1,
                    "growth_outlook": 0.0,
                    "forward_guidance_strength": 0.4,
                    "uncertainty_index": 0.3,
                    "rate_hike_signal": 0.55,
                    "rate_cut_signal": 0.05,
                    "policy_flexibility": 0.5,
                    "inflation_commitment_credibility": 0.7,
                }
            )
            return {"choices": [{"message": {"content": content}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(semantic_pipeline.requests, "post", fake_post)
    monkeypatch.setattr(semantic_pipeline.time, "sleep", lambda _seconds: None)
    return calls


def test_agent_facades_expose_concrete_runtime_classes() -> None:
    assert AGENT_CLASSES == [
        DataPerceptionAgent,
        SemanticExtractionAgent,
        DecisionReasoningAgent,
        MultiClusterGameAgent,
        EvidenceChainAgent,
    ]
    for agent_cls, expected_name in zip(AGENT_CLASSES, AGENT_ORDER):
        agent = agent_cls()
        assert agent.name == expected_name
        assert callable(agent.run)


def test_langchain_runtime_is_real_runnable_sequence(tmp_path: Path, monkeypatch) -> None:
    calls = _install_fake_deepseek(monkeypatch)
    pages = {
        "https://example.test/fomc": """
        <html><head><title>FOMC index</title></head><body>
          <a href="/fomc/20240612a.htm">June statement</a>
        </body></html>
        """,
        "https://example.test/fomc/20240612a.htm": """
        <html><head><title>June FOMC statement</title></head>
        <body><time datetime="2024-06-12"></time>
        <p>Inflation remains elevated. The Committee will assess incoming data and risks.</p>
        </body></html>
        """,
    }
    spec = PolicySourceSpec(
        source_id="fed_test",
        url="https://example.test/fomc",
        max_depth=1,
        max_pages=3,
        allowed_url_patterns=[r"/fomc/\d{8}a\.htm$"],
        metadata={"institution": "Federal Reserve"},
    )
    chain = build_policy_forecasting_chain()

    assert isinstance(chain, Runnable)

    result = chain.invoke(
        {
            "source_specs": [spec.to_dict()],
            "http_get": lambda url: pages[url],
            "policy_store_path": tmp_path / "policy_documents.jsonl",
            "dry_run": True,
            "decision_context": {
                "fed_prediction": {"hike_25bp": 0.2, "hold": 0.7, "cut_25bp": 0.1}
            },
            "game_context": {"quarter": "2024Q2"},
            "semantic_api_key": "test-key",
            "semantic_base_url": "https://api.example.test",
        }
    )

    assert result["runtime"]["orchestrator"] == "langchain_runnable_sequence"
    assert [item["agent"] for item in result["audit_trail"]] == AGENT_ORDER
    assert result["data_perception"]["status"] == "fetched"
    assert result["data_perception"]["new_documents"] == 2
    assert result["semantic_extraction"]["signal_count"] == 2
    assert result["semantic_extraction"]["llm_count"] == 2
    assert result["semantic_extraction"]["tfidf_filtered_count"] == 2
    assert result["semantic_extraction"]["fallback_disabled"] is True
    assert len(calls) == 2
    assert result["decision_context"]["status"] == "prepared"
    assert result["game_context"]["self_play_engine"] == "fed_game.self_play.RollingSelfPlayEngine"
    assert result["evidence_report"]["analysis_scope"] == "langchain_runtime_evidence"


def test_langchain_runtime_helper_invokes_all_five_agents(monkeypatch) -> None:
    _install_fake_deepseek(monkeypatch)
    result = invoke_policy_runtime(
        {
            "documents": [
                {
                    "source_id": "demo_fomc",
                    "url": "https://www.federalreserve.gov/demo",
                    "title": "Demo FOMC statement",
                    "published_at": "2024-06-12",
                    "text_hash": "demo",
                    "text": "Inflation remains elevated and policy remains data dependent.",
                }
            ],
            "decision_context": {
                "fed_prediction": {"hike_25bp": 0.25, "hold": 0.65, "cut_25bp": 0.10}
            },
            "game_context": {"quarter": "2024Q2"},
            "semantic_api_key": "test-key",
            "semantic_base_url": "https://api.example.test",
        }
    )

    assert [item["agent"] for item in result["audit_trail"]] == AGENT_ORDER
    assert result["event"]["event_type"] == "langchain_runtime_tick"
    assert result["evidence_report"]["status"] == "built"


def test_semantic_agent_requires_deepseek_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    try:
        invoke_policy_runtime(
            {
                "documents": [
                    {
                        "source_id": "demo_fomc",
                        "published_at": "2024-06-12",
                        "text": "Inflation remains elevated.",
                    }
                ],
            }
        )
    except RuntimeError as exc:
        assert "DeepSeek semantic extraction is required" in str(exc)
    else:
        raise AssertionError("strict semantic LLM mode should fail without an API key")

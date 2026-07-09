from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_source_monitor() -> None:
    from data_engineering.source_monitor import PolicySourceSpec, crawl_policy_source, incremental_fetch

    pages = {
        "https://example.test/fomc": """
    <html>
      <head><title>FOMC index</title></head>
      <body>
        <a href="/fomc/20260617a.htm">June statement</a>
        <a href="/fomc/20260729a.htm">July statement</a>
        <a href="https://outside.test/ignore.htm">outside</a>
      </body>
    </html>
    """,
        "https://example.test/fomc/20260617a.htm": """
    <html><head><title>June FOMC statement</title></head>
      <body><time datetime="2026-06-17"></time>
      <article><p>Inflation remains elevated and policy remains data dependent.</p></article></body></html>
    """,
        "https://example.test/fomc/20260729a.htm": """
    <html><head><title>July FOMC statement</title></head>
      <body><time datetime="2026-07-29"></time>
      <article><p>The Committee will assess incoming data and risks.</p></article></body></html>
    """,
    }

    spec = PolicySourceSpec(
        source_id="fed_test",
        url="https://example.test/fomc",
        parser="generic_html",
        max_pages=5,
        max_depth=1,
        allowed_url_patterns=[r"/fomc/\d{8}a\.htm$"],
        metadata={"institution": "Federal Reserve"},
    )
    fake_get = lambda url: pages[url]
    crawled = crawl_policy_source(spec, http_get=fake_get)
    assert crawled["pages_fetched"] == 3
    assert len(crawled["documents"]) == 3
    assert "https://example.test/fomc/20260617a.htm" in crawled["visited_urls"]

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "docs.jsonl"
        first = incremental_fetch([spec], store_path=store, http_get=fake_get)
        second = incremental_fetch([spec], store_path=store, http_get=fake_get)
    assert first["new_documents"] == 3
    assert second["duplicate_documents"] == 3
    doc = first["documents"][0]
    for key in ["source_id", "url", "title", "published_at", "fetched_at", "text", "text_hash", "metadata"]:
        assert key in doc


def test_event_pipeline() -> None:
    from models.event_pipeline import FileEventBus, InMemoryEventBus, KafkaEventBus, RedisEventBus, RollingPredictor

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bus = FileEventBus(root / "triggers")
        predictor = RollingPredictor(archive_dir=root / "predictions")
        event = bus.publish("policy_document_update", {"count": 1}, source_ids=["fed_test"], as_of="2026-06-17")
        polled = bus.poll()
        assert len(polled) == 1
        result = predictor.handle_event(polled[0])
        bus.ack(event.event_id)
        assert result["event"]["event_id"] == event.event_id
        assert result["input_documents"] == ["fed_test"]
        assert "evidence_chain" in result
        assert "risk_attribution" in result

    memory_bus = InMemoryEventBus()
    memory_event = memory_bus.publish("semantic_signal_update", {"count": 2}, source_ids=["rag_index"])
    assert memory_bus.poll(max_events=1)[0].event_id == memory_event.event_id
    memory_bus.ack(memory_event.event_id)
    assert memory_bus.poll() == []

    class FakeRedisClient:
        def __init__(self) -> None:
            self.entries = []
            self.acked = []

        def xgroup_create(self, *_args, **_kwargs) -> None:
            return None

        def xadd(self, _stream, fields):
            entry_id = f"{len(self.entries) + 1}-0"
            self.entries.append((entry_id, fields))
            return entry_id

        def xreadgroup(self, _group, _consumer, streams, count=None, block=None):
            stream = next(iter(streams.keys()))
            return [(stream, self.entries[:count])]

        def xack(self, _stream, _group, entry_id) -> None:
            self.acked.append(entry_id)

    redis_client = FakeRedisClient()
    redis_bus = RedisEventBus(client=redis_client, stream="events", group="workers", block_ms=0)
    redis_event = redis_bus.publish("policy_document_update", {"count": 3}, source_ids=["fed_test"])
    assert redis_bus.poll()[0].event_id == redis_event.event_id
    redis_bus.ack(redis_event.event_id)
    assert redis_client.acked == ["1-0"]

    class FakeKafkaMessage:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def error(self):
            return None

    class FakeKafkaProducer:
        def __init__(self) -> None:
            self.messages = []

        def produce(self, _topic, key, value):
            self.messages.append(FakeKafkaMessage(value))

        def poll(self, _timeout):
            return None

        def flush(self, _timeout=None):
            return None

    class FakeKafkaConsumer:
        def __init__(self, producer):
            self.producer = producer
            self.committed = []

        def poll(self, _timeout):
            return self.producer.messages.pop(0) if self.producer.messages else None

        def commit(self, message, asynchronous=False):
            self.committed.append(message)

    kafka_producer = FakeKafkaProducer()
    kafka_consumer = FakeKafkaConsumer(kafka_producer)
    kafka_bus = KafkaEventBus(producer=kafka_producer, consumer=kafka_consumer, topic="events")
    kafka_event = kafka_bus.publish("macro_data_update", {"count": 4}, source_ids=["fred_macro"])
    assert kafka_bus.poll()[0].event_id == kafka_event.event_id
    kafka_bus.ack(kafka_event.event_id)
    assert len(kafka_consumer.committed) == 1


def test_identification_diagnostics() -> None:
    from models.identification_diagnostics import run_diagnostics

    report = run_diagnostics()
    assert report["analysis_scope"].startswith("Forecast diagnostics")
    assert report["status"] == "diagnostic_scaffold"
    assert "event_study" in report


def test_counterfactual_report() -> None:
    from fed_game.counterfactual import CounterfactualResult, build_counterfactual_briefing, parse_override_assignments

    overrides = parse_override_assignments(
        ["inflation_cpi_yoy=4.5", "energy_risk=0.1", "warsh_replaced_by_powell=true"]
    )
    briefing = build_counterfactual_briefing("2024Q2", overrides)
    assert briefing["inflation_cpi_yoy"] == 4.5
    assert briefing["energy_risk_from_vecm"] == 0.1
    assert briefing["fed_chair"] == "powell"
    assert "taylor_gap_pp" in briefing

    result = CounterfactualResult(
        quarter="2024Q2",
        scenario_name="high_inflation",
        overrides=overrides,
        factual_prediction={"hike_25bp": 0.2, "hold": 0.7, "cut_25bp": 0.1},
        counterfactual_prediction={"hike_25bp": 0.5, "hold": 0.45, "cut_25bp": 0.05},
        delta={"hike_25bp": 0.3, "hold": -0.25, "cut_25bp": -0.05},
        factual_evidence=["base"],
        counterfactual_evidence=["base", "counterfactual"],
        briefing_delta={"inflation_cpi_yoy": {"factual": 3.3, "counterfactual": 4.5}},
        strategy_delta=[
            {
                "scope": "cluster:USA",
                "field": "rate_hike_25bp_prob",
                "factual": 0.2,
                "counterfactual": 0.5,
                "delta": 0.3,
            }
        ],
        belief_delta=[
            {
                "role_id": "usa_warsh",
                "field": "inflation_persistence",
                "factual": 0.4,
                "counterfactual": 0.7,
                "delta": 0.3,
            }
        ],
        evidence_delta={"added": ["counterfactual"], "removed": []},
    )
    markdown = result.to_markdown()
    assert "Fed Decision Delta" in markdown
    assert "+30.00pp" in markdown


def test_llm_payoff_and_equilibrium_judge() -> None:
    from fed_game.clusters import get_role
    from fed_game.persona import load_policy_persona
    from fed_game.schemas import AgentMemory, StrategyVector
    from fed_game.skills import LLMEquilibriumJudge, LLMBestResponseSkill

    class FakeTeacher:
        def __init__(self) -> None:
            self.schema_types = []
            self.messages = []

        def chat_json(self, messages, schema_hint=None):
            schema_type = (schema_hint or {}).get("type")
            self.schema_types.append(schema_type)
            self.messages.append(messages)
            if schema_type == "policy_best_response":
                return {
                    "strategy": {
                        "hawkish_signal_prob": 0.66,
                        "rate_hike_25bp_prob": 0.31,
                        "hold_with_hawkish_statement_prob": 0.72,
                        "remove_forward_guidance_prob": 0.61,
                        "easing_signal_prob": 0.08,
                        "liquidity_support_prob": 0.16,
                        "trade_or_sanction_pressure_prob": 0.23,
                    },
                    "rationale": "Inflation credibility dominates, but the role keeps optionality.",
                }
            if schema_type == "policy_payoff_judgement":
                return {
                    "payoff": 0.74,
                    "regret_estimate": 0.01,
                    "best_unilateral_deviation": {
                        "strategy_change": "slightly lower hike probability",
                        "alternative_payoff": 0.75,
                    },
                    "reasoning": "The strategy is close to the role objective with little profitable deviation.",
                    "confidence": 0.8,
                }
            if schema_type == "nash_equilibrium_check":
                return {
                    "is_nash_equilibrium": "true",
                    "max_profitable_deviation_gain": 0.01,
                    "profitable_deviations": [],
                    "reasoning": "No role has a material unilateral improvement.",
                    "confidence": 0.77,
                }
            raise AssertionError(f"unexpected schema type: {schema_type}")

    teacher = FakeTeacher()
    skill = LLMBestResponseSkill(teacher)
    persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")
    context = {
        "quarter": "2024Q2",
        "var_vecm_briefing": {
            "inflation_cpi_yoy": 3.4,
            "unemployment": 4.0,
            "taylor_gap_pp": 0.8,
            "ect_signal": "near_equilibrium",
        },
        "policy_persona": persona.to_prompt_payload(),
    }
    proposal = skill.run(
        role=get_role("usa_warsh"),
        memory=AgentMemory(role_id="usa_warsh", cluster_id="USA"),
        context=context,
        opponent_strategies={"CHN": StrategyVector(trade_or_sanction_pressure_prob=0.4)},
        evidence=[{"title": "FOMC statement", "text": "Inflation remains elevated."}],
        round_id=1,
    )
    assert proposal.payoff_source == "deepseek_payoff_judge"
    assert proposal.payoff_estimate == 0.74
    assert proposal.regret_estimate == 0.01
    assert proposal.deviation_candidate["alternative_payoff"] == 0.75

    judge = LLMEquilibriumJudge(teacher)
    check = judge.check(
        quarter="2024Q2",
        round_id=10,
        cluster_strategies={"USA": proposal.strategy, "CHN": StrategyVector()},
        proposals=[proposal],
        var_vecm_briefing=context["var_vecm_briefing"],
        heuristic={
            "max_strategy_distance": 0.01,
            "strategy_epsilon": 0.025,
            "max_deviation_gain": 0.01,
            "deviation_gain_tau": 0.02,
            "stable_rounds": 10,
            "stable_rounds_required": 10,
            "heuristic_passed": True,
        },
        role_personas={"usa_warsh": persona.to_prompt_payload(max_markdown_chars=1200)},
    )
    assert check["source"] == "deepseek_equilibrium_judge"
    assert check["checked"] is True
    assert check["is_nash_equilibrium"] is True
    assert "policy_best_response" in teacher.schema_types
    assert "policy_payoff_judgement" in teacher.schema_types
    assert "nash_equilibrium_check" in teacher.schema_types
    prompt_text = json.dumps(teacher.messages, ensure_ascii=False)
    assert "policy_persona" in prompt_text
    assert "Credibility is the real policy multiplier" in prompt_text


def test_policy_persona_loader() -> None:
    from fed_game.config import load_config
    from fed_game.persona import load_configured_personas, load_policy_persona
    from fed_game.schemas import StrategyVector

    persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")
    assert persona.role_id == "usa_warsh"
    assert len(persona.nuwa_dimensions) == 6
    assert len(persona.mental_models) >= 3
    assert len(persona.decision_heuristics) >= 5
    assert len(persona.evidence_sources) >= 6
    payload = persona.to_prompt_payload(max_markdown_chars=800)
    assert payload["priors"]["hawkish_signal_prob"] > payload["priors"]["easing_signal_prob"]
    score = persona.consistency_score(StrategyVector.from_dict(persona.priors))
    assert score["score"] == 1.0

    configured = load_configured_personas(load_config())
    assert configured["usa_warsh"].persona_id == "kevin-warsh-policy-persona"


def test_forecast_metrics() -> None:
    from fed_game.evaluation import evaluate_forecasting_traces

    rows = [
        {
            "quarter": "2024Q1",
            "converged": True,
            "rounds": 10,
            "fed_prediction": {"hike_25bp": 0.1, "hold": 0.8, "cut_25bp": 0.1},
        },
        {
            "quarter": "2024Q3",
            "converged": True,
            "rounds": 10,
            "fed_prediction": {"hike_25bp": 0.1, "hold": 0.2, "cut_25bp": 0.7},
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traces.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        report = evaluate_forecasting_traces(path)
    metrics = report["forecast_metrics"]
    for key in ["accuracy", "brier_score", "log_loss", "balanced_accuracy", "macro_f1", "hold_collapse_index"]:
        assert key in metrics
    assert metrics["non_hold_recall"] is not None
    assert "equilibrium_checks" in report


def test_agent_imports() -> None:
    import agents.data_perception  # noqa: F401
    import agents.decision_reasoning  # noqa: F401
    import agents.evidence_chain  # noqa: F401
    import agents.multi_cluster_game  # noqa: F401
    import agents.semantic_extraction  # noqa: F401
    import fed_game.cli  # noqa: F401


def main() -> int:
    test_source_monitor()
    test_event_pipeline()
    test_identification_diagnostics()
    test_counterfactual_report()
    test_policy_persona_loader()
    test_llm_payoff_and_equilibrium_judge()
    test_forecast_metrics()
    test_agent_imports()
    print("hardening_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

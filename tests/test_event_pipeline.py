from __future__ import annotations

from pathlib import Path

from models.event_pipeline import (
    FileEventBus,
    ForecastEvent,
    InMemoryEventBus,
    KafkaEventBus,
    RedisEventBus,
    RollingPredictor,
    as_of_to_quarter,
    build_counterfactual_request,
)
from scripts.smoke_realtime_pipeline import run_realtime_smoke


def test_file_and_memory_event_buses_drive_rolling_prediction_record(tmp_path: Path) -> None:
    bus = FileEventBus(tmp_path / "triggers")
    predictor = RollingPredictor(archive_dir=tmp_path / "predictions")

    event = bus.publish("policy_document_update", {"count": 1}, source_ids=["fed_test"], as_of="2026-06-17")
    polled = bus.poll()
    assert len(polled) == 1

    result = predictor.handle_event(polled[0])
    bus.ack(event.event_id)

    assert result["event"]["event_id"] == event.event_id
    assert result["input_documents"] == ["fed_test"]
    assert "evidence_chain" in result
    assert result["risk_attribution"]["analysis_scope"] == "rolling_prediction_attribution"

    memory_bus = InMemoryEventBus()
    memory_event = memory_bus.publish("semantic_signal_update", {"count": 2}, source_ids=["rag_index"])
    assert memory_bus.poll(max_events=1)[0].event_id == memory_event.event_id
    memory_bus.ack(memory_event.event_id)
    assert memory_bus.poll() == []


def test_redis_event_bus_adapter_contract() -> None:
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

    client = FakeRedisClient()
    bus = RedisEventBus(client=client, stream="events", group="workers", block_ms=0)
    event = bus.publish("policy_document_update", {"count": 3}, source_ids=["fed_test"])

    assert bus.poll()[0].event_id == event.event_id
    bus.ack(event.event_id)
    assert client.acked == ["1-0"]


def test_event_counterfactual_request_from_daily_policy_shock() -> None:
    event = ForecastEvent.create(
        "geopolitical_shock",
        {
            "scenario_name": "energy_supply_stress",
            "energy_risk": 0.82,
            "geopolitical_escalation": 0.74,
            "max_rounds": 2,
        },
        as_of="2026-05-06T13:45:00Z",
        source_ids=["policy_wire"],
    )

    request = build_counterfactual_request(event)

    assert request is not None
    assert request["quarter"] == "2026Q2"
    assert request["scenario_name"] == "energy_supply_stress"
    assert request["overrides"]["energy_risk"] == 0.82
    assert request["overrides"]["geopolitical_escalation"] == 0.74
    assert request["max_rounds"] == 2
    assert as_of_to_quarter("2026-12-31T23:59:00Z") == "2026Q4"


def test_rolling_predictor_runs_p5_counterfactual_from_event(tmp_path: Path) -> None:
    calls = []

    def fake_counterfactual_runner(**request):
        calls.append(request)
        return {
            "quarter": request["quarter"],
            "scenario_name": request["scenario_name"],
            "overrides": request["overrides"],
            "delta": {"hike_25bp": 0.08, "hold": -0.05, "cut_25bp": -0.03},
            "p5_impact_summary": [
                {
                    "cluster_id": "RUS",
                    "impact_label": "external-pressure shock",
                    "strategy_shift_l1": 0.31,
                    "hawkish_pressure_delta": 0.04,
                    "external_pressure_delta": 0.22,
                }
            ],
            "strategy_delta": [{"scope": "cluster:RUS", "field": "trade_or_sanction_pressure_prob", "delta": 0.22}],
            "belief_delta": [{"role_id": "russia_energy", "field": "energy_shock_risk", "delta": 0.18}],
            "evidence_delta": {"added": ["event shock"], "removed": []},
        }

    predictor = RollingPredictor(
        archive_dir=tmp_path / "predictions",
        counterfactual_runner=fake_counterfactual_runner,
    )
    event = ForecastEvent.create(
        "p5_game_counterfactual",
        {
            "counterfactual": {
                "quarter": "2024Q2",
                "scenario_name": "shipping_lane_stress",
                "overrides": {"energy_risk": 0.75},
                "max_rounds": 1,
                "stable_rounds_required": 1,
            }
        },
        as_of="2024-05-15T09:30:00Z",
        source_ids=["market_news"],
    )

    record = predictor.handle_event(event)

    assert calls[0]["quarter"] == "2024Q2"
    assert calls[0]["max_rounds"] == 1
    assert record["strategic_counterfactual"]["status"] == "completed"
    assert record["strategic_counterfactual"]["p5_impact_summary"][0]["cluster_id"] == "RUS"
    assert record["risk_attribution"]["analysis_scope"] == "rolling_prediction_attribution_with_p5_counterfactual"
    assert record["risk_attribution"]["strategic_counterfactual"]["delta"]["hike_25bp"] == 0.08


def test_kafka_event_bus_adapter_contract() -> None:
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

    producer = FakeKafkaProducer()
    consumer = FakeKafkaConsumer(producer)
    bus = KafkaEventBus(producer=producer, consumer=consumer, topic="events")
    event = bus.publish("macro_data_update", {"count": 4}, source_ids=["fred_macro"])

    assert bus.poll()[0].event_id == event.event_id
    bus.ack(event.event_id)
    assert len(consumer.committed) == 1


def test_realtime_closed_loop_smoke_all_transports() -> None:
    report = run_realtime_smoke()

    assert report["status"] == "realtime_closed_loop_smoke_ok"
    assert set(report["transports"]) == {"file", "memory", "redis", "kafka"}
    for transport, result in report["transports"].items():
        assert result["event_type"] == "policy_document_update"
        assert result["risk_attribution_method"] == "predictive_attribution"
        assert result["analysis_scope"] == "rolling_prediction_attribution"
        assert result["archived"] is True
        assert result["input_documents"] == [f"{transport}_policy_source"]
    assert report["transports"]["redis"]["acked_entries"] == ["1-0"]
    assert report["transports"]["kafka"]["committed_messages"] == 1
    assert report["p5_counterfactual"]["event_type"] == "p5_game_counterfactual"
    assert report["p5_counterfactual"]["analysis_scope"] == "rolling_prediction_attribution_with_p5_counterfactual"
    assert report["p5_counterfactual"]["quarter"] == "2026Q2"
    assert report["p5_counterfactual"]["top_cluster"] == "RUS"

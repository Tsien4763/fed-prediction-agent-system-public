from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.event_pipeline import FileEventBus, InMemoryEventBus, KafkaEventBus, RedisEventBus, RollingPredictor


class FakeRedisClient:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, str]]] = []
        self.acked: list[str] = []

    def xgroup_create(self, *_args, **_kwargs) -> None:
        return None

    def xadd(self, _stream, fields):
        entry_id = f"{len(self.entries) + 1}-0"
        self.entries.append((entry_id, fields))
        return entry_id

    def xreadgroup(self, _group, _consumer, streams, count=None, block=None):
        stream = next(iter(streams.keys()))
        pending = [entry for entry in self.entries if entry[0] not in self.acked]
        return [(stream, pending[:count])]

    def xack(self, _stream, _group, entry_id) -> None:
        self.acked.append(entry_id)


class FakeKafkaMessage:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def error(self):
        return None


class FakeKafkaProducer:
    def __init__(self) -> None:
        self.messages: list[FakeKafkaMessage] = []

    def produce(self, _topic, key, value):
        self.messages.append(FakeKafkaMessage(value))

    def poll(self, _timeout):
        return None

    def flush(self, _timeout=None):
        return None


class FakeKafkaConsumer:
    def __init__(self, producer: FakeKafkaProducer):
        self.producer = producer
        self.committed: list[FakeKafkaMessage] = []

    def poll(self, _timeout):
        return self.producer.messages.pop(0) if self.producer.messages else None

    def commit(self, message, asynchronous=False):
        self.committed.append(message)


def fake_p5_counterfactual_runner(**request: Any) -> dict[str, Any]:
    return {
        "quarter": request["quarter"],
        "scenario_name": request["scenario_name"],
        "overrides": request["overrides"],
        "delta": {"hike_25bp": 0.06, "hold": -0.04, "cut_25bp": -0.02},
        "p5_impact_summary": [
            {
                "cluster_id": "RUS",
                "impact_label": "external-pressure shock",
                "strategy_shift_l1": 0.28,
                "hawkish_pressure_delta": 0.03,
                "external_pressure_delta": 0.21,
            },
            {
                "cluster_id": "USA",
                "impact_label": "hawkish policy pressure",
                "strategy_shift_l1": 0.14,
                "hawkish_pressure_delta": 0.11,
                "external_pressure_delta": 0.0,
            },
        ],
        "strategy_delta": [{"scope": "cluster:RUS", "field": "trade_or_sanction_pressure_prob", "delta": 0.21}],
        "belief_delta": [{"role_id": "russia_energy", "field": "energy_shock_risk", "delta": 0.16}],
        "evidence_delta": {"added": ["event-level energy shock"], "removed": []},
    }


def run_realtime_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        redis_client = FakeRedisClient()
        kafka_producer = FakeKafkaProducer()
        kafka_consumer = FakeKafkaConsumer(kafka_producer)
        transports = {
            "file": FileEventBus(root / "triggers"),
            "memory": InMemoryEventBus(),
            "redis": RedisEventBus(client=redis_client, stream="events", group="workers", block_ms=0),
            "kafka": KafkaEventBus(producer=kafka_producer, consumer=kafka_consumer, topic="events", poll_timeout=0.0),
        }
        results = {}
        for name, bus in transports.items():
            predictor = RollingPredictor(archive_dir=root / f"predictions_{name}")
            event = bus.publish(
                "policy_document_update",
                {"transport": name, "source_count": 1},
                source_ids=[f"{name}_policy_source"],
                as_of="2026-06-17",
            )
            polled = bus.poll(max_events=1)
            if len(polled) != 1:
                raise AssertionError(f"{name} bus did not return exactly one event")
            record = predictor.handle_event(polled[0])
            bus.ack(event.event_id)
            risk = record.get("risk_attribution", {})
            if risk.get("method") != "predictive_attribution":
                raise AssertionError(f"{name} bus did not produce risk attribution")
            if record.get("event", {}).get("event_id") != event.event_id:
                raise AssertionError(f"{name} bus returned mismatched event id")
            results[name] = {
                "event_id": event.event_id,
                "event_type": record["event"]["event_type"],
                "input_documents": record["input_documents"],
                "archived": True,
                "risk_attribution_method": risk.get("method"),
                "analysis_scope": risk.get("analysis_scope"),
            }
        p5_bus = InMemoryEventBus()
        p5_predictor = RollingPredictor(
            archive_dir=root / "predictions_p5_counterfactual",
            counterfactual_runner=fake_p5_counterfactual_runner,
        )
        p5_event = p5_bus.publish(
            "p5_game_counterfactual",
            {
                "scenario_name": "event_level_energy_stress",
                "energy_risk": 0.8,
                "geopolitical_escalation": 0.7,
                "max_rounds": 1,
            },
            source_ids=["event_wire"],
            as_of="2026-05-06T13:45:00Z",
        )
        p5_record = p5_predictor.handle_event(p5_bus.poll(max_events=1)[0])
        p5_bus.ack(p5_event.event_id)
        results["redis"]["acked_entries"] = list(redis_client.acked)
        results["kafka"]["committed_messages"] = len(kafka_consumer.committed)
        return {
            "status": "realtime_closed_loop_smoke_ok",
            "transports": results,
            "p5_counterfactual": {
                "event_type": p5_record["event"]["event_type"],
                "analysis_scope": p5_record["risk_attribution"]["analysis_scope"],
                "quarter": p5_record["strategic_counterfactual"]["quarter"],
                "scenario_name": p5_record["strategic_counterfactual"]["scenario_name"],
                "top_cluster": p5_record["strategic_counterfactual"]["p5_impact_summary"][0]["cluster_id"],
                "delta": p5_record["strategic_counterfactual"]["delta"],
            },
            "boundary": (
                "Redis/Kafka are exercised with in-process fake clients in CI. "
                "Use --extra realtime and real broker URLs for deployment smoke."
            ),
        }


def main() -> int:
    print(json.dumps(run_realtime_smoke(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

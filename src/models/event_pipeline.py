"""Event-driven rolling prediction pipeline.

The default transport is file based so the public repo is runnable on one
machine. The same EventBus protocol also has real Redis Streams and Kafka
adapters for deployment-style rolling prediction.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from data_engineering.config import REPO_ROOT, ensure_dir


TRIGGER_DIR = REPO_ROOT / "data" / "triggers"
ARCHIVE_DIR = REPO_ROOT / "data" / "predictions"
COUNTERFACTUAL_EVENT_TYPES = {
    "strategic_counterfactual",
    "p5_game_counterfactual",
    "geopolitical_shock",
    "macro_shock",
    "policy_shock",
}
COUNTERFACTUAL_OVERRIDE_KEYS = {
    "inflation_cpi_yoy",
    "gdp_growth_qoq_ann",
    "unemployment",
    "fedfunds",
    "energy_risk",
    "energy_price_risk",
    "energy_risk_from_vecm",
    "geopolitical_escalation",
    "dollar_liquidity_pressure",
    "policy_credibility_prior",
    "fed_chair",
    "warsh_replaced_by_powell",
}
CounterfactualRunner = Callable[..., Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ForecastEvent:
    event_id: str
    event_type: str
    as_of: str
    source_ids: list[str]
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
        event_id: str | None = None,
    ) -> "ForecastEvent":
        return cls(
            event_id=event_id or uuid.uuid4().hex,
            event_type=event_type,
            as_of=as_of or utc_now(),
            source_ids=source_ids or [],
            payload=payload or {},
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForecastEvent":
        payload = data.get("payload", {})
        return cls(
            event_id=str(data.get("event_id") or data.get("id") or uuid.uuid4().hex),
            event_type=str(data.get("event_type", "unknown")),
            as_of=str(data.get("as_of") or data.get("timestamp") or data.get("created_at") or utc_now()),
            source_ids=[str(item) for item in data.get("source_ids", [])],
            payload=payload if isinstance(payload, dict) else {"raw": payload},
            created_at=str(data.get("created_at") or data.get("timestamp") or utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus(Protocol):
    def poll(self, max_events: int | None = None) -> list[ForecastEvent]:
        ...

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
    ) -> ForecastEvent:
        ...

    def ack(self, event_id: str) -> None:
        ...


class FileEventBus:
    """File transport for local demos and tests."""

    def __init__(self, watch_dir: Path = TRIGGER_DIR):
        self.watch_dir = ensure_dir(watch_dir)
        self.processed: set[str] = set()

    def poll(self, max_events: int | None = None) -> list[ForecastEvent]:
        events: list[ForecastEvent] = []
        for path in sorted(self.watch_dir.glob("*.json")):
            if path.name in self.processed:
                continue
            event = ForecastEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
            events.append(event)
            self.processed.add(path.name)
            if max_events is not None and len(events) >= max_events:
                break
        return events

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
    ) -> ForecastEvent:
        event = ForecastEvent.create(event_type, payload, as_of=as_of, source_ids=source_ids)
        path = self.watch_dir / f"{event.event_type}_{event.event_id}.json"
        path.write_text(json.dumps(event.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return event

    def ack(self, event_id: str) -> None:
        self.processed.add(event_id)
        for path in self.watch_dir.glob(f"*{event_id}.json"):
            path.unlink(missing_ok=True)
            self.processed.add(path.name)


class InMemoryEventBus:
    """In-process transport used by tests and notebooks."""

    def __init__(self) -> None:
        self._events: list[ForecastEvent] = []
        self._acked: set[str] = set()

    def poll(self, max_events: int | None = None) -> list[ForecastEvent]:
        events = [event for event in self._events if event.event_id not in self._acked]
        return events[:max_events] if max_events is not None else events

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
    ) -> ForecastEvent:
        event = ForecastEvent.create(event_type, payload, as_of=as_of, source_ids=source_ids)
        self._events.append(event)
        return event

    def ack(self, event_id: str) -> None:
        self._acked.add(event_id)


class RedisEventBus:
    """Redis Streams transport using XADD, XREADGROUP, and XACK."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        stream: str = "fed_policy_events",
        group: str = "rolling_predictors",
        consumer: str | None = None,
        block_ms: int = 1000,
        count: int = 10,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("Install realtime dependencies with: uv sync --extra realtime") from exc
            client = redis.from_url(redis_url)
        self.client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer or f"consumer-{uuid.uuid4().hex[:8]}"
        self.block_ms = block_ms
        self.count = count
        self._pending: dict[str, str] = {}
        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def poll(self, max_events: int | None = None) -> list[ForecastEvent]:
        response = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=max_events or self.count,
            block=self.block_ms,
        )
        events: list[ForecastEvent] = []
        for _stream_name, messages in response or []:
            for entry_id, fields in messages:
                payload = _redis_event_payload(fields)
                event = ForecastEvent.from_dict(json.loads(payload))
                entry_id_str = _decode(entry_id)
                self._pending[event.event_id] = entry_id_str
                events.append(event)
        return events

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
    ) -> ForecastEvent:
        event = ForecastEvent.create(event_type, payload, as_of=as_of, source_ids=source_ids)
        encoded = json.dumps(event.to_dict(), ensure_ascii=False)
        self.client.xadd(
            self.stream,
            {
                "event": encoded,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "as_of": event.as_of,
            },
        )
        return event

    def ack(self, event_id: str) -> None:
        entry_id = self._pending.pop(event_id, None)
        if entry_id is not None:
            self.client.xack(self.stream, self.group, entry_id)


class KafkaEventBus:
    """Kafka transport using confluent-kafka with manual commits."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        *,
        topic: str = "fed-policy-events",
        group_id: str = "rolling-predictors",
        client_id: str | None = None,
        poll_timeout: float = 1.0,
        flush_timeout: float = 5.0,
        max_poll_records: int = 10,
        producer: Any | None = None,
        consumer: Any | None = None,
    ) -> None:
        self.topic = topic
        self.poll_timeout = poll_timeout
        self.flush_timeout = flush_timeout
        self.max_poll_records = max_poll_records
        self._pending: dict[str, Any] = {}

        if producer is None or consumer is None:
            try:
                from confluent_kafka import Consumer, Producer
            except ImportError as exc:
                raise RuntimeError("Install realtime dependencies with: uv sync --extra realtime") from exc
            client = client_id or f"fed-policy-{uuid.uuid4().hex[:8]}"
            producer = producer or Producer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "client.id": f"{client}-producer",
                }
            )
            consumer = consumer or Consumer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "group.id": group_id,
                    "client.id": f"{client}-consumer",
                    "enable.auto.commit": False,
                    "auto.offset.reset": "earliest",
                }
            )
            consumer.subscribe([topic])

        self.producer = producer
        self.consumer = consumer

    def poll(self, max_events: int | None = None) -> list[ForecastEvent]:
        target = max_events or self.max_poll_records
        events: list[ForecastEvent] = []
        while len(events) < target:
            message = self.consumer.poll(self.poll_timeout)
            if message is None:
                break
            error = message.error() if hasattr(message, "error") else None
            if error:
                raise RuntimeError(f"Kafka consumer error: {error}")
            raw_value = message.value()
            payload = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else str(raw_value)
            event = ForecastEvent.from_dict(json.loads(payload))
            self._pending[event.event_id] = message
            events.append(event)
        return events

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        source_ids: list[str] | None = None,
    ) -> ForecastEvent:
        event = ForecastEvent.create(event_type, payload, as_of=as_of, source_ids=source_ids)
        encoded = json.dumps(event.to_dict(), ensure_ascii=False).encode("utf-8")
        self.producer.produce(self.topic, key=event.event_id, value=encoded)
        self.producer.poll(0)
        self.producer.flush(self.flush_timeout)
        return event

    def ack(self, event_id: str) -> None:
        message = self._pending.pop(event_id, None)
        if message is not None:
            self.consumer.commit(message=message, asynchronous=False)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _redis_event_payload(fields: dict[Any, Any]) -> str:
    normalized = {_decode(key): _decode(value) for key, value in fields.items()}
    if "event" not in normalized:
        raise KeyError("Redis stream entry is missing the 'event' field")
    return normalized["event"]


def build_event_bus(kind: str = "file", **kwargs: Any) -> EventBus:
    """Create an EventBus by transport name."""
    normalized = kind.lower().strip()
    if normalized == "file":
        return FileEventBus(**kwargs)
    if normalized == "memory":
        return InMemoryEventBus()
    if normalized == "redis":
        return RedisEventBus(**kwargs)
    if normalized == "kafka":
        return KafkaEventBus(**kwargs)
    raise ValueError(f"Unknown event bus kind: {kind}")


def event_bus_from_env() -> EventBus:
    """Build an EventBus from environment variables.

    Supported values:
      MAE_CPS_EVENT_BUS=file|memory|redis|kafka
      REDIS_URL, MAE_CPS_REDIS_STREAM, MAE_CPS_REDIS_GROUP
      KAFKA_BOOTSTRAP_SERVERS, MAE_CPS_KAFKA_TOPIC, MAE_CPS_KAFKA_GROUP
    """
    kind = os.getenv("MAE_CPS_EVENT_BUS", "file").lower()
    if kind == "redis":
        return RedisEventBus(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            stream=os.getenv("MAE_CPS_REDIS_STREAM", "fed_policy_events"),
            group=os.getenv("MAE_CPS_REDIS_GROUP", "rolling_predictors"),
            consumer=os.getenv("MAE_CPS_REDIS_CONSUMER"),
        )
    if kind == "kafka":
        return KafkaEventBus(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic=os.getenv("MAE_CPS_KAFKA_TOPIC", "fed-policy-events"),
            group_id=os.getenv("MAE_CPS_KAFKA_GROUP", "rolling-predictors"),
            client_id=os.getenv("MAE_CPS_KAFKA_CLIENT_ID"),
        )
    if kind == "memory":
        return InMemoryEventBus()
    return FileEventBus(TRIGGER_DIR)


class RollingPredictor:
    """Execute prediction pipeline on each trigger event."""

    def __init__(
        self,
        *,
        archive_dir: Path = ARCHIVE_DIR,
        enable_counterfactuals: bool = True,
        counterfactual_runner: CounterfactualRunner | None = None,
    ):
        self.archive_dir = ensure_dir(archive_dir)
        self.enable_counterfactuals = enable_counterfactuals
        self.counterfactual_runner = counterfactual_runner
        self.run_count = 0

    def handle_event(self, event: ForecastEvent | Path | dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_event(event)
        print(f"\n{'=' * 50}")
        print(f"EVENT: {normalized.event_type} @ {normalized.as_of}")
        print(f"{'=' * 50}")

        self.run_count += 1
        fresh_data = self._check_fresh_data()
        semantic_signals = self._load_semantic_signals()
        prediction = self._run_prediction(semantic_signals)
        evidence = self._generate_evidence(prediction)
        strategic_counterfactual = self._run_event_counterfactual(normalized)
        risk = self._generate_risk_attribution(semantic_signals, evidence, strategic_counterfactual)

        result = {
            "run_id": f"run_{self.run_count:04d}",
            "event": normalized.to_dict(),
            "input_documents": normalized.source_ids,
            "fresh_data_available": fresh_data,
            "semantic_signals": semantic_signals,
            "prediction": prediction,
            "evidence_chain": evidence,
            "risk_attribution": risk,
            "prediction_timestamp": utc_now(),
        }
        if strategic_counterfactual is not None:
            result["strategic_counterfactual"] = strategic_counterfactual

        archive_path = self.archive_dir / f"prediction_{self.run_count:04d}.json"
        archive_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            display_path = archive_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = archive_path
        print(f"  Archived -> {display_path}")
        return result

    def _check_fresh_data(self) -> bool:
        macro_path = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
        if macro_path.exists():
            age_hours = (time.time() - macro_path.stat().st_mtime) / 3600
            return age_hours < 24
        return False

    def _load_semantic_signals(self) -> dict[str, float]:
        full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
        if not full_path.exists():
            return {}
        df = pd.read_parquet(full_path)
        game_cols = [column for column in df.columns if column.startswith("game_")]
        if not game_cols:
            return {}
        latest = df.iloc[-1]
        return {column: float(latest[column]) for column in game_cols if not pd.isna(latest[column])}

    def _run_prediction(self, signals: dict[str, float]) -> dict[str, float]:
        pred_files = sorted(self.archive_dir.glob("prediction_*.json"))
        if pred_files:
            try:
                last = json.loads(pred_files[-1].read_text(encoding="utf-8"))
                prediction = last.get("prediction")
                if isinstance(prediction, dict):
                    return prediction
            except Exception:
                pass
        return {"hold": 0.65, "hike": 0.20, "cut": 0.15}

    def _generate_evidence(self, prediction: dict[str, float]) -> list[str]:
        evidence: list[str] = []
        try:
            full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
            df_full = pd.read_parquet(full_path)
            latest = df_full.iloc[-1]
            ect = latest.get("ect_combined", 0) if not pd.isna(latest.get("ect_combined", 0)) else 0
            if ect < -0.5:
                evidence.append(f"VECM ECT={ect:.2f} < 0: below long-run equilibrium, upward pressure.")
            elif ect > 0.5:
                evidence.append(f"VECM ECT={ect:.2f} > 0: above long-run equilibrium, downward pressure.")

            hawkish = latest.get("game_warsh_hawkish", 0) if not pd.isna(latest.get("game_warsh_hawkish", 0)) else 0
            if hawkish > 0.5:
                evidence.append(f"Game signal: Warsh hawkish index={hawkish:.2f}.")

            energy = latest.get("game_energy_risk", 0) if not pd.isna(latest.get("game_energy_risk", 0)) else 0
            if energy > 0.3:
                evidence.append(f"Game signal: energy risk={energy:.2f}.")
        except Exception as exc:
            evidence.append(f"Evidence fallback: {type(exc).__name__}")
        if not evidence:
            evidence.append("No fresh model artifacts found; using fallback prediction prior.")
        return evidence

    def _generate_risk_attribution(
        self,
        semantic_signals: dict[str, float],
        evidence: list[str],
        strategic_counterfactual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        risk = {
            "method": "predictive_attribution",
            "top_signals": sorted(
                semantic_signals.items(),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:5],
            "evidence_count": len(evidence),
            "analysis_scope": "rolling_prediction_attribution",
        }
        if strategic_counterfactual is not None:
            risk["analysis_scope"] = "rolling_prediction_attribution_with_p5_counterfactual"
            risk["strategic_counterfactual"] = {
                "scenario_name": strategic_counterfactual.get("scenario_name"),
                "quarter": strategic_counterfactual.get("quarter"),
                "delta": strategic_counterfactual.get("delta", {}),
                "top_p5_impacts": strategic_counterfactual.get("p5_impact_summary", [])[:3],
            }
        return risk

    def _run_event_counterfactual(self, event: ForecastEvent) -> dict[str, Any] | None:
        request = build_counterfactual_request(event)
        if not self.enable_counterfactuals or request is None:
            return None
        runner = self.counterfactual_runner or self._default_counterfactual_runner
        try:
            raw = runner(**request)
            data = raw.to_dict() if hasattr(raw, "to_dict") else raw
            if not isinstance(data, dict):
                data = {"raw_result": str(data)}
            return summarize_event_counterfactual(data)
        except Exception as exc:
            return {
                "status": "failed",
                "quarter": request["quarter"],
                "scenario_name": request["scenario_name"],
                "overrides": request["overrides"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }

    @staticmethod
    def _default_counterfactual_runner(**request: Any) -> Any:
        from fed_game.config import load_config
        from fed_game.counterfactual import run_counterfactual

        config = load_config()
        self_play = config.raw.setdefault("self_play", {})
        self_play["max_rounds"] = int(request.pop("max_rounds"))
        self_play["stable_rounds_required"] = int(request.pop("stable_rounds_required"))
        return run_counterfactual(config, **request)


def build_counterfactual_request(event: ForecastEvent) -> dict[str, Any] | None:
    payload = event.payload
    explicit = payload.get("counterfactual")
    trigger = event.event_type in COUNTERFACTUAL_EVENT_TYPES or bool(payload.get("run_counterfactual"))
    if not trigger and not isinstance(explicit, dict):
        return None

    source = explicit if isinstance(explicit, dict) else payload
    overrides = _counterfactual_overrides(source, payload)
    if not overrides and not bool(source.get("run_without_overrides", False)):
        return None
    return {
        "quarter": str(source.get("quarter") or payload.get("quarter") or as_of_to_quarter(event.as_of)),
        "scenario_name": str(
            source.get("scenario_name")
            or payload.get("scenario_name")
            or event.event_type
        ),
        "overrides": overrides,
        "max_context_docs": int(source.get("max_context_docs", payload.get("max_context_docs", 1))),
        "max_rounds": int(source.get("max_rounds", payload.get("max_rounds", 3))),
        "stable_rounds_required": int(
            source.get("stable_rounds_required", payload.get("stable_rounds_required", 1))
        ),
    }


def summarize_event_counterfactual(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": data.get("status", "completed"),
        "quarter": data.get("quarter"),
        "scenario_name": data.get("scenario_name"),
        "overrides": data.get("overrides", {}),
        "delta": data.get("delta", {}),
        "p5_impact_summary": data.get("p5_impact_summary", []),
        "top_strategy_shifts": data.get("strategy_delta", [])[:5],
        "top_belief_shifts": data.get("belief_delta", [])[:5],
        "evidence_delta": data.get("evidence_delta", {}),
        "scenario_scope": data.get("scenario_scope"),
    }


def _counterfactual_overrides(source: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    raw = source.get("overrides", payload.get("overrides", {}))
    overrides = dict(raw) if isinstance(raw, dict) else {}
    for key in COUNTERFACTUAL_OVERRIDE_KEYS:
        if key in source and key not in overrides:
            overrides[key] = source[key]
        elif key in payload and key not in overrides:
            overrides[key] = payload[key]
    return overrides


def as_of_to_quarter(as_of: str) -> str:
    try:
        stamp = pd.Timestamp(as_of)
    except Exception:
        stamp = pd.Timestamp.utcnow()
    quarter = ((int(stamp.month) - 1) // 3) + 1
    return f"{int(stamp.year)}Q{quarter}"


def normalize_event(event: ForecastEvent | Path | dict[str, Any]) -> ForecastEvent:
    if isinstance(event, ForecastEvent):
        return event
    if isinstance(event, Path):
        return ForecastEvent.from_dict(json.loads(event.read_text(encoding="utf-8")))
    return ForecastEvent.from_dict(event)


def run_once() -> None:
    bus = FileEventBus(TRIGGER_DIR)
    predictor = RollingPredictor()
    bus.publish(
        "macro_data_update",
        {
            "source": "FRED",
            "series_updated": ["CPIAUCSL", "UNRATE"],
            "note": "Simulated event trigger for local smoke test",
        },
        source_ids=["fred_macro"],
    )
    for event in bus.poll():
        predictor.handle_event(event)
        bus.ack(event.event_id)


def run_watcher(poll_interval: int = 10, bus: EventBus | None = None) -> None:
    bus = bus or event_bus_from_env()
    predictor = RollingPredictor()
    print(f"Event watcher started with {bus.__class__.__name__}.")
    if isinstance(bus, FileEventBus):
        print(f"Polling {TRIGGER_DIR} every {poll_interval}s.")
        print("Drop a .json file in data/triggers/ to simulate an event.\n")
    try:
        while True:
            for event in bus.poll():
                predictor.handle_event(event)
                bus.ack(event.event_id)
            if isinstance(bus, FileEventBus):
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        run_once()
    else:
        run_watcher()

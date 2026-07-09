"""Agent 1: Data Perception — automated data crawling, cleaning, and event triggering.

Responsible for:
  - FRED macro series download (FRED API)
  - IMF WEO / World Bank WDI bulk downloads
  - Registry-based policy source monitoring
  - Policy document inventory building (BIS, Fed, PBOC, ECB, BoE, BoR)
  - Event-driven trigger pipeline

Entry points:
  from agents.data_perception import download_macro, trigger_event, build_inventory

Implementation:
  Core logic in data_engineering/download_fred_macro.py, download_bulk.py,
  build_policy_inventory.py, build_weo_panel.py, build_wdi_top100.py
  Source monitoring in data_engineering/source_monitor.py
  Event pipeline in models/event_pipeline.py
"""
from data_engineering.download_fred_macro import run as download_macro
from data_engineering.download_bulk import download_bulk
from data_engineering.build_policy_inventory import main as build_inventory
from data_engineering.source_monitor import (
    NormalizedPolicyDocument,
    PolicySourceSpec,
    crawl_policy_source,
    incremental_fetch,
    load_policy_source_specs,
)
from models.event_pipeline import (
    EventBus,
    FileEventBus,
    ForecastEvent,
    InMemoryEventBus,
    KafkaEventBus,
    RedisEventBus,
    RollingPredictor,
    build_event_bus,
    event_bus_from_env,
    run_once as trigger_event,
)
from typing import Any

from agents.runtime_support import append_audit


class DataPerceptionAgent:
    """Agent boundary for source monitoring, document normalization, and triggers."""

    name = "data_perception"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        documents = list(state.get("documents", []))
        specs = [_coerce_source_spec(item) for item in state.get("source_specs", [])]
        runtime_inputs = state.get("runtime", {}).get("extra_inputs", {})
        http_get = runtime_inputs.get("http_get")
        store_path = runtime_inputs.get("policy_store_path")
        dry_run = bool(runtime_inputs.get("dry_run", True))

        result: dict[str, Any] = {
            "status": "ready",
            "source_count": len(specs),
            "existing_document_count": len(documents),
        }
        if specs:
            fetched = incremental_fetch(
                specs,
                store_path=store_path,
                http_get=http_get if callable(http_get) else None,
                dry_run=dry_run,
            )
            documents.extend(fetched.get("documents", []))
            result.update(
                {
                    "status": "fetched",
                    "new_documents": fetched.get("new_documents", 0),
                    "duplicate_documents": fetched.get("duplicate_documents", 0),
                    "store_path": fetched.get("store_path"),
                    "dry_run": fetched.get("dry_run"),
                }
            )

        if not state.get("event"):
            event = ForecastEvent.create(
                "langchain_runtime_tick",
                {"document_count": len(documents)},
                source_ids=sorted({str(doc.get("source_id", "unknown")) for doc in documents}),
            )
            state["event"] = event.to_dict()

        state["documents"] = documents
        state[self.name] = result
        return append_audit(state, self.name, result)


def _coerce_source_spec(value: Any) -> PolicySourceSpec:
    if isinstance(value, PolicySourceSpec):
        return value
    if isinstance(value, dict):
        return PolicySourceSpec(**value)
    raise TypeError(f"Unsupported source spec type: {type(value).__name__}")

__all__ = [
    "DataPerceptionAgent",
    "download_macro",
    "download_bulk", 
    "build_inventory",
    "PolicySourceSpec",
    "NormalizedPolicyDocument",
    "crawl_policy_source",
    "incremental_fetch",
    "load_policy_source_specs",
    "EventBus",
    "FileEventBus",
    "InMemoryEventBus",
    "RedisEventBus",
    "KafkaEventBus",
    "build_event_bus",
    "event_bus_from_env",
    "ForecastEvent",
    "RollingPredictor",
    "trigger_event",
]

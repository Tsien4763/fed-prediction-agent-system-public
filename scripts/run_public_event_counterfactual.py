from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.event_pipeline import InMemoryEventBus, RollingPredictor


PUBLIC_EVAL_DIR = REPO_ROOT / "examples" / "public_eval"
MACRO_FIXTURE = PUBLIC_EVAL_DIR / "fred_macro_quarterly_2000_2026.csv"
LABEL_FIXTURE = PUBLIC_EVAL_DIR / "fomc_quarter_labels_2000_2026.csv"
OUTPUT_PATH = PUBLIC_EVAL_DIR / "event_counterfactual_result.json"


def main() -> int:
    build_macro_panel()
    build_public_rag_index()

    with tempfile.TemporaryDirectory() as tmp:
        bus = InMemoryEventBus()
        predictor = RollingPredictor(archive_dir=Path(tmp) / "predictions")
        event = bus.publish(
            "p5_game_counterfactual",
            {
                "scenario_name": "event_level_energy_and_geopolitical_stress",
                "overrides": {
                    "energy_risk": 0.82,
                    "geopolitical_escalation": 0.74,
                    "dollar_liquidity_pressure": 0.62,
                },
                "max_context_docs": 2,
                "max_rounds": 1,
                "stable_rounds_required": 1,
            },
            source_ids=["public_event_wire", "public_fomc_context"],
            as_of="2026-05-06T13:45:00Z",
        )
        polled = bus.poll(max_events=1)
        if len(polled) != 1:
            raise RuntimeError("InMemoryEventBus did not return the published event.")
        record = predictor.handle_event(polled[0])
        bus.ack(event.event_id)

    result = build_public_result(record)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": relative_path(OUTPUT_PATH)}, indent=2))
    return 0


def build_macro_panel() -> None:
    macro = pd.read_csv(MACRO_FIXTURE)
    idx = pd.PeriodIndex(macro["quarter"], freq="Q").to_timestamp(how="end").normalize()
    panel = macro.set_index(idx).drop(columns=["quarter"])
    panel["gdp_growth_qoq_ann"] = 2.0
    out = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)


def build_public_rag_index() -> None:
    labels = pd.read_csv(LABEL_FIXTURE)
    macro = pd.read_csv(MACRO_FIXTURE).set_index("quarter")
    out = REPO_ROOT / "data" / "index" / "policy_context_index.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in labels.to_dict(orient="records"):
            quarter = str(item["quarter"])
            macro_row = macro.loc[quarter].to_dict() if quarter in macro.index else {}
            row = {
                "doc_id": f"public_fomc_{quarter}",
                "date": quarter_end_date(quarter),
                "country": "USA",
                "actor": "FOMC",
                "strategy_key": "fed_policy_direction",
                "title": f"Public FOMC label and macro context {quarter}",
                "url": "public://examples/public_eval/fomc_quarter_labels_2000_2026.csv",
                "text": (
                    f"Public FOMC label for {quarter}: {item.get('actual_class', '')}. "
                    f"Meetings: {item.get('meeting_dates', '')}. "
                    f"Macro snapshot: fedfunds={macro_row.get('fedfunds')}, "
                    f"inflation_cpi_yoy={macro_row.get('inflation_cpi_yoy')}, "
                    f"unemployment={macro_row.get('unemployment')}, gs10={macro_row.get('gs10')}."
                ),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_public_result(record: dict[str, Any]) -> dict[str, Any]:
    strategic = record.get("strategic_counterfactual", {})
    risk = record.get("risk_attribution", {})
    p5_rows = list(strategic.get("p5_impact_summary", []))
    top_impacts = [
        {
            "cluster_id": row.get("cluster_id"),
            "impact_label": row.get("impact_label"),
            "strategy_shift_l1": row.get("strategy_shift_l1"),
            "hawkish_pressure_delta": row.get("hawkish_pressure_delta"),
            "external_pressure_delta": row.get("external_pressure_delta"),
            "top_strategy_shift": row.get("top_strategy_shift"),
        }
        for row in p5_rows[:5]
    ]
    return {
        "artifact_type": "public_event_counterfactual_result",
        "status": strategic.get("status", "missing_counterfactual"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event": {
            "event_type": record.get("event", {}).get("event_type"),
            "as_of": record.get("event", {}).get("as_of"),
            "source_ids": record.get("input_documents", []),
        },
        "event_frequency_claim": (
            "The forecasting labels remain quarterly. The event runtime is event-driven and "
            "accepts daily or intraday timestamps, then maps each shock to the active quarter."
        ),
        "counterfactual": {
            "quarter": strategic.get("quarter"),
            "scenario_name": strategic.get("scenario_name"),
            "overrides": strategic.get("overrides", {}),
            "fed_probability_delta": strategic.get("delta", {}),
            "top_p5_impacts": top_impacts,
            "top_strategy_shifts": strategic.get("top_strategy_shifts", []),
            "top_belief_shifts": strategic.get("top_belief_shifts", []),
            "evidence_delta": strategic.get("evidence_delta", {}),
        },
        "risk_attribution": {
            "method": risk.get("method"),
            "analysis_scope": risk.get("analysis_scope"),
            "top_counterfactual_impacts": (risk.get("strategic_counterfactual", {}) or {}).get(
                "top_p5_impacts", []
            ),
        },
        "archive_record": {
            "run_id": record.get("run_id"),
            "has_prediction": isinstance(record.get("prediction"), dict),
            "has_evidence_chain": isinstance(record.get("evidence_chain"), list),
            "has_strategic_counterfactual": isinstance(strategic, dict) and bool(strategic),
        },
        "boundaries": [
            "This artifact validates event-triggered five-cluster counterfactual routing.",
            "It is a bounded one-round public rerun using public fixtures and rule fallback when no DeepSeek key is present.",
            "It does not claim intraday Fed label evaluation because the public FOMC labels are quarterly.",
        ],
    }


def quarter_end_date(quarter: str) -> str:
    year = int(quarter[:4])
    q = int(quarter[-1])
    month = q * 3
    day = 31 if month in {3, 12} else 30
    return f"{year}-{month:02d}-{day:02d}"


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())

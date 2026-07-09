"""Identification diagnostics for policy-event forecast narratives."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_engineering.config import REPO_ROOT, ensure_parent
from models.fomc_labels import build_fomc_label_df


@dataclass(frozen=True)
class EventStudyResult:
    event_count: int
    window: list[int]
    average_decision_by_relative_meeting: dict[str, float]
    pre_trend_mean_abs_decision: float
    post_event_mean_abs_decision: float
    lead_lag_placebo_gap: float
    diagnostic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_event_study(window: range = range(-4, 5)) -> EventStudyResult:
    labels = build_fomc_label_df().sort_values("date").reset_index(drop=True)
    event_indices = labels.index[labels["decision"] != 0].tolist()
    buckets: dict[int, list[float]] = {offset: [] for offset in window}

    for idx in event_indices:
        for offset in window:
            target = idx + offset
            if 0 <= target < len(labels):
                buckets[offset].append(float(labels.loc[target, "decision"]))

    averages = {
        str(offset): round(float(np.mean(values)), 6) if values else 0.0
        for offset, values in buckets.items()
    }
    pre_values = [abs(value) for offset, values in buckets.items() if offset < 0 for value in values]
    post_values = [abs(value) for offset, values in buckets.items() if offset >= 0 for value in values]
    pre = float(np.mean(pre_values)) if pre_values else 0.0
    post = float(np.mean(post_values)) if post_values else 0.0
    return EventStudyResult(
        event_count=len(event_indices),
        window=list(window),
        average_decision_by_relative_meeting=averages,
        pre_trend_mean_abs_decision=round(pre, 6),
        post_event_mean_abs_decision=round(post, 6),
        lead_lag_placebo_gap=round(post - pre, 6),
        diagnostic=(
            "Event-study scaffold over realized FOMC direction labels. "
            "Useful for stress-testing policy-event narratives and timing assumptions."
        ),
    )


def run_diagnostics(output_path: str | Path | None = None) -> dict[str, Any]:
    event_study = run_event_study()
    report = {
        "report_type": "identification_diagnostics",
        "analysis_scope": (
            "Forecast diagnostics with event-study, lead-lag placebo, and pre-trend checks "
            "for policy-event design review."
        ),
        "event_study": event_study.to_dict(),
        "placebo": {
            "lead_lag_placebo_gap": event_study.lead_lag_placebo_gap,
            "interpretation": "Large pre-event movement flags timing assumptions for review.",
        },
        "pre_trend_check": {
            "pre_trend_mean_abs_decision": event_study.pre_trend_mean_abs_decision,
            "interpretation": "Pre-trend summary for event-window design review.",
        },
        "next_identification_candidates": [
            "FOMC voting-member rotation as an instrument candidate",
            "scheduled meeting timing for event-study windows",
            "cross-country policy divergence for comparative designs",
        ],
        "status": "diagnostic_scaffold",
    }
    if output_path is not None:
        path = ensure_parent(output_path)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    path = REPO_ROOT / "data" / "processed" / "identification_diagnostics.json"
    report = run_diagnostics(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

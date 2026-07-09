from __future__ import annotations

import json
from pathlib import Path

from fed_game.counterfactual import (
    CounterfactualResult,
    build_counterfactual_briefing,
    parse_override_assignments,
    summarize_p5_counterfactual_impact,
)
from fed_game.evaluation import evaluate_forecasting_traces, evaluate_traces
from fed_game.schemas import GameTrace
from fed_game.training.rewards import score_completion_components
from fed_game.training.splits import split_jsonl_temporally


def test_temporal_split_contract_and_leakage_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": json.dumps({"quarter": "2019Q4"})}], "metadata": {}},
        {"messages": [{"role": "user", "content": json.dumps({"quarter": "2020Q1"})}], "metadata": {}},
        {"messages": [{"role": "user", "content": json.dumps({"quarter": "2024Q2"})}], "metadata": {}},
        {"messages": [{"role": "user", "content": "missing"}], "metadata": {}},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = split_jsonl_temporally(source, tmp_path / "splits")

    assert summary["total_rows"] == 4
    assert summary["splits"]["train"]["rows"] == 1
    assert summary["splits"]["val"]["rows"] == 1
    assert summary["splits"]["test"]["rows"] == 1
    assert summary["unassigned"]["rows"] == 1


def test_reward_components_score_schema_evidence_leakage_and_hold_collapse() -> None:
    completion = {
        "equilibrium_strategy": {
            "hawkish_signal_prob": 0.7,
            "rate_hike_25bp_prob": 0.4,
            "hold_with_hawkish_statement_prob": 0.5,
            "remove_forward_guidance_prob": 0.6,
            "easing_signal_prob": 0.1,
            "liquidity_support_prob": 0.2,
            "trade_or_sanction_pressure_prob": 0.2,
        },
        "fed_prediction": {"hike_25bp": 0.55, "hold": 0.35, "cut_25bp": 0.10},
        "converged": True,
        "rounds": 10,
        "evidence_chain": [
            {
                "source_id": "fed_fomc",
                "url": "https://www.federalreserve.gov/example",
                "published_at": "2024-06-12",
                "claim": "Inflation remains elevated.",
            }
        ],
    }

    scores = score_completion_components(
        json.dumps(completion),
        quarter="2024Q2",
        role_id="usa_warsh",
        current_round=10,
        quarter_end="2024-06-30",
    )

    assert scores["schema_valid"] == 1.0
    assert scores["no_future_leakage"] == 1.0
    assert scores["evidence_traceable"] > 0
    assert scores["not_always_hold"] > 0
    assert "weighted_total" in scores

    leaky = dict(completion)
    leaky["evidence_chain"] = [{"published_at": "2024-12-18", "claim": "future meeting"}]
    assert (
        score_completion_components(json.dumps(leaky), quarter="2024Q2", quarter_end="2024-06-30")[
            "no_future_leakage"
        ]
        < 1.0
    )


def test_forecast_metrics_include_anti_collapse_diagnostics(tmp_path: Path) -> None:
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
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = evaluate_forecasting_traces(path)
    metrics = report["forecast_metrics"]

    for key in ["accuracy", "brier_score", "log_loss", "balanced_accuracy", "macro_f1", "hold_collapse_index"]:
        assert key in metrics
    assert metrics["non_hold_recall"] is not None
    assert "equilibrium_checks" in report


def test_forecast_leakage_check_ignores_persona_audit_cutoff_not_evidence(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        {
            "quarter": "2024Q1",
            "fed_prediction": {"hike_25bp": 0.1, "hold": 0.8, "cut_25bp": 0.1},
            "personas_used": {"usa_warsh": {"research_cutoff": "2026-07-08"}},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = evaluate_forecasting_traces(path)

    assert report["future_leakage"]["future_leakage_rate"] == 0.0
    assert report["future_leakage"]["future_date_mentions"] == 0


def test_forecast_leakage_check_still_flags_future_evidence_dates(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        {
            "quarter": "2024Q1",
            "fed_prediction": {"hike_25bp": 0.1, "hold": 0.8, "cut_25bp": 0.1},
            "evidence_chain": [{"published_at": "2024-12-18", "claim": "future meeting"}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = evaluate_forecasting_traces(path)

    assert report["future_leakage"]["future_leakage_rate"] == 1.0
    assert report["future_leakage"]["examples"][0]["source_path"] == "$.evidence_chain[0].published_at"


def test_trace_evaluators_handle_missing_public_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "missing_traces.jsonl"

    trace_report = evaluate_traces(missing)
    forecast_report = evaluate_forecasting_traces(missing)

    assert trace_report["missing_trace"] is True
    assert trace_report["trace_count"] == 0
    assert forecast_report["missing_trace"] is True
    assert forecast_report["trace_count"] == 0
    assert forecast_report["forecast_metrics"]["accuracy"] == 0.0


def test_counterfactual_report_marks_boundary_and_policy_deltas() -> None:
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
    assert "Five-Cluster Impact" in markdown
    assert "Scenario Scope" in markdown
    assert "structured simulation contrast" in markdown


def test_p5_counterfactual_summary_tracks_cluster_strategy_pressure() -> None:
    base = _trace_with_clusters(
        {
            "USA": {"hawkish_signal_prob": 0.5, "rate_hike_25bp_prob": 0.2, "trade_or_sanction_pressure_prob": 0.1},
            "RUS": {"hawkish_signal_prob": 0.4, "rate_hike_25bp_prob": 0.2, "trade_or_sanction_pressure_prob": 0.2},
        }
    )
    shocked = _trace_with_clusters(
        {
            "USA": {"hawkish_signal_prob": 0.6, "rate_hike_25bp_prob": 0.3, "trade_or_sanction_pressure_prob": 0.1},
            "RUS": {"hawkish_signal_prob": 0.4, "rate_hike_25bp_prob": 0.2, "trade_or_sanction_pressure_prob": 0.7},
        }
    )

    summary = summarize_p5_counterfactual_impact(
        base,
        shocked,
        {"hike_25bp": 0.1, "hold": -0.1, "cut_25bp": 0.0},
    )

    assert summary[0]["cluster_id"] == "RUS"
    assert summary[0]["impact_label"] == "external-pressure shock"
    assert summary[0]["external_pressure_delta"] == 0.5
    assert summary[0]["fed_decision_delta"]["hike_25bp"] == 0.1


def _trace_with_clusters(strategies: dict[str, dict[str, float]]) -> GameTrace:
    return GameTrace(
        quarter="2024Q2",
        converged=True,
        rounds=1,
        max_strategy_distance=0.0,
        max_deviation_gain=0.0,
        cluster_strategies=[
            {"cluster_id": cluster_id, "round_id": 1, "strategy": strategy, "members": [], "internal_dispersion": 0.0}
            for cluster_id, strategy in strategies.items()
        ],
        shock_variables=[],
        proposals=[],
        critiques=[],
        fed_prediction={"hike_25bp": 0.2, "hold": 0.7, "cut_25bp": 0.1},
        evidence_chain=[],
    )

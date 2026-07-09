from __future__ import annotations

import json
from pathlib import Path


PRIVATE_PATH_MARKER = "C:" + "\\Users"
PRIVATE_USER_MARKER = "e139" + "7077"


def test_public_forecasting_comparison_artifact_reports_agent_and_statistical_baselines() -> None:
    path = Path("examples/public_eval/forecasting_comparison.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_text = path.read_text(encoding="utf-8")

    assert report["artifact_type"] == "public_forecasting_comparison"
    assert report["validation_window"] == "2020Q1-2023Q4"
    assert report["validation_quarters"] == 10
    assert set(report["models"]) >= {
        "train_prior_majority",
        "previous_quarter_persistence",
        "naive_macro_logistic",
        "naive_macro_ordered_probit",
        "rolling_var_rate_direction",
        "rule_self_play_agent",
    }

    agent_metrics = report["models"]["rule_self_play_agent"]["metrics"]
    probit_metrics = report["models"]["naive_macro_ordered_probit"]["metrics"]
    var_metrics = report["models"]["rolling_var_rate_direction"]["metrics"]
    assert agent_metrics["accuracy"] >= 0.7
    assert agent_metrics["predicted_class_counts"] != {"hold": 10}
    assert agent_metrics["predicted_class_counts"]
    assert probit_metrics["accuracy"] >= 0.7
    assert var_metrics["accuracy"] >= agent_metrics["accuracy"]
    assert PRIVATE_PATH_MARKER not in raw_text
    assert PRIVATE_USER_MARKER not in raw_text


def test_public_rerun_manifest_is_machine_readable_and_path_sanitized() -> None:
    path = Path("examples/public_eval/public_rerun_manifest.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_text = path.read_text(encoding="utf-8")

    assert report["artifact_type"] == "public_eval_rerun_manifest"
    assert report["status"] == "completed"
    assert report["forecast_eval_summary"]["evaluated_quarters"] == 10
    assert report["forecast_eval_summary"]["future_leakage_rate"] == 0.0
    assert "rule_self_play_agent" in report["comparison_metrics"]
    assert PRIVATE_PATH_MARKER not in raw_text
    assert PRIVATE_USER_MARKER not in raw_text


def test_public_training_debug_manifest_reports_split_and_reward_contract() -> None:
    path = Path("examples/public_eval/public_training_debug_manifest.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_text = path.read_text(encoding="utf-8")

    assert report["artifact_type"] == "public_training_debug_manifest"
    assert report["status"] == "completed"
    assert report["counts"]["trace_rows"] == 106
    assert report["counts"]["compact_train_rows"] == 880
    assert report["counts"]["compact_val_rows"] == 176
    assert report["counts"]["compact_test_rows"] == 110
    assert report["temporal_split"]["leakage_check"]["passed"] is True
    assert report["reward_dry_run"]["rows_scored"] == 176
    assert report["reward_dry_run"]["average_components"]["schema_valid"] == 1.0
    assert report["reward_dry_run"]["average_components"]["no_future_leakage"] == 1.0
    assert report["adapter_training"]["status"] == "smoke_completed"
    assert report["adapter_training"]["sft"]["command_status"] == "completed"
    assert report["adapter_training"]["grpo"]["command_status"] == "completed"
    assert report["adapter_training"]["sft"]["max_steps"] == 80
    assert report["adapter_training"]["grpo"]["max_steps"] == 12
    assert report["adapter_training"]["torch"]["cuda_available"] is True
    assert "RTX 3070" in report["adapter_training"]["torch"]["cuda_device"]
    assert "adapter_model.safetensors" in report["adapter_training"]["sft"]["key_files"]
    assert "adapter_model.safetensors" in report["adapter_training"]["grpo"]["key_files"]
    assert report["adapter_inference"]["status"] == "completed"
    assert report["adapter_inference"]["samples"] == 16
    assert report["adapter_inference"]["valid_json_rate"] == 1.0
    assert report["adapter_inference"]["predicted_class_counts"] == {"hold": 16}
    assert report["adapter_inference"]["forecast_row_accuracy"] == 0.3125
    assert report["adapter_inference"]["forecast_future_leakage_rate"] == 0.0
    assert PRIVATE_PATH_MARKER not in raw_text
    assert PRIVATE_USER_MARKER not in raw_text


def test_public_event_counterfactual_artifact_reports_p5_strategy_delta() -> None:
    path = Path("examples/public_eval/event_counterfactual_result.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_text = path.read_text(encoding="utf-8")

    assert report["artifact_type"] == "public_event_counterfactual_result"
    assert report["status"] == "completed"
    assert report["event"]["event_type"] == "p5_game_counterfactual"
    assert report["event"]["as_of"] == "2026-05-06T13:45:00Z"
    assert report["counterfactual"]["quarter"] == "2026Q2"
    assert report["risk_attribution"]["analysis_scope"] == "rolling_prediction_attribution_with_p5_counterfactual"

    impacts = report["counterfactual"]["top_p5_impacts"]
    assert {item["cluster_id"] for item in impacts} >= {"USA", "CHN", "RUS", "GBR", "FRA"}
    assert impacts[0]["strategy_shift_l1"] > 0.3
    assert impacts[0]["top_strategy_shift"]["field"] == "trade_or_sanction_pressure_prob"
    assert abs(report["counterfactual"]["fed_probability_delta"]["hike_25bp"]) > 0
    assert "quarterly" in report["event_frequency_claim"].lower()
    assert PRIVATE_PATH_MARKER not in raw_text
    assert PRIVATE_USER_MARKER not in raw_text

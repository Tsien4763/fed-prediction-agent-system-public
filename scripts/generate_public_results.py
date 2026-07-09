from __future__ import annotations

import argparse
import csv
import io
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tsa.api import VAR

from fed_game.evaluation import (
    CLASS_ORDER,
    DECISION_TO_CLASS,
    _aggregate_quarter_decision,
    _forecast_metrics,
    _read_trace_rows,
    normalize_fed_prediction,
)
from models.fomc_labels import build_fomc_label_df


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS,UNRATE,CPIAUCSL,GS10"
DEFAULT_OUTPUT_DIR = Path("examples/public_eval")
MACRO_HISTORY_FILE = "fred_macro_quarterly_2000_2026.csv"
FORECASTING_COMPARISON_FILE = "forecasting_comparison.json"
CLASS_TO_INDEX = {klass: idx for idx, klass in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS = {idx: klass for klass, idx in CLASS_TO_INDEX.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate small public evaluation artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--refresh-fred-sample", action="store_true")
    parser.add_argument("--trace-path", default="artifacts/first_version/traces/rolling_self_play_traces.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = build_quarter_labels()
    write_csv(output_dir / "fomc_quarter_labels_2000_2026.csv", labels)

    macro_history = ensure_macro_history(output_dir / MACRO_HISTORY_FILE, refresh=args.refresh_fred_sample)
    baseline = build_baseline_result(labels)
    (output_dir / "baseline_results.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    comparison = build_forecasting_comparison(labels, macro_history, trace_path=Path(args.trace_path))
    (output_dir / FORECASTING_COMPARISON_FILE).write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    adapter_card = {
        "artifact_type": "local_adapter_result_card",
        "note": "Weights and generated training data are not committed; this card preserves the headline local GRPO validation result.",
        "adapter": "artifacts/first_version/adapters/grpo_from_sft_v2_lr1e4_steps12",
        "test_file": "compact_equilibrium_sft.test.jsonl",
        "test_rows_scored": 128,
        "valid_json_rate": 1.0,
        "weighted_reward": 0.483267,
        "fed_probability_calibrated": 0.459647,
        "not_always_hold": 0.776133,
        "predicted_class_counts": {"hike_25bp": 76, "hold": 52},
    }
    (output_dir / "local_adapter_result_card.json").write_text(json.dumps(adapter_card, indent=2), encoding="utf-8")

    if args.refresh_fred_sample:
        write_fred_sample(output_dir / "fred_macro_sample_2020_2024.csv")

    print(json.dumps({"output_dir": str(output_dir), "files": sorted(path.name for path in output_dir.iterdir())}, indent=2))


def build_quarter_labels() -> list[dict[str, Any]]:
    meetings = build_fomc_label_df()
    meetings["quarter_id"] = meetings["date"].dt.to_period("Q").astype(str)
    rows = []
    for quarter, group in meetings.groupby("quarter_id"):
        decisions = [int(item) for item in group["decision"].tolist()]
        actual_decision = _aggregate_quarter_decision(decisions)
        rows.append(
            {
                "quarter": quarter,
                "actual_decision": actual_decision,
                "actual_class": DECISION_TO_CLASS[actual_decision],
                "meetings": len(decisions),
                "meeting_dates": ";".join(str(item.date()) for item in group["date"].tolist()),
                "meeting_descriptions": ";".join(str(item) for item in group.get("desc", []).tolist()),
            }
        )
    return sorted(rows, key=lambda row: row["quarter"])


def build_baseline_result(labels: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in labels if "2000Q1" <= row["quarter"] <= "2019Q4"]
    val = [row for row in labels if "2020Q1" <= row["quarter"] <= "2023Q4"]
    class_counts = Counter(row["actual_class"] for row in train)
    total = sum(class_counts.values())
    train_prior = {klass: class_counts[klass] / total for klass in CLASS_ORDER}

    majority_rows = evaluate_rows(val, lambda _idx, _row: train_prior)

    def persistence_probs(idx: int, _row: dict[str, Any]) -> dict[str, float]:
        previous_class = train[-1]["actual_class"] if idx == 0 else val[idx - 1]["actual_class"]
        return {klass: (0.8 if klass == previous_class else 0.1) for klass in CLASS_ORDER}

    persistence_rows = evaluate_rows(val, persistence_probs)
    return {
        "artifact_type": "public_baseline_eval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_fixture": "examples/public_eval/fomc_quarter_labels_2000_2026.csv",
        "train_window": "2000Q1-2019Q4",
        "validation_window": "2020Q1-2023Q4",
        "train_quarters": len(train),
        "validation_quarters": len(val),
        "train_class_counts": dict(class_counts),
        "validation_class_counts": dict(Counter(row["actual_class"] for row in val)),
        "baselines": {
            "train_prior_majority": {
                "description": "Uses train-window class frequencies as the probability vector for every validation quarter.",
                "probabilities": train_prior,
                "metrics": _forecast_metrics(majority_rows),
            },
            "previous_quarter_persistence": {
                "description": "Assigns 0.80 probability to the previous quarter's realized class and 0.10 to each other class.",
                "metrics": _forecast_metrics(persistence_rows),
            },
        },
    }


def build_forecasting_comparison(
    labels: list[dict[str, Any]],
    macro_history: pd.DataFrame,
    *,
    trace_path: Path,
) -> dict[str, Any]:
    train = [row for row in labels if "2000Q1" <= row["quarter"] <= "2019Q4"]
    val = [row for row in labels if "2020Q1" <= row["quarter"] <= "2023Q4"]
    models: dict[str, dict[str, Any]] = {}

    train_counts = Counter(row["actual_class"] for row in train)
    total = sum(train_counts.values())
    train_prior = {klass: train_counts[klass] / total for klass in CLASS_ORDER}
    models["train_prior_majority"] = {
        "description": "Train-window class-frequency probability vector applied to every validation quarter.",
        "rows": evaluate_rows(val, lambda _idx, _row: train_prior),
    }

    def persistence_probs(idx: int, _row: dict[str, Any]) -> dict[str, float]:
        previous_class = train[-1]["actual_class"] if idx == 0 else val[idx - 1]["actual_class"]
        return {klass: (0.8 if klass == previous_class else 0.1) for klass in CLASS_ORDER}

    models["previous_quarter_persistence"] = {
        "description": "Assigns 0.80 probability to the previous observed quarter's class.",
        "rows": evaluate_rows(val, persistence_probs),
    }
    macro_logit = macro_logistic_rows(train, val, macro_history)
    models["naive_macro_logistic"] = {
        "description": (
            "Multinomial logistic baseline on lagged public FRED quarterly macro features "
            "(FEDFUNDS, CPI inflation, unemployment, GS10, changes, real policy rate, yield gap)."
        ),
        **macro_logit,
    }
    macro_probit = macro_ordered_probit_rows(train, val, macro_history)
    models["naive_macro_ordered_probit"] = {
        "description": (
            "Ordered probit baseline on lagged public FRED quarterly macro features, treating "
            "cut < hold < hike as an ordered policy direction."
        ),
        **macro_probit,
    }
    rolling_var = rolling_var_rows(val, macro_history, train_prior)
    models["rolling_var_rate_direction"] = {
        "description": (
            "Rolling VAR on public FRED macro variables through the prior quarter; one-step "
            "fed funds forecast delta is mapped to hike/hold/cut probabilities."
        ),
        **rolling_var,
    }
    agent_rows, agent_metadata = agent_trace_rows(val, trace_path)
    models["rule_self_play_agent"] = {
        "description": (
            "Rule-fallback multi-agent self-play trace on the validation window. Public rerun uses "
            "bounded rounds and no private DeepSeek key or LoRA weights."
        ),
        "rows": agent_rows,
        "metadata": agent_metadata,
    }

    summarized = {
        name: {
            key: value
            for key, value in report.items()
            if key not in {"rows"}
        }
        | {"metrics": _forecast_metrics(report.get("rows", []))}
        for name, report in models.items()
    }
    return {
        "artifact_type": "public_forecasting_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_fixture": "examples/public_eval/fomc_quarter_labels_2000_2026.csv",
        "macro_fixture": f"examples/public_eval/{MACRO_HISTORY_FILE}",
        "agent_trace_path": relative_path(trace_path),
        "train_window": "2000Q1-2019Q4",
        "validation_window": "2020Q1-2023Q4",
        "validation_quarters": len(val),
        "validation_class_counts": dict(Counter(row["actual_class"] for row in val)),
        "small_sample_note": (
            "The validation fixture has 10 labeled quarters. Treat accuracy deltas as diagnostics, "
            "not production performance claims."
        ),
        "models": summarized,
        "per_quarter": {
            name: report.get("rows", [])
            for name, report in models.items()
        },
    }


def evaluate_rows(labels: list[dict[str, Any]], probs_fn) -> list[dict[str, Any]]:
    evaluated = []
    for idx, row in enumerate(labels):
        probs = probs_fn(idx, row)
        predicted_class = max(CLASS_ORDER, key=lambda klass: probs[klass])
        evaluated.append(
            {
                "quarter": row["quarter"],
                "probabilities": probs,
                "predicted_class": predicted_class,
                "actual_class": row["actual_class"],
                "correct": predicted_class == row["actual_class"],
            }
        )
    return evaluated


def macro_logistic_rows(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    macro_history: pd.DataFrame,
) -> dict[str, Any]:
    feature_names = [
        "fedfunds",
        "unemployment",
        "inflation_cpi_yoy",
        "gs10",
        "fedfunds_qoq",
        "unemployment_qoq",
        "inflation_yoy_qoq",
        "gs10_qoq",
        "real_policy_rate",
        "yield_gap",
    ]
    train_rows = feature_rows(train, macro_history, feature_names)
    val_rows = feature_rows(val, macro_history, feature_names)
    if len(train_rows) < 12 or not val_rows:
        return {"rows": [], "metadata": {"status": "skipped", "reason": "insufficient macro feature rows"}}

    x_train = np.asarray([row["features"] for row in train_rows], dtype=float)
    y_train = np.asarray([CLASS_TO_INDEX[row["actual_class"]] for row in train_rows], dtype=int)
    x_val = np.asarray([row["features"] for row in val_rows], dtype=float)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    probas = model.predict_proba(x_val)
    classes = [int(item) for item in model.named_steps["logisticregression"].classes_]
    rows = []
    for source, proba in zip(val_rows, probas):
        probs = {klass: 0.0 for klass in CLASS_ORDER}
        for idx, cls_idx in enumerate(classes):
            probs[INDEX_TO_CLASS[cls_idx]] = float(proba[idx])
        probs = normalize_probability_vector(probs)
        rows.append(evaluated_row(source["label"], probs))
    return {
        "rows": rows,
        "metadata": {
            "status": "completed",
            "feature_names": feature_names,
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "model": "sklearn LogisticRegression(class_weight='balanced')",
        },
    }


def macro_ordered_probit_rows(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    macro_history: pd.DataFrame,
) -> dict[str, Any]:
    feature_names = [
        "fedfunds",
        "unemployment",
        "inflation_cpi_yoy",
        "fedfunds_qoq",
        "inflation_yoy_qoq",
        "real_policy_rate",
        "yield_gap",
    ]
    train_rows = feature_rows(train, macro_history, feature_names)
    val_rows = feature_rows(val, macro_history, feature_names)
    if len(train_rows) < 12 or not val_rows:
        return {"rows": [], "metadata": {"status": "skipped", "reason": "insufficient macro feature rows"}}
    try:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.asarray([row["features"] for row in train_rows], dtype=float))
        x_val = scaler.transform(np.asarray([row["features"] for row in val_rows], dtype=float))
        y_train = np.asarray([CLASS_TO_INDEX[row["actual_class"]] for row in train_rows], dtype=int)
        train_frame = pd.DataFrame(x_train, columns=feature_names)
        val_frame = pd.DataFrame(x_val, columns=feature_names)
        model = OrderedModel(y_train, train_frame, distr="probit")
        fit_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", HessianInversionWarning)
            fitted = model.fit(method="bfgs", disp=False, maxiter=300)
            fit_warnings = [
                {"category": item.category.__name__, "message": str(item.message)[:240]}
                for item in caught
            ]
        predicted = fitted.model.predict(fitted.params, exog=val_frame)
        rows = []
        for source, proba in zip(val_rows, np.asarray(predicted)):
            probs = {INDEX_TO_CLASS[idx]: float(proba[idx]) for idx in range(min(len(proba), len(CLASS_ORDER)))}
            rows.append(evaluated_row(source["label"], probs))
        return {
            "rows": rows,
            "metadata": {
                "status": "completed",
                "feature_names": feature_names,
                "train_rows": len(train_rows),
                "validation_rows": len(val_rows),
                "model": "statsmodels OrderedModel(distr='probit')",
                "converged": bool(fitted.mle_retvals.get("converged", False)),
                "fit_warnings": fit_warnings,
            },
        }
    except Exception as exc:
        return {
            "rows": [],
            "metadata": {
                "status": "failed",
                "reason": type(exc).__name__,
                "message": str(exc)[:300],
            },
        }


def rolling_var_rows(
    val: list[dict[str, Any]],
    macro_history: pd.DataFrame,
    fallback_probs: dict[str, float],
) -> dict[str, Any]:
    variables = ["fedfunds", "inflation_cpi_yoy", "unemployment", "gs10"]
    rows = []
    failures = []
    var_df = macro_history.copy()
    var_df["quarter_end"] = pd.PeriodIndex(var_df["quarter"], freq="Q").to_timestamp(how="end").normalize()
    var_df = var_df.set_index("quarter_end")[variables].dropna().asfreq("QE")
    for label in val:
        prior_quarter = previous_quarter(label["quarter"])
        prior_date = pd.Period(prior_quarter, freq="Q").to_timestamp(how="end").normalize()
        history = var_df[var_df.index <= prior_date]
        if len(history) < 16:
            rows.append(evaluated_row(label, fallback_probs))
            failures.append({"quarter": label["quarter"], "reason": "insufficient_history"})
            continue
        try:
            maxlags = min(4, max(1, len(history) // 12))
            fitted = VAR(history).fit(maxlags=maxlags, ic="aic")
            forecast = fitted.forecast(history.values[-fitted.k_ar :], steps=1)[0]
            fed_idx = variables.index("fedfunds")
            rate_delta = float(forecast[fed_idx] - history.iloc[-1]["fedfunds"])
            probs = rate_delta_to_probs(rate_delta)
        except Exception as exc:
            probs = fallback_probs
            failures.append({"quarter": label["quarter"], "reason": type(exc).__name__, "message": str(exc)[:200]})
        rows.append(evaluated_row(label, probs))
    return {
        "rows": rows,
        "metadata": {
            "status": "completed_with_fallbacks" if failures else "completed",
            "variables": variables,
            "fallback_count": len(failures),
            "fallback_examples": failures[:5],
        },
    }


def agent_trace_rows(labels: list[dict[str, Any]], trace_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not trace_path.exists():
        return [], {"status": "missing_trace", "trace_path": relative_path(trace_path)}
    trace_by_quarter = {str(row.get("quarter")): row for row in _read_trace_rows(trace_path)}
    rows = []
    missing = []
    for label in labels:
        trace = trace_by_quarter.get(label["quarter"])
        if trace is None:
            missing.append(label["quarter"])
            continue
        rows.append(evaluated_row(label, normalize_fed_prediction(trace.get("fed_prediction", {}))))
    return rows, {
        "status": "completed" if rows else "empty",
        "trace_path": relative_path(trace_path),
        "trace_rows": len(trace_by_quarter),
        "evaluated_rows": len(rows),
        "missing_quarters": missing,
        "mode": "rule-fallback bounded-round public rerun",
    }


def feature_rows(
    labels: list[dict[str, Any]],
    macro_history: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    macro_by_quarter = macro_history.set_index("quarter")
    rows = []
    for label in labels:
        prior = previous_quarter(label["quarter"])
        if prior not in macro_by_quarter.index:
            continue
        series = macro_by_quarter.loc[prior, feature_names]
        if series.isna().any():
            continue
        rows.append({"label": label, "actual_class": label["actual_class"], "features": [float(item) for item in series]})
    return rows


def evaluated_row(label: dict[str, Any], probs: dict[str, float]) -> dict[str, Any]:
    probs = normalize_probability_vector(probs)
    predicted_class = max(CLASS_ORDER, key=lambda klass: probs[klass])
    return {
        "quarter": label["quarter"],
        "probabilities": probs,
        "predicted_class": predicted_class,
        "actual_class": label["actual_class"],
        "correct": predicted_class == label["actual_class"],
    }


def normalize_probability_vector(probs: dict[str, float]) -> dict[str, float]:
    values = {klass: max(0.0, float(probs.get(klass, 0.0))) for klass in CLASS_ORDER}
    total = sum(values.values()) or 1.0
    return {klass: round(values[klass] / total, 6) for klass in CLASS_ORDER}


def rate_delta_to_probs(rate_delta: float) -> dict[str, float]:
    scaled = max(-2.0, min(2.0, rate_delta / 0.25))
    hike = 0.10 + max(0.0, scaled)
    cut = 0.10 + max(0.0, -scaled)
    hold = 0.20 + max(0.0, 1.0 - abs(scaled))
    return normalize_probability_vector({"hike_25bp": hike, "hold": hold, "cut_25bp": cut})


def previous_quarter(quarter: str) -> str:
    year = int(quarter[:4])
    q = int(quarter[-1])
    if q == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{q - 1}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ensure_macro_history(path: Path, *, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path)
    response = requests.get(FRED_GRAPH_URL, timeout=30)
    response.raise_for_status()
    monthly = pd.read_csv(io.StringIO(response.text))
    date_col = monthly.columns[0]
    monthly[date_col] = pd.to_datetime(monthly[date_col])
    monthly = monthly.rename(
        columns={
            date_col: "date",
            "UNRATE": "unemployment",
            "CPIAUCSL": "cpi",
        }
    )
    for column in ["FEDFUNDS", "unemployment", "cpi", "GS10"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    quarterly = (
        monthly.set_index("date")[["FEDFUNDS", "unemployment", "cpi", "GS10"]]
        .resample("QE")
        .agg({"FEDFUNDS": "mean", "unemployment": "mean", "cpi": "last", "GS10": "mean"})
    )
    quarterly = quarterly.rename(columns={"FEDFUNDS": "fedfunds", "GS10": "gs10"})
    quarterly["inflation_cpi_yoy"] = (quarterly["cpi"] / quarterly["cpi"].shift(4) - 1.0) * 100.0
    quarterly["fedfunds_qoq"] = quarterly["fedfunds"].diff()
    quarterly["unemployment_qoq"] = quarterly["unemployment"].diff()
    quarterly["inflation_yoy_qoq"] = quarterly["inflation_cpi_yoy"].diff()
    quarterly["gs10_qoq"] = quarterly["gs10"].diff()
    quarterly["real_policy_rate"] = quarterly["fedfunds"] - quarterly["inflation_cpi_yoy"]
    quarterly["yield_gap"] = quarterly["gs10"] - quarterly["fedfunds"]
    quarterly["quarter"] = quarterly.index.to_period("Q").astype(str)
    out = quarterly[
        [
            "quarter",
            "fedfunds",
            "unemployment",
            "inflation_cpi_yoy",
            "gs10",
            "fedfunds_qoq",
            "unemployment_qoq",
            "inflation_yoy_qoq",
            "gs10_qoq",
            "real_policy_rate",
            "yield_gap",
        ]
    ]
    out = out[out["quarter"].between("2000Q1", "2026Q4")]
    out = out.round(6)
    out.to_csv(path, index=False)
    return out


def write_fred_sample(path: Path) -> None:
    response = requests.get(FRED_GRAPH_URL, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    header, data = lines[0], lines[1:]
    sample = [line for line in data if "2020-01-01" <= line.split(",", 1)[0] <= "2024-12-01"]
    path.write_text("\n".join([header, *sample]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

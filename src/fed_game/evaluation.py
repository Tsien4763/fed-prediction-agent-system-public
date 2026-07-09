from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .config import repo_path


CLASS_ORDER = ["hike_25bp", "hold", "cut_25bp"]
DECISION_TO_CLASS = {1: "hike_25bp", 0: "hold", -1: "cut_25bp"}
CLASS_TO_DECISION = {value: key for key, value in DECISION_TO_CLASS.items()}
NON_EVIDENCE_DATE_PATH_SUFFIXES = (
    ".research_cutoff",
)


def brier_score(prob: float, outcome: int) -> float:
    return (float(prob) - int(outcome)) ** 2


def log_loss(prob: float, outcome: int, eps: float = 1e-6) -> float:
    prob = max(eps, min(1.0 - eps, float(prob)))
    return -(outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob))


def evaluate_traces(trace_path: str | Path) -> dict[str, Any]:
    resolved = repo_path(trace_path)
    if not resolved.exists():
        return {
            "trace_path": str(resolved),
            "trace_count": 0,
            "missing_trace": True,
            "message": "No self-play trace file found. Run `python -m fed_game.cli self-play ...` first.",
        }
    rows = []
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {"trace_count": 0}
    convergence_rate = sum(1 for row in rows if row.get("converged")) / len(rows)
    avg_rounds = sum(float(row.get("rounds", 0)) for row in rows) / len(rows)
    avg_hold = sum(float(row.get("fed_prediction", {}).get("hold", 0)) for row in rows) / len(rows)
    equilibrium = _equilibrium_check_metrics(rows)
    return {
        "trace_count": len(rows),
        "convergence_rate": round(convergence_rate, 4),
        "avg_rounds": round(avg_rounds, 4),
        "avg_hold_probability": round(avg_hold, 4),
        "equilibrium_checks": equilibrium,
        "notes": "First-version trace evaluation; plug in realized FOMC labels for Brier/log-loss backtest.",
    }


def evaluate_forecasting_traces(
    trace_path: str | Path,
    *,
    calibration_bins: int = 10,
    leakage_examples_limit: int = 25,
) -> dict[str, Any]:
    resolved = repo_path(trace_path)
    if not resolved.exists():
        return _empty_forecasting_report(resolved, calibration_bins=calibration_bins, missing_trace=True)

    rows = _read_trace_rows(trace_path)
    labels = _quarter_labels()
    evaluated: list[dict[str, Any]] = []
    missing_label_quarters: list[str] = []
    leakage = _empty_leakage_summary()
    convergence_rows: list[dict[str, Any]] = []

    for row in rows:
        quarter = str(row.get("quarter", ""))
        if not quarter:
            continue
        convergence_rows.append(row)
        leakage_row = _future_leakage_for_trace(row, examples_limit=leakage_examples_limit - len(leakage["examples"]))
        _merge_leakage_summary(leakage, leakage_row)
        label = labels.get(quarter)
        if label is None:
            missing_label_quarters.append(quarter)
            continue
        probs = normalize_fed_prediction(row.get("fed_prediction", {}))
        predicted_class = max(CLASS_ORDER, key=lambda key: probs[key])
        actual_class = label["actual_class"]
        evaluated.append(
            {
                "quarter": quarter,
                "probabilities": probs,
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "actual_decision": label["actual_decision"],
                "correct": predicted_class == actual_class,
                "meetings": label["meetings"],
                "meeting_dates": label["meeting_dates"],
                "meeting_descriptions": label["meeting_descriptions"],
                "converged": bool(row.get("converged")),
                "rounds": int(row.get("rounds", 0) or 0),
                "max_strategy_distance": float(row.get("max_strategy_distance", 0.0) or 0.0),
                "max_deviation_gain": float(row.get("max_deviation_gain", 0.0) or 0.0),
            }
        )

    forecast_metrics = _forecast_metrics(evaluated)
    convergence_metrics = _convergence_metrics(convergence_rows)
    calibration = _calibration_metrics(evaluated, bins=calibration_bins)
    split_metrics = _split_metrics(evaluated)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(repo_path(trace_path)),
        "trace_count": len(rows),
        "evaluated_quarters": len(evaluated),
        "missing_label_quarters": sorted(missing_label_quarters),
        "forecast_metrics": forecast_metrics,
        "split_metrics": split_metrics,
        "calibration": calibration,
        "future_leakage": leakage,
        "convergence": convergence_metrics,
        "equilibrium_checks": _equilibrium_check_metrics(convergence_rows),
        "per_quarter": evaluated,
        "label_policy": (
            "Quarter label is the net direction of FOMC decisions in that quarter; "
            "if net is zero, all-hold quarters map to hold and mixed quarters use the last non-hold decision."
        ),
    }
    return result


def normalize_fed_prediction(prediction: dict[str, Any]) -> dict[str, float]:
    values = {
        "hike_25bp": _as_probability(prediction.get("hike_25bp", prediction.get("hike", 0.0))),
        "hold": _as_probability(prediction.get("hold", 0.0)),
        "cut_25bp": _as_probability(prediction.get("cut_25bp", prediction.get("cut", 0.0))),
    }
    total = sum(values.values())
    if total <= 0:
        return {"hike_25bp": 1.0 / 3.0, "hold": 1.0 / 3.0, "cut_25bp": 1.0 / 3.0}
    return {key: value / total for key, value in values.items()}


def _read_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    resolved = repo_path(trace_path)
    rows = []
    with resolved.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _empty_forecasting_report(
    trace_path: Path,
    *,
    calibration_bins: int,
    missing_trace: bool,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(trace_path),
        "trace_count": 0,
        "evaluated_quarters": 0,
        "missing_trace": missing_trace,
        "message": "No self-play trace file found. Run `python -m fed_game.cli self-play ...` first.",
        "missing_label_quarters": [],
        "forecast_metrics": _forecast_metrics([]),
        "split_metrics": {},
        "calibration": _calibration_metrics([], bins=calibration_bins),
        "future_leakage": _empty_leakage_summary(),
        "convergence": _convergence_metrics([]),
        "equilibrium_checks": _equilibrium_check_metrics([]),
        "per_quarter": [],
        "label_policy": (
            "Quarter label is the net direction of FOMC decisions in that quarter; "
            "if net is zero, all-hold quarters map to hold and mixed quarters use the last non-hold decision."
        ),
    }


def _forecast_metrics(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluated:
        return {
            "accuracy": 0.0,
            "direction_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "non_hold_recall": None,
            "hold_collapse_index": 0.0,
            "brier_score": None,
            "log_loss": None,
            "confusion_matrix": {},
            "class_counts": {},
            "predicted_class_counts": {},
        }
    correct = sum(1 for row in evaluated if row["correct"])
    brier_total = 0.0
    log_total = 0.0
    per_class_brier = {key: 0.0 for key in CLASS_ORDER}
    confusion = {actual: {pred: 0 for pred in CLASS_ORDER} for actual in CLASS_ORDER}
    class_counts = Counter()
    predicted_counts = Counter()
    hold_prob_total = 0.0
    for row in evaluated:
        actual = row["actual_class"]
        predicted = row["predicted_class"]
        probs = row["probabilities"]
        hold_prob_total += float(probs.get("hold", 0.0))
        class_counts[actual] += 1
        predicted_counts[predicted] += 1
        confusion[actual][predicted] += 1
        for klass in CLASS_ORDER:
            outcome = 1 if klass == actual else 0
            value = brier_score(probs[klass], outcome)
            per_class_brier[klass] += value
            brier_total += value
        log_total += -math.log(max(1e-12, min(1.0, probs[actual])))
    n = len(evaluated)
    derived = _classification_diagnostics(confusion, class_counts, predicted_counts, hold_prob_total, n)
    return {
        "accuracy": round(correct / n, 6),
        "direction_accuracy": round(correct / n, 6),
        "balanced_accuracy": derived["balanced_accuracy"],
        "macro_f1": derived["macro_f1"],
        "non_hold_recall": derived["non_hold_recall"],
        "hold_collapse_index": derived["hold_collapse_index"],
        "brier_score": round(brier_total / n, 6),
        "brier_score_per_class": {key: round(value / n, 6) for key, value in per_class_brier.items()},
        "log_loss": round(log_total / n, 6),
        "confusion_matrix": confusion,
        "class_counts": dict(class_counts),
        "predicted_class_counts": dict(predicted_counts),
        "majority_class_baseline_accuracy": round(max(class_counts.values()) / n, 6),
    }


def _classification_diagnostics(
    confusion: dict[str, dict[str, int]],
    class_counts: Counter,
    predicted_counts: Counter,
    hold_prob_total: float,
    n: int,
) -> dict[str, Any]:
    recalls = []
    f1_scores = []
    for klass in CLASS_ORDER:
        support = int(class_counts.get(klass, 0))
        predicted = int(predicted_counts.get(klass, 0))
        tp = int(confusion.get(klass, {}).get(klass, 0))
        fp = predicted - tp
        fn = support - tp
        if support > 0:
            recalls.append(tp / support)
        denom = 2 * tp + fp + fn
        if support > 0 or predicted > 0:
            f1_scores.append((2 * tp / denom) if denom else 0.0)

    non_hold_support = sum(int(class_counts.get(klass, 0)) for klass in CLASS_ORDER if klass != "hold")
    non_hold_correct = sum(int(confusion.get(klass, {}).get(klass, 0)) for klass in CLASS_ORDER if klass != "hold")
    hold_argmax_share = int(predicted_counts.get("hold", 0)) / max(1, n)
    avg_hold_prob = hold_prob_total / max(1, n)
    return {
        "balanced_accuracy": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 6) if f1_scores else 0.0,
        "non_hold_recall": round(non_hold_correct / non_hold_support, 6) if non_hold_support else None,
        "hold_collapse_index": round(hold_argmax_share * avg_hold_prob, 6),
    }


def _split_metrics(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        split = _split_for_quarter(row["quarter"])
        by_split[split].append(row)
    return {split: _forecast_metrics(rows) for split, rows in sorted(by_split.items())}


def _calibration_metrics(evaluated: list[dict[str, Any]], *, bins: int) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("calibration_bins must be positive")
    one_vs_rest = {klass: _class_calibration(evaluated, klass, bins=bins) for klass in CLASS_ORDER}
    top_label = _top_label_calibration(evaluated, bins=bins)
    return {
        "bins": bins,
        "one_vs_rest": one_vs_rest,
        "top_label": top_label,
        "macro_ece": round(sum(item["ece"] for item in one_vs_rest.values()) / len(CLASS_ORDER), 6)
        if evaluated
        else 0.0,
    }


def _class_calibration(evaluated: list[dict[str, Any]], klass: str, *, bins: int) -> dict[str, Any]:
    bucket_rows = [[] for _ in range(bins)]
    for row in evaluated:
        prob = row["probabilities"][klass]
        idx = min(bins - 1, int(prob * bins))
        bucket_rows[idx].append((prob, 1 if row["actual_class"] == klass else 0))
    return _summarize_calibration_buckets(bucket_rows)


def _top_label_calibration(evaluated: list[dict[str, Any]], *, bins: int) -> dict[str, Any]:
    bucket_rows = [[] for _ in range(bins)]
    for row in evaluated:
        prob = row["probabilities"][row["predicted_class"]]
        idx = min(bins - 1, int(prob * bins))
        bucket_rows[idx].append((prob, 1 if row["correct"] else 0))
    return _summarize_calibration_buckets(bucket_rows)


def _summarize_calibration_buckets(bucket_rows: list[list[tuple[float, int]]]) -> dict[str, Any]:
    total = sum(len(rows) for rows in bucket_rows)
    buckets = []
    ece = 0.0
    for idx, rows in enumerate(bucket_rows):
        lower = idx / len(bucket_rows)
        upper = (idx + 1) / len(bucket_rows)
        if rows:
            mean_pred = sum(prob for prob, _ in rows) / len(rows)
            empirical = sum(outcome for _, outcome in rows) / len(rows)
            gap = abs(mean_pred - empirical)
            ece += gap * len(rows) / max(1, total)
            buckets.append(
                {
                    "bin": idx,
                    "lower": round(lower, 4),
                    "upper": round(upper, 4),
                    "count": len(rows),
                    "mean_predicted_probability": round(mean_pred, 6),
                    "empirical_rate": round(empirical, 6),
                    "abs_gap": round(gap, 6),
                }
            )
        else:
            buckets.append({"bin": idx, "lower": round(lower, 4), "upper": round(upper, 4), "count": 0})
    return {"ece": round(ece, 6), "buckets": buckets}


def _convergence_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trace_count": 0,
            "convergence_rate": 0.0,
            "avg_rounds": 0.0,
            "median_rounds": 0.0,
            "avg_max_strategy_distance": 0.0,
            "avg_max_deviation_gain": 0.0,
        }
    rounds = sorted(int(row.get("rounds", 0) or 0) for row in rows)
    n = len(rows)
    median = rounds[n // 2] if n % 2 else (rounds[n // 2 - 1] + rounds[n // 2]) / 2
    return {
        "trace_count": n,
        "convergence_rate": round(sum(1 for row in rows if row.get("converged")) / n, 6),
        "avg_rounds": round(sum(float(row.get("rounds", 0) or 0) for row in rows) / n, 6),
        "median_rounds": round(median, 6),
        "avg_max_strategy_distance": round(
            sum(float(row.get("max_strategy_distance", 0.0) or 0.0) for row in rows) / n, 8
        ),
        "avg_max_deviation_gain": round(
            sum(float(row.get("max_deviation_gain", 0.0) or 0.0) for row in rows) / n, 8
        ),
        "round_histogram": dict(Counter(str(value) for value in rounds)),
    }


def _equilibrium_check_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trace_count": 0,
            "checked_rate": 0.0,
            "deepseek_checked_rate": 0.0,
            "accepted_rate": 0.0,
            "avg_max_profitable_deviation_gain": 0.0,
            "source_counts": {},
        }
    checks = [row.get("equilibrium_check") for row in rows if isinstance(row.get("equilibrium_check"), dict)]
    checked = [item for item in checks if item.get("checked")]
    deepseek_checked = [item for item in checked if item.get("source") == "deepseek_equilibrium_judge"]
    accepted = [item for item in checked if item.get("is_nash_equilibrium")]
    gains = [
        float(item.get("max_profitable_deviation_gain", 0.0) or 0.0)
        for item in checked
    ]
    return {
        "trace_count": len(rows),
        "checked_rate": round(len(checked) / len(rows), 6),
        "deepseek_checked_rate": round(len(deepseek_checked) / len(rows), 6),
        "accepted_rate": round(len(accepted) / len(rows), 6),
        "avg_max_profitable_deviation_gain": round(sum(gains) / max(1, len(gains)), 8),
        "source_counts": dict(Counter(str(item.get("source", "missing")) for item in checks)),
    }


def _quarter_labels() -> dict[str, dict[str, Any]]:
    try:
        from models.fomc_labels import build_fomc_label_df
    except Exception as exc:
        raise RuntimeError("Unable to load FOMC labels from models.fomc_labels") from exc
    df = build_fomc_label_df()
    df["quarter_id"] = df["date"].dt.to_period("Q").astype(str)
    labels: dict[str, dict[str, Any]] = {}
    for quarter, group in df.groupby("quarter_id"):
        decisions = [int(item) for item in group["decision"].tolist()]
        actual_decision = _aggregate_quarter_decision(decisions)
        labels[quarter] = {
            "actual_decision": actual_decision,
            "actual_class": DECISION_TO_CLASS[actual_decision],
            "meetings": len(decisions),
            "meeting_dates": [str(item.date()) for item in group["date"].tolist()],
            "meeting_descriptions": [str(item) for item in group.get("desc", []).tolist()],
        }
    return labels


def _aggregate_quarter_decision(decisions: list[int]) -> int:
    net = sum(decisions)
    if net > 0:
        return 1
    if net < 0:
        return -1
    non_hold = [item for item in decisions if item != 0]
    if not non_hold:
        return 0
    return non_hold[-1]


def _future_leakage_for_trace(row: dict[str, Any], *, examples_limit: int) -> dict[str, Any]:
    quarter = str(row.get("quarter", ""))
    cutoff = _quarter_end_date(quarter)
    summary = _empty_leakage_summary()
    summary["trace_count"] = 1
    trace_has_leak = False
    for source_path, value in _walk_json(row):
        if _is_non_evidence_date_path(source_path):
            continue
        parsed_dates = _dates_from_value(value)
        for raw, parsed in parsed_dates:
            summary["date_mentions_checked"] += 1
            if parsed > cutoff:
                trace_has_leak = True
                summary["future_date_mentions"] += 1
                if len(summary["examples"]) < examples_limit:
                    summary["examples"].append(
                        {
                            "quarter": quarter,
                            "quarter_end": cutoff.isoformat(),
                            "future_date": parsed.isoformat(),
                            "raw_value": raw[:240],
                            "source_path": source_path,
                        }
                    )
    if trace_has_leak:
        summary["traces_with_future_leakage"] = 1
    return summary


def _is_non_evidence_date_path(source_path: str) -> bool:
    """Ignore audit metadata dates that are not evidence available to the forecast."""
    return source_path.startswith("$.personas_used.") and source_path.endswith(NON_EVIDENCE_DATE_PATH_SUFFIXES)


def _empty_leakage_summary() -> dict[str, Any]:
    return {
        "trace_count": 0,
        "traces_with_future_leakage": 0,
        "date_mentions_checked": 0,
        "future_date_mentions": 0,
        "future_leakage_rate": 0.0,
        "examples": [],
    }


def _merge_leakage_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ["trace_count", "traces_with_future_leakage", "date_mentions_checked", "future_date_mentions"]:
        target[key] += source[key]
    target["examples"].extend(source["examples"])
    target["future_leakage_rate"] = round(
        target["traces_with_future_leakage"] / target["trace_count"], 6
    ) if target["trace_count"] else 0.0


def _walk_json(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk_json(item, f"{path}[{idx}]")
    else:
        yield path, value


def _dates_from_value(value: Any) -> list[tuple[str, date]]:
    if not isinstance(value, str):
        return []
    dates = []
    for match in re.finditer(r"\b(20\d{2}|19\d{2})[-/](\d{2})[-/](\d{2})\b", value):
        raw = match.group(0)
        try:
            dates.append((raw, date(int(match.group(1)), int(match.group(2)), int(match.group(3)))))
        except ValueError:
            continue
    return dates


def _quarter_end_date(quarter: str) -> date:
    year = int(quarter[:4])
    q = int(quarter[-1])
    if q == 1:
        return date(year, 3, 31)
    if q == 2:
        return date(year, 6, 30)
    if q == 3:
        return date(year, 9, 30)
    if q == 4:
        return date(year, 12, 31)
    raise ValueError(f"Invalid quarter: {quarter}")


def _split_for_quarter(quarter: str) -> str:
    year = int(quarter[:4])
    if 2000 <= year <= 2019:
        return "train"
    if 2020 <= year <= 2023:
        return "val"
    if 2024 <= year <= 2026:
        return "test"
    return "outside"


def _as_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return max(0.0, min(1.0, number))

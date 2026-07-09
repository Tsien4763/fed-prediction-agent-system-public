from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from fed_game.config import repo_path
from fed_game.evaluation import (
    CLASS_ORDER,
    normalize_fed_prediction,
    _calibration_metrics,
    _empty_leakage_summary,
    _forecast_metrics,
    _future_leakage_for_trace,
    _merge_leakage_summary,
    _quarter_labels,
)


REQUIRED_STRATEGY_KEYS = {
    "hawkish_signal_prob",
    "rate_hike_25bp_prob",
    "hold_with_hawkish_statement_prob",
    "remove_forward_guidance_prob",
    "easing_signal_prob",
    "liquidity_support_prob",
    "trade_or_sanction_pressure_prob",
}
REQUIRED_FED_KEYS = {"hike_25bp", "hold", "cut_25bp"}
ROLE_BIASES = {
    "hawk": {"hawkish_signal_prob": 0.55, "rate_hike_25bp_prob": 0.2, "easing_signal_prob": 0.25},
    "dove": {"hawkish_signal_prob": 0.75, "rate_hike_25bp_prob": 0.45},
    "warsh": {"hawkish_signal_prob": 0.75, "rate_hike_25bp_prob": 0.5, "easing_signal_prob": 0.35},
    "euro": {"trade_or_sanction_pressure_prob": 0.7},
    "energy": {"trade_or_sanction_pressure_prob": 0.7},
}

DEFAULT_REWARD_WEIGHTS = {
    "schema_valid": 1.0,
    "no_future_leakage": 1.0,
    "evidence_traceable": 0.8,
    "fed_probability_calibrated": 1.2,
    "lower_regret": 0.8,
    "fewer_convergence_rounds": 0.5,
    "stable_after_10_rounds": 0.7,
    "role_consistency": 0.8,
    "unsupported_claims": 0.7,
    "overconfident_wrong_prediction": 1.0,
    "not_always_hold": 1.0,
}


def parse_json_completion(text: Any) -> dict[str, Any] | None:
    if isinstance(text, list):
        if text and isinstance(text[-1], dict):
            text = text[-1].get("content", "")
        else:
            text = "".join(str(item) for item in text)
    if not isinstance(text, str):
        text = str(text)
    parsed = _extract_first_json_object(text)
    return parsed


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : idx + 1])
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def score_completion_components(
    completion: Any,
    *,
    target_strategy: dict[str, Any] | None = None,
    target_fed_prediction: dict[str, Any] | None = None,
    quarter: str | None = None,
    role_id: str | None = None,
    current_round: int | None = None,
    quarter_end: str | None = None,
) -> dict[str, float]:
    data = parse_json_completion(completion)
    if data is None:
        return _invalid_component_scores()

    strategy = _strategy_payload(data)
    target_strategy = _json_from_string(target_strategy)
    target_fed_prediction = _json_from_string(target_fed_prediction)
    fed_prediction = normalize_fed_prediction(_json_from_string(data.get("fed_prediction")) or target_fed_prediction or {})
    actual_class = _actual_class_for_quarter(quarter)
    scores = {
        "schema_valid": _schema_score(data),
        "no_future_leakage": _future_leakage_score(data, quarter=quarter, quarter_end=quarter_end),
        "evidence_traceable": _evidence_traceability_score(data),
        "fed_probability_calibrated": _fed_calibration_score(fed_prediction, actual_class),
        "lower_regret": _lower_regret_score(data, strategy, target_strategy),
        "fewer_convergence_rounds": _convergence_round_score(data, current_round=current_round),
        "stable_after_10_rounds": _stable_after_10_rounds_score(data),
        "role_consistency": _role_consistency_score(strategy, role_id=role_id),
        "unsupported_claims": _unsupported_claims_score(data),
        "overconfident_wrong_prediction": _overconfident_wrong_prediction_score(fed_prediction, actual_class),
        "not_always_hold": _not_always_hold_score(fed_prediction, actual_class),
    }
    scores["weighted_total"] = weighted_reward(scores)
    return scores


def weighted_reward(component_scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    active_weights = weights or DEFAULT_REWARD_WEIGHTS
    total_weight = sum(abs(value) for value in active_weights.values()) or 1.0
    total = sum(component_scores.get(key, 0.0) * weight for key, weight in active_weights.items())
    return round(total / total_weight, 6)


def grpo_reward_suite(completions: list[str], **kwargs: Any) -> list[float]:
    targets = _broadcast(kwargs.get("target_strategy"), completions, default={})
    target_fed = _broadcast(kwargs.get("target_fed_prediction"), completions, default={})
    quarters = _broadcast(kwargs.get("quarter"), completions, default=None)
    role_ids = _broadcast(kwargs.get("role_id"), completions, default=None)
    current_rounds = _broadcast(kwargs.get("current_round"), completions, default=None)
    component_rows = [
        score_completion_components(
            completion,
            target_strategy=target,
            target_fed_prediction=fed,
            quarter=quarter,
            role_id=role_id,
            current_round=current_round,
        )
        for completion, target, fed, quarter, role_id, current_round in zip(
            completions, targets, target_fed, quarters, role_ids, current_rounds
        )
    ]
    hold_collapse = _batch_hold_collapse_penalty(completions)
    rewards = []
    for scores in component_rows:
        adjusted = dict(scores)
        adjusted["not_always_hold"] = min(adjusted["not_always_hold"], hold_collapse)
        adjusted["weighted_total"] = weighted_reward(adjusted)
        rewards.append(adjusted["weighted_total"])
    return rewards


def score_reward_file(
    train_file: str | Path,
    *,
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    completions = []
    path = repo_path(train_file)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            scored = score_training_row(row)
            rows.append(scored)
            messages = row.get("messages", [])
            completions.append(messages[2].get("content", "") if len(messages) > 2 and isinstance(messages[2], dict) else "")
            if limit is not None and len(rows) >= limit:
                break
    batch_hold_penalty = _batch_hold_collapse_penalty(completions)
    for row in rows:
        components = row["components"]
        components["not_always_hold"] = min(components["not_always_hold"], batch_hold_penalty)
        components["weighted_total"] = weighted_reward(components)

    count = len(rows)
    component_sums = Counter()
    issue_counts = Counter()
    for row in rows:
        for key, value in row["components"].items():
            component_sums[key] += float(value)
            if key != "weighted_total" and value < 0:
                issue_counts[key] += 1
    report = {
        "train_file": str(path),
        "rows_scored": count,
        "batch_hold_collapse_penalty": batch_hold_penalty,
        "predicted_class_counts": _predicted_class_counts(completions),
        "average_components": {
            key: round(component_sums[key] / count, 6) for key in sorted(component_sums) if count
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "score_histogram": _score_histogram([row["components"]["weighted_total"] for row in rows]),
        "sample_rows": rows[:5],
    }
    if output_path is not None:
        out = repo_path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def score_prediction_file(
    prediction_file: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    completions = []
    prediction_records = []
    path = repo_path(prediction_file)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            scored = score_prediction_row(row)
            rows.append(scored)
            completions.append(str(row.get("completion", "")))
            prediction_records.append(row)

    batch_hold_penalty = _batch_hold_collapse_penalty(completions)
    for row in rows:
        components = row["components"]
        components["not_always_hold"] = min(components["not_always_hold"], batch_hold_penalty)
        components["weighted_total"] = weighted_reward(components)

    count = len(rows)
    component_sums = Counter()
    issue_counts = Counter()
    valid_json = 0
    for row in rows:
        valid_json += int(bool(row.get("valid_json")))
        for key, value in row["components"].items():
            component_sums[key] += float(value)
            if key != "weighted_total" and value < 0:
                issue_counts[key] += 1
    report = {
        "prediction_file": str(path),
        "rows_scored": count,
        "valid_json": valid_json,
        "valid_json_rate": round(valid_json / count, 6) if count else 0.0,
        "batch_hold_collapse_penalty": batch_hold_penalty,
        "predicted_class_counts": _predicted_class_counts(completions),
        "average_components": {
            key: round(component_sums[key] / count, 6) for key in sorted(component_sums) if count
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "score_histogram": _score_histogram([row["components"]["weighted_total"] for row in rows]),
        "forecasting": _forecasting_report_from_predictions(prediction_records),
        "sample_rows": rows[:5],
    }
    if output_path is not None:
        out = repo_path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def score_training_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", [])
    metadata = row.get("metadata", {})
    user_payload = _message_json(messages, 1)
    assistant_payload = _message_json(messages, 2)
    quarter = str(metadata.get("quarter") or user_payload.get("quarter") or "")
    role_id = str(metadata.get("role_id") or user_payload.get("role_id") or "")
    components = score_completion_components(
        messages[2].get("content", "") if len(messages) > 2 and isinstance(messages[2], dict) else assistant_payload,
        target_strategy=assistant_payload.get("equilibrium_strategy"),
        target_fed_prediction=assistant_payload.get("fed_prediction"),
        quarter=quarter,
        role_id=role_id,
        current_round=user_payload.get("round_id"),
    )
    return {
        "quarter": quarter,
        "role_id": role_id,
        "task": row.get("task"),
        "components": components,
    }


def score_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    target = _json_from_string(row.get("target"))
    prompt = row.get("prompt") or []
    user_payload = {}
    if isinstance(prompt, list) and len(prompt) > 1 and isinstance(prompt[1], dict):
        user_payload = _json_from_string(prompt[1].get("content"))
    quarter = str(metadata.get("quarter") or user_payload.get("quarter") or "")
    role_id = str(metadata.get("role_id") or user_payload.get("role_id") or "")
    completion = str(row.get("completion", ""))
    components = score_completion_components(
        completion,
        target_strategy=target.get("equilibrium_strategy"),
        target_fed_prediction=target.get("fed_prediction"),
        quarter=quarter,
        role_id=role_id,
        current_round=user_payload.get("round_id"),
    )
    return {
        "quarter": quarter,
        "role_id": role_id,
        "task": row.get("task"),
        "valid_json": parse_json_completion(completion) is not None,
        "components": components,
    }


def _forecasting_report_from_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = _quarter_labels()
    evaluated = []
    invalid_rows = 0
    missing_label_quarters = set()
    leakage = _empty_leakage_summary()
    per_quarter_probs: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        metadata = row.get("metadata", {})
        prompt = row.get("prompt") or []
        user_payload = {}
        if isinstance(prompt, list) and len(prompt) > 1 and isinstance(prompt[1], dict):
            user_payload = _json_from_string(prompt[1].get("content"))
        quarter = str(metadata.get("quarter") or user_payload.get("quarter") or "")
        parsed = parse_json_completion(row.get("completion", ""))
        if parsed is None:
            invalid_rows += 1
            continue
        leakage_row = _future_leakage_for_trace({"quarter": quarter, "completion": parsed}, examples_limit=10)
        _merge_leakage_summary(leakage, leakage_row)
        label = labels.get(quarter)
        if label is None:
            missing_label_quarters.add(quarter)
            continue
        probs = normalize_fed_prediction(_json_from_string(parsed.get("fed_prediction")) or {})
        evaluated.append(_forecast_eval_row(quarter, probs, label))
        per_quarter_probs.setdefault(quarter, []).append(probs)

    quarter_evaluated = []
    for quarter, prob_rows in sorted(per_quarter_probs.items()):
        label = labels.get(quarter)
        if label is None:
            continue
        mean_probs = {
            klass: sum(row.get(klass, 0.0) for row in prob_rows) / len(prob_rows)
            for klass in CLASS_ORDER
        }
        quarter_evaluated.append(_forecast_eval_row(quarter, mean_probs, label))

    return {
        "rows": len(rows),
        "invalid_rows": invalid_rows,
        "evaluated_rows": len(evaluated),
        "evaluated_quarters": len(quarter_evaluated),
        "missing_label_quarters": sorted(item for item in missing_label_quarters if item),
        "row_metrics": _forecast_metrics(evaluated),
        "quarter_mean_metrics": _forecast_metrics(quarter_evaluated),
        "calibration": _calibration_metrics(evaluated, bins=10),
        "future_leakage": leakage,
    }


def _forecast_eval_row(quarter: str, probs: dict[str, float], label: dict[str, Any]) -> dict[str, Any]:
    predicted_class = max(CLASS_ORDER, key=lambda key: probs[key])
    actual_class = label["actual_class"]
    return {
        "quarter": quarter,
        "probabilities": probs,
        "predicted_class": predicted_class,
        "actual_class": actual_class,
        "actual_decision": label["actual_decision"],
        "correct": predicted_class == actual_class,
    }


def schema_reward(completions: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        data = parse_json_completion(completion)
        if data is None:
            rewards.append(-1.0)
            continue
        strategy = data.get("strategy") or data.get("equilibrium_strategy") or {}
        if REQUIRED_STRATEGY_KEYS.issubset(strategy):
            rewards.append(1.0)
        else:
            rewards.append(-0.3)
    return rewards


def convergence_reward(completions: list[str], **kwargs: Any) -> list[float]:
    targets = kwargs.get("target_strategy") or [{} for _ in completions]
    if isinstance(targets, dict):
        targets = [targets for _ in completions]
    rewards = []
    for completion, target in zip(completions, targets):
        data = parse_json_completion(completion)
        if data is None:
            rewards.append(-1.0)
            continue
        strategy = data.get("strategy") or data.get("equilibrium_strategy") or {}
        distances = [
            abs(float(strategy.get(key, 0.0)) - float(target.get(key, 0.0)))
            for key in REQUIRED_STRATEGY_KEYS
            if key in target
        ]
        if not distances:
            rewards.append(0.0)
            continue
        rewards.append(max(-1.0, 1.0 - max(distances) * 3.0))
    return rewards


def evidence_reward(completions: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        data = parse_json_completion(completion)
        if data is None:
            rewards.append(-1.0)
            continue
        evidence = data.get("evidence") or data.get("evidence_chain") or []
        rewards.append(0.5 if evidence else -0.2)
    return rewards


def no_future_leakage_reward(completions: list[str], **kwargs: Any) -> list[float]:
    quarters = _broadcast(kwargs.get("quarter"), completions, default=None)
    return [
        score_completion_components(completion, quarter=quarter)["no_future_leakage"]
        for completion, quarter in zip(completions, quarters)
    ]


def calibration_reward(completions: list[str], **kwargs: Any) -> list[float]:
    quarters = _broadcast(kwargs.get("quarter"), completions, default=None)
    target_fed = _broadcast(kwargs.get("target_fed_prediction"), completions, default={})
    return [
        score_completion_components(completion, quarter=quarter, target_fed_prediction=fed)["fed_probability_calibrated"]
        for completion, quarter, fed in zip(completions, quarters, target_fed)
    ]


def regret_reward(completions: list[str], **kwargs: Any) -> list[float]:
    targets = _broadcast(kwargs.get("target_strategy"), completions, default={})
    return [
        score_completion_components(completion, target_strategy=target)["lower_regret"]
        for completion, target in zip(completions, targets)
    ]


def role_consistency_reward(completions: list[str], **kwargs: Any) -> list[float]:
    role_ids = _broadcast(kwargs.get("role_id"), completions, default=None)
    return [
        score_completion_components(completion, role_id=role_id)["role_consistency"]
        for completion, role_id in zip(completions, role_ids)
    ]


def _invalid_component_scores() -> dict[str, float]:
    scores = {key: -1.0 for key in DEFAULT_REWARD_WEIGHTS}
    scores["weighted_total"] = weighted_reward(scores)
    return scores


def _schema_score(data: dict[str, Any]) -> float:
    strategy = _strategy_payload(data)
    fed = data.get("fed_prediction") or {}
    if not isinstance(strategy, dict) or not isinstance(fed, dict):
        return -1.0
    missing_strategy = REQUIRED_STRATEGY_KEYS - set(strategy)
    missing_fed = REQUIRED_FED_KEYS - set(fed)
    values = list(strategy.values()) + list(fed.values())
    numeric = all(_is_finite_number(value) for value in values)
    in_range = all(0.0 <= float(value) <= 1.0 for value in values if _is_finite_number(value))
    if not missing_strategy and not missing_fed and numeric and in_range:
        return 1.0
    if len(missing_strategy) <= 2 and len(missing_fed) <= 1 and numeric:
        return 0.0
    return -1.0


def _future_leakage_score(data: dict[str, Any], *, quarter: str | None, quarter_end: str | None = None) -> float:
    cutoff = quarter_end or _quarter_end_iso(quarter)
    if not cutoff:
        return 0.0
    future_mentions = 0
    for value in _walk_json(data):
        if not isinstance(value, str):
            continue
        for raw in re.findall(r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b", value):
            if raw.replace("/", "-") > cutoff:
                future_mentions += 1
    if future_mentions == 0:
        return 1.0
    return max(-1.0, -0.25 * future_mentions)


def _evidence_traceability_score(data: dict[str, Any]) -> float:
    evidence = data.get("evidence") or data.get("evidence_chain") or []
    if isinstance(evidence, dict):
        evidence = evidence.get("items") or evidence.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return -0.3
    traceable = 0
    for item in evidence:
        if isinstance(item, dict) and any(item.get(key) for key in ["source_id", "source", "url", "date", "title"]):
            traceable += 1
        elif isinstance(item, str) and len(item.strip()) > 20:
            traceable += 0.5
    return max(-1.0, min(1.0, traceable / max(1, len(evidence))))


def _fed_calibration_score(fed_prediction: dict[str, float], actual_class: str | None) -> float:
    if actual_class not in CLASS_ORDER:
        return 0.0
    prob = float(fed_prediction.get(actual_class, 0.0))
    brier = sum((float(fed_prediction.get(key, 0.0)) - (1.0 if key == actual_class else 0.0)) ** 2 for key in CLASS_ORDER)
    return round(max(-1.0, 1.0 - brier), 6)


def _lower_regret_score(data: dict[str, Any], strategy: dict[str, Any], target_strategy: dict[str, Any] | None) -> float:
    if _is_finite_number(data.get("regret_estimate")):
        regret = max(0.0, float(data["regret_estimate"]))
        return round(max(-1.0, 1.0 - 4.0 * regret), 6)
    if not target_strategy:
        return 0.0
    distances = [
        abs(float(strategy.get(key, 0.0)) - float(target_strategy.get(key, 0.0)))
        for key in REQUIRED_STRATEGY_KEYS
        if key in strategy and key in target_strategy and _is_finite_number(strategy.get(key)) and _is_finite_number(target_strategy.get(key))
    ]
    if not distances:
        return 0.0
    return round(max(-1.0, 1.0 - max(distances) * 4.0), 6)


def _convergence_round_score(data: dict[str, Any], *, current_round: int | None) -> float:
    rounds = data.get("rounds_to_equilibrium") or data.get("rounds")
    if not _is_finite_number(rounds):
        rounds = current_round
    if not _is_finite_number(rounds):
        return 0.0
    rounds_float = max(1.0, float(rounds))
    return round(max(-1.0, min(1.0, 1.0 - (rounds_float - 1.0) / 20.0)), 6)


def _stable_after_10_rounds_score(data: dict[str, Any]) -> float:
    if data.get("converged") is True:
        return 1.0
    stable_rounds = data.get("stable_rounds")
    if _is_finite_number(stable_rounds):
        return 1.0 if float(stable_rounds) >= 10 else round(float(stable_rounds) / 10.0 - 0.5, 6)
    return 0.0


def _role_consistency_score(strategy: dict[str, Any], *, role_id: str | None) -> float:
    if not role_id:
        return 0.0
    role = role_id.lower()
    penalty = 0.0
    checks = 0
    for token, thresholds in ROLE_BIASES.items():
        if token not in role:
            continue
        for key, threshold in thresholds.items():
            checks += 1
            value = float(strategy.get(key, 0.0)) if _is_finite_number(strategy.get(key)) else 0.0
            if token in {"hawk", "warsh"} and value < threshold:
                penalty += threshold - value
            elif token in {"dove"} and value > threshold:
                penalty += value - threshold
            elif token in {"euro", "energy"} and value > threshold:
                penalty += value - threshold
    if checks == 0:
        return 0.0
    return round(max(-1.0, 1.0 - penalty * 3.0), 6)


def _unsupported_claims_score(data: dict[str, Any]) -> float:
    text = " ".join(str(data.get(key, "")) for key in ["rationale", "analysis", "notes", "claim"])
    if not text.strip():
        return 0.0
    claim_markers = len(re.findall(r"\b(will|must|clearly|because|therefore|due to|caused by|必然|因为|导致|证明)\b", text, re.I))
    evidence = data.get("evidence") or data.get("evidence_chain") or []
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    unsupported = max(0, claim_markers - evidence_count)
    if unsupported == 0:
        return 1.0 if claim_markers else 0.2
    return max(-1.0, 0.5 - 0.3 * unsupported)


def _overconfident_wrong_prediction_score(fed_prediction: dict[str, float], actual_class: str | None) -> float:
    if actual_class not in CLASS_ORDER:
        return 0.0
    predicted = max(CLASS_ORDER, key=lambda key: fed_prediction.get(key, 0.0))
    confidence = float(fed_prediction.get(predicted, 0.0))
    if predicted == actual_class:
        return round(min(1.0, confidence), 6)
    if confidence >= 0.75:
        return -1.0
    if confidence >= 0.6:
        return -0.6
    return -0.2


def _not_always_hold_score(fed_prediction: dict[str, float], actual_class: str | None) -> float:
    hold_prob = float(fed_prediction.get("hold", 0.0))
    non_hold_prob = max(float(fed_prediction.get("hike_25bp", 0.0)), float(fed_prediction.get("cut_25bp", 0.0)))
    predicted = max(CLASS_ORDER, key=lambda key: fed_prediction.get(key, 0.0))
    if actual_class in {"hike_25bp", "cut_25bp"}:
        if predicted == "hold" and hold_prob >= 0.6:
            return -0.8
        if predicted == "hold" and hold_prob >= 0.5:
            return -0.5
        if hold_prob >= 0.75:
            return -1.0
        if hold_prob >= 0.6 and non_hold_prob < 0.25:
            return -0.6
        return max(-0.2, round(1.0 - hold_prob, 6))
    if actual_class == "hold":
        if hold_prob > 0.9:
            return 0.2
        return min(1.0, round(0.3 + hold_prob, 6))
    if hold_prob >= 0.85 and non_hold_prob < 0.1:
        return -0.6
    return 0.3


def _batch_hold_collapse_penalty(completions: list[str]) -> float:
    predictions = []
    for completion in completions:
        data = parse_json_completion(completion)
        if data is None:
            continue
        predictions.append(normalize_fed_prediction(_json_from_string(data.get("fed_prediction")) or {}))
    if len(predictions) < 2:
        return 1.0
    hold_argmax_share = sum(
        1 for prediction in predictions if max(CLASS_ORDER, key=lambda key: prediction.get(key, 0.0)) == "hold"
    ) / len(predictions)
    avg_hold_prob = sum(float(prediction.get("hold", 0.0)) for prediction in predictions) / len(predictions)
    if hold_argmax_share >= 0.95 and avg_hold_prob >= 0.6:
        return -1.0
    if hold_argmax_share >= 0.85 and avg_hold_prob >= 0.55:
        return -0.6
    if hold_argmax_share >= 0.75:
        return -0.2
    return 1.0


def _predicted_class_counts(completions: list[str]) -> dict[str, int]:
    counts = Counter()
    for completion in completions:
        data = parse_json_completion(completion)
        if data is None:
            counts["invalid"] += 1
            continue
        prediction = normalize_fed_prediction(_json_from_string(data.get("fed_prediction")) or {})
        counts[max(CLASS_ORDER, key=lambda key: prediction.get(key, 0.0))] += 1
    return dict(counts)


def _strategy_payload(data: dict[str, Any]) -> dict[str, Any]:
    strategy = data.get("strategy") or data.get("equilibrium_strategy") or {}
    return _json_from_string(strategy)


def _message_json(messages: list[Any], idx: int) -> dict[str, Any]:
    if len(messages) <= idx or not isinstance(messages[idx], dict):
        return {}
    try:
        value = json.loads(messages[idx].get("content", "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _json_from_string(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _broadcast(value: Any, completions: list[str], *, default: Any) -> list[Any]:
    if isinstance(value, list) and len(value) == len(completions):
        return value
    if value is None:
        value = default
    return [value for _ in completions]


def _score_histogram(values: list[float]) -> dict[str, int]:
    buckets = Counter()
    for value in values:
        if value < -0.5:
            buckets["[-1.0,-0.5)"] += 1
        elif value < 0.0:
            buckets["[-0.5,0.0)"] += 1
        elif value < 0.5:
            buckets["[0.0,0.5)"] += 1
        else:
            buckets["[0.5,1.0]"] += 1
    return dict(sorted(buckets.items()))


def _walk_json(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)
    else:
        yield value


def _quarter_end_iso(quarter: str | None) -> str | None:
    if not quarter or len(quarter) != 6 or quarter[4] != "Q":
        return None
    year = int(quarter[:4])
    quarter_num = int(quarter[5])
    return {
        1: f"{year}-03-31",
        2: f"{year}-06-30",
        3: f"{year}-09-30",
        4: f"{year}-12-31",
    }.get(quarter_num)


@lru_cache(maxsize=1)
def _quarter_actual_classes() -> dict[str, str]:
    try:
        from models.fomc_labels import build_fomc_label_df
    except Exception:
        return {}
    df = build_fomc_label_df()
    df["quarter_id"] = df["date"].dt.to_period("Q").astype(str)
    result = {}
    for quarter, group in df.groupby("quarter_id"):
        decisions = [int(item) for item in group["decision"].tolist()]
        net = sum(decisions)
        if net > 0:
            result[quarter] = "hike_25bp"
        elif net < 0:
            result[quarter] = "cut_25bp"
        else:
            non_hold = [item for item in decisions if item != 0]
            if not non_hold:
                result[quarter] = "hold"
            else:
                result[quarter] = "hike_25bp" if non_hold[-1] > 0 else "cut_25bp"
    return result


def _actual_class_for_quarter(quarter: str | None) -> str | None:
    if not quarter:
        return None
    return _quarter_actual_classes().get(quarter)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)

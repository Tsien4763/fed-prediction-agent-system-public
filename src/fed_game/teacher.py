from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import RuntimeConfig, ensure_parent
from .data_sources import load_context_snapshots, load_strategy_cards, write_jsonl
from .llm import TeacherClient
from .schemas import TrainingExample


BALANCED_TEACHER_WINDOWS = [
    {
        "name": "train",
        "as_of_start": "2000-01-01",
        "as_of_end": "2019-12-31",
        "sample_per_year": 5,
    },
    {
        "name": "val",
        "as_of_start": "2020-01-01",
        "as_of_end": "2023-12-31",
        "sample_per_year": 10,
    },
    {
        "name": "test",
        "as_of_start": "2024-01-01",
        "as_of_end": "2026-12-31",
        "sample_per_year": 10,
    },
]


def build_teacher_client(config: RuntimeConfig) -> TeacherClient:
    return TeacherClient(
        base_url=config.teacher_base_url,
        model=config.teacher_model,
        api_key=config.teacher_api_key,
        timeout_seconds=config.teacher_timeout,
        allow_mock=config.allow_mock_teacher,
    )


def generate_semantic_sft(
    config: RuntimeConfig,
    *,
    limit: int | None = 200,
    use_teacher: bool = True,
    as_of_start: str | None = None,
    as_of_end: str | None = None,
    sample_per_year: int | None = None,
    append: bool = False,
) -> Path:
    paths = config.paths
    contexts = load_teacher_contexts(
        config,
        limit=limit,
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        sample_per_year=sample_per_year,
    )
    teacher = build_teacher_client(config) if use_teacher else None
    out_path = ensure_parent(paths["train_dir"] / "semantic_sft.jsonl")
    rows = semantic_examples(contexts, teacher)
    write_jsonl(out_path, (row.to_json() for row in rows), append=append)
    return out_path


def generate_role_sft(
    config: RuntimeConfig,
    *,
    limit: int | None = 200,
    use_teacher: bool = True,
    as_of_start: str | None = None,
    as_of_end: str | None = None,
    sample_per_year: int | None = None,
    append: bool = False,
) -> Path:
    paths = config.paths
    contexts = load_teacher_contexts(
        config,
        limit=limit,
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        sample_per_year=sample_per_year,
    )
    cards = load_strategy_cards(paths["strategy_cards"])
    teacher = build_teacher_client(config) if use_teacher else None
    out_path = ensure_parent(paths["train_dir"] / "role_best_response_sft.jsonl")
    write_jsonl(out_path, (row.to_json() for row in role_examples(contexts, cards, teacher)), append=append)
    return out_path


def generate_critique_sft(
    config: RuntimeConfig,
    *,
    limit: int | None = 200,
    use_teacher: bool = True,
    as_of_start: str | None = None,
    as_of_end: str | None = None,
    sample_per_year: int | None = None,
    append: bool = False,
) -> Path:
    paths = config.paths
    contexts = load_teacher_contexts(
        config,
        limit=limit,
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        sample_per_year=sample_per_year,
    )
    cards = load_strategy_cards(paths["strategy_cards"])
    teacher = build_teacher_client(config) if use_teacher else None
    out_path = ensure_parent(paths["train_dir"] / "critique_traces_sft.jsonl")
    write_jsonl(out_path, (row.to_json() for row in critique_examples(contexts, cards, teacher)), append=append)
    return out_path


def generate_evidence_chain_sft(
    config: RuntimeConfig,
    *,
    limit: int | None = 200,
    use_teacher: bool = True,
    as_of_start: str | None = None,
    as_of_end: str | None = None,
    sample_per_year: int | None = None,
    append: bool = False,
) -> Path:
    paths = config.paths
    contexts = load_teacher_contexts(
        config,
        limit=limit,
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        sample_per_year=sample_per_year,
    )
    teacher = build_teacher_client(config) if use_teacher else None
    out_path = ensure_parent(paths["train_dir"] / "evidence_chain_sft.jsonl")
    write_jsonl(out_path, (row.to_json() for row in evidence_chain_examples(contexts, teacher)), append=append)
    return out_path


def generate_balanced_teacher_sft(
    config: RuntimeConfig,
    *,
    tasks: set[str] | None = None,
    use_semantic_teacher: bool = True,
    use_role_teacher: bool = True,
    use_critique_teacher: bool = True,
    use_evidence_teacher: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_tasks = tasks or {"semantic", "role", "critique", "evidence"}
    outputs: dict[str, Any] = {
        "windows": [],
        "dry_run": dry_run,
        "policy": "balanced temporal teacher sampling; no head limit is applied",
    }
    for idx, window in enumerate(BALANCED_TEACHER_WINDOWS):
        contexts = load_teacher_contexts(
            config,
            limit=None,
            as_of_start=window["as_of_start"],
            as_of_end=window["as_of_end"],
            sample_per_year=int(window["sample_per_year"]),
        )
        window_report = {
            **window,
            "contexts": len(contexts),
            "first_as_of_date": _as_text(contexts[0].get("as_of_date")) if contexts else None,
            "last_as_of_date": _as_text(contexts[-1].get("as_of_date")) if contexts else None,
            "append": idx > 0,
            "outputs": {},
        }
        if not dry_run:
            append = idx > 0
            if "semantic" in selected_tasks:
                window_report["outputs"]["semantic_sft"] = str(
                    generate_semantic_sft(
                        config,
                        limit=None,
                        use_teacher=use_semantic_teacher,
                        as_of_start=window["as_of_start"],
                        as_of_end=window["as_of_end"],
                        sample_per_year=int(window["sample_per_year"]),
                        append=append,
                    )
                )
            if "role" in selected_tasks:
                window_report["outputs"]["role_sft"] = str(
                    generate_role_sft(
                        config,
                        limit=None,
                        use_teacher=use_role_teacher,
                        as_of_start=window["as_of_start"],
                        as_of_end=window["as_of_end"],
                        sample_per_year=int(window["sample_per_year"]),
                        append=append,
                    )
                )
            if "critique" in selected_tasks:
                window_report["outputs"]["critique_sft"] = str(
                    generate_critique_sft(
                        config,
                        limit=None,
                        use_teacher=use_critique_teacher,
                        as_of_start=window["as_of_start"],
                        as_of_end=window["as_of_end"],
                        sample_per_year=int(window["sample_per_year"]),
                        append=append,
                    )
                )
            if "evidence" in selected_tasks:
                window_report["outputs"]["evidence_chain_sft"] = str(
                    generate_evidence_chain_sft(
                        config,
                        limit=None,
                        use_teacher=use_evidence_teacher,
                        as_of_start=window["as_of_start"],
                        as_of_end=window["as_of_end"],
                        sample_per_year=int(window["sample_per_year"]),
                        append=append,
                    )
                )
        outputs["windows"].append(window_report)
    outputs["total_contexts"] = sum(int(window["contexts"]) for window in outputs["windows"])
    return outputs


def load_teacher_contexts(
    config: RuntimeConfig,
    *,
    limit: int | None = 200,
    as_of_start: str | None = None,
    as_of_end: str | None = None,
    sample_per_year: int | None = None,
) -> list[dict[str, Any]]:
    if sample_per_year is not None and limit is not None:
        raise ValueError("Do not combine sample_per_year with limit; use temporal sampling without a head limit.")
    needs_full_scan = bool(as_of_start or as_of_end or sample_per_year)
    paths = config.paths
    contexts = load_context_snapshots(paths["context_snapshots"], limit=None if needs_full_scan else limit)
    contexts = sorted(contexts, key=lambda row: _as_text(row.get("as_of_date")))
    if as_of_start:
        contexts = [row for row in contexts if _as_text(row.get("as_of_date")) >= as_of_start]
    if as_of_end:
        contexts = [row for row in contexts if _as_text(row.get("as_of_date")) <= as_of_end]
    if sample_per_year is not None:
        contexts = _sample_contexts_per_year(contexts, sample_per_year)
    if limit is not None and (sample_per_year is None or len(contexts) > limit):
        contexts = contexts[:limit]
    return contexts


def semantic_examples(contexts: list[dict[str, Any]], teacher: TeacherClient | None) -> Iterable[TrainingExample]:
    for row in contexts:
        payload = {
            "snapshot_id": row.get("snapshot_id"),
            "as_of_date": _as_text(row.get("as_of_date")),
            "target_country": row.get("target_country"),
            "target_actor": row.get("target_actor"),
            "target_strategy_key": row.get("target_strategy_key"),
            "macro_context": _truncate(row.get("macro_context"), 4000),
            "rag_text": _truncate(row.get("rag_text"), 2000),
        }
        fallback = _heuristic_semantic_answer()
        if teacher is None:
            answer = fallback
        else:
            raw_answer = teacher.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract calibrated macro-policy semantic labels. Return strict JSON only. "
                            "Use only information available at or before as_of_date. Do not repeat the input."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "semantic_extraction",
                                "expected_schema": {
                                    "hawkish_dovish_score": "float_minus_1_to_1",
                                    "inflation_concern": "float_0_to_1",
                                    "growth_outlook": "float_minus_1_to_1",
                                    "forward_guidance_strength": "float_0_to_1",
                                    "strategic_ambiguity": "float_0_to_1",
                                    "policy_stickiness": "float_0_to_1",
                                    "semantic_rationale": "string",
                                    "evidence_chain": [
                                        {
                                            "claim": "string",
                                            "evidence": "string",
                                            "source_date": "YYYY-MM-DD or null",
                                            "confidence": "float_0_to_1",
                                        }
                                    ],
                                },
                                "input": payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema_hint={"task": "semantic_sft"},
            )
            answer = _normalize_semantic_answer(raw_answer, fallback)
        yield TrainingExample(
            task="semantic_extraction",
            messages=[
                {
                    "role": "system",
                    "content": "You are a macro policy semantic extraction model. Return strict JSON.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
            metadata={"snapshot_id": row.get("snapshot_id"), "as_of_date": payload["as_of_date"]},
        )


def role_examples(
    contexts: list[dict[str, Any]],
    strategy_cards: list[dict[str, Any]],
    teacher: TeacherClient | None,
) -> Iterable[TrainingExample]:
    card_by_key = {str(card.get("strategy_key")): card for card in strategy_cards}
    for row in contexts:
        strategy_key = str(row.get("target_strategy_key", "communication.signal.communication"))
        as_of_date = _as_text(row.get("as_of_date"))
        card = _as_of_strategy_card(card_by_key.get(strategy_key, {}), as_of_date)
        user_payload = {
            "as_of_date": as_of_date,
            "target_country": row.get("target_country"),
            "target_actor": row.get("target_actor"),
            "own_previous_strategy_sequence": row.get("own_previous_strategy_sequence"),
            "other_p4_previous_strategy_sequence": row.get("other_p4_previous_strategy_sequence"),
            "macro_context": row.get("macro_context"),
            "rag_text": _truncate(row.get("rag_text"), 3000),
            "candidate_strategy_card": card,
        }
        fallback = _heuristic_role_answer(strategy_key)
        if teacher is None:
            assistant_payload = fallback
        else:
            raw_answer = teacher.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior macro-policy game teacher. Produce a role best-response trace "
                            "for supervised fine-tuning. Return strict JSON only. Do not use any information "
                            "after as_of_date. Include calibrated probabilities, a concise rationale, and an "
                            "evidence_chain grounded in the provided context. Do not default to hold simply because "
                            "it is common; assign non-hold probability when inflation, labor, liquidity, or crisis "
                            "evidence supports hike or cut risk, and penalize unsupported always-hold reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "role_best_response",
                                "expected_schema": {
                                    "selected_strategy_key": "string",
                                    "strategy": {
                                        "hawkish_signal_prob": "float_0_to_1",
                                        "rate_hike_25bp_prob": "float_0_to_1",
                                        "hold_with_hawkish_statement_prob": "float_0_to_1",
                                        "remove_forward_guidance_prob": "float_0_to_1",
                                        "easing_signal_prob": "float_0_to_1",
                                        "liquidity_support_prob": "float_0_to_1",
                                        "trade_or_sanction_pressure_prob": "float_0_to_1",
                                    },
                                    "rationale": "string",
                                    "evidence_chain": [
                                        {
                                            "claim": "string",
                                            "evidence": "string",
                                            "source_date": "YYYY-MM-DD or null",
                                            "source_actor": "string or null",
                                            "confidence": "float_0_to_1",
                                        }
                                    ],
                                    "counterfactual_risks": ["string"],
                                    "fed_probability_discipline": {
                                        "why_not_always_hold": "string",
                                        "hike_or_cut_evidence": ["string"],
                                        "overconfidence_check": "string",
                                    },
                                },
                                "input": user_payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema_hint={"task": "role_best_response"},
            )
            assistant_payload = _normalize_role_answer(raw_answer, strategy_key, fallback)
        yield TrainingExample(
            task="role_best_response",
            messages=[
                {
                    "role": "system",
                    "content": "You are a policy role agent. Produce a structured best-response strategy.",
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            metadata={"snapshot_id": row.get("snapshot_id"), "as_of_date": as_of_date, "strategy_key": strategy_key},
        )


def critique_examples(
    contexts: list[dict[str, Any]],
    strategy_cards: list[dict[str, Any]],
    teacher: TeacherClient | None,
) -> Iterable[TrainingExample]:
    card_by_key = {str(card.get("strategy_key")): card for card in strategy_cards}
    for row in contexts:
        strategy_key = str(row.get("target_strategy_key", "communication.signal.communication"))
        as_of_date = _as_text(row.get("as_of_date"))
        card = _as_of_strategy_card(card_by_key.get(strategy_key, {}), as_of_date)
        candidate = _heuristic_role_answer(strategy_key)
        user_payload = {
            "as_of_date": as_of_date,
            "target_country": row.get("target_country"),
            "target_actor": row.get("target_actor"),
            "macro_context": row.get("macro_context"),
            "rag_text": _truncate(row.get("rag_text"), 3000),
            "candidate_strategy_card": card,
            "candidate_best_response": candidate,
        }
        fallback = {
            "critic_id": "deepseek_policy_critic",
            "target_strategy_key": strategy_key,
            "issues": ["No teacher critique was available; review evidence coverage and feasibility manually."],
            "feasibility_score": 0.5,
            "revision_hint": "Ground the strategy in as-of evidence and report uncertainty explicitly.",
            "evidence_gaps": [],
            "future_leakage_check": {"contains_future_info": False, "notes": "No future evidence was supplied."},
        }
        if teacher is None:
            assistant_payload = fallback
        else:
            raw_answer = teacher.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a macro policy critic for a multi-agent Fed prediction game. "
                            "Critique the candidate best response using only information at or before as_of_date. "
                            "Return strict JSON only. Explicitly flag always-hold collapse: if the candidate treats "
                            "hold as a default without evidence, mark it as a serious calibration issue."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "critique_trace",
                                "expected_schema": {
                                    "critic_id": "string",
                                    "target_strategy_key": "string",
                                    "issues": ["string"],
                                    "feasibility_score": "float_0_to_1",
                                    "revision_hint": "string",
                                    "evidence_gaps": ["string"],
                                    "always_hold_collapse_check": {
                                        "is_always_hold_collapse": "boolean",
                                        "notes": "string",
                                    },
                                    "future_leakage_check": {
                                        "contains_future_info": "boolean",
                                        "notes": "string",
                                    },
                                },
                                "input": user_payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema_hint={"task": "critique_trace"},
            )
            assistant_payload = _normalize_critique_answer(raw_answer, strategy_key, fallback)
        yield TrainingExample(
            task="critique_trace",
            messages=[
                {
                    "role": "system",
                    "content": "You are a macro policy critic. Return structured critique JSON.",
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            metadata={"snapshot_id": row.get("snapshot_id"), "as_of_date": as_of_date, "strategy_key": strategy_key},
        )


def evidence_chain_examples(
    contexts: list[dict[str, Any]],
    teacher: TeacherClient | None,
) -> Iterable[TrainingExample]:
    for row in contexts:
        as_of_date = _as_text(row.get("as_of_date"))
        user_payload = {
            "as_of_date": as_of_date,
            "target_country": row.get("target_country"),
            "target_actor": row.get("target_actor"),
            "target_strategy_key": row.get("target_strategy_key"),
            "macro_context": row.get("macro_context"),
            "rag_text": _truncate(row.get("rag_text"), 4000),
        }
        fallback = {
            "evidence_chain": [
                {
                    "step_id": 1,
                    "claim": "Evidence must be extracted from as-of policy context.",
                    "evidence": _truncate(row.get("rag_text"), 240),
                    "source_date": as_of_date,
                    "source_actor": row.get("target_actor"),
                    "supports": "policy_context",
                    "confidence": 0.5,
                }
            ],
            "information_cutoff": as_of_date,
            "future_leakage_check": {"contains_future_info": False, "notes": "No future evidence was supplied."},
        }
        if teacher is None:
            assistant_payload = fallback
        else:
            raw_answer = teacher.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract an evidence chain for a macro policy game agent. Return strict JSON only. "
                            "Use only the provided as-of context and explicitly check for future leakage."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": "evidence_chain",
                                "expected_schema": {
                                    "evidence_chain": [
                                        {
                                            "step_id": "integer",
                                            "claim": "string",
                                            "evidence": "string",
                                            "source_date": "YYYY-MM-DD or null",
                                            "source_actor": "string or null",
                                            "supports": "string",
                                            "confidence": "float_0_to_1",
                                        }
                                    ],
                                    "information_cutoff": "YYYY-MM-DD",
                                    "future_leakage_check": {
                                        "contains_future_info": "boolean",
                                        "notes": "string",
                                    },
                                },
                                "input": user_payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                schema_hint={"task": "evidence_chain"},
            )
            assistant_payload = _normalize_evidence_answer(raw_answer, as_of_date, fallback)
        yield TrainingExample(
            task="evidence_chain",
            messages=[
                {
                    "role": "system",
                    "content": "You extract auditable as-of evidence chains for macro policy reasoning.",
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            metadata={
                "snapshot_id": row.get("snapshot_id"),
                "as_of_date": as_of_date,
                "strategy_key": row.get("target_strategy_key"),
            },
        )


def _heuristic_role_answer(strategy_key: str) -> dict[str, Any]:
    return {
            "selected_strategy_key": strategy_key,
            "strategy": {
                "hawkish_signal_prob": 0.65 if "tightening" in strategy_key else 0.42,
                "rate_hike_25bp_prob": 0.32 if "tightening" in strategy_key else 0.08,
                "hold_with_hawkish_statement_prob": 0.68 if "communication" in strategy_key else 0.45,
                "remove_forward_guidance_prob": 0.55 if "communication" in strategy_key else 0.35,
                "easing_signal_prob": 0.38 if "easing" in strategy_key else 0.08,
                "liquidity_support_prob": 0.30 if "easing" in strategy_key else 0.12,
                "trade_or_sanction_pressure_prob": 0.25,
            },
            "rationale": "Choose the historically observed strategy as supervised teacher signal under the as-of context.",
            "evidence_chain": [],
            "counterfactual_risks": [],
        }


def _heuristic_semantic_answer() -> dict[str, Any]:
    return {
        "hawkish_dovish_score": 0.0,
        "inflation_concern": 0.5,
        "growth_outlook": 0.0,
        "forward_guidance_strength": 0.4,
        "strategic_ambiguity": 0.5,
        "policy_stickiness": 0.6,
        "semantic_rationale": "Fallback neutral semantic label.",
        "evidence_chain": [],
    }


def _sample_contexts_per_year(contexts: list[dict[str, Any]], sample_per_year: int) -> list[dict[str, Any]]:
    if sample_per_year <= 0:
        raise ValueError("sample_per_year must be positive")
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in contexts:
        year = _as_text(row.get("as_of_date"))[:4]
        if len(year) == 4:
            by_year.setdefault(year, []).append(row)
    sampled: list[dict[str, Any]] = []
    for year in sorted(by_year):
        rows = sorted(by_year[year], key=lambda row: _as_text(row.get("as_of_date")))
        if len(rows) <= sample_per_year:
            sampled.extend(rows)
            continue
        if sample_per_year == 1:
            sampled.append(rows[len(rows) // 2])
            continue
        positions = [
            round(idx * (len(rows) - 1) / (sample_per_year - 1))
            for idx in range(sample_per_year)
        ]
        seen: set[int] = set()
        for pos in positions:
            if pos in seen:
                continue
            sampled.append(rows[pos])
            seen.add(pos)
    return sampled


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)[:10]


def _truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text[:limit]


def _as_of_strategy_card(card: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    keep = {
        "strategy_id",
        "strategy_key",
        "strategy_name",
        "actor_type",
        "instrument",
        "action",
        "stance",
        "preconditions",
        "context_required",
        "expected_channels",
        "selection_prompt",
    }
    sanitized = {key: card.get(key) for key in keep if key in card}
    historical = []
    for item in card.get("historical_examples", []) or []:
        item_date = _as_text(item.get("date"))
        if as_of_date and item_date and item_date > as_of_date:
            continue
        historical.append(item)
    if historical:
        sanitized["historical_examples_as_of"] = historical[:5]
    return sanitized


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 4)


def _clamp_range(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number < lower:
        return lower
    if number > upper:
        return upper
    return round(number, 4)


def _normalize_semantic_answer(answer: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in answer.items() if value is not None})
    merged["hawkish_dovish_score"] = _clamp_range(
        merged.get("hawkish_dovish_score"), fallback["hawkish_dovish_score"], -1.0, 1.0
    )
    for key in [
        "inflation_concern",
        "forward_guidance_strength",
        "strategic_ambiguity",
        "policy_stickiness",
    ]:
        merged[key] = _clamp_float(merged.get(key), fallback[key])
    merged["growth_outlook"] = _clamp_range(merged.get("growth_outlook"), fallback["growth_outlook"], -1.0, 1.0)
    merged["semantic_rationale"] = str(merged.get("semantic_rationale") or fallback["semantic_rationale"])
    if not isinstance(merged.get("evidence_chain"), list):
        merged["evidence_chain"] = []
    return {
        "hawkish_dovish_score": merged["hawkish_dovish_score"],
        "inflation_concern": merged["inflation_concern"],
        "growth_outlook": merged["growth_outlook"],
        "forward_guidance_strength": merged["forward_guidance_strength"],
        "strategic_ambiguity": merged["strategic_ambiguity"],
        "policy_stickiness": merged["policy_stickiness"],
        "semantic_rationale": merged["semantic_rationale"],
        "evidence_chain": merged["evidence_chain"],
    }


def _normalize_role_answer(answer: dict[str, Any], strategy_key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in answer.items() if value is not None})
    merged["selected_strategy_key"] = str(merged.get("selected_strategy_key") or strategy_key)
    raw_strategy = merged.get("strategy") if isinstance(merged.get("strategy"), dict) else {}
    defaults = fallback["strategy"]
    merged["strategy"] = {key: _clamp_float(raw_strategy.get(key), default) for key, default in defaults.items()}
    if not isinstance(merged.get("evidence_chain"), list):
        merged["evidence_chain"] = []
    if not isinstance(merged.get("counterfactual_risks"), list):
        merged["counterfactual_risks"] = []
    merged["rationale"] = str(merged.get("rationale") or fallback["rationale"])
    return merged


def _normalize_critique_answer(answer: dict[str, Any], strategy_key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in answer.items() if value is not None})
    merged["critic_id"] = str(merged.get("critic_id") or "deepseek_policy_critic")
    merged["target_strategy_key"] = str(merged.get("target_strategy_key") or strategy_key)
    if not isinstance(merged.get("issues"), list):
        merged["issues"] = [str(merged.get("issues") or "No issues provided.")]
    if not isinstance(merged.get("evidence_gaps"), list):
        merged["evidence_gaps"] = []
    merged["feasibility_score"] = _clamp_float(merged.get("feasibility_score"), fallback["feasibility_score"])
    merged["revision_hint"] = str(merged.get("revision_hint") or fallback["revision_hint"])
    leakage = merged.get("future_leakage_check")
    if not isinstance(leakage, dict):
        leakage = fallback["future_leakage_check"]
    merged["future_leakage_check"] = leakage
    return merged


def _normalize_evidence_answer(answer: dict[str, Any], as_of_date: str, fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    merged.update({key: value for key, value in answer.items() if value is not None})
    if not isinstance(merged.get("evidence_chain"), list):
        merged["evidence_chain"] = fallback["evidence_chain"]
    merged["information_cutoff"] = str(merged.get("information_cutoff") or as_of_date)
    leakage = merged.get("future_leakage_check")
    if not isinstance(leakage, dict):
        leakage = fallback["future_leakage_check"]
    merged["future_leakage_check"] = leakage
    return merged

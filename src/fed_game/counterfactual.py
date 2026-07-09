from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, ensure_parent
from .schemas import GameTrace
from .self_play import RollingSelfPlayEngine, _MACRO_PANEL, _build_var_vecm_briefing


CLASS_ORDER = ["hike_25bp", "hold", "cut_25bp"]
STRATEGY_KEYS = [
    "hawkish_signal_prob",
    "rate_hike_25bp_prob",
    "hold_with_hawkish_statement_prob",
    "remove_forward_guidance_prob",
    "easing_signal_prob",
    "liquidity_support_prob",
    "trade_or_sanction_pressure_prob",
]
BELIEF_KEYS = [
    "inflation_persistence",
    "labor_softening",
    "energy_shock_risk",
    "geopolitical_escalation",
    "dollar_liquidity_pressure",
    "policy_credibility_risk",
]
P5_CLUSTERS = ("USA", "CHN", "RUS", "GBR", "FRA")


@dataclass
class CounterfactualResult:
    quarter: str
    scenario_name: str
    overrides: dict[str, Any]
    factual_prediction: dict[str, float]
    counterfactual_prediction: dict[str, float]
    delta: dict[str, float]
    factual_evidence: list[str]
    counterfactual_evidence: list[str]
    briefing_delta: dict[str, dict[str, Any]]
    strategy_delta: list[dict[str, Any]] = field(default_factory=list)
    belief_delta: list[dict[str, Any]] = field(default_factory=list)
    evidence_delta: dict[str, list[str]] = field(default_factory=dict)
    p5_impact_summary: list[dict[str, Any]] = field(default_factory=list)
    factual_trace: dict[str, Any] = field(default_factory=dict)
    counterfactual_trace: dict[str, Any] = field(default_factory=dict)
    scenario_scope: str = (
        "Model-based counterfactual self-play. The delta is a structured "
        "simulation contrast under the stated scenario overrides."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        rows = [
            "# Counterfactual Self-Play Report",
            "",
            f"Quarter: `{self.quarter}`",
            f"Scenario: `{self.scenario_name}`",
            "",
            "## Overrides",
            "",
        ]
        for key, value in self.overrides.items():
            rows.append(f"- `{key}` = `{value}`")
        rows.extend(
            [
                "",
                "## Fed Decision Delta",
                "",
                "| Scenario | P(hike) | P(hold) | P(cut) |",
                "| --- | ---: | ---: | ---: |",
                _prediction_row("factual", self.factual_prediction),
                _prediction_row("counterfactual", self.counterfactual_prediction),
                _prediction_row("delta", self.delta, signed=True),
                "",
                "## Largest Strategy Shifts",
                "",
            ]
        )
        if self.strategy_delta:
            rows.extend(
                f"- `{item['scope']}.{item['field']}`: "
                f"{item['factual']:.3f} -> {item['counterfactual']:.3f} "
                f"({item['delta']:+.3f})"
                for item in self.strategy_delta[:8]
            )
        else:
            rows.append("- No material strategy shift detected.")

        rows.extend(["", "## Largest Belief Shifts", ""])
        if self.belief_delta:
            rows.extend(
                f"- `{item['role_id']}.{item['field']}`: "
                f"{item['factual']:.3f} -> {item['counterfactual']:.3f} "
                f"({item['delta']:+.3f})"
                for item in self.belief_delta[:8]
            )
        else:
            rows.append("- No material belief shift detected.")

        rows.extend(["", "## Five-Cluster Impact", ""])
        if self.p5_impact_summary:
            rows.extend(
                f"- `{item['cluster_id']}`: {item['impact_label']}; "
                f"strategy L1={item['strategy_shift_l1']:.3f}, "
                f"hawkish pressure delta={item['hawkish_pressure_delta']:+.3f}, "
                f"external pressure delta={item['external_pressure_delta']:+.3f}"
                for item in self.p5_impact_summary
            )
        else:
            rows.append("- Five-cluster impact summary is unavailable.")

        rows.extend(["", "## Evidence Chain Changes", ""])
        added = self.evidence_delta.get("added", [])
        removed = self.evidence_delta.get("removed", [])
        if added:
            rows.append("Added:")
            rows.extend(f"- {item}" for item in added[:5])
        if removed:
            rows.append("Removed:")
            rows.extend(f"- {item}" for item in removed[:5])
        if not added and not removed:
            rows.append("- Evidence chain text is unchanged or unavailable.")

        rows.extend(
            [
                "",
                "## Scenario Scope",
                "",
                self.scenario_scope,
            ]
        )
        return "\n".join(rows) + "\n"


def run_counterfactual(
    config: RuntimeConfig,
    *,
    quarter: str,
    scenario_name: str,
    overrides: dict[str, Any],
    max_context_docs: int = 4,
) -> CounterfactualResult:
    factual_engine = RollingSelfPlayEngine(config)
    counterfactual_engine = RollingSelfPlayEngine(config)

    factual_trace = factual_engine.run_quarter(quarter, max_context_docs=max_context_docs)
    factual_briefing = _build_var_vecm_briefing(quarter, _MACRO_PANEL)
    counterfactual_briefing = build_counterfactual_briefing(quarter, overrides)
    counterfactual_trace = counterfactual_engine.run_quarter(
        quarter,
        max_context_docs=max_context_docs,
        briefing_override=counterfactual_briefing,
        scenario_context={"name": scenario_name, **overrides},
    )

    factual_prediction = _normalize_prediction(factual_trace.fed_prediction)
    counterfactual_prediction = _normalize_prediction(counterfactual_trace.fed_prediction)
    delta = {
        key: round(counterfactual_prediction.get(key, 0.0) - factual_prediction.get(key, 0.0), 6)
        for key in CLASS_ORDER
    }

    return CounterfactualResult(
        quarter=quarter,
        scenario_name=scenario_name,
        overrides=overrides,
        factual_prediction=factual_prediction,
        counterfactual_prediction=counterfactual_prediction,
        delta=delta,
        factual_evidence=list(factual_trace.evidence_chain),
        counterfactual_evidence=list(counterfactual_trace.evidence_chain),
        briefing_delta=diff_mapping(factual_briefing, counterfactual_briefing),
        strategy_delta=compare_strategy_shifts(factual_trace, counterfactual_trace),
        belief_delta=compare_belief_shifts(factual_trace, counterfactual_trace),
        evidence_delta=diff_evidence(factual_trace.evidence_chain, counterfactual_trace.evidence_chain),
        p5_impact_summary=summarize_p5_counterfactual_impact(factual_trace, counterfactual_trace, delta),
        factual_trace=compact_trace(factual_trace),
        counterfactual_trace=compact_trace(counterfactual_trace),
    )


def build_counterfactual_briefing(quarter: str, overrides: dict[str, Any]) -> dict[str, Any]:
    briefing = dict(_build_var_vecm_briefing(quarter, _MACRO_PANEL))
    briefing.update(overrides)

    if "energy_risk" in overrides and "energy_risk_from_vecm" not in overrides:
        briefing["energy_risk_from_vecm"] = overrides["energy_risk"]
    if "energy_price_risk" in overrides and "energy_risk_from_vecm" not in overrides:
        briefing["energy_risk_from_vecm"] = overrides["energy_price_risk"]

    if _truthy(briefing.get("warsh_replaced_by_powell")):
        briefing.setdefault("fed_chair", "powell")
        briefing.setdefault("policy_credibility_prior", 0.34)

    chair = str(briefing.get("fed_chair", "")).lower()
    if chair in {"powell", "yellen", "dove", "dovish"}:
        briefing.setdefault("policy_credibility_prior", 0.34)
    elif chair in {"warsh", "hawk", "hawkish"}:
        briefing.setdefault("policy_credibility_prior", 0.48)

    _refresh_taylor_fields(briefing)
    _refresh_signal_fields(briefing)
    briefing["counterfactual_override_keys"] = sorted(overrides)
    return briefing


def parse_override_assignments(assignments: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        overrides[key] = parse_override_value(raw_value.strip())
    return overrides


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def save_counterfactual_outputs(
    result: CounterfactualResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, str]:
    json_out = ensure_parent(Path(json_path))
    json_out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    outputs = {"json_path": str(json_out)}
    if markdown_path is not None:
        md_out = ensure_parent(Path(markdown_path))
        md_out.write_text(result.to_markdown(), encoding="utf-8")
        outputs["markdown_path"] = str(md_out)
    return outputs


def compact_trace(trace: GameTrace) -> dict[str, Any]:
    latest_roles = _latest_role_proposals(trace)
    payoff_sources: dict[str, int] = {}
    for proposal in trace.proposals:
        source = str(proposal.get("payoff_source", "missing"))
        payoff_sources[source] = payoff_sources.get(source, 0) + 1
    return {
        "quarter": trace.quarter,
        "converged": trace.converged,
        "rounds": trace.rounds,
        "fed_prediction": trace.fed_prediction,
        "cluster_strategies": trace.cluster_strategies,
        "equilibrium_check": trace.equilibrium_check,
        "personas_used": trace.personas_used,
        "warsh_consistency_score": trace.warsh_consistency_score,
        "payoff_source_counts": payoff_sources,
        "latest_role_proposals": [
            {
                "role_id": proposal.get("role_id"),
                "cluster_id": proposal.get("cluster_id"),
                "payoff_source": proposal.get("payoff_source"),
                "payoff_estimate": proposal.get("payoff_estimate"),
                "regret_estimate": proposal.get("regret_estimate"),
                "payoff_reasoning": str(proposal.get("payoff_reasoning", ""))[:600],
                "deviation_candidate": proposal.get("deviation_candidate", {}),
                "strategy": proposal.get("strategy", {}),
            }
            for proposal in latest_roles.values()
        ],
    }


def compare_strategy_shifts(factual: GameTrace, counterfactual: GameTrace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factual_clusters = {
        item.get("cluster_id"): item.get("strategy", {})
        for item in factual.cluster_strategies
    }
    counterfactual_clusters = {
        item.get("cluster_id"): item.get("strategy", {})
        for item in counterfactual.cluster_strategies
    }
    for cluster_id in sorted(set(factual_clusters) | set(counterfactual_clusters)):
        rows.extend(
            _numeric_delta_rows(
                f"cluster:{cluster_id}",
                factual_clusters.get(cluster_id, {}),
                counterfactual_clusters.get(cluster_id, {}),
                STRATEGY_KEYS,
            )
        )
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows


def compare_belief_shifts(factual: GameTrace, counterfactual: GameTrace) -> list[dict[str, Any]]:
    factual_roles = _latest_role_proposals(factual)
    counterfactual_roles = _latest_role_proposals(counterfactual)
    rows: list[dict[str, Any]] = []
    for role_id in sorted(set(factual_roles) | set(counterfactual_roles)):
        factual_belief = factual_roles.get(role_id, {}).get("belief", {})
        counterfactual_belief = counterfactual_roles.get(role_id, {}).get("belief", {})
        for item in _numeric_delta_rows(role_id, factual_belief, counterfactual_belief, BELIEF_KEYS):
            item["role_id"] = item.pop("scope")
            rows.append(item)
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows


def summarize_p5_counterfactual_impact(
    factual: GameTrace,
    counterfactual: GameTrace,
    prediction_delta: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Summarize how an event scenario changes each five-cluster strategy."""
    factual_clusters = _cluster_strategy_map(factual)
    counterfactual_clusters = _cluster_strategy_map(counterfactual)
    prediction_delta = prediction_delta or {}
    rows: list[dict[str, Any]] = []
    for cluster_id in P5_CLUSTERS:
        factual_strategy = factual_clusters.get(cluster_id, {})
        counterfactual_strategy = counterfactual_clusters.get(cluster_id, {})
        deltas = _numeric_delta_rows(
            f"cluster:{cluster_id}",
            factual_strategy,
            counterfactual_strategy,
            STRATEGY_KEYS,
        )
        deltas.sort(key=lambda item: abs(item["delta"]), reverse=True)
        strategy_l1 = round(sum(abs(float(item["delta"])) for item in deltas), 6)
        hawkish_delta = _strategy_pressure_delta(
            factual_strategy,
            counterfactual_strategy,
            positive_keys=[
                "hawkish_signal_prob",
                "rate_hike_25bp_prob",
                "hold_with_hawkish_statement_prob",
                "remove_forward_guidance_prob",
            ],
            negative_keys=["easing_signal_prob", "liquidity_support_prob"],
        )
        external_delta = _strategy_pressure_delta(
            factual_strategy,
            counterfactual_strategy,
            positive_keys=["trade_or_sanction_pressure_prob"],
            negative_keys=[],
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "strategy_shift_l1": strategy_l1,
                "hawkish_pressure_delta": round(hawkish_delta, 6),
                "external_pressure_delta": round(external_delta, 6),
                "top_strategy_shift": deltas[0] if deltas else None,
                "impact_label": _impact_label(strategy_l1, hawkish_delta, external_delta),
                "fed_decision_delta": {
                    key: float(prediction_delta.get(key, 0.0))
                    for key in CLASS_ORDER
                },
            }
        )
    rows.sort(
        key=lambda item: (
            float(item["strategy_shift_l1"]),
            abs(float(item["hawkish_pressure_delta"])),
            abs(float(item["external_pressure_delta"])),
        ),
        reverse=True,
    )
    return rows


def _cluster_strategy_map(trace: GameTrace) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("cluster_id")): item.get("strategy", {})
        for item in trace.cluster_strategies
        if item.get("cluster_id")
    }


def _strategy_pressure_delta(
    factual: dict[str, Any],
    counterfactual: dict[str, Any],
    *,
    positive_keys: list[str],
    negative_keys: list[str],
) -> float:
    pos = sum((_as_float(counterfactual.get(key)) or 0.0) - (_as_float(factual.get(key)) or 0.0) for key in positive_keys)
    neg = sum((_as_float(counterfactual.get(key)) or 0.0) - (_as_float(factual.get(key)) or 0.0) for key in negative_keys)
    return pos - neg


def _impact_label(strategy_l1: float, hawkish_delta: float, external_delta: float) -> str:
    if strategy_l1 < 0.03:
        return "little strategic change"
    if abs(external_delta) >= max(abs(hawkish_delta), 0.05):
        return "external-pressure shock" if external_delta > 0 else "external-pressure easing"
    if hawkish_delta > 0.05:
        return "hawkish policy pressure"
    if hawkish_delta < -0.05:
        return "dovish or liquidity-support pressure"
    return "mixed strategic repositioning"


def diff_mapping(factual: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(factual) | set(counterfactual)):
        old = factual.get(key)
        new = counterfactual.get(key)
        if old != new:
            out[key] = {"factual": old, "counterfactual": new}
    return out


def diff_evidence(factual: list[str], counterfactual: list[str]) -> dict[str, list[str]]:
    factual_set = set(factual)
    counterfactual_set = set(counterfactual)
    return {
        "added": sorted(counterfactual_set - factual_set),
        "removed": sorted(factual_set - counterfactual_set),
    }


def _numeric_delta_rows(
    scope: str,
    factual: dict[str, Any],
    counterfactual: dict[str, Any],
    keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in keys:
        old = _as_float(factual.get(key))
        new = _as_float(counterfactual.get(key))
        if old is None or new is None:
            continue
        delta = round(new - old, 6)
        if abs(delta) > 1e-6:
            rows.append(
                {
                    "scope": scope,
                    "field": key,
                    "factual": round(old, 6),
                    "counterfactual": round(new, 6),
                    "delta": delta,
                }
            )
    return rows


def _latest_role_proposals(trace: GameTrace) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for proposal in trace.proposals:
        role_id = proposal.get("role_id")
        if role_id:
            latest[str(role_id)] = proposal
    return latest


def _normalize_prediction(prediction: dict[str, Any]) -> dict[str, float]:
    values = {key: max(0.0, float(prediction.get(key, 0.0))) for key in CLASS_ORDER}
    total = sum(values.values()) or 1.0
    return {key: round(value / total, 6) for key, value in values.items()}


def _refresh_taylor_fields(briefing: dict[str, Any]) -> None:
    inflation = float(briefing.get("inflation_cpi_yoy", 2.5) or 2.5)
    gdp_growth = float(briefing.get("gdp_growth_qoq_ann", 2.0) or 2.0)
    ff_rate = float(briefing.get("fedfunds", 3.5) or 3.5)
    natural_rate = 2.5
    inflation_target = 2.0
    output_gap = gdp_growth - 2.0
    taylor_rate = natural_rate + inflation + 1.5 * (inflation - inflation_target) + 0.5 * output_gap
    briefing["taylor_rule_implied_rate"] = round(taylor_rate, 2)
    briefing["taylor_gap_pp"] = round(taylor_rate - ff_rate, 2)


def _refresh_signal_fields(briefing: dict[str, Any]) -> None:
    ect = float(briefing.get("ect_combined", 0.0) or 0.0)
    taylor_gap = float(briefing.get("taylor_gap_pp", 0.0) or 0.0)
    if ect < -0.5:
        briefing["ect_signal"] = "below_equilibrium"
    elif ect > 0.5:
        briefing["ect_signal"] = "above_equilibrium"
    else:
        briefing["ect_signal"] = "near_equilibrium"

    if taylor_gap > 1.0:
        briefing["taylor_signal"] = "policy_too_loose"
    elif taylor_gap < -1.0:
        briefing["taylor_signal"] = "policy_too_tight"
    else:
        briefing["taylor_signal"] = "policy_neutral"


def _prediction_row(label: str, values: dict[str, float], *, signed: bool = False) -> str:
    def fmt(key: str) -> str:
        value = values.get(key, 0.0)
        if signed:
            return f"{value * 100:+.2f}pp"
        return f"{value:.2%}"

    return f"| {label} | {fmt('hike_25bp')} | {fmt('hold')} | {fmt('cut_25bp')} |"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

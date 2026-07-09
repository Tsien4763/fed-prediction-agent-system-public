from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if math.isnan(value):
        return lower
    return max(lower, min(upper, float(value)))


@dataclass
class StrategyVector:
    hawkish_signal_prob: float = 0.5
    rate_hike_25bp_prob: float = 0.2
    hold_with_hawkish_statement_prob: float = 0.5
    remove_forward_guidance_prob: float = 0.4
    easing_signal_prob: float = 0.1
    liquidity_support_prob: float = 0.1
    trade_or_sanction_pressure_prob: float = 0.2

    def normalized(self) -> "StrategyVector":
        data = {key: clamp(value) for key, value in asdict(self).items()}
        return StrategyVector(**data)

    def distance(self, other: "StrategyVector") -> float:
        own = asdict(self.normalized())
        peer = asdict(other.normalized())
        return max(abs(float(own[key]) - float(peer[key])) for key in own)

    def blend(self, other: "StrategyVector", weight: float) -> "StrategyVector":
        weight = clamp(weight)
        own = asdict(self.normalized())
        peer = asdict(other.normalized())
        return StrategyVector(**{key: own[key] * (1.0 - weight) + peer[key] * weight for key in own})

    def to_dict(self) -> dict[str, float]:
        return asdict(self.normalized())

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StrategyVector":
        if not data:
            return cls()
        defaults = asdict(cls())
        values = {key: clamp(float(data.get(key, defaults[key]))) for key in defaults}
        return cls(**values)


@dataclass
class BeliefState:
    inflation_persistence: float = 0.5
    labor_softening: float = 0.3
    energy_shock_risk: float = 0.3
    geopolitical_escalation: float = 0.3
    dollar_liquidity_pressure: float = 0.2
    policy_credibility_risk: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {key: clamp(value) for key, value in asdict(self).items()}

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> "BeliefState":
        # --- 1. Use VAR/VECM structured data if available ---
        briefing = context.get("var_vecm_briefing", {})
        
        # --- 1b. Cross-quarter memory: policy inertia from past strategies ---
        past_text = str(context.get("past_strategies", ""))
        has_past = "Past strategies" in past_text
        # If we have past strategies, agents should be more inertial (less likely to swing wildly)
        inertia_bonus = 0.15 if has_past else 0.0
        if isinstance(briefing, dict) and briefing:
            ect = float(briefing.get("ect_combined", 0))
            taylor_gap = float(briefing.get("taylor_gap_pp", 0))
            inflation_now = float(briefing.get("inflation_cpi_yoy", 2.5))
            labor_now = float(briefing.get("unemployment", 4.0))
            energy_risk = float(
                briefing.get(
                    "energy_risk_from_vecm",
                    briefing.get("energy_risk", briefing.get("energy_price_risk", 0.3)),
                )
            )
            credibility_prior = briefing.get("policy_credibility_prior")
            credibility_base = float(credibility_prior) if credibility_prior is not None else 0.4
            
            inflation_excess = max(0.0, inflation_now - 2.0) / 6.0
            taylor_pressure = max(0.0, taylor_gap) / 4.0
            # ECT < 0 means below long-run equilibrium; inflation above target and
            # a positive Taylor gap are stronger no-key fallback signals.
            inflation_signal = 0.25 + 0.45 * inflation_excess + 0.20 * taylor_pressure + 0.10 * max(0, -ect / 5.0)
            # Taylor gap > 0 → policy too loose → need to hike
            hawkish_pressure = 0.1 + 0.35 * taylor_pressure
            # Labor: unemployment below NAIRU (~4.5%) → tight
            labor_signal = 0.15 + 0.20 * max(0, (4.5 - labor_now) / 3.0)
            # Event-level shocks can arrive between quarterly macro releases.
            # Treat them as overrides to the quarterly VAR/VECM briefing.
            energy_signal = 0.2 + 0.15 * energy_risk
            geopolitical_signal = briefing.get("geopolitical_escalation")
            if geopolitical_signal is None:
                geopolitical_signal = 0.3 + 0.1 * hawkish_pressure
            liquidity_signal = briefing.get("dollar_liquidity_pressure")
            if liquidity_signal is None:
                liquidity_signal = 0.25 + 0.1 * (taylor_gap / 4.0)
            
            return cls(
                inflation_persistence=clamp(inflation_signal),
                labor_softening=clamp(1.0 - labor_signal),
                energy_shock_risk=clamp(energy_signal),
                geopolitical_escalation=clamp(float(geopolitical_signal)),
                dollar_liquidity_pressure=clamp(float(liquidity_signal)),
                policy_credibility_risk=clamp(credibility_base + 0.25 * max(0, inflation_now - 2.0) / 5.0),
            )
        
        # --- 2. Fallback: keyword-based belief from context text ---
        text = json.dumps(context, ensure_ascii=False).lower()
        inflation = 0.55 + 0.18 * ("inflation" in text or "cpi" in text)
        labor = 0.28 + 0.14 * ("unemployment" in text or "labor" in text)
        energy = 0.25 + 0.18 * ("energy" in text or "oil" in text)
        geopol = 0.25 + 0.20 * ("sanction" in text or "war" in text or "geopolitical" in text)
        liquidity = 0.20 + 0.15 * ("liquidity" in text or "dollar" in text)
        credibility = 0.45 + 0.20 * ("target" in text or "credibility" in text)
        return cls(inflation, labor, energy, geopol, liquidity, credibility)


@dataclass
class PolicyCost:
    credibility_loss_if_dovish: float = 0.5
    growth_cost_if_hike: float = 0.4
    political_cost_if_tight: float = 0.3
    external_retaliation_cost: float = 0.2
    policy_turn_cost: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {key: clamp(value) for key, value in asdict(self).items()}


@dataclass
class AgentMemory:
    role_id: str
    cluster_id: str
    previous_strategy: StrategyVector = field(default_factory=StrategyVector)
    failed_strategies: list[dict[str, Any]] = field(default_factory=list)
    stable_rounds: int = 0
    payoff_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "cluster_id": self.cluster_id,
            "previous_strategy": self.previous_strategy.to_dict(),
            "failed_strategies": self.failed_strategies[-10:],
            "stable_rounds": self.stable_rounds,
            "payoff_history": self.payoff_history[-20:],
        }


@dataclass
class AgentProposal:
    role_id: str
    cluster_id: str
    round_id: int
    strategy: StrategyVector
    belief: BeliefState
    policy_cost: PolicyCost
    rationale: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    payoff_estimate: float = 0.0
    regret_estimate: float = 0.0
    payoff_source: str = "rule"
    payoff_reasoning: str = ""
    deviation_candidate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "cluster_id": self.cluster_id,
            "round_id": self.round_id,
            "strategy": self.strategy.to_dict(),
            "belief": self.belief.to_dict(),
            "policy_cost": self.policy_cost.to_dict(),
            "rationale": self.rationale,
            "evidence": [summarize_evidence(item) for item in self.evidence],
            "payoff_estimate": float(self.payoff_estimate),
            "regret_estimate": float(self.regret_estimate),
            "payoff_source": self.payoff_source,
            "payoff_reasoning": self.payoff_reasoning,
            "deviation_candidate": self.deviation_candidate,
        }


@dataclass
class Critique:
    critic_id: str
    target_role_id: str
    issues: list[str]
    feasibility_score: float
    revision_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic_id": self.critic_id,
            "target_role_id": self.target_role_id,
            "issues": self.issues,
            "feasibility_score": clamp(self.feasibility_score),
            "revision_hint": self.revision_hint,
        }


@dataclass
class ClusterStrategy:
    cluster_id: str
    round_id: int
    strategy: StrategyVector
    members: list[str]
    internal_dispersion: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "round_id": self.round_id,
            "strategy": self.strategy.to_dict(),
            "members": self.members,
            "internal_dispersion": float(self.internal_dispersion),
            "notes": self.notes,
        }


@dataclass
class GameTrace:
    quarter: str
    converged: bool
    rounds: int
    max_strategy_distance: float
    max_deviation_gain: float
    cluster_strategies: list[dict[str, Any]]
    shock_variables: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    critiques: list[dict[str, Any]]
    fed_prediction: dict[str, float]
    evidence_chain: list[str]
    equilibrium_check: dict[str, Any] = field(default_factory=dict)
    personas_used: dict[str, Any] = field(default_factory=dict)
    warsh_consistency_score: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingExample:
    task: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def summarize_evidence(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text", ""))
    return {
        "doc_id": item.get("doc_id"),
        "date": item.get("date"),
        "country": item.get("country"),
        "actor": item.get("actor"),
        "strategy_key": item.get("strategy_key"),
        "title": item.get("title"),
        "url": item.get("url"),
        "snippet": text[:500],
    }

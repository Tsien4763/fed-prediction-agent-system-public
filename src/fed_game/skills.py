from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .clusters import RoleCard
from .llm import TeacherClient
from .schemas import AgentMemory, AgentProposal, BeliefState, Critique, PolicyCost, StrategyVector


class SemanticExtractionSkill:
    def __init__(self, teacher: TeacherClient | None = None) -> None:
        if teacher is None:
            raise RuntimeError("SemanticExtractionSkill requires a DeepSeek teacher; no fallback is allowed.")
        self.teacher = teacher

    def run(self, policy_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract macro-policy semantic signals as strict JSON. "
                    "Use only information visible in the supplied text and metadata."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"metadata": metadata, "policy_text": policy_text[:8000]}, ensure_ascii=False),
            },
        ]
        return self.teacher.chat_json(messages, schema_hint={"type": "semantic_policy_signals"})


class BestResponseSkill:
    """Disabled rule strategy mode.

    The project contract requires DeepSeek for strategy generation, payoff
    judging, and equilibrium auditing. This class remains only as an explicit
    guard against accidental reintroduction of rule fallback.
    """

    def run(
        self,
        *,
        role: RoleCard,
        memory: AgentMemory,
        context: dict[str, Any],
        opponent_strategies: dict[str, StrategyVector],
        evidence: list[dict[str, Any]],
        round_id: int,
    ) -> AgentProposal:
        raise RuntimeError(
            "BestResponseSkill rule mode is disabled. "
            "Use LLMBestResponseSkill with a real DeepSeek teacher; no fallback is allowed."
        )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clamp_float(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    if number != number:
        number = lower
    return max(lower, min(upper, number))


def _required_float_field(data: Any, key: str) -> float:
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"DeepSeek JSON missing required numeric field: {key}")
    return _clamp_float(data[key])


def _strict_strategy_from_payload(data: Any) -> StrategyVector:
    if not isinstance(data, dict):
        raise ValueError("DeepSeek policy_best_response must return a JSON object or {'strategy': object}.")
    required = [
        "hawkish_signal_prob",
        "rate_hike_25bp_prob",
        "hold_with_hawkish_statement_prob",
        "remove_forward_guidance_prob",
        "easing_signal_prob",
        "liquidity_support_prob",
        "trade_or_sanction_pressure_prob",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"DeepSeek policy_best_response missing required StrategyVector fields: {missing}")
    return StrategyVector(**{key: _required_float_field(data, key) for key in required})


def _compact_evidence(item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text", item.get("snippet", "")))
    return {
        "doc_id": item.get("doc_id"),
        "title": item.get("title"),
        "date": item.get("date") or item.get("published_at"),
        "url": item.get("url"),
        "country": item.get("country"),
        "snippet": text[:600],
    }


def _compact_json_dict(data: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    if not data:
        return {}
    encoded = json.dumps(data, ensure_ascii=False)
    if len(encoded) <= max_chars:
        return data
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            compact[key] = value[:300]
        elif isinstance(value, dict):
            compact[key] = _compact_json_dict(value, max_chars=500)
        elif isinstance(value, list):
            compact[key] = value[:3]
        else:
            compact[key] = value
        if len(json.dumps(compact, ensure_ascii=False)) >= max_chars:
            break
    return compact


def _persona_prompt_payload(context: dict[str, Any]) -> dict[str, Any]:
    persona = context.get("policy_persona")
    return persona if isinstance(persona, dict) else {}


class LLMBestResponseSkill:
    """LLM-powered best response: reads context, past strategies, and VAR/VECM data.
    
    Uses the teacher LLM (DeepSeek) to reason about:
      - Past strategies (cross-quarter memory → policy inertia)
      - VAR/VECM briefing (economic equilibrium → data-aware decisions)
      - Opponent strategies (game-theoretic adaptation)

    This class is fail-closed: strategy generation and payoff judging must both
    come from DeepSeek. Rule fallback is deliberately disabled.
    """
    def __init__(self, teacher: TeacherClient | None = None, *, use_llm_payoff: bool = True):
        if teacher is None:
            raise RuntimeError("LLMBestResponseSkill requires a DeepSeek teacher; no fallback is allowed.")
        if not use_llm_payoff:
            raise RuntimeError("DeepSeek payoff judging is required; no rule payoff fallback is allowed.")
        self.teacher = teacher
        self.use_llm_payoff = use_llm_payoff
    
    def run(
        self,
        *,
        role: RoleCard,
        memory: AgentMemory,
        context: dict[str, Any],
        opponent_strategies: dict[str, StrategyVector],
        evidence: list[dict[str, Any]],
        round_id: int,
    ) -> AgentProposal:
        try:
            return self._llm_propose(role, memory, context, opponent_strategies, evidence, round_id)
        except Exception as exc:
            raise RuntimeError(f"DeepSeek best-response failed for {role.role_id}; no fallback is allowed.") from exc
    
    def _llm_propose(
        self,
        role: RoleCard,
        memory: AgentMemory,
        context: dict[str, Any],
        opponent_strategies: dict[str, StrategyVector],
        evidence: list[dict[str, Any]],
        round_id: int,
    ) -> AgentProposal:
        """Use LLM to reason about optimal strategy given full context."""
        # Build rich context prompt
        var_vecm = context.get("var_vecm_briefing", {})
        past = context.get("past_strategies", "No past strategies available.")
        policy_persona = _persona_prompt_payload(context)
        
        prompt = json.dumps({
            "task": "policy_best_response",
            "role": {
                "id": role.role_id,
                "name": role.name,
                "cluster": role.cluster_id,
                "objective": role.objective,
                "constraints": role.constraints,
                "preferred_signals": role.preferred_signals,
            },
            "policy_persona": policy_persona,
            "economic_context": {
                "quarter": context.get("quarter", "?"),
                "var_vecm_briefing": var_vecm,
                "past_strategies": past[:2000],  # truncate for token limit
            },
            "opponent_strategies": {
                cid: s.to_dict() for cid, s in opponent_strategies.items()
            },
            "game_state": {
                "round_id": round_id,
                "previous_own_strategy": memory.previous_strategy.to_dict(),
                "payoff_history": memory.payoff_history[-3:] if memory.payoff_history else [],
                "equilibrium_feedback": context.get("equilibrium_feedback", {}),
            },
            "instruction": (
                "Based on your role, economic data (VAR/VECM), past strategies, and opponent moves, "
                "output a JSON StrategyVector with these 7 fields (all 0-1): "
                "hawkish_signal_prob, rate_hike_25bp_prob, hold_with_hawkish_statement_prob, "
                "remove_forward_guidance_prob, easing_signal_prob, liquidity_support_prob, "
                "trade_or_sanction_pressure_prob. "
                "If past strategies show policy inertia, adjust gradually. "
                "If VAR/VECM shows economy below equilibrium (ECT<0), lean hawkish. "
                "If policy_persona is provided, use it as this role's cognitive operating system: "
                "mental models, value ordering, decision heuristics, expression DNA, anti-patterns, and honest boundaries. "
                "Include a 'rationale' string explaining your reasoning."
            ),
        }, ensure_ascii=False)
        
        messages = [
            {"role": "system", "content": "You are a macro-policy AI agent in a multi-country game-theoretic simulation. Output only valid JSON."},
            {"role": "user", "content": prompt[:8000]},
        ]
        
        result = self.teacher.chat_json(messages, schema_hint={"type": "policy_best_response"})
        
        # Parse LLM output into StrategyVector
        strategy_payload = result.get("strategy") if isinstance(result.get("strategy"), dict) else result
        strategy = _strict_strategy_from_payload(strategy_payload).normalized()
        
        belief = BeliefState.from_context(context)
        cost = PolicyCost(
            credibility_loss_if_dovish=belief.policy_credibility_risk,
            growth_cost_if_hike=belief.labor_softening + strategy.rate_hike_25bp_prob * 0.25,
            political_cost_if_tight=0.25 + strategy.rate_hike_25bp_prob * 0.4,
            external_retaliation_cost=strategy.trade_or_sanction_pressure_prob * 0.5,
            policy_turn_cost=memory.previous_strategy.distance(strategy),
        )
        assessment = self._score_payoff_with_llm(
            role=role,
            memory=memory,
            context=context,
            opponent_strategies=opponent_strategies,
            evidence=evidence,
            strategy=strategy,
            belief=belief,
            cost=cost,
            round_id=round_id,
        )
        payoff = assessment["payoff"]
        regret = assessment["regret_estimate"]
        rationale = result.get("rationale", f"LLM best-response for {role.name} under {role.objective}.")
        
        return AgentProposal(
            role_id=role.role_id,
            cluster_id=role.cluster_id,
            round_id=round_id,
            strategy=strategy,
            belief=belief,
            policy_cost=cost,
            rationale=rationale,
            evidence=evidence,
            payoff_estimate=payoff,
            regret_estimate=regret,
            payoff_source=assessment["source"],
            payoff_reasoning=assessment["reasoning"],
            deviation_candidate=assessment["deviation_candidate"],
        )

    def _score_payoff_with_llm(
        self,
        *,
        role: RoleCard,
        memory: AgentMemory,
        context: dict[str, Any],
        opponent_strategies: dict[str, StrategyVector],
        evidence: list[dict[str, Any]],
        strategy: StrategyVector,
        belief: BeliefState,
        cost: PolicyCost,
        round_id: int,
    ) -> dict[str, Any]:
        if self.teacher is None or not self.use_llm_payoff:
            raise RuntimeError("DeepSeek payoff judge is required; no fallback is allowed.")

        prompt = json.dumps(
            {
                "task": "policy_payoff_judgement",
                "role": {
                    "id": role.role_id,
                    "name": role.name,
                    "cluster": role.cluster_id,
                    "objective": role.objective,
                    "constraints": role.constraints,
                    "preferred_signals": role.preferred_signals,
                },
                "policy_persona": _persona_prompt_payload(context),
                "economic_context": {
                    "quarter": context.get("quarter", "?"),
                    "var_vecm_briefing": context.get("var_vecm_briefing", {}),
                    "past_strategies": str(context.get("past_strategies", ""))[:1800],
                    "counterfactual_scenario": context.get("counterfactual_scenario", {}),
                },
                "game_state": {
                    "round_id": round_id,
                    "candidate_strategy": strategy.to_dict(),
                    "belief": belief.to_dict(),
                    "policy_cost": cost.to_dict(),
                    "previous_own_strategy": memory.previous_strategy.to_dict(),
                    "opponent_cluster_strategies": {
                        cid: item.to_dict() for cid, item in opponent_strategies.items()
                    },
                    "equilibrium_feedback": context.get("equilibrium_feedback", {}),
                },
                "evidence": [_compact_evidence(item) for item in evidence[:5]],
                "instruction": (
                    "Judge this candidate strategy as the named role. Rate role payoff on a 0-1 scale. "
                    "If a policy_persona is supplied, score against its value ordering, mental models, "
                    "decision heuristics, anti-patterns, and honest boundaries. "
                    "Then check whether a unilateral alternative strategy would improve this role's payoff "
                    "while all opponent cluster strategies remain fixed. Return strict JSON with keys: "
                    "payoff, regret_estimate, best_unilateral_deviation, reasoning, confidence. "
                    "regret_estimate must be max(0, best_alternative_payoff - payoff)."
                ),
            },
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DeepSeek acting as a macro policy game payoff judge. "
                    "Use role objectives, macro evidence, policy costs, and unilateral-deviation logic. "
                    "Respect any supplied policy_persona as a public-information cognitive profile, "
                    "not as evidence of private beliefs. "
                    "Output only valid JSON."
                ),
            },
            {"role": "user", "content": prompt[:10000]},
        ]
        result = self.teacher.chat_json(messages, schema_hint={"type": "policy_payoff_judgement"})

        deviation = result.get("best_unilateral_deviation")
        if not isinstance(deviation, dict):
            deviation = {}
        payoff = _required_float_field(result, "payoff")
        regret = result.get("regret_estimate", result.get("profitable_deviation_gain"))
        if regret is None:
            alternative_payoff = deviation.get("alternative_payoff", deviation.get("deviation_payoff"))
            if alternative_payoff is not None:
                regret = max(0.0, _clamp_float(alternative_payoff) - payoff)
            else:
                raise ValueError("DeepSeek payoff judgement missing regret_estimate.")
        reasoning = str(result.get("reasoning", result.get("rationale", ""))).strip()
        return {
            "payoff": round(payoff, 4),
            "regret_estimate": round(_clamp_float(regret), 4),
            "source": "deepseek_payoff_judge",
            "reasoning": reasoning[:1200] or "DeepSeek payoff judge returned a numeric score without rationale.",
            "deviation_candidate": _compact_json_dict(deviation, max_chars=1400),
        }


class LLMEquilibriumJudge:
    """DeepSeek-backed check for profitable unilateral deviations.

    The self-play loop still uses a cheap stability rule to decide when to ask
    for a judge pass. DeepSeek then checks the Nash-style question directly:
    holding all other cluster strategies fixed, does any role have a profitable
    unilateral deviation?
    """

    def __init__(self, teacher: TeacherClient | None = None) -> None:
        if teacher is None:
            raise RuntimeError("LLMEquilibriumJudge requires a DeepSeek teacher; no heuristic fallback is allowed.")
        self.teacher = teacher

    def check(
        self,
        *,
        quarter: str,
        round_id: int,
        cluster_strategies: dict[str, StrategyVector],
        proposals: list[AgentProposal],
        var_vecm_briefing: dict[str, Any],
        heuristic: dict[str, Any],
        role_personas: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.teacher is None:
            raise RuntimeError("DeepSeek equilibrium judge is required; no heuristic fallback is allowed.")

        recent_by_role: dict[str, AgentProposal] = {}
        for proposal in proposals:
            recent_by_role[proposal.role_id] = proposal
        prompt = json.dumps(
            {
                "task": "nash_equilibrium_check",
                "definition": (
                    "A strategy profile is a Nash-style equilibrium only if no single role can improve "
                    "its payoff through a unilateral strategy deviation while all other cluster strategies "
                    "remain fixed."
                ),
                "quarter": quarter,
                "round_id": round_id,
                "heuristic_metrics": heuristic,
                "var_vecm_briefing": var_vecm_briefing,
                "role_personas": role_personas or {},
                "cluster_strategies": {key: value.to_dict() for key, value in cluster_strategies.items()},
                "latest_role_proposals": [
                    {
                        "role_id": proposal.role_id,
                        "cluster_id": proposal.cluster_id,
                        "strategy": proposal.strategy.to_dict(),
                        "payoff_estimate": proposal.payoff_estimate,
                        "regret_estimate": proposal.regret_estimate,
                        "payoff_source": proposal.payoff_source,
                        "rationale": proposal.rationale[:700],
                        "payoff_reasoning": proposal.payoff_reasoning[:700],
                    }
                    for proposal in recent_by_role.values()
                ],
                "instruction": (
                    "Check for profitable unilateral deviations role by role. "
                    "When a role persona exists, judge deviations against that persona's value ordering "
                    "and anti-patterns, while keeping the Nash-style unilateral-deviation definition. "
                    "If any role can improve payoff by changing only its own strategy, set "
                    "is_nash_equilibrium=false and list deviations. "
                    "Return strict JSON with keys: is_nash_equilibrium, "
                    "max_profitable_deviation_gain, profitable_deviations, reasoning, confidence. "
                    "Use 0 gain when no profitable deviation exists."
                ),
            },
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DeepSeek acting as a game-theoretic equilibrium auditor for a macro policy game. "
                    "Be conservative: if profitable unilateral deviation is plausible, reject equilibrium. "
                    "Output only valid JSON."
                ),
            },
            {"role": "user", "content": prompt[:12000]},
        ]
        result = self.teacher.chat_json(messages, schema_hint={"type": "nash_equilibrium_check"})

        deviations = result.get("profitable_deviations", [])
        if not isinstance(deviations, list):
            deviations = []
        max_gain = _required_float_field(result, "max_profitable_deviation_gain")
        is_equilibrium = _truthy(result.get("is_nash_equilibrium", False)) and max_gain <= float(
            heuristic.get("deviation_gain_tau", 0.02) or 0.02
        )
        return {
            "checked": True,
            "source": "deepseek_equilibrium_judge",
            "quarter": quarter,
            "round_id": round_id,
            "is_nash_equilibrium": is_equilibrium,
            "max_profitable_deviation_gain": round(max_gain, 4),
            "profitable_deviations": [_compact_json_dict(item, max_chars=1200) for item in deviations[:8] if isinstance(item, dict)],
            "reasoning": str(result.get("reasoning", "")).strip()[:1600],
            "confidence": _clamp_float(result.get("confidence", 0.5)),
            "heuristic": heuristic,
        }


class CriticSkill:
    def run(self, proposal: AgentProposal) -> Critique:
        issues: list[str] = []
        strategy = proposal.strategy.normalized()
        if strategy.rate_hike_25bp_prob > 0.55 and proposal.belief.labor_softening > 0.55:
            issues.append("tightening may conflict with labor-market deterioration")
        if strategy.easing_signal_prob > 0.35 and proposal.belief.inflation_persistence > 0.6:
            issues.append("easing signal may damage inflation credibility")
        if proposal.policy_cost.policy_turn_cost > 0.35:
            issues.append("large turn from prior strategy violates internal stickiness")
        if not issues:
            issues.append("no blocking inconsistency; monitor evidence freshness")
        feasibility = max(0.0, 1.0 - 0.22 * (len(issues) - 1) - proposal.policy_cost.policy_turn_cost * 0.35)
        return Critique(
            critic_id=f"{proposal.cluster_id}_critic",
            target_role_id=proposal.role_id,
            issues=issues,
            feasibility_score=feasibility,
            revision_hint="reduce abrupt turns and cite stronger as-of evidence when feasibility falls",
        )


class CoordinatorSkill:
    def run(self, cluster_id: str, proposals: list[AgentProposal], round_id: int) -> tuple[StrategyVector, float, str]:
        if not proposals:
            return StrategyVector(), 0.0, "empty cluster"
        weights = [max(0.05, proposal.payoff_estimate) for proposal in proposals]
        total = sum(weights)
        aggregate = {}
        for key in asdict(StrategyVector()).keys():
            aggregate[key] = sum(getattr(proposal.strategy.normalized(), key) * weight for proposal, weight in zip(proposals, weights)) / total
        strategy = StrategyVector(**aggregate).normalized()
        dispersion = max((proposal.strategy.distance(strategy) for proposal in proposals), default=0.0)
        notes = f"{cluster_id} coordinator synthesized {len(proposals)} role strategies with payoff weighting."
        return strategy, dispersion, notes


class EvidenceSkill:
    def run(self, fed_path: dict[str, float], shocks: dict[str, float], proposals: list[AgentProposal]) -> list[str]:
        top_roles = sorted(proposals, key=lambda item: item.payoff_estimate, reverse=True)[:3]
        chain = [
            f"FOMC path: hold={fed_path.get('hold', 0):.2f}, hike={fed_path.get('hike_25bp', 0):.2f}, cut={fed_path.get('cut_25bp', 0):.2f}.",
            f"External shock layer: energy={shocks.get('energy_price_risk', 0):.2f}, geopol={shocks.get('geopol_escalation_prob', 0):.2f}, term_premium={shocks.get('term_premium_pressure', 0):.2f}.",
        ]
        for proposal in top_roles:
            chain.append(
                f"{proposal.role_id}: hawkish={proposal.strategy.hawkish_signal_prob:.2f}, "
                f"hike={proposal.strategy.rate_hike_25bp_prob:.2f}, payoff={proposal.payoff_estimate:.2f}."
            )
        return chain

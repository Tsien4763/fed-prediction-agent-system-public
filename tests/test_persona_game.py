from __future__ import annotations

import json

from fed_game.clusters import get_role
from fed_game.config import RuntimeConfig, load_config
from fed_game.persona import load_configured_personas, load_policy_persona
from fed_game.schemas import AgentMemory, StrategyVector
from fed_game.self_play import RollingSelfPlayEngine
from fed_game.skills import BestResponseSkill, LLMBestResponseSkill, LLMEquilibriumJudge


class FakeTeacher:
    def __init__(self) -> None:
        self.schema_types: list[str | None] = []
        self.messages: list[list[dict[str, str]]] = []

    def chat_json(self, messages, schema_hint=None):
        schema_type = (schema_hint or {}).get("type")
        self.schema_types.append(schema_type)
        self.messages.append(messages)
        if schema_type == "policy_best_response":
            return {
                "strategy": {
                    "hawkish_signal_prob": 0.66,
                    "rate_hike_25bp_prob": 0.31,
                    "hold_with_hawkish_statement_prob": 0.72,
                    "remove_forward_guidance_prob": 0.61,
                    "easing_signal_prob": 0.08,
                    "liquidity_support_prob": 0.16,
                    "trade_or_sanction_pressure_prob": 0.23,
                },
                "rationale": "Inflation credibility dominates, but the role keeps optionality.",
            }
        if schema_type == "policy_payoff_judgement":
            return {
                "payoff": 0.74,
                "regret_estimate": 0.01,
                "best_unilateral_deviation": {
                    "strategy_change": "slightly lower hike probability",
                    "alternative_payoff": 0.75,
                },
                "reasoning": "The strategy is close to the role objective with little profitable deviation.",
                "confidence": 0.8,
            }
        if schema_type == "nash_equilibrium_check":
            return {
                "is_nash_equilibrium": "true",
                "max_profitable_deviation_gain": 0.01,
                "profitable_deviations": [],
                "reasoning": "No role has a material unilateral improvement.",
                "confidence": 0.77,
            }
        raise AssertionError(f"unexpected schema type: {schema_type}")


def test_warsh_persona_loader_and_consistency() -> None:
    persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")

    assert persona.role_id == "usa_warsh"
    assert len(persona.nuwa_dimensions) == 6
    assert len(persona.mental_models) >= 3
    assert len(persona.decision_heuristics) >= 5
    assert len(persona.evidence_sources) >= 6

    payload = persona.to_prompt_payload(max_markdown_chars=800)
    assert payload["priors"]["hawkish_signal_prob"] > payload["priors"]["easing_signal_prob"]
    assert persona.consistency_score(StrategyVector.from_dict(persona.priors))["score"] == 1.0

    configured = load_configured_personas(load_config())
    assert configured["usa_warsh"].persona_id == "kevin-warsh-policy-persona"


def test_deepseek_payoff_and_equilibrium_contracts_use_persona_context() -> None:
    teacher = FakeTeacher()
    skill = LLMBestResponseSkill(teacher)
    persona = load_policy_persona("policy_personas/warsh_policy/SKILL.md")
    context = {
        "quarter": "2024Q2",
        "var_vecm_briefing": {
            "inflation_cpi_yoy": 3.4,
            "unemployment": 4.0,
            "taylor_gap_pp": 0.8,
            "ect_signal": "near_equilibrium",
        },
        "policy_persona": persona.to_prompt_payload(),
    }

    proposal = skill.run(
        role=get_role("usa_warsh"),
        memory=AgentMemory(role_id="usa_warsh", cluster_id="USA"),
        context=context,
        opponent_strategies={"CHN": StrategyVector(trade_or_sanction_pressure_prob=0.4)},
        evidence=[{"title": "FOMC statement", "text": "Inflation remains elevated."}],
        round_id=1,
    )

    assert proposal.payoff_source == "deepseek_payoff_judge"
    assert proposal.payoff_estimate == 0.74
    assert proposal.regret_estimate == 0.01
    assert proposal.deviation_candidate["alternative_payoff"] == 0.75

    judge = LLMEquilibriumJudge(teacher)
    check = judge.check(
        quarter="2024Q2",
        round_id=10,
        cluster_strategies={"USA": proposal.strategy, "CHN": StrategyVector()},
        proposals=[proposal],
        var_vecm_briefing=context["var_vecm_briefing"],
        heuristic={
            "max_strategy_distance": 0.01,
            "strategy_epsilon": 0.025,
            "max_deviation_gain": 0.01,
            "deviation_gain_tau": 0.02,
            "stable_rounds": 10,
            "stable_rounds_required": 10,
            "heuristic_passed": True,
        },
        role_personas={"usa_warsh": persona.to_prompt_payload(max_markdown_chars=1200)},
    )

    assert check["source"] == "deepseek_equilibrium_judge"
    assert check["checked"] is True
    assert check["is_nash_equilibrium"] is True
    assert "policy_best_response" in teacher.schema_types
    assert "policy_payoff_judgement" in teacher.schema_types
    assert "nash_equilibrium_check" in teacher.schema_types

    prompt_text = json.dumps(teacher.messages, ensure_ascii=False)
    assert "policy_persona" in prompt_text
    assert "Credibility is the real policy multiplier" in prompt_text


def test_rule_best_response_is_disabled() -> None:
    context = {
        "quarter": "2022Q2",
        "var_vecm_briefing": {
            "inflation_cpi_yoy": 8.5,
            "unemployment": 3.6,
            "fedfunds": 1.0,
            "taylor_gap_pp": 5.0,
            "energy_risk_from_vecm": 0.3,
        },
    }

    try:
        BestResponseSkill().run(
            role=get_role("usa_warsh"),
            memory=AgentMemory(role_id="usa_warsh", cluster_id="USA"),
            context=context,
            opponent_strategies={"CHN": StrategyVector()},
            evidence=[],
            round_id=1,
        )
    except RuntimeError as exc:
        assert "rule mode is disabled" in str(exc)
    else:
        raise AssertionError("BestResponseSkill must be disabled; no rule fallback is allowed.")


def test_llm_best_response_rejects_missing_teacher_and_disabled_payoff() -> None:
    try:
        LLMBestResponseSkill(None)
    except RuntimeError as exc:
        assert "requires a DeepSeek teacher" in str(exc)
    else:
        raise AssertionError("LLMBestResponseSkill must require a DeepSeek teacher.")

    try:
        LLMBestResponseSkill(FakeTeacher(), use_llm_payoff=False)
    except RuntimeError as exc:
        assert "payoff judging is required" in str(exc)
    else:
        raise AssertionError("LLMBestResponseSkill must require DeepSeek payoff judging.")

    try:
        LLMEquilibriumJudge(None)
    except RuntimeError as exc:
        assert "requires a DeepSeek teacher" in str(exc)
    else:
        raise AssertionError("LLMEquilibriumJudge must require a DeepSeek teacher.")


def test_self_play_engine_requires_real_deepseek_key(monkeypatch) -> None:
    raw = json.loads(json.dumps(load_config().raw))
    raw["teacher"]["api_key_env"] = "MAE_CPS_TEST_MISSING_DEEPSEEK_KEY"
    raw["teacher"]["allow_mock_without_key"] = False
    monkeypatch.delenv("MAE_CPS_TEST_MISSING_DEEPSEEK_KEY", raising=False)

    try:
        RollingSelfPlayEngine(RuntimeConfig(raw=raw))
    except RuntimeError as exc:
        assert "requires DEEPSEEK_API_KEY" in str(exc)
        assert "fallback is disabled" in str(exc)
    else:
        raise AssertionError("RollingSelfPlayEngine must require a real DeepSeek key.")

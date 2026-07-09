from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agents import PolicyAgent
from .clusters import CLUSTER_MEMBERS, WARSH_POLICY_PROFILE, get_role
from .config import SELF_PLAY_TRACE_FILENAME, RuntimeConfig, ensure_dir, ensure_parent, repo_path
from .data_sources import RagIndex
from .persona import PolicyPersona, load_configured_personas
from .schemas import AgentMemory, AgentProposal, ClusterStrategy, GameTrace, StrategyVector
from .skills import BestResponseSkill, CoordinatorSkill, CriticSkill, EvidenceSkill, LLMEquilibriumJudge, LLMBestResponseSkill
from .tools import PolicyTools


def _load_macro_panel() -> pd.DataFrame | None:
    """Load US macro panel + VECM ECT for VAR/VECM context injection."""
    root = repo_path(".")
    macro_path = root / "data" / "processed" / "us_macro_panel.parquet"
    ect_path = root / "data" / "processed" / "vecm_ect.parquet"
    try:
        macro = pd.read_parquet(macro_path)
        if ect_path.exists():
            ect = pd.read_parquet(ect_path)
            macro = macro.join(ect, how="left")
        return macro
    except Exception:
        return None


def _build_var_vecm_briefing(quarter: str, macro_panel: pd.DataFrame | None) -> dict[str, float]:
    """Build VAR/VECM briefing for a given quarter (e.g., '2024Q1').
    
    Returns structured data about:
    - VECM Error Correction Term (how far from long-run equilibrium)
    - Taylor Rule gap
    - Current macro snapshot
    """
    if macro_panel is None:
        return {}
    
    # Convert quarter to end-of-quarter date
    y, q = int(quarter[:4]), int(quarter[-1])
    from pandas.tseries.offsets import MonthEnd
    q_end = pd.Timestamp(year=y, month=q*3, day=1) + MonthEnd(1)
    
    # Find closest macro data (use current quarter or previous if not available)
    mask = macro_panel.index <= q_end
    if not mask.any():
        return {}
    
    row = macro_panel[mask].iloc[-1]
    
    # Taylor Rule: r = r* + π + 0.5(π-π*) + 0.5(y-y*)
    inflation = float(row.get("inflation_cpi_yoy", 2.5)) if not pd.isna(row.get("inflation_cpi_yoy", np.nan)) else 2.5
    gdp_growth = float(row.get("gdp_growth_qoq_ann", 2.0)) if not pd.isna(row.get("gdp_growth_qoq_ann", np.nan)) else 2.0
    ff_rate = float(row.get("fedfunds", 3.5)) if not pd.isna(row.get("fedfunds", np.nan)) else 3.5
    unemp = float(row.get("unemployment", 4.0)) if not pd.isna(row.get("unemployment", np.nan)) else 4.0
    
    natural_rate = 2.5
    inflation_target = 2.0
    output_gap = gdp_growth - 2.0
    taylor_rate = natural_rate + inflation + 1.5 * (inflation - inflation_target) + 0.5 * output_gap
    taylor_gap = taylor_rate - ff_rate
    
    ect_val = float(row.get("ect_combined", 0)) if not pd.isna(row.get("ect_combined", np.nan)) else 0
    
    briefing = {
        "quarter": quarter,
        "fedfunds": round(ff_rate, 2),
        "inflation_cpi_yoy": round(inflation, 2),
        "gdp_growth_qoq_ann": round(gdp_growth, 2),
        "unemployment": round(unemp, 2),
        "taylor_rule_implied_rate": round(taylor_rate, 2),
        "taylor_gap_pp": round(taylor_gap, 2),
        "ect_combined": round(ect_val, 4),
        "energy_risk_from_vecm": round(abs(ect_val) / 10.0, 3) if abs(ect_val) > 1 else 0.3,
    }
    
    # Add interpretation hints
    if ect_val < -0.5:
        briefing["ect_signal"] = "below_equilibrium"  # upward pressure
    elif ect_val > 0.5:
        briefing["ect_signal"] = "above_equilibrium"  # downward pressure
    else:
        briefing["ect_signal"] = "near_equilibrium"
    
    if taylor_gap > 1.0:
        briefing["taylor_signal"] = "policy_too_loose"
    elif taylor_gap < -1.0:
        briefing["taylor_signal"] = "policy_too_tight"
    else:
        briefing["taylor_signal"] = "policy_neutral"
    
    return briefing


def _quarter_end_iso(quarter: str) -> str:
    year, q = int(quarter[:4]), int(quarter[-1])
    if q == 1:
        return f"{year}-03-31"
    if q == 2:
        return f"{year}-06-30"
    if q == 3:
        return f"{year}-09-30"
    if q == 4:
        return f"{year}-12-31"
    raise ValueError(f"Invalid quarter: {quarter}")


# Pre-load macro panel once at module level
_MACRO_PANEL = _load_macro_panel()


class CrossQuarterMemory:
    """RAG-like memory: store and retrieve past strategies across quarters.
    
    Enables agents to:
      - Recall their previous quarter's strategy (policy inertia)
      - Retrieve similar historical situations by keyword/embedding
      - Adapt based on what worked/failed in the past
    """
    def __init__(self):
        self._history: list[dict] = []  # list of {quarter, cluster_id, strategy, fed_prediction, outcome}
    
    def record(self, quarter: str, cluster_strategies: list[dict], fed_prediction: dict):
        """Store a completed quarter's strategies."""
        for cs in cluster_strategies:
            self._history.append({
                "quarter": quarter,
                "cluster_id": cs.get("cluster_id", "?"),
                "strategy": cs.get("strategy", {}),
                "fed_prediction": fed_prediction,
            })
    
    def retrieve(self, cluster_id: str, current_quarter: str, top_k: int = 3) -> list[dict]:
        """Retrieve past strategies for a cluster, ordered by recency + similarity.
        
        Priority: (1) same cluster, (2) recent quarters first, (3) similar macro conditions.
        For v1: simple recency-weighted retrieval — most recent same-cluster strategies.
        """
        same_cluster = [h for h in self._history if h["cluster_id"] == cluster_id and h["quarter"] < current_quarter]
        # Most recent first
        same_cluster.sort(key=lambda x: x["quarter"], reverse=True)
        return same_cluster[:top_k]
    
    def build_context_text(self, cluster_id: str, current_quarter: str) -> str:
        """Build a natural-language summary of past strategies for agent context."""
        past = self.retrieve(cluster_id, current_quarter)
        if not past:
            return "No prior strategy history available."
        
        lines = [f"Past strategies for {cluster_id} (most recent first):"]
        for h in past:
            s = h["strategy"]
            fp = h.get("fed_prediction", {})
            lines.append(
                f"  {h['quarter']}: hawkish={s.get('hawkish_signal_prob',0):.2f}, "
                f"hike_prob={s.get('rate_hike_25bp_prob',0):.2f}, "
                f"hold_prob={s.get('hold_with_hawkish_statement_prob',0):.2f}, "
                f"easing={s.get('easing_signal_prob',0):.2f}. "
                f"Result: P(hike)={fp.get('hike_25bp',0):.2f}, P(hold)={fp.get('hold',0):.2f}"
            )
        return "\n".join(lines)


@dataclass
class SelfPlayResult:
    traces: list[GameTrace]
    trace_path: Path
    distill_path: Path


def quarter_range(start: str, end: str) -> list[str]:
    start_year, start_quarter = _parse_quarter(start)
    end_year, end_quarter = _parse_quarter(end)
    if (start_year, start_quarter) > (end_year, end_quarter):
        raise ValueError(f"quarter start must be <= end: {start} > {end}")
    quarters: list[str] = []
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        quarters.append(f"{year}Q{quarter}")
        quarter += 1
        if quarter > 4:
            year += 1
            quarter = 1
    return quarters


def _parse_quarter(value: str) -> tuple[int, int]:
    if len(value) != 6 or value[4] != "Q":
        raise ValueError(f"quarter must look like YYYYQn: {value}")
    year = int(value[:4])
    quarter = int(value[5])
    if quarter < 1 or quarter > 4:
        raise ValueError(f"quarter must be between Q1 and Q4: {value}")
    return year, quarter


def warsh_consistency_score(proposals: list[AgentProposal], persona: PolicyPersona | None = None) -> dict[str, Any]:
    warsh = next((proposal for proposal in reversed(proposals) if proposal.role_id == "usa_warsh"), None)
    if warsh is None:
        return {
            "score": 0.0,
            "distance": None,
            "profile": WARSH_POLICY_PROFILE.rationale,
            "note": "Warsh role proposal not available.",
        }
    if persona is not None:
        return persona.consistency_score(warsh.strategy)
    strategy = warsh.strategy.to_dict()
    priors = WARSH_POLICY_PROFILE.priors
    distances = {
        key: abs(float(strategy.get(key, 0.0)) - float(value))
        for key, value in priors.items()
    }
    avg_distance = sum(distances.values()) / max(1, len(distances))
    score = max(0.0, 1.0 - avg_distance / max(1e-6, WARSH_POLICY_PROFILE.tolerance))
    return {
        "score": round(score, 6),
        "distance": round(avg_distance, 6),
        "feature_distances": {key: round(value, 6) for key, value in distances.items()},
        "profile": WARSH_POLICY_PROFILE.rationale,
        "role_id": WARSH_POLICY_PROFILE.role_id,
    }


class RollingSelfPlayEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        paths = config.paths
        self.policy_tools = PolicyTools(RagIndex(paths["rag_index"]))
        self.personas = load_configured_personas(config)
        sp_cfg = config.raw.get("self_play", {})
        self.use_llm_equilibrium_check = bool(sp_cfg.get("llm_equilibrium_check", True))
        self.require_llm_equilibrium_for_convergence = bool(
            sp_cfg.get("require_llm_equilibrium_for_convergence", True)
        )
        # Try LLM-powered best response; fall back to rule-based
        teacher = None
        try:
            from .teacher import build_teacher_client
            teacher = build_teacher_client(config)
            if config.allow_mock_teacher and not config.teacher_api_key:
                teacher = None  # mock mode → use rule-based for reliability
        except Exception:
            pass
        self.teacher = teacher
        
        if teacher is not None:
            self.best_response = LLMBestResponseSkill(
                teacher,
                use_llm_payoff=bool(sp_cfg.get("llm_payoff_judge", True)),
            )
            print("  [Game Engine] LLM best response + payoff judge (DeepSeek teacher)")
        else:
            self.best_response = BestResponseSkill()
            print("  [Game Engine] Rule-based best response (no LLM teacher)")
            
        self.critic = CriticSkill()
        self.coordinator = CoordinatorSkill()
        self.evidence = EvidenceSkill()
        self.equilibrium_judge = LLMEquilibriumJudge(teacher if self.use_llm_equilibrium_check else None)
        self.agents = self._build_agents()
        self.memory = CrossQuarterMemory()

    def _build_agents(self) -> dict[str, PolicyAgent]:
        agents: dict[str, PolicyAgent] = {}
        for cluster_id, members in CLUSTER_MEMBERS.items():
            for role_id in members:
                agents[role_id] = PolicyAgent(
                    role=get_role(role_id),
                    memory=AgentMemory(role_id=role_id, cluster_id=cluster_id),
                    best_response_skill=self.best_response,
                )
        return agents

    def run(self, quarters: list[str] | None = None, *, max_context_docs: int = 4) -> SelfPlayResult:
        sp_cfg = self.config.raw["self_play"]
        quarters = quarters or list(sp_cfg["quarters"])
        traces = [self.run_quarter(quarter, max_context_docs=max_context_docs) for quarter in quarters]
        trace_dir = ensure_dir(self.config.paths["trace_dir"])
        trace_path = trace_dir / SELF_PLAY_TRACE_FILENAME
        with trace_path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        distill_path = ensure_parent(self.config.paths["train_dir"] / "equilibrium_distill.jsonl")
        self.write_distillation_examples(traces, distill_path)
        return SelfPlayResult(traces=traces, trace_path=trace_path, distill_path=distill_path)

    def run_quarter(
        self,
        quarter: str,
        *,
        max_context_docs: int = 4,
        briefing_override: dict[str, Any] | None = None,
        scenario_context: dict[str, Any] | None = None,
    ) -> GameTrace:
        sp_cfg = self.config.raw["self_play"]
        max_rounds = int(sp_cfg["max_rounds"])
        stable_required = int(sp_cfg["stable_rounds_required"])
        epsilon = float(sp_cfg["strategy_epsilon"])
        tau = float(sp_cfg["deviation_gain_tau"])
        previous_cluster_strategies = {cluster_id: StrategyVector() for cluster_id in CLUSTER_MEMBERS}
        stable_rounds = 0
        all_proposals: list[AgentProposal] = []
        all_critiques = []
        final_clusters: list[ClusterStrategy] = []
        max_distance = 1.0
        max_deviation_gain = 1.0
        deviation_gain_method = "pending"
        equilibrium_feedback: dict[str, Any] = {}
        equilibrium_check: dict[str, Any] = {
            "checked": False,
            "source": "pending",
            "quarter": quarter,
            "round_id": 0,
            "is_nash_equilibrium": False,
            "max_profitable_deviation_gain": 1.0,
            "profitable_deviations": [],
            "reasoning": "Self-play has not reached an equilibrium check point yet.",
            "heuristic": {},
        }
        quarter_end = _quarter_end_iso(quarter)
        var_vecm_briefing = _build_var_vecm_briefing(quarter, _MACRO_PANEL)
        if briefing_override:
            var_vecm_briefing = {**var_vecm_briefing, **briefing_override}
        scenario_context = scenario_context or {}

        for round_id in range(1, max_rounds + 1):
            round_clusters: list[ClusterStrategy] = []
            round_proposals: list[AgentProposal] = []
            for cluster_id, member_ids in CLUSTER_MEMBERS.items():
                evidence = self.policy_tools.retrieve_policy_cases(
                    query=f"{quarter} {cluster_id} inflation energy policy strategy",
                    country=cluster_id if cluster_id in {"USA", "CHN", "GBR", "RUS", "FRA"} else None,
                    before_date=quarter_end,
                    top_k=max_context_docs,
                )
                # --- INJECT VAR/VECM briefing + cross-quarter memory ---
                past_summary = self.memory.build_context_text(cluster_id, quarter)
                context = {
                    "quarter": quarter,
                    "cluster_id": cluster_id,
                    "evidence": evidence,
                    "var_vecm_briefing": var_vecm_briefing,
                    "counterfactual_scenario": scenario_context,
                    "past_strategies": past_summary,  # ← RAG memory of previous quarters
                    "previous_cluster_strategy": previous_cluster_strategies[cluster_id].to_dict(),
                    "opponent_cluster_strategies": {
                        key: value.to_dict() for key, value in previous_cluster_strategies.items() if key != cluster_id
                    },
                    "equilibrium_feedback": equilibrium_feedback,
                }
                member_proposals: list[AgentProposal] = []
                for role_id in member_ids:
                    opponents = {key: value for key, value in previous_cluster_strategies.items() if key != cluster_id}
                    role_context = dict(context)
                    if role_id in self.personas:
                        role_context["policy_persona"] = self.personas[role_id].to_prompt_payload()
                    proposal = self.agents[role_id].propose(
                        context=role_context,
                        opponent_strategies=opponents,
                        evidence=evidence,
                        round_id=round_id,
                    )
                    critique = self.critic.run(proposal)
                    member_proposals.append(proposal)
                    round_proposals.append(proposal)
                    all_critiques.append(critique.to_dict())
                strategy, dispersion, notes = self.coordinator.run(cluster_id, member_proposals, round_id)
                round_clusters.append(
                    ClusterStrategy(
                        cluster_id=cluster_id,
                        round_id=round_id,
                        strategy=strategy,
                        members=member_ids,
                        internal_dispersion=dispersion,
                        notes=notes,
                    )
                )
            round_map = {item.cluster_id: item.strategy for item in round_clusters}
            max_distance = max(
                previous_cluster_strategies[cluster_id].distance(strategy)
                for cluster_id, strategy in round_map.items()
            )
            if any(proposal.payoff_source == "deepseek_payoff_judge" for proposal in round_proposals):
                max_deviation_gain = max((proposal.regret_estimate for proposal in round_proposals), default=0.0)
                deviation_gain_method = "deepseek_role_regret"
            else:
                max_deviation_gain = max(
                    (proposal.regret_estimate * max_distance for proposal in round_proposals),
                    default=0.0,
                )
                deviation_gain_method = "fallback_regret_times_strategy_distance"
            all_proposals.extend(round_proposals)
            final_clusters = round_clusters
            previous_cluster_strategies = round_map
            stable_rounds = stable_rounds + 1 if max_distance < epsilon and max_deviation_gain < tau else 0
            heuristic = {
                "max_strategy_distance": round(max_distance, 6),
                "strategy_epsilon": epsilon,
                "max_deviation_gain": round(max_deviation_gain, 6),
                "deviation_gain_method": deviation_gain_method,
                "deviation_gain_tau": tau,
                "stable_rounds": stable_rounds,
                "stable_rounds_required": stable_required,
                "heuristic_passed": stable_rounds >= stable_required,
            }
            if stable_rounds >= stable_required:
                if self.use_llm_equilibrium_check:
                    equilibrium_check = self.equilibrium_judge.check(
                        quarter=quarter,
                        round_id=round_id,
                        cluster_strategies=previous_cluster_strategies,
                        proposals=all_proposals,
                        var_vecm_briefing=var_vecm_briefing,
                        heuristic=heuristic,
                        role_personas={
                            role_id: persona.to_prompt_payload(max_markdown_chars=1800)
                            for role_id, persona in self.personas.items()
                        },
                    )
                    max_deviation_gain = max(
                        max_deviation_gain,
                        float(equilibrium_check.get("max_profitable_deviation_gain", 0.0) or 0.0),
                    )
                    llm_required = self.require_llm_equilibrium_for_convergence and self.teacher is not None
                    llm_accepts = bool(equilibrium_check.get("checked")) and bool(
                        equilibrium_check.get("is_nash_equilibrium")
                    )
                    if llm_accepts or not llm_required:
                        break
                    equilibrium_feedback = equilibrium_check
                    stable_rounds = 0
                else:
                    equilibrium_check = {
                        "checked": False,
                        "source": "heuristic_only_disabled",
                        "quarter": quarter,
                        "round_id": round_id,
                        "is_nash_equilibrium": True,
                        "max_profitable_deviation_gain": round(max_deviation_gain, 6),
                        "profitable_deviations": [],
                        "reasoning": "LLM equilibrium judge disabled; accepted heuristic convergence.",
                        "heuristic": heuristic,
                    }
                    break

        if (
            self.use_llm_equilibrium_check
            and self.teacher is not None
            and final_clusters
            and (
                not bool(equilibrium_check.get("checked"))
                or int(equilibrium_check.get("round_id", 0) or 0) != final_clusters[-1].round_id
            )
        ):
            heuristic = {
                "max_strategy_distance": round(max_distance, 6),
                "strategy_epsilon": epsilon,
                "max_deviation_gain": round(max_deviation_gain, 6),
                "deviation_gain_method": deviation_gain_method,
                "deviation_gain_tau": tau,
                "stable_rounds": stable_rounds,
                "stable_rounds_required": stable_required,
                "heuristic_passed": stable_rounds >= stable_required,
            }
            equilibrium_check = self.equilibrium_judge.check(
                quarter=quarter,
                round_id=final_clusters[-1].round_id,
                cluster_strategies=previous_cluster_strategies,
                proposals=all_proposals,
                var_vecm_briefing=var_vecm_briefing,
                heuristic=heuristic,
                role_personas={
                    role_id: persona.to_prompt_payload(max_markdown_chars=1800)
                    for role_id, persona in self.personas.items()
                },
            )
            max_deviation_gain = max(
                max_deviation_gain,
                float(equilibrium_check.get("max_profitable_deviation_gain", 0.0) or 0.0),
            )

        shocks = self.policy_tools.estimate_external_shocks(previous_cluster_strategies)
        shock_variables = self.policy_tools.estimate_external_shock_variables(
            previous_cluster_strategies,
            shocks=shocks,
        )
        usa_strategy = previous_cluster_strategies.get("USA", StrategyVector())
        usa_belief = next((proposal.belief for proposal in reversed(all_proposals) if proposal.cluster_id == "USA"), None)
        if usa_belief is None:
            usa_belief = all_proposals[-1].belief
        fed_path = self.policy_tools.estimate_fomc_path(usa_strategy, usa_belief)
        evidence_chain = self.evidence.run(fed_path, shocks, [p for p in all_proposals if p.cluster_id == "USA"])
        warsh_score = warsh_consistency_score(all_proposals, persona=self.personas.get("usa_warsh"))
        
        # --- Record in cross-quarter memory for future quarters ---
        self.memory.record(quarter, [c.to_dict() for c in final_clusters], fed_path)
        llm_required = (
            self.use_llm_equilibrium_check
            and self.require_llm_equilibrium_for_convergence
            and self.teacher is not None
        )
        converged = stable_rounds >= stable_required
        if llm_required:
            converged = converged and bool(equilibrium_check.get("checked")) and bool(
                equilibrium_check.get("is_nash_equilibrium")
            )
        
        return GameTrace(
            quarter=quarter,
            converged=converged,
            rounds=final_clusters[-1].round_id if final_clusters else 0,
            max_strategy_distance=round(max_distance, 6),
            max_deviation_gain=round(max_deviation_gain, 6),
            cluster_strategies=[cluster.to_dict() for cluster in final_clusters],
            shock_variables=shock_variables,
            proposals=[proposal.to_dict() for proposal in all_proposals],
            critiques=all_critiques,
            fed_prediction=fed_path,
            evidence_chain=evidence_chain,
            equilibrium_check=equilibrium_check,
            personas_used={
                role_id: {
                    "persona_id": persona.persona_id,
                    "name": persona.name,
                    "skill_path": persona.skill_path,
                    "research_cutoff": persona.research_cutoff,
                    "mental_models": [item.name for item in persona.mental_models],
                    "evidence_source_count": len(persona.evidence_sources),
                }
                for role_id, persona in self.personas.items()
            },
            warsh_consistency_score=warsh_score,
        )

    @staticmethod
    def write_distillation_examples(traces: list[GameTrace], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                final_by_cluster = {
                    row["cluster_id"]: row["strategy"] for row in trace.cluster_strategies
                }
                for proposal in trace.proposals:
                    target = final_by_cluster.get(proposal["cluster_id"], proposal["strategy"])
                    example = {
                        "task": "equilibrium_distillation",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Learn to move a macro policy agent toward the quarter's approximate empirical equilibrium.",
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "quarter": trace.quarter,
                                        "role_id": proposal["role_id"],
                                        "round_id": proposal["round_id"],
                                        "current_strategy": proposal["strategy"],
                                        "belief": proposal["belief"],
                                        "policy_cost": proposal["policy_cost"],
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "equilibrium_strategy": target,
                                        "fed_prediction": trace.fed_prediction,
                                        "converged": trace.converged,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        "metadata": {
                            "quarter": trace.quarter,
                            "role_id": proposal["role_id"],
                            "cluster_id": proposal["cluster_id"],
                        },
                    }
                    fh.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
        return count

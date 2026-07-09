from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .data_sources import RagIndex
from .schemas import BeliefState, StrategyVector


ToolFn = Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)


@dataclass
class PolicyTools:
    rag_index: RagIndex

    def retrieve_policy_cases(
        self,
        query: str,
        country: str | None = None,
        before_date: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return [
            record.to_dict()
            for record in self.rag_index.search(query, country=country, before_date=before_date, top_k=top_k)
        ]

    def build_macro_snapshot(self, raw_context: dict[str, Any]) -> dict[str, Any]:
        macro = raw_context.get("macro_context", {})
        if isinstance(macro, str):
            try:
                macro = json.loads(macro)
            except json.JSONDecodeError:
                macro = {"raw": macro[:2000]}
        belief = BeliefState.from_context(macro)
        return {
            "macro_context": macro,
            "belief": belief.to_dict(),
        }

    def estimate_fomc_path(self, cluster_strategy: StrategyVector, belief: BeliefState) -> dict[str, float]:
        strategy = cluster_strategy.normalized()
        inflation = belief.inflation_persistence
        labor = belief.labor_softening
        energy = belief.energy_shock_risk
        credibility = belief.policy_credibility_risk
        hike = (
            0.02
            + 0.55 * strategy.rate_hike_25bp_prob
            + 0.30 * inflation
            + 0.10 * credibility
            + 0.08 * energy
            - 0.10 * labor
        )
        cut = 0.02 + 0.35 * strategy.easing_signal_prob + 0.30 * labor - 0.25 * inflation - 0.10 * credibility
        hike = max(0.0, min(0.95, hike))
        cut = max(0.0, min(0.60, cut))
        hold = max(0.0, 1.0 - hike - cut)
        total = hike + hold + cut
        return {
            "hike_25bp": round(hike / total, 4),
            "hold": round(hold / total, 4),
            "cut_25bp": round(cut / total, 4),
        }

    def estimate_external_shocks(self, cluster_strategies: dict[str, StrategyVector]) -> dict[str, float]:
        values = {key: strategy.normalized() for key, strategy in cluster_strategies.items()}
        geopol = max((strategy.trade_or_sanction_pressure_prob for strategy in values.values()), default=0.2)
        energy = max(values.get("RUS", StrategyVector()).trade_or_sanction_pressure_prob, geopol * 0.8)
        dollar = max(values.get("CHN", StrategyVector()).liquidity_support_prob, values.get("USA", StrategyVector()).hawkish_signal_prob * 0.4)
        return {
            "geopol_escalation_prob": round(geopol, 4),
            "energy_price_risk": round(energy, 4),
            "tariff_risk": round(max(0.1, geopol * 0.75), 4),
            "supply_chain_risk": round(max(0.1, geopol * 0.65), 4),
            "dollar_liquidity_pressure": round(dollar, 4),
            "term_premium_pressure": round(values.get("USA", StrategyVector()).hawkish_signal_prob * 0.55 + energy * 0.25, 4),
        }

    def estimate_external_shock_variables(
        self,
        cluster_strategies: dict[str, StrategyVector],
        shocks: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert the strategic game state into explicit macro shock variables.

        The policy game is not the final Fed predictor. It is an external
        strategic shock layer: each variable is a bounded risk signal with
        source clusters and a channel into the Fed decision problem.
        """
        shocks = shocks or self.estimate_external_shocks(cluster_strategies)
        strategies = {key: value.normalized().to_dict() for key, value in cluster_strategies.items()}

        def row(
            name: str,
            source_clusters: list[str],
            fed_channel: str,
            mechanism: str,
        ) -> dict[str, Any]:
            return {
                "name": name,
                "value": float(shocks.get(name, 0.0)),
                "source_clusters": source_clusters,
                "fed_channel": fed_channel,
                "mechanism": mechanism,
                "source_strategy": {
                    cluster: strategies.get(cluster, {})
                    for cluster in source_clusters
                    if cluster in strategies
                },
            }

        return [
            row(
                "energy_price_risk",
                ["RUS", "USA"],
                "inflation_upside",
                "sanction or supply pressure raises energy prices and inflation persistence",
            ),
            row(
                "geopol_escalation_prob",
                ["RUS", "CHN", "USA", "GBR", "FRA"],
                "risk_premium",
                "strategic confrontation raises geopolitical and safe-haven risk premia",
            ),
            row(
                "tariff_risk",
                ["CHN", "USA"],
                "goods_inflation",
                "trade restrictions raise import prices and supply friction",
            ),
            row(
                "supply_chain_risk",
                ["CHN", "RUS", "USA"],
                "growth_inflation_mix",
                "supply disruptions weaken growth while lifting selected prices",
            ),
            row(
                "dollar_liquidity_pressure",
                ["USA", "CHN"],
                "financial_conditions",
                "dollar funding pressure tightens financial conditions outside the policy rate",
            ),
            row(
                "term_premium_pressure",
                ["USA", "RUS"],
                "long_rate_channel",
                "hawkish credibility and external risk lift term premia",
            ),
        ]


def build_tool_registry(policy_tools: PolicyTools) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("retrieve_policy_cases", policy_tools.retrieve_policy_cases)
    registry.register("build_macro_snapshot", policy_tools.build_macro_snapshot)
    registry.register("estimate_fomc_path", policy_tools.estimate_fomc_path)
    registry.register("estimate_external_shocks", policy_tools.estimate_external_shocks)
    registry.register("estimate_external_shock_variables", policy_tools.estimate_external_shock_variables)
    return registry

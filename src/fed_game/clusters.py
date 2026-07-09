from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleCard:
    role_id: str
    cluster_id: str
    name: str
    objective: str
    constraints: list[str]
    preferred_signals: list[str]
    persona_id: str | None = None
    persona_path: str | None = None


@dataclass(frozen=True)
class WarshPolicyProfile:
    role_id: str
    priors: dict[str, float]
    rationale: dict[str, str]
    tolerance: float = 0.25


ROLE_CARDS: dict[str, RoleCard] = {
    "usa_warsh": RoleCard(
        "usa_warsh",
        "USA",
        "Fed Chair Warsh",
        "Protect inflation-fighting credibility, Fed independence, and meeting-by-meeting policy optionality.",
        [
            "FOMC consensus",
            "central bank credibility",
            "market communication cost",
            "monetary-fiscal boundary",
            "balance-sheet exit discipline",
        ],
        [
            "data dependence",
            "reduced forward guidance",
            "anti-inflation credibility",
            "mission-boundary discipline",
            "temporary liquidity support over broad easing",
        ],
        "kevin-warsh-policy-persona",
        "policy_personas/warsh_policy/SKILL.md",
    ),
    "usa_fomc_hawk": RoleCard(
        "usa_fomc_hawk",
        "USA",
        "FOMC Hawk",
        "Prevent inflation expectations from becoming unanchored.",
        ["growth slowdown risk", "financial stability risk"],
        ["higher-for-longer", "possible hike", "tight financial conditions"],
    ),
    "usa_fomc_dove": RoleCard(
        "usa_fomc_dove",
        "USA",
        "FOMC Dove",
        "Avoid unnecessary labor-market damage and excessive real-rate tightening.",
        ["inflation still above target", "credibility risk"],
        ["patience", "hold", "watch labor softening"],
    ),
    "usa_treasury": RoleCard(
        "usa_treasury",
        "USA",
        "US Treasury",
        "Maintain debt-market functioning and fiscal sustainability.",
        ["term premium", "auction demand", "political pressure"],
        ["liquidity stability", "term-premium containment"],
    ),
    "usa_white_house": RoleCard(
        "usa_white_house",
        "USA",
        "White House Pressure",
        "Support growth and political approval without openly compromising Fed independence.",
        ["Fed independence", "inflation voter salience"],
        ["growth risk", "household costs", "employment"],
    ),
    "usa_market_stability": RoleCard(
        "usa_market_stability",
        "USA",
        "Market Stability",
        "Avoid disorderly credit, rates, and equity-market repricing.",
        ["credit spreads", "liquidity", "volatility"],
        ["financial conditions", "risk premia", "liquidity support"],
    ),
    "chn_pboc": RoleCard(
        "chn_pboc",
        "CHN",
        "PBOC",
        "Balance growth support, currency stability, and capital-flow control.",
        ["property stress", "CNY pressure", "external demand"],
        ["targeted easing", "FX stability", "liquidity support"],
    ),
    "chn_trade": RoleCard(
        "chn_trade",
        "CHN",
        "China Trade Policy",
        "Protect supply chains and export competitiveness.",
        ["tariff risk", "external retaliation", "domestic employment"],
        ["trade defense", "supply-chain resilience"],
    ),
    "rus_energy": RoleCard(
        "rus_energy",
        "RUS",
        "Russia Energy-Fiscal",
        "Preserve energy revenue and strategic leverage under sanctions.",
        ["sanctions", "oil price", "fiscal pressure"],
        ["energy supply risk", "sanction response"],
    ),
    "gbr_boe": RoleCard(
        "gbr_boe",
        "GBR",
        "Bank of England",
        "Balance inflation persistence against weak growth and currency pressure.",
        ["sterling", "mortgage channel", "growth"],
        ["inflation persistence", "policy caution"],
    ),
    "fra_euro": RoleCard(
        "fra_euro",
        "FRA",
        "France / Euro Area Policy",
        "Coordinate euro-area price stability, fiscal stress, and energy exposure.",
        ["ECB reaction function", "energy import exposure", "fiscal rules"],
        ["energy risk", "euro financial conditions"],
    ),
}


WARSH_POLICY_PROFILE = WarshPolicyProfile(
    role_id="usa_warsh",
    priors={
        "hawkish_signal_prob": 0.76,
        "rate_hike_25bp_prob": 0.38,
        "hold_with_hawkish_statement_prob": 0.74,
        "remove_forward_guidance_prob": 0.72,
        "easing_signal_prob": 0.08,
        "liquidity_support_prob": 0.20,
        "trade_or_sanction_pressure_prob": 0.25,
    },
    rationale={
        "inflation_credibility": "credibility is treated as the real policy multiplier",
        "fed_independence": "protects monetary policy from fiscal or political dominance",
        "forward_guidance_skepticism": "prefers flexibility over strong calendar guidance",
        "exit_discipline": "treats crisis tools and balance-sheet expansion as temporary",
        "communication_style": "leans institutional and hawkish when inflation risk persists",
    },
)


CLUSTER_MEMBERS: dict[str, list[str]] = {
    "USA": [
        "usa_warsh",
        "usa_fomc_hawk",
        "usa_fomc_dove",
        "usa_treasury",
        "usa_white_house",
        "usa_market_stability",
    ],
    "CHN": ["chn_pboc", "chn_trade"],
    "RUS": ["rus_energy"],
    "GBR": ["gbr_boe"],
    "FRA": ["fra_euro"],
}


def get_role(role_id: str) -> RoleCard:
    return ROLE_CARDS[role_id]

from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import ensure_parent, load_registry


PRECONDITIONS = {
    "tightening": [
        "inflation_above_target",
        "credibility_risk_high",
        "financial_conditions_too_loose",
    ],
    "easing": [
        "growth_risk_high",
        "labor_market_softening",
        "financial_stability_stress",
    ],
    "hold": [
        "mixed_macro_signals",
        "wait_for_more_data",
        "policy_lag_uncertainty",
    ],
    "communication": [
        "policy_path_uncertain",
        "market_expectations_need_management",
        "institutional_reaction_function_update",
    ],
}

CHANNELS = {
    "tightening": [
        "tighten_financial_conditions",
        "anchor_inflation_expectations",
        "support_currency",
        "raise_growth_risk",
    ],
    "easing": [
        "ease_financial_conditions",
        "support_growth_and_employment",
        "raise_inflation_expectation_risk",
        "pressure_currency",
    ],
    "hold": [
        "preserve_policy_option_value",
        "wait_for_macro_confirmation",
        "maintain_current_financial_conditions",
    ],
    "communication": [
        "shape_market_expectations",
        "signal_reaction_function",
        "reduce_or_increase_forward_guidance",
    ],
}


def build_strategy_cards() -> list[dict[str, object]]:
    registry = load_registry()
    events = pd.read_parquet(registry["outputs"]["policy_events"])
    cards: list[dict[str, object]] = []
    for strategy_key, group in events.groupby("strategy_key", dropna=False):
        stance = str(group["stance"].mode().iloc[0]) if not group["stance"].empty else "communication"
        instrument = str(group["instrument"].mode().iloc[0]) if not group["instrument"].empty else "communication"
        action = str(group["action"].mode().iloc[0]) if not group["action"].empty else "signal"
        examples = (
            group.sort_values("date")
            .tail(8)[["event_id", "date", "country", "actor", "title", "url"]]
            .to_dict(orient="records")
        )
        card = {
            "strategy_id": str(strategy_key).replace(".", "_"),
            "strategy_key": strategy_key,
            "strategy_name": " ".join(part.capitalize() for part in str(strategy_key).split(".")),
            "actor_type": "central_bank_or_financial_policy_authority",
            "instrument": instrument,
            "action": action,
            "stance": stance,
            "preconditions": PRECONDITIONS.get(stance, PRECONDITIONS["communication"]),
            "context_required": [
                "own_previous_strategy_sequence",
                "other_p4_previous_strategy_sequence",
                "policy_text_signals",
                "weo_macro_factors",
                "wdi_development_factors_top100",
                "market_expectations",
                "energy_trade_geopolitical_shocks",
            ],
            "expected_channels": CHANNELS.get(stance, CHANNELS["communication"]),
            "selection_prompt": (
                "Select this strategy only when the as-of context supports its preconditions. "
                "Use RAG evidence to cite the exact policy text and panel context used."
            ),
            "historical_examples": examples,
            "support_count": int(len(group)),
            "countries_observed": sorted(str(item) for item in group["country"].dropna().unique()),
        }
        cards.append(card)

    out_path = ensure_parent(registry["outputs"]["strategy_cards"])
    with out_path.open("w", encoding="utf-8") as fh:
        for card in sorted(cards, key=lambda item: (-int(item["support_count"]), str(item["strategy_id"]))):
            fh.write(json.dumps(card, ensure_ascii=False) + "\n")
    return cards


def main() -> None:
    argparse.ArgumentParser(description="Build strategy cards from normalized policy events.").parse_args()
    cards = build_strategy_cards()
    print(f"Wrote strategy cards: {len(cards)}")


if __name__ == "__main__":
    main()


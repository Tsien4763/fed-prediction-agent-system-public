from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import ensure_parent, load_registry


def _safe_date(value: object) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def _latest_values_by_country_year(
    path: str,
    countries: list[str],
    start_year: int,
    end_year: int,
    code_col: str = "indicator_code",
) -> dict[str, dict[int, dict[str, dict[str, object]]]]:
    empty = {country: {year: {} for year in range(start_year, end_year + 1)} for country in countries}
    panel_path = Path(path)
    if not panel_path.exists():
        return empty

    panel = pd.read_parquet(panel_path)
    required = {"country", code_col, "year", "value"}
    if panel.empty or not required.issubset(panel.columns):
        return empty

    panel = panel[["country", code_col, "year", "value"]].copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["year", "value"])
    panel["year"] = panel["year"].astype(int)

    result = empty
    for country in countries:
        country_panel = panel[panel["country"] == country].sort_values([code_col, "year"])
        for year in range(start_year, end_year + 1):
            eligible = country_panel[country_panel["year"] <= year]
            if eligible.empty:
                continue
            latest = eligible.groupby(code_col, as_index=False).tail(1)
            result[country][year] = {
                str(row[code_col]): {"value": float(row["value"]), "year": int(row["year"])}
                for _, row in latest.iterrows()
                if not pd.isna(row["value"])
            }
    return result


def build_context_snapshots() -> pd.DataFrame:
    registry = load_registry()
    events = pd.read_parquet(registry["outputs"]["policy_events"])
    events["date_ts"] = events["date"].map(_safe_date)
    events = events.dropna(subset=["date_ts"]).copy()
    events = events.sort_values(["date_ts", "country", "actor", "event_id"])
    countries = [item["code"] for item in registry["project"]["countries"]]
    country_set = set(countries)
    start_year = int(registry["project"]["date_range"]["start_year"])
    end_year = int(registry["project"]["date_range"]["end_year"])
    wdi_by_country_year = _latest_values_by_country_year(
        registry["outputs"].get("wdi_top100_panel", ""),
        countries,
        start_year,
        end_year,
    )
    weo_by_country_year = _latest_values_by_country_year(
        registry["outputs"].get("weo_panel", ""),
        countries,
        start_year,
        end_year,
    )

    snapshots = []
    history_by_country: dict[str, list[dict[str, object]]] = defaultdict(list)
    all_history: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        as_of = row.date_ts
        own_prior = history_by_country[str(row.country)][-12:]
        other_prior = [
            item for item in reversed(all_history) if item["country"] in country_set and item["country"] != row.country
        ][:20]
        other_prior = list(reversed(other_prior))
        as_of_year = min(max(int(as_of.year), start_year), end_year)
        snapshots.append(
            {
                "snapshot_id": f"{row.event_id}_ctx",
                "as_of_date": as_of.date().isoformat(),
                "target_event_id": row.event_id,
                "target_country": row.country,
                "target_actor": row.actor,
                "target_strategy_key": row.strategy_key,
                "own_previous_strategy_sequence": json.dumps(
                    [
                        {
                            "date": item["date"],
                            "actor": item["actor"],
                            "strategy_key": item["strategy_key"],
                            "title": item["title"],
                        }
                        for item in own_prior
                    ],
                    ensure_ascii=False,
                ),
                "other_p4_previous_strategy_sequence": json.dumps(
                    [
                        {
                            "date": item["date"],
                            "country": item["country"],
                            "actor": item["actor"],
                            "strategy_key": item["strategy_key"],
                            "title": item["title"],
                        }
                        for item in other_prior
                    ],
                    ensure_ascii=False,
                ),
                "macro_context": json.dumps(
                    {
                        "as_of_year": as_of_year,
                        "target_country": row.country,
                        "wdi_top100": wdi_by_country_year.get(str(row.country), {}).get(as_of_year, {}),
                        "weo": weo_by_country_year.get(str(row.country), {}).get(as_of_year, {}),
                        "note": "MVP yearly latest-observation join; exact release-lag calendar is not modeled yet.",
                    },
                    ensure_ascii=False,
                ),
                "rag_text": " | ".join(
                    [
                        str(row.date),
                        str(row.country),
                        str(row.actor),
                        str(row.strategy_key),
                        str(row.title),
                    ]
                ),
            }
        )
        history_item = {
            "date": row.date,
            "country": row.country,
            "actor": row.actor,
            "strategy_key": row.strategy_key,
            "title": row.title,
        }
        history_by_country[str(row.country)].append(history_item)
        all_history.append(history_item)
    df = pd.DataFrame(snapshots)
    out_path = ensure_parent(registry["outputs"]["context_snapshots"])
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    argparse.ArgumentParser(description="Build as-of context snapshots for policy events.").parse_args()
    df = build_context_snapshots()
    print(f"Wrote context snapshots: {len(df)}")


if __name__ == "__main__":
    main()

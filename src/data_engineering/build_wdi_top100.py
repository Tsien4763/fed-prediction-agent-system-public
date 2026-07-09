from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from .config import ensure_parent, load_registry


ECONOMIC_KEYWORDS = {
    "gdp": 1.0,
    "growth": 1.0,
    "inflation": 1.0,
    "consumer price": 1.0,
    "unemployment": 0.95,
    "employment": 0.8,
    "labor": 0.8,
    "interest": 0.9,
    "exchange rate": 0.9,
    "credit": 0.9,
    "debt": 0.9,
    "fiscal": 0.85,
    "tax": 0.75,
    "current account": 0.9,
    "trade": 0.85,
    "export": 0.8,
    "import": 0.8,
    "investment": 0.85,
    "savings": 0.8,
    "energy": 0.75,
    "oil": 0.75,
    "industry": 0.7,
    "manufacturing": 0.7,
    "population": 0.65,
    "urban": 0.55,
    "internet": 0.5,
    "education": 0.45,
}


def economic_relevance(indicator_name: str, indicator_code: str, seed_indicators: set[str]) -> float:
    if indicator_code in seed_indicators:
        return 1.0
    lowered = indicator_name.lower()
    score = 0.0
    for keyword, value in ECONOMIC_KEYWORDS.items():
        if keyword in lowered:
            score = max(score, value)
    return score


def build_wdi_top100() -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = load_registry()
    source = next(item for item in registry["bulk_downloads"] if item["id"] == "wdi_bulk_csv")
    zip_path = Path(source["target_path"])
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing WDI ZIP: {zip_path}")

    countries = registry["wdi_indicator_selection"]["countries"]
    start_year = registry["project"]["date_range"]["start_year"]
    end_year = min(registry["project"]["date_range"]["end_year"], 2025)
    years = [str(year) for year in range(start_year, end_year + 1)]
    seed_indicators = set(registry["wdi_indicator_selection"].get("seed_indicators", []))
    weights = registry["wdi_indicator_selection"]["score_weights"]

    chunks = []
    usecols = ["Country Code", "Indicator Name", "Indicator Code", *years]
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("WDICSV.csv") as fh:
            reader = pd.read_csv(fh, usecols=usecols, chunksize=50000, low_memory=False)
            for chunk in reader:
                subset = chunk[chunk["Country Code"].isin(countries)]
                if not subset.empty:
                    chunks.append(subset)
    if not chunks:
        raise ValueError("No WDI rows found for configured countries")
    panel_wide = pd.concat(chunks, ignore_index=True)

    metrics = []
    total_cells = len(countries) * len(years)
    for (code, name), group in panel_wide.groupby(["Indicator Code", "Indicator Name"], dropna=False):
        values = group[years]
        non_null = int(values.notna().sum().sum())
        countries_with_data = int(group.set_index("Country Code")[years].notna().any(axis=1).sum())
        country_coverage = countries_with_data / len(countries)
        time_coverage = non_null / total_cells
        relevance = economic_relevance(str(name), str(code), seed_indicators)
        non_redundancy = 1.0
        predictive_gain = 0.0
        score = (
            weights["economic_relevance"] * relevance
            + weights["country_coverage"] * country_coverage
            + weights["time_coverage"] * time_coverage
            + weights["non_redundancy"] * non_redundancy
            + weights["policy_predictive_gain"] * predictive_gain
        )
        metrics.append(
            {
                "indicator_code": code,
                "indicator_name": name,
                "score": round(float(score), 6),
                "economic_relevance": round(float(relevance), 6),
                "country_coverage": round(float(country_coverage), 6),
                "time_coverage": round(float(time_coverage), 6),
                "non_redundancy": non_redundancy,
                "policy_predictive_gain": predictive_gain,
                "countries_with_data": countries_with_data,
                "non_null_cells": non_null,
                "start_year": start_year,
                "end_year": end_year,
            }
        )
    metrics_df = pd.DataFrame(metrics).sort_values(
        ["score", "country_coverage", "time_coverage", "indicator_code"],
        ascending=[False, False, False, True],
    )
    top_n = int(registry["wdi_indicator_selection"]["top_n"])
    top_indicators = metrics_df.head(top_n).copy()

    top_codes = set(top_indicators["indicator_code"])
    top_panel = panel_wide[panel_wide["Indicator Code"].isin(top_codes)].copy()
    top_panel = top_panel.melt(
        id_vars=["Country Code", "Indicator Name", "Indicator Code"],
        value_vars=years,
        var_name="year",
        value_name="value",
    )
    top_panel = top_panel.rename(
        columns={
            "Country Code": "country",
            "Indicator Name": "indicator_name",
            "Indicator Code": "indicator_code",
        }
    )
    top_panel["year"] = top_panel["year"].astype(int)
    top_panel = top_panel.sort_values(["country", "indicator_code", "year"])

    indicators_path = ensure_parent(registry["outputs"]["wdi_top100_indicators"])
    panel_path = ensure_parent(registry["outputs"]["wdi_top100_panel"])
    top_indicators.to_csv(indicators_path, index=False)
    top_panel.to_parquet(panel_path, index=False)
    return top_indicators, top_panel


def main() -> None:
    argparse.ArgumentParser(description="Select top 100 WDI development indicators for five powers.").parse_args()
    indicators, panel = build_wdi_top100()
    print(f"Wrote WDI top indicators: {len(indicators)}")
    print(f"Wrote WDI panel rows: {len(panel)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import pandas as pd

from .config import ensure_parent, load_registry
from .http_utils import fetch_bytes


DATA360_BASE_URL = "https://data360api.worldbank.org/data360/data"


def _indicator_id(indicator: str) -> str:
    return indicator if indicator.startswith("IMF_WEO_") else f"IMF_WEO_{indicator}"


def fetch_data360_records(
    countries: list[str],
    indicators: list[str],
    start_year: int,
    end_year: int,
    sleep_sec: float = 0.15,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for country in countries:
        for indicator in indicators:
            params = {
                "DATABASE_ID": "IMF_WEO",
                "INDICATOR": _indicator_id(indicator),
                "REF_AREA": country,
                "timePeriodFrom": str(start_year),
                "timePeriodTo": str(end_year),
                "skip": "0",
            }
            url = f"{DATA360_BASE_URL}?{urllib.parse.urlencode(params)}"
            payload = json.loads(fetch_bytes(url, timeout=120).decode("utf-8"))
            values = payload.get("value", [])
            records.extend(values)
            time.sleep(sleep_sec)
    return records


def build_weo_panel() -> pd.DataFrame:
    registry = load_registry()
    countries = [item["code"] for item in registry["project"]["countries"]]
    indicators = registry["weo_indicators"]
    start_year = registry["project"]["date_range"]["start_year"]
    end_year = registry["project"]["date_range"]["end_year"]

    records = fetch_data360_records(countries, indicators, start_year, end_year)

    raw_path = ensure_parent("raw/weo/weo_data360_panel.json")
    raw_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    if not records:
        df = pd.DataFrame(
            columns=[
                "country",
                "indicator_code",
                "year",
                "value",
                "unit_measure",
                "obs_status",
                "source",
            ]
        )
    else:
        df = pd.DataFrame(records)
        df = df.rename(
            columns={
                "REF_AREA": "country",
                "INDICATOR": "indicator_code",
                "TIME_PERIOD": "year",
                "OBS_VALUE": "value",
                "UNIT_MEASURE": "unit_measure",
                "OBS_STATUS": "obs_status",
            }
        )
        keep = ["country", "indicator_code", "year", "value", "unit_measure", "obs_status"]
        df = df[keep].copy()
        df["indicator_code"] = df["indicator_code"].str.replace("IMF_WEO_", "", regex=False)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["source"] = "World Bank Data360 IMF_WEO"
        df = df.sort_values(["country", "indicator_code", "year"])

    out_path = ensure_parent(registry["outputs"]["weo_panel"])
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    argparse.ArgumentParser(description="Build five-power WEO panel through Data360 IMF_WEO API.").parse_args()
    df = build_weo_panel()
    print(f"Wrote WEO panel rows: {len(df)}")


if __name__ == "__main__":
    main()

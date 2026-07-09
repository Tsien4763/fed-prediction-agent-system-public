"""Build US quarterly macro panel from FRED downloads + WEO data.

Produces data/processed/us_macro_panel.parquet with these columns:
  date (quarter-end), fedfunds, inflation_cpi_yoy, inflation_pce_yoy,
  gdp_growth_qoq_ann, unemployment, industrial_prod_yoy,
  m2_yoy, gs10, term_spread_10y2y, usd_index_yoy,
  breakeven_10y, hy_spread, vix, payrolls_yoy, housing_starts_yoy

Usage:
    python -m data_engineering.build_us_macro_panel
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, ensure_parent, load_registry


def _load_fred_series(series_id: str) -> pd.Series:
    """Load a FRED JSON file and return a pandas Series with datetime index."""
    fred_dir = REPO_ROOT / "raw" / "market" / "fred"
    path = fred_dir / f"{series_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"FRED series not downloaded: {path}. Run download_fred_macro first.")

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("observations", [])
    dates, values = [], []
    for r in records:
        val = r.get("value", ".")
        if val == ".":
            continue
        try:
            dates.append(pd.Timestamp(r["date"]))
            values.append(float(val))
        except (ValueError, KeyError):
            continue
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id).sort_index()


def _quarterly_from_monthly(series: pd.Series, agg: str = "mean") -> pd.Series:
    """Aggregate monthly series to quarterly (end-of-quarter)."""
    quarterly = series.resample("QE").agg(agg)
    quarterly.index = quarterly.index.to_period("Q").to_timestamp(how="end")
    quarterly.name = series.name
    return quarterly


def _to_quarter_end(series: pd.Series) -> pd.Series:
    """Normalize any datetime index to quarter-end dates."""
    series = series.copy()
    series.index = series.index.to_period("Q").to_timestamp(how="end")
    return series


def _monthly_to_quarterly(series: pd.Series, agg: str = "mean") -> pd.Series:
    """Aggregate monthly series to quarterly (quarter-end dates)."""
    quarterly = series.resample("QE").agg(agg)
    quarterly.index = quarterly.index.to_period("Q").to_timestamp(how="end")
    quarterly.name = series.name
    return quarterly


def _quarterly_yoy_pct(series: pd.Series) -> pd.Series:
    """Year-over-year % change for quarterly series (4 lags = 1 year)."""
    result = series.pct_change(periods=4) * 100
    result.name = series.name
    return result


def _qoq_annualized_pct(series: pd.Series) -> pd.Series:
    """Quarter-over-quarter annualized % change."""
    result = ((series / series.shift(1)) ** 4 - 1) * 100
    result.name = series.name
    return result


def build_us_macro_panel() -> pd.DataFrame:
    registry = load_registry()
    start_year = registry["project"]["date_range"]["start_year"]
    end_year = registry["project"]["date_range"]["end_year"]

    # --- Load all FRED series ---
    print("Loading FRED series...")
    series_map: dict[str, pd.Series] = {}
    for item in registry.get("fred_macro_series", []):
        sid = item["id"]
        try:
            series_map[sid] = _load_fred_series(sid)
            print(f"  OK {sid}")
        except FileNotFoundError:
            print(f"  SKIP {sid} not downloaded")
        except Exception as exc:
            print(f"  ERROR {sid}: {exc}")

    # --- Build quarterly panel ---
    print("\nBuilding quarterly panel...")

    # Policy rate: monthly → quarterly (last value)
    fedfunds_m = series_map.get("FEDFUNDS")
    fedfunds_q = _monthly_to_quarterly(fedfunds_m, agg="last") if fedfunds_m is not None else None
    if fedfunds_q is not None:
        fedfunds_q.name = "fedfunds"

    # CPI: monthly price level → quarterly (mean) → YoY % (4Q lag)
    cpi_m = series_map.get("CPIAUCSL")
    if cpi_m is not None:
        cpi_q = _monthly_to_quarterly(cpi_m, agg="mean")
        cpi_yoy = _quarterly_yoy_pct(cpi_q)
        cpi_yoy.name = "inflation_cpi_yoy"
    else:
        cpi_yoy = None

    # PCE: monthly price level → quarterly (mean) → YoY % (4Q lag)
    pce_m = series_map.get("PCEPI")
    if pce_m is not None:
        pce_q = _monthly_to_quarterly(pce_m, agg="mean")
        pce_yoy = _quarterly_yoy_pct(pce_q)
        pce_yoy.name = "inflation_pce_yoy"
    else:
        pce_yoy = None

    # Real GDP: already quarterly → normalize to quarter-end → QoQ annualized % growth
    gdp_q = series_map.get("GDPC1")
    if gdp_q is not None:
        gdp_qe = _to_quarter_end(gdp_q)
        gdp_growth = _qoq_annualized_pct(gdp_qe)
        gdp_growth.name = "gdp_growth_qoq_ann"
    else:
        gdp_growth = None

    # Unemployment: monthly → quarterly (mean)
    unrate_m = series_map.get("UNRATE")
    if unrate_m is not None:
        unrate_q = _monthly_to_quarterly(unrate_m, agg="mean")
        unrate_q.name = "unemployment"
    else:
        unrate_q = None

    # Industrial production: monthly → quarterly (mean) → YoY %
    ip_m = series_map.get("INDPRO")
    if ip_m is not None:
        ip_q = _monthly_to_quarterly(ip_m, agg="mean")
        ip_yoy = _quarterly_yoy_pct(ip_q)
        ip_yoy.name = "industrial_prod_yoy"
    else:
        ip_yoy = None

    # Payrolls: monthly → quarterly (mean) → YoY %
    payrolls_m = series_map.get("PAYEMS")
    if payrolls_m is not None:
        pay_q = _monthly_to_quarterly(payrolls_m, agg="mean")
        payrolls_yoy = _quarterly_yoy_pct(pay_q)
        payrolls_yoy.name = "payrolls_yoy"
    else:
        payrolls_yoy = None

    # Housing starts: monthly → quarterly (mean) → YoY %
    housing_m = series_map.get("HOUST")
    if housing_m is not None:
        hous_q = _monthly_to_quarterly(housing_m, agg="mean")
        housing_yoy = _quarterly_yoy_pct(hous_q)
        housing_yoy.name = "housing_starts_yoy"
    else:
        housing_yoy = None

    # M2: monthly → quarterly (mean) → YoY %
    m2_m = series_map.get("M2SL")
    if m2_m is not None:
        m2_q = _monthly_to_quarterly(m2_m, agg="mean")
        m2_yoy = _quarterly_yoy_pct(m2_q)
        m2_yoy.name = "m2_yoy"
    else:
        m2_yoy = None

    # 10Y yield: monthly → quarterly (last)
    gs10_m = series_map.get("GS10")
    if gs10_m is not None:
        gs10_q = _monthly_to_quarterly(gs10_m, agg="last")
        gs10_q.name = "gs10"
    else:
        gs10_q = None

    # Term spread: daily → quarterly (mean)
    spread_d = series_map.get("T10Y2Y")
    if spread_d is not None:
        spread_q = _monthly_to_quarterly(spread_d, agg="mean")
        spread_q.name = "term_spread_10y2y"
    else:
        spread_q = None

    # USD index: daily → quarterly (last) → YoY %
    usd_d = series_map.get("DTWEXBGS")
    if usd_d is not None:
        usd_q = _monthly_to_quarterly(usd_d, agg="last")
        usd_yoy = _quarterly_yoy_pct(usd_q)
        usd_yoy.name = "usd_index_yoy"
    else:
        usd_yoy = None

    # Breakeven 10Y: daily → quarterly (mean)
    be_d = series_map.get("T10YIE")
    if be_d is not None:
        be_q = _monthly_to_quarterly(be_d, agg="mean")
        be_q.name = "breakeven_10y"
    else:
        be_q = None

    # HY spread: daily → quarterly (mean)
    hy_d = series_map.get("BAMLH0A0HYM2")
    if hy_d is not None:
        hy_q = _monthly_to_quarterly(hy_d, agg="mean")
        hy_q.name = "hy_spread"
    else:
        hy_q = None

    # VIX: daily → quarterly (mean)
    vix_d = series_map.get("VIXCLS")
    if vix_d is not None:
        vix_q = _monthly_to_quarterly(vix_d, agg="mean")
        vix_q.name = "vix"
    else:
        vix_q = None

    # --- Merge all into one DataFrame ---
    cols = [
        fedfunds_q, cpi_yoy, pce_yoy, gdp_growth, unrate_q,
        ip_yoy, payrolls_yoy, housing_yoy, m2_yoy,
        gs10_q, spread_q, usd_yoy, be_q, hy_q, vix_q,
    ]
    df = pd.concat([c for c in cols if c is not None], axis=1)
    df.index.name = "date"
    df = df.sort_index()

    # Filter to date range
    mask = (df.index.year >= start_year) & (df.index.year <= end_year)
    df = df.loc[mask].copy()

    # Drop rows where core variables are missing
    core_cols = ["fedfunds", "inflation_cpi_yoy", "gdp_growth_qoq_ann", "unemployment"]
    available_core = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=available_core)

    print(f"\nPanel shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Columns: {list(df.columns)}")

    # --- Save ---
    out_path = ensure_parent(REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet")
    df.to_parquet(out_path)
    print(f"\nSaved → {out_path.relative_to(REPO_ROOT)}")

    return df


if __name__ == "__main__":
    build_us_macro_panel()

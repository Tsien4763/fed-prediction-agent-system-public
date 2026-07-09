"""VAR/VECM → TFT Alignment Layer

The core technical innovation: fuse traditional econometric models (VAR, VECM)
with deep learning (TFT) to get both accuracy and economic plausibility.

Three alignment strategies, from simplest to most sophisticated:
  A. VARX:     Add VAR residuals as exogenous features to TFT
  B. Cross-Att: TFT's attention queries VAR residuals + VECM error correction terms
  C. Penalty:   Add Taylor-rule + VECM constraints to TFT loss function

This module implements Strategy A (VARX) and Strategy C (Penalty).
Strategy B (Cross-Attention) is implemented by
models.tft_model.EconometricCrossAttention.

Input:
  - data/processed/us_macro_panel.parquet   ← macro variables
  - data/processed/var_residuals.parquet    ← VAR residuals
  - data/processed/vecm_ect.parquet         ← VECM error correction terms
  - data/processed/context_snapshots.parquet ← semantic features (from Agent 2)

Output:
  - data/processed/alignment_features.parquet  ← fused feature matrix

Usage:
    python -m models.alignment_layer
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_engineering.config import REPO_ROOT, ensure_parent


def load_all_data() -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load macro panel, VAR residuals, VECM ECT, and semantic features."""
    macro_path = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    var_resid_path = REPO_ROOT / "data" / "processed" / "var_residuals.parquet"
    vecm_ect_path = REPO_ROOT / "data" / "processed" / "vecm_ect.parquet"
    semantic_path = REPO_ROOT / "data" / "processed" / "context_snapshots.parquet"

    macro = pd.read_parquet(macro_path) if macro_path.exists() else None
    var_resid = pd.read_parquet(var_resid_path) if var_resid_path.exists() else None
    vecm_ect = pd.read_parquet(vecm_ect_path) if vecm_ect_path.exists() else None
    semantic = pd.read_parquet(semantic_path) if semantic_path.exists() else None

    if macro is None:
        raise FileNotFoundError("Macro panel not found. Run build_us_macro_panel first.")

    return macro, var_resid, vecm_ect, semantic


# ─── Strategy A: VARX (VAR + eXogenous semantic features) ───

def build_varx_features(
    macro: pd.DataFrame,
    var_resid: pd.DataFrame,
    vecm_ect: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    VARX strategy: append VAR residuals and VECM error correction terms
    as additional features alongside raw macro variables.
    
    This is the simplest alignment — the TFT (or any downstream model) sees:
      [macro_lags, var_residuals, vecm_ect, semantic_features]
    
    The VAR residuals capture "what the linear model couldn't explain."
    The VECM ECT captures "how far from long-run equilibrium we are."
    """
    print("=" * 60)
    print("Strategy A: VARX Feature Fusion")
    print("=" * 60)
    
    # Align all DataFrames to common quarterly index
    dfs = [macro]
    labels = ["macro"]
    
    if var_resid is not None:
        dfs.append(var_resid)
        labels.append("var_resid")
    
    if vecm_ect is not None:
        dfs.append(vecm_ect)
        labels.append("vecm_ect")
    
    # Inner join on date index
    aligned = macro.copy()
    for extra, label in zip(dfs[1:], labels[1:]):
        aligned = aligned.join(extra, how="inner")
    
    aligned = aligned.dropna()
    
    print(f"  Macro variables:  {macro.shape[1]}")
    print(f"  VAR residuals:    {var_resid.shape[1] if var_resid is not None else 0}")
    print(f"  VECM ECT:         {vecm_ect.shape[1] if vecm_ect is not None else 0}")
    print(f"  Fused features:   {aligned.shape[1]}")
    print(f"  Time span:        {aligned.index[0].date()} → {aligned.index[-1].date()}")
    print(f"  Observations:     {aligned.shape[0]}")
    
    return aligned


# ─── Strategy C: Economics Penalty Function ───

def compute_economics_penalty(
    tft_forecast_ff: np.ndarray,      # (T,) — TFT's fed funds forecast
    taylor_rule_ff: np.ndarray,       # (T,) — Taylor Rule implied rate
    vecm_ect: Optional[np.ndarray] = None,  # (T,) — error correction term
    lambda_taylor: float = 0.1,
    lambda_vecm: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """
    Strategy C: Economics penalty added to TFT's loss function.
    
    This penalizes forecasts that violate economic theory:
    
      Loss_total = MSE(ŷ, y)                  ← prediction accuracy
                 + λ_Taylor × |ŷ - r_Taylor|  ← Taylor Rule anchor
                 + λ_VECM × |ŷ_change - ECT|  ← long-run equilibrium constraint
    
    The VECM constraint says: if ECT > 0 (economy above equilibrium),
    the fed funds rate should tend to converge DOWNWARD over time.
    If the TFT predicts a sharp RISE when ECT says "above equilibrium",
    it pays a penalty.
    """
    # Taylor rule penalty
    taylor_diff = np.abs(tft_forecast_ff - taylor_rule_ff)
    taylor_penalty = lambda_taylor * np.mean(taylor_diff)
    
    # VECM equilibrium penalty
    vecm_penalty = 0.0
    if vecm_ect is not None and len(vecm_ect) > 1:
        # ECT > 0 → downward pressure on rates → forecast should not rise sharply
        # ECT < 0 → upward pressure on rates   → forecast should not fall sharply
        ff_change = np.diff(tft_forecast_ff)
        ect_valid = vecm_ect[1:]  # align with diff
        
        # Penalty when forecast change goes against ECT direction
        misalignment = np.maximum(0, -ect_valid * ff_change)  # > 0 when they disagree
        vecm_penalty = lambda_vecm * np.mean(misalignment)
    
    total = taylor_penalty + vecm_penalty
    
    return total, {
        "taylor_penalty": round(float(taylor_penalty), 6),
        "vecm_penalty": round(float(vecm_penalty), 6),
        "total_penalty": round(float(total), 6),
    }


def compute_taylor_rule_rate(
    inflation: np.ndarray,         # YoY CPI %
    output_gap_pct: np.ndarray,    # % deviation from potential GDP
    inflation_target: float = 2.0,
    natural_rate: float = 2.5,     # r*
    phi_pi: float = 1.5,           # Taylor coefficient on inflation
    phi_y: float = 0.5,            # Taylor coefficient on output gap
) -> np.ndarray:
    """
    Standard Taylor Rule:
      r = r* + π + 0.5·(π - π*) + 0.5·(y - y*)
      r = r* + π + φ_π·(π - π*) + φ_y·(y - y*)
    """
    return natural_rate + inflation + phi_pi * (inflation - inflation_target) + phi_y * output_gap_pct


def run_alignment_demo() -> None:
    """Demonstrate the full VAR/VECM → TFT alignment pipeline."""
    macro, var_resid, vecm_ect, semantic = load_all_data()
    
    # ─── Strategy A: VARX ───
    fused = build_varx_features(macro, var_resid, vecm_ect)
    
    # Save fused features
    out_path = ensure_parent(REPO_ROOT / "data" / "processed" / "alignment_features.parquet")
    fused.to_parquet(out_path)
    print(f"\nFused features → {out_path.relative_to(REPO_ROOT)}")
    
    # ─── Strategy C: Economics Penalty Demo ───
    print("\n" + "=" * 60)
    print("Strategy C: Economics Penalty Demo")
    print("=" * 60)
    
    # Compute Taylor Rule implied rate (using CPI inflation + a simple output gap proxy)
    if "inflation_cpi_yoy" in fused.columns and "gdp_growth_qoq_ann" in fused.columns:
        inflation = fused["inflation_cpi_yoy"].values
        # Crude output gap proxy: deviation of GDP growth from 2% trend
        output_gap = fused["gdp_growth_qoq_ann"].values - 2.0
        
        taylor_rate = compute_taylor_rule_rate(inflation, output_gap)
        
        # Simulate a "naive TFT forecast" — just use the last observed fed funds rate
        tft_forecast = fused["fedfunds"].values
        
        # ECT for penalty
        ect_values = fused["ect_combined"].values if "ect_combined" in fused.columns else None
        
        penalty, details = compute_economics_penalty(tft_forecast, taylor_rate, ect_values)
        
        print(f"\n  Latest quarter: {fused.index[-1].date()}")
        print(f"  Fed Funds actual:           {tft_forecast[-1]:.2f}%")
        print(f"  Taylor Rule implied:        {taylor_rate[-1]:.2f}%")
        print(f"  Taylor Rule gap:            {tft_forecast[-1] - taylor_rate[-1]:.2f} pp")
        if ect_values is not None:
            print(f"  VECM ECT (combined):        {ect_values[-1]:.4f}")
        print(f"\n  Economics penalty breakdown:")
        for k, v in details.items():
            print(f"    {k}: {v}")
        
        # Interpretation
        gap = tft_forecast[-1] - taylor_rate[-1]
        if gap > 0.5:
            print(f"\n  Fed Funds {gap:.1f}pp ABOVE Taylor Rule -> policy is restrictive")
        elif gap < -0.5:
            print(f"\n  Fed Funds {abs(gap):.1f}pp BELOW Taylor Rule -> policy is accommodative")
        else:
            print(f"\n  Fed Funds near Taylor Rule -> policy is neutral")
    
    print("\nAlignment layer demo complete.")


if __name__ == "__main__":
    run_alignment_demo()

"""Build and save VECM model for US macro cointegration relationships.

Input:  data/processed/us_macro_panel.parquet
Output: artifacts/vecm/us_vecm_model.pkl
        data/processed/vecm_ect.parquet   ← Error Correction Terms

Key insight for Fed prediction:
  - The VECM's error correction term (ECT) tells you how far the economy is
    from its long-run equilibrium.
  - When ECT < 0 (below equilibrium), the system tends to mean-revert upward.
  - When ECT > 0 (above equilibrium), the system tends to mean-revert downward.
  - This is the "long-run anchor" that the TFT alignment layer uses to
    prevent economically nonsensical forecasts.

Usage:
    python -m models.vecm_model
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

from data_engineering.config import REPO_ROOT, ensure_parent

# VECM variables — subset that are likely cointegrated
VECM_VARIABLES = ["fedfunds", "inflation_cpi_yoy", "gdp_growth_qoq_ann"]


def load_panel() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Macro panel not found: {path}. Run build_us_macro_panel first.")
    return pd.read_parquet(path)


def build_vecm(
    df: pd.DataFrame,
    variables: list[str],
    deterministic: str = "ci",  # constant inside cointegration relation
    maxlags: int = 4,
) -> tuple[VECM, pd.DataFrame]:
    """
    Build a VECM and extract the Error Correction Term (ECT).
    
    Returns:
        model:     fitted VECM results
        ect_df:    DataFrame with error correction term(s) indexed by date
    """
    data = df[variables].dropna()
    
    # Step 1: select cointegration rank (Johansen trace test)
    print("Selecting cointegration rank...")
    rank_result = select_coint_rank(data, det_order=0, k_ar_diff=maxlags, method="trace")
    coint_rank = min(rank_result.rank, len(variables) - 1)  # cap at k-1
    print(f"  Johansen trace test → raw rank = {rank_result.rank}, capped = {coint_rank}")
    print(f"  (rank=0 → no cointegration, rank=1 → one long-run relationship, etc.)")
    
    if coint_rank == 0:
        print("  No cointegration found. ECT will be all zeros. Consider different variables.")
        coint_rank = 1
    
    # Step 2: fit VECM
    print(f"\nFitting VECM(coint_rank={coint_rank}, k_ar_diff={maxlags})...")
    vecm = VECM(data, k_ar_diff=maxlags, coint_rank=coint_rank, deterministic=deterministic)
    results = vecm.fit()
    try:
        llf = float(getattr(results, 'llf', 0) or 0)
    except (TypeError, ValueError):
        llf = results.llf if hasattr(results, 'llf') else 0
        if hasattr(llf, 'real'):
            llf = float(llf.real)
    print(f"  Log-likelihood = {llf:.2f}")
    
    try:
        info = results.info_criteria
        if info is not None:
            print(f"  AIC = {info.get('aic', 'N/A')}, BIC = {info.get('bic', 'N/A')}")
    except Exception:
        print("  (info criteria unavailable)")
    
    # Step 3: extract error correction terms
    # beta: (k, r) cointegrating vectors
    # ECT_t = beta' @ Y_{t-1}  →  (r, T)
    beta = results.beta  # (k, r)
    beta_real = np.real(beta) if np.iscomplexobj(beta) else beta
    ect_raw = data.values @ beta_real  # (T, r) — this uses Y_t, approximate as Y_{t-1}
    
    # Proper ECT uses lagged Y: shift by 1
    ect_values = np.roll(ect_raw, shift=1, axis=0)
    ect_values[0, :] = np.nan  # first row is invalid
    
    ect_cols = [f"ect_{i+1}" for i in range(coint_rank)]
    ect_df = pd.DataFrame(ect_values, index=data.index, columns=ect_cols)
    
    # Add a combined ECT (first principal component)
    if coint_rank > 0:
        ect_df["ect_combined"] = ect_df[ect_cols].mean(axis=1)
    
    # Drop NaN (first row)
    ect_df = ect_df.dropna()
    
    # Print cointegration vectors
    print("\nCointegration vector(s) β (normalized):")
    for i in range(coint_rank):
        vec = np.real(beta[:, i]) if np.iscomplexobj(beta) else beta[:, i]
        vec_str = " + ".join(f"{vec[j]:.4f} * {variables[j]}" for j in range(len(variables)))
        print(f"  β_{i+1}: {vec_str}")
        print(f"  → Interpretation: {_interpret_cointegration(vec, variables)}")
    
    # Print adjustment speeds
    print("\nAdjustment speeds α (how fast each variable corrects):")
    alpha = results.alpha
    for i, var in enumerate(variables):
        a_row = np.real(alpha[i, :]) if np.iscomplexobj(alpha) else alpha[i, :]
        speeds = ", ".join(f"α_{j+1}={a_row[j]:.4f}" for j in range(coint_rank))
        print(f"  {var}: {speeds}")
    
    return results, ect_df


def _interpret_cointegration(beta_vec: np.ndarray, variables: list[str]) -> str:
    """Heuristic interpretation of a single cointegration vector."""
    fedfunds_idx = variables.index("fedfunds") if "fedfunds" in variables else 0
    inflation_idx = variables.index("inflation_cpi_yoy") if "inflation_cpi_yoy" in variables else 1
    
    # Normalize so fedfunds coefficient = -1 (if possible)
    if abs(beta_vec[fedfunds_idx]) > 1e-6:
        norm = -beta_vec[fedfunds_idx]
        normalized = beta_vec / norm
        return (
            f"Fed Funds ≈ {normalized[inflation_idx]:.2f} × CPI_inflation "
            f"+ {normalized[2]:.2f} × GDP_growth (long-run Taylor-rule-like relationship)"
        )
    return "Cannot interpret — fedfunds coefficient near zero."


def run() -> None:
    df = load_panel()
    
    vecm_vars = [v for v in VECM_VARIABLES if v in df.columns]
    if len(vecm_vars) < 3:
        print(f"Only {len(vecm_vars)} VECM variables available; need 3. Aborting.")
        return
    
    print("=" * 60)
    print("Building VECM for US macro cointegration")
    print(f"Variables: {vecm_vars}")
    print("=" * 60 + "\n")
    
    vecm_model, ect_df = build_vecm(df, vecm_vars)
    
    # --- Save ---
    vecm_dir = ensure_parent(REPO_ROOT / "artifacts" / "vecm" / "_placeholder")
    vecm_dir = REPO_ROOT / "artifacts" / "vecm"
    vecm_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = vecm_dir / "us_vecm_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(vecm_model, fh)
    print(f"\nVECM model → {model_path.relative_to(REPO_ROOT)}")
    
    ect_path = ensure_parent(REPO_ROOT / "data" / "processed" / "vecm_ect.parquet")
    ect_df.to_parquet(ect_path)
    print(f"VECM ECT    → {ect_path.relative_to(REPO_ROOT)}")
    
    # --- Show ECT sample ---
    print("\n" + "-" * 40)
    print("Error Correction Term (ECT) — last 8 quarters:")
    print(ect_df.tail(8).round(4).to_string())
    
    # Interpret current state
    if "ect_combined" in ect_df.columns:
        latest_ect = ect_df["ect_combined"].iloc[-1]
        if latest_ect > 0.5:
            print(f"\n🔴 ECT = {latest_ect:.3f} → Economy ABOVE long-run equilibrium")
            print("   → System tends to mean-revert DOWNWARD (disinflationary pressure)")
            print("   → Fed may be more likely to hold or cut")
        elif latest_ect < -0.5:
            print(f"\n🟢 ECT = {latest_ect:.3f} → Economy BELOW long-run equilibrium")
            print("   → System tends to mean-revert UPWARD (inflationary pressure)")
            print("   → Fed may be more likely to hold or hike")
        else:
            print(f"\n🟡 ECT = {latest_ect:.3f} → Economy NEAR long-run equilibrium")
            print("   → No strong mean-reversion signal")


if __name__ == "__main__":
    run()

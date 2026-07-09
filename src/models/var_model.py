"""Build and save VAR model for US macro variables.

Input:  data/processed/us_macro_panel.parquet
Output: artifacts/var/us_var_model.pkl
        data/processed/var_residuals.parquet

Usage:
    python -m models.var_model
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from data_engineering.config import REPO_ROOT, ensure_parent


# Core VAR variables — the "textbook Fed forecasting" set
VAR_VARIABLES = ["fedfunds", "inflation_cpi_yoy", "gdp_growth_qoq_ann", "unemployment"]

# Extended VAR variables — adds financial conditions
VAR_EXTENDED = VAR_VARIABLES + ["gs10", "term_spread_10y2y", "breakeven_10y", "hy_spread", "vix"]


def load_panel() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Macro panel not found: {path}. Run build_us_macro_panel first.")
    return pd.read_parquet(path)


def build_var(
    df: pd.DataFrame,
    variables: list[str],
    maxlags: int = 8,
    ic: str = "aic",
) -> tuple[VAR, pd.DataFrame, pd.DataFrame]:
    """
    Build a VAR model on the selected variables.
    
    Returns:
        model:      fitted VAR results
        residuals:  (T, k) DataFrame of residuals
        fitted_vals: (T, k) DataFrame of fitted values
    """
    data = df[variables].dropna()
    model = VAR(data)
    results = model.fit(maxlags=maxlags, ic=ic)
    
    residuals = pd.DataFrame(
        np.asarray(results.resid),
        index=data.index[results.k_ar:],
        columns=[f"{v}_resid" for v in variables],
    )
    fitted_vals = pd.DataFrame(
        results.fittedvalues,
        index=data.index[results.k_ar:],
        columns=[f"{v}_fitted" for v in variables],
    )
    
    print(f"VAR({results.k_ar}) with {len(variables)} variables")
    print(f"  T = {results.nobs}, log-likelihood = {results.llf:.2f}")
    print(f"  AIC = {results.aic:.2f}, BIC = {results.bic:.2f}")
    
    return results, residuals, fitted_vals


def run() -> None:
    df = load_panel()
    
    # --- Core VAR ---
    print("=" * 60)
    print("Building CORE VAR (4-variable textbook model)")
    print("=" * 60)
    
    core_vars = [v for v in VAR_VARIABLES if v in df.columns]
    if len(core_vars) < 3:
        print(f"Only {len(core_vars)} core variables available; need at least 3. Aborting.")
        return
    
    core_model, core_resid, core_fitted = build_var(df, core_vars)
    
    # --- Extended VAR ---
    print("\n" + "=" * 60)
    print("Building EXTENDED VAR (core + financial conditions)")
    print("=" * 60)
    
    ext_vars = [v for v in VAR_EXTENDED if v in df.columns]
    if len(ext_vars) > len(core_vars):
        ext_model, ext_resid, ext_fitted = build_var(df, ext_vars)
    else:
        ext_model, ext_resid, ext_fitted = None, None, None
        print("Skipped extended VAR; not enough extended variables available.")
    
    # --- Save ---
    var_dir = ensure_parent(REPO_ROOT / "artifacts" / "var" / "_placeholder")
    var_dir = REPO_ROOT / "artifacts" / "var"
    var_dir.mkdir(parents=True, exist_ok=True)
    
    # Save core model
    core_path = var_dir / "us_var_model.pkl"
    with open(core_path, "wb") as fh:
        pickle.dump(core_model, fh)
    print(f"\nVAR model → {core_path.relative_to(REPO_ROOT)}")
    
    # Save residuals
    resid_path = ensure_parent(REPO_ROOT / "data" / "processed" / "var_residuals.parquet")
    core_resid.to_parquet(resid_path)
    print(f"VAR residuals → {resid_path.relative_to(REPO_ROOT)}")
    
    # Save extended if built
    if ext_model is not None:
        ext_path = var_dir / "us_var_extended_model.pkl"
        with open(ext_path, "wb") as fh:
            pickle.dump(ext_model, fh)
        print(f"VAR extended → {ext_path.relative_to(REPO_ROOT)}")
        
        ext_resid_path = ensure_parent(REPO_ROOT / "data" / "processed" / "var_extended_residuals.parquet")
        ext_resid.to_parquet(ext_resid_path)
        print(f"VAR ext residuals → {ext_resid_path.relative_to(REPO_ROOT)}")
    
    # --- Print impulse responses for fedfunds shock ---
    print("\n" + "-" * 40)
    print("Impulse Response: 1pp shock to Fed Funds →")
    irf = core_model.irf(periods=12)
    irf_df = pd.DataFrame({
        "horizon": range(13),
        **{v: irf.irfs[:, core_vars.index(v), core_vars.index("fedfunds")]
           for v in core_vars}
    })
    print(irf_df.round(4).to_string(index=False))


if __name__ == "__main__":
    run()

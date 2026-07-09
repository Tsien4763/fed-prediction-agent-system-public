"""End-to-end US macro VAR/VECM pipeline — run everything in order.

Usage:
    # Step 0: Set FRED_API_KEY in your shell or .env file.
    
    # Step 1: Download FRED data
    python -m data_engineering.download_fred_macro
    
    # Step 2: Build macro panel
    python -m data_engineering.build_us_macro_panel
    
    # Step 3: Build VAR + VECM + Alignment
    python scripts/run_var_pipeline.py

Or run this script directly to do Steps 2-3:
    python scripts/run_var_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("US Macro VAR/VECM Alignment Pipeline\n")

    # ── Step 1: Build US macro panel ──
    print("[1/4] Building US quarterly macro panel...\n")
    from data_engineering.build_us_macro_panel import build_us_macro_panel
    panel = build_us_macro_panel()
    print(f"\nMacro panel: {panel.shape[0]} quarters, {panel.shape[1]} variables\n")

    # ── Step 2: Build VAR model ──
    print("[2/4] Building VAR model...\n")
    from models.var_model import run as run_var
    run_var()
    print("\nVAR model saved\n")

    # ── Step 3: Build VECM model ──
    print("[3/4] Building VECM model...\n")
    from models.vecm_model import run as run_vecm
    run_vecm()
    print("\nVECM model saved\n")

    # ── Step 4: Run alignment demo ──
    print("[4/4] Running Alignment Layer demo...\n")
    from models.alignment_layer import run_alignment_demo
    run_alignment_demo()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print("""
Outputs produced:
  data/processed/us_macro_panel.parquet     ← Quarterly macro panel (2000-2026)
  artifacts/var/us_var_model.pkl            ← VAR model (core 4-variable)
  artifacts/var/us_var_extended_model.pkl   ← VAR model (extended with financial)
  data/processed/var_residuals.parquet      ← VAR residuals
  artifacts/vecm/us_vecm_model.pkl          ← VECM model
  data/processed/vecm_ect.parquet           ← Error Correction Terms
  data/processed/alignment_features.parquet ← Fused VARX features

Next steps:
  1. Review the VAR impulse responses and VECM cointegration vectors
  2. Feed alignment_features.parquet into the TFT training pipeline
  3. Use the economics penalty in the TFT loss function
  4. Connect with semantic features from Agent 2 for full VARX alignment
""")


if __name__ == "__main__":
    main()

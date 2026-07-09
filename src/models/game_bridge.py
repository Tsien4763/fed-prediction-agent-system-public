"""Bridge: fed_game LLM traces → macro panel features.

Extracts quarterly game-theoretic features from rolling_self_play_traces.jsonl
and merges them with us_macro_panel.parquet.

Output: data/processed/full_features.parquet
        (macro variables + VECM ECT + LLM game features)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_engineering.config import REPO_ROOT, ensure_parent
from fed_game.config import default_self_play_trace_path, load_config


def parse_evidence_metrics(evidence_chain: list[str]) -> dict[str, float]:
    """Extract numerical metrics from evidence chain text lines.
    
    Examples:
      "External shock layer: energy=0.45, geopol=0.45, term_premium=0.38."
      → {"energy_risk": 0.45, "geopol_risk": 0.45, "term_premium_risk": 0.38}
      
      "usa_warsh: hawkish=0.63, hike=0.31, payoff=0.80."
      → {"warsh_hawkish": 0.63, "warsh_hike_propensity": 0.31, "warsh_payoff": 0.80}
    """
    metrics = {}
    for line in evidence_chain:
        # Parse "key=value" pairs
        pairs = re.findall(r'(\w+)=(\d+\.?\d*)', line.lower())
        for key, val in pairs:
            metrics[key] = float(val)
    return metrics


def parse_shock_variable_metrics(shock_variables: list[dict] | None) -> dict[str, float]:
    """Extract named shock variables from self-play traces.

    New traces expose the external strategic shock layer as structured JSON.
    Older traces only have textual evidence lines, so build_game_features keeps
    a fallback parser for those.
    """
    metrics: dict[str, float] = {}
    for item in shock_variables or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            metrics[name] = float(item.get("value", 0.0))
        except (TypeError, ValueError):
            metrics[name] = 0.0
    return metrics


def metric_value(metrics: dict[str, float], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return default


def build_game_features() -> pd.DataFrame:
    """Parse fed_game traces into a quarterly DataFrame of game-theoretic features."""
    trace_path = default_self_play_trace_path(load_config())
    if not trace_path.exists():
        print(f"fed_game traces not found at {trace_path}")
        return pd.DataFrame()
    
    with open(trace_path, encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]
    
    print(f"Processing {len(traces)} fed_game traces...")
    
    rows = []
    for t in traces:
        quarter = t["quarter"]
        year, q = int(quarter[:4]), int(quarter[-1])
        q_end = pd.Timestamp(year=year, month=q*3, day=1) + pd.offsets.MonthEnd(1)
        
        row = {"quarter": q_end, "quarter_str": quarter}
        
        fp = t.get("fed_prediction", {})
        row["game_p_hike"] = float(fp.get("hike_25bp", 0))
        row["game_p_hold"] = float(fp.get("hold", 0))
        row["game_p_cut"] = float(fp.get("cut_25bp", 0))
        row["game_converged"] = int(t.get("converged", 0))
        row["game_rounds"] = int(t.get("rounds", 0))
        
        ec = t.get("evidence_chain", [])
        metrics = parse_evidence_metrics(ec)
        metrics.update(parse_shock_variable_metrics(t.get("shock_variables")))
        row["game_energy_risk"] = metric_value(metrics, "energy_price_risk", "energy")
        row["game_geopol_risk"] = metric_value(metrics, "geopol_escalation_prob", "geopol")
        row["game_term_premium_risk"] = metric_value(metrics, "term_premium_pressure", "term_premium")
        row["game_warsh_hawkish"] = metric_value(metrics, "hawkish")
        row["game_warsh_hike_propensity"] = metric_value(metrics, "hike")
        row["game_external_shock_count"] = float(len(t.get("shock_variables") or []))
        row["game_highest_shock_value"] = max(
            [float(item.get("value", 0.0)) for item in t.get("shock_variables", []) if isinstance(item, dict)]
            or [0.0]
        )
        warsh_consistency = t.get("warsh_consistency_score") or {}
        row["game_warsh_consistency_score"] = float(warsh_consistency.get("score", 0.0)) if isinstance(warsh_consistency, dict) else 0.0
        
        row["game_tariff_risk"] = metric_value(metrics, "tariff_risk", "tariff")
        row["game_supply_chain_risk"] = metric_value(metrics, "supply_chain_risk", "supply_chain")
        row["game_dollar_liquidity_pressure"] = metric_value(
            metrics,
            "dollar_liquidity_pressure",
            "liquidity",
            "capital_flow",
        )
        row["game_sanction_risk"] = metric_value(metrics, "sanction")
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("quarter").sort_index()
    df.index.name = "date"
    print(f"  Game features: {len(df)} quarters, {df.shape[1]} features")
    return df


def build_per_meeting_game_predictions() -> pd.DataFrame:
    """Map quarterly game equilibrium to individual FOMC meetings.
    
    Strategy: use the game's quarterly equilibrium as the base prediction
    for each meeting within that quarter. Meetings inherit the strategic
    equilibrium the game computed — individual meetings are tactical
    executions of the quarterly strategic stance.
    
    This transforms the game model from quarterly to per-meeting frequency,
    enabling direct comparison with meeting-level models (GB, Logistic).
    """
    from models.fomc_labels import build_fomc_label_df
    
    # Load quarterly game features
    game_df = build_game_features()
    if len(game_df) == 0:
        return pd.DataFrame()
    
    # Load FOMC labels
    labels = build_fomc_label_df()
    
    # Map each meeting to its containing quarter
    rows = []
    for _, meeting in labels.iterrows():
        mdate = pd.Timestamp(meeting["date"])
        # Find the game quarter that contains this meeting
        # Game quarters go: 2024Q1 (Jan-Mar), 2024Q2 (Apr-Jun), etc.
        meeting_year = mdate.year
        meeting_month = mdate.month
        meeting_q = (meeting_month - 1) // 3 + 1  # 1,2,3,4
        quarter_str = f"{meeting_year}Q{meeting_q}"
        
        # Find matching game trace
        matching = game_df[game_df["quarter_str"] == quarter_str]
        if len(matching) == 0:
            continue
        
        game_row = matching.iloc[0]
        row = {
            "meeting_date": mdate,
            "quarter": quarter_str,
            "actual": meeting["decision"],
            "game_p_hike": game_row["game_p_hike"],
            "game_p_hold": game_row["game_p_hold"],
            "game_p_cut": game_row["game_p_cut"],
            "game_warsh_hawkish": game_row["game_warsh_hawkish"],
            "game_energy_risk": game_row["game_energy_risk"],
            "game_geopol_risk": game_row["game_geopol_risk"],
            "game_warsh_consistency_score": game_row.get("game_warsh_consistency_score", 0.0),
        }
        
        # Per-meeting adjustment: meetings later in the quarter drift slightly
        # from the quarterly equilibrium based on accumulated information
        month_in_q = (meeting_month - 1) % 3  # 0=first month, 1=second, 2=third
        if month_in_q > 0:
            # Small drift: later meetings may reflect more information
            drift = month_in_q * 0.02  # 2% drift per month
            # Drift toward hawkish if energy risk high
            energy = game_row["game_energy_risk"]
            hawkish = game_row["game_warsh_hawkish"]
            row["game_p_hike"] = min(1.0, row["game_p_hike"] + drift * hawkish * energy)
            row["game_p_cut"] = max(0.0, row["game_p_cut"] - drift * (1 - hawkish))
            row["game_p_hold"] = max(0.0, 1.0 - row["game_p_hike"] - row["game_p_cut"])
            # Renormalize
            total = row["game_p_hike"] + row["game_p_hold"] + row["game_p_cut"]
            row["game_p_hike"] /= total
            row["game_p_hold"] /= total
            row["game_p_cut"] /= total
        
        rows.append(row)
    
    result = pd.DataFrame(rows).set_index("meeting_date").sort_index()
    print(f"  Per-meeting game predictions: {len(result)} meetings across {result['quarter'].nunique()} quarters")
    return result


def compare_game_vs_traditional() -> dict:
    """Compare 5-cluster game model accuracy vs traditional GB on per-meeting basis."""
    from models.fomc_predictor import load_data, prepare_train_val_test, train_models_with_validation
    
    # Get per-meeting game predictions
    game_meetings = build_per_meeting_game_predictions()
    if len(game_meetings) == 0:
        return {"error": "No game predictions available"}
    
    # Get traditional model for comparison
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, test_df, val_df = prepare_train_val_test(df)
    gb_models = train_models_with_validation(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    gb = gb_models["gradient_boosting"]
    
    # Align game predictions with test set
    test_dates = set(test_df.index)
    game_test = game_meetings[game_meetings.index.isin(test_dates)]
    
    if len(game_test) == 0:
        return {"error": "No overlapping test meetings between game and traditional models"}
    
    # Game model: predict highest-probability class
    game_preds = []
    for _, row in game_test.iterrows():
        probs = [row["game_p_cut"], row["game_p_hold"], row["game_p_hike"]]
        game_preds.append(np.argmax(probs) - 1)  # -1, 0, 1
    
    game_acc = np.mean(np.array(game_preds) == game_test["actual"].values)
    
    # Traditional GB accuracy on same meetings
    test_mask = [i for i, idx in enumerate(test_df.index) if idx in test_dates]
    if test_mask:
        gb_preds = gb.predict(X_test[test_mask])
        gb_acc = np.mean(gb_preds == y_test[test_mask])
    else:
        gb_acc = gb.score(X_test, y_test)
    
    return {
        "n_meetings": len(game_test),
        "game_accuracy": round(float(game_acc), 4),
        "gb_accuracy": round(float(gb_acc), 4),
        "game_avg_hike": round(float(game_test["game_p_hike"].mean()), 4),
        "game_avg_hold": round(float(game_test["game_p_hold"].mean()), 4),
        "game_quarters_covered": list(game_test["quarter"].unique()),
    }


def merge_all_features() -> pd.DataFrame:
    """Merge macro panel + VECM ECT + LLM game features into one DataFrame."""
    # Load macro panel
    macro_path = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    macro = pd.read_parquet(macro_path)
    
    # Load VECM ECT
    ect_path = REPO_ROOT / "data" / "processed" / "vecm_ect.parquet"
    if ect_path.exists():
        ect = pd.read_parquet(ect_path)
        macro = macro.join(ect, how="left")
    
    # Load VAR residuals (Layer 1 output → Layer 2 input)
    resid_path = REPO_ROOT / "data" / "processed" / "var_residuals.parquet"
    if resid_path.exists():
        resid = pd.read_parquet(resid_path)
        # Normalize both indices for reliable join
        macro.index = pd.to_datetime(macro.index).normalize()
        resid.index = pd.to_datetime(resid.index).normalize()
        macro = macro.join(resid, how="left")
        print(f"  VAR residuals loaded: {list(resid.columns)}")
    
    # Load game features
    game = build_game_features()
    
    if len(game) > 0:
        # Normalize indices to date-only for reliable merging
        macro.index = pd.to_datetime(macro.index).normalize()
        game.index = pd.to_datetime(game.index).normalize()
        # Merge: game features aligned by quarter-end date
        merged = macro.join(game, how="left")
        print(f"\nMerged panel: {len(merged)} quarters, {merged.shape[1]} features")
        game_cols = [c for c in game.columns]
        n_game_filled = merged[game_cols[0]].notna().sum() if game_cols else 0
        print(f"  Game features available for {n_game_filled}/{len(merged)} quarters")
    else:
        merged = macro
        print("\nNo game features available; using macro-only panel")
    
    # Save
    out_path = ensure_parent(REPO_ROOT / "data" / "processed" / "full_features.parquet")
    merged.to_parquet(out_path)
    print(f"  Saved → {out_path.relative_to(REPO_ROOT)}")
    
    return merged


if __name__ == "__main__":
    merge_all_features()

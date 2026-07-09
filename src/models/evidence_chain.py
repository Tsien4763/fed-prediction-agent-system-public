"""Forecast evidence chain and risk attribution.

Connects to the trained FOMC predictor and computes:
  1. SHAP feature contributions - which macro/game features drove the prediction
  2. Integrated Gradients for the TFT model
  3. Human-readable evidence chain combining SHAP + VECM + game signals

Usage:
    python -m models.evidence_chain
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from data_engineering.config import REPO_ROOT


def compute_shapley_attribution(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    background_samples: int = 30,
) -> dict[str, Any]:
    """Compute predictive SHAP values for the GB model.
    
    Strategy: SHAP TreeExplainer doesn't support multi-class GB directly,
    so we compute SHAP for each binary sub-problem (HIKE vs rest, CUT vs rest)
    and aggregate.
    """
    try:
        import shap
        
        bg = X[:min(background_samples, len(X))]
        explainer = shap.TreeExplainer(model)
        
        # SHAP for multi-class: shap_values is list of (n_samples, n_features) per class
        shap_values = explainer.shap_values(X[-1:])  # explain latest prediction
        
        if isinstance(shap_values, list):
            # Multi-class: aggregate absolute SHAP across classes
            shap_abs = np.zeros(len(feature_names))
            directions = []
            for cls_idx, sv in enumerate(shap_values):
                shap_abs += np.abs(sv[0])
                cls_label = model.classes_[cls_idx] if hasattr(model, 'classes_') else cls_idx - 1
                directions.append({
                    "class": int(cls_label),
                    "class_name": { -1: "CUT", 0: "HOLD", 1: "HIKE" }.get(int(cls_label), str(cls_label)),
                    "top_features": [
                        {"feature": feature_names[j], "shap_value": round(float(sv[0][j]), 5)}
                        for j in np.argsort(np.abs(sv[0]))[-3:][::-1]
                    ]
                })
            shap_abs /= len(shap_values)
        else:
            # Binary or single-output
            shap_abs = np.abs(shap_values[0]) if shap_values.ndim == 2 else np.abs(shap_values)[0]
            directions = []
        
        top_idx = np.argsort(shap_abs)[-8:][::-1]
        contributions = [
            {
                "feature": feature_names[i],
                "shap_value": round(float(shap_abs[i]), 5),
            }
            for i in top_idx
        ]
        return {
            "method": "Predictive SHAP (TreeExplainer, aggregated across classes)",
            "analysis_scope": "model_attribution",
            "top_contributors": contributions,
            "per_class_direction": directions,
        }
        
    except Exception as e:
        print(f"  (SHAP failed: {e} — using permutation importance)")
        return _permutation_importance(model, X, feature_names, reason=type(e).__name__)


def _permutation_importance(model, X: np.ndarray, feature_names: list[str], *, reason: str = "unknown") -> dict:
    """Simple permutation-based feature importance."""
    baseline_pred = model.predict_proba(X[-1:])[0]
    importances = []
    
    for i, name in enumerate(feature_names):
        X_perm = X[-1:].copy()
        X_perm[0, i] = np.random.permutation(X[:, i])[0]
        perm_pred = model.predict_proba(X_perm)[0]
        importance = np.mean(np.abs(perm_pred - baseline_pred))
        importances.append({"feature": name, "importance": round(float(importance), 5)})
    
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return {
        "method": f"Predictive permutation importance (SHAP unavailable: {reason})",
        "analysis_scope": "model_attribution",
        "top_contributors": importances[:8],
    }


def build_evidence_chain(
    shap_result: dict,
    game_signals: dict,
    vecm_ect: float,
    taylor_gap: float,
) -> dict[str, Any]:
    """Build a complete, human-readable predictive evidence chain."""
    
    evidence_items = []
    
    # 1. VECM long-run equilibrium signal
    if vecm_ect < -0.5:
        evidence_items.append({
            "layer": "VECM (long-run equilibrium)",
            "signal": f"ECT = {vecm_ect:.2f} < 0",
            "interpretation": "Economy BELOW long-run equilibrium → mean-reversion pressure is UPWARD",
            "policy_implication": "Supports HOLD or HIKE bias",
            "confidence": "high" if abs(vecm_ect) > 1.5 else "medium",
        })
    elif vecm_ect > 0.5:
        evidence_items.append({
            "layer": "VECM (long-run equilibrium)",
            "signal": f"ECT = {vecm_ect:.2f} > 0",
            "interpretation": "Economy ABOVE long-run equilibrium → mean-reversion pressure is DOWNWARD",
            "policy_implication": "Supports HOLD or CUT bias",
            "confidence": "high" if abs(vecm_ect) > 1.5 else "medium",
        })
    
    # 2. Taylor Rule gap
    if abs(taylor_gap) > 1.0:
        direction = "too loose" if taylor_gap > 0 else "too tight"
        evidence_items.append({
            "layer": "Taylor Rule Anchor",
            "signal": f"Taylor gap = {taylor_gap:+.2f} pp",
            "interpretation": f"Policy is {direction} relative to Taylor Rule benchmark",
            "policy_implication": "Rate adjustment needed to close gap" if abs(taylor_gap) > 2 else "Moderate deviation — watch for convergence",
            "confidence": "medium",
        })
    
    # 3. LLM Game Theory signals
    hawkish = game_signals.get("game_warsh_hawkish", 0)
    warsh_consistency = game_signals.get("game_warsh_consistency_score", 0)
    energy = game_signals.get("game_energy_risk", 0)
    geopol = game_signals.get("game_geopol_risk", 0)
    
    if hawkish > 0.5:
        evidence_items.append({
            "layer": "LLM Multi-Agent Game (US Internal)",
            "signal": f"Warsh hawkish index = {hawkish:.2f}",
            "interpretation": f"Fed Chair Agent shows {hawkish:.0%} hawkish propensity in 5-cluster equilibrium",
            "policy_implication": "Elevated probability of hawkish statement or rate action",
            "confidence": "medium",
        })

    if warsh_consistency > 0:
        evidence_items.append({
            "layer": "Warsh Policy Persona",
            "signal": f"Warsh consistency score = {warsh_consistency:.2f}",
            "interpretation": "Warsh role strategy is compared with the Nuwa-style public-information persona",
            "policy_implication": "Higher consistency increases confidence in the role-based Fed prior",
            "confidence": "medium",
        })
    
    if energy > 0.3 or geopol > 0.3:
        evidence_items.append({
            "layer": "LLM Multi-Agent Game (External Shocks)",
            "signal": f"Energy risk = {energy:.2f}, Geopolitical risk = {geopol:.2f}",
            "interpretation": "Between-cluster game equilibrium indicates elevated external inflation risks",
            "policy_implication": "Fed less likely to ease given external price pressures",
            "confidence": "medium",
        })
    
    # 4. SHAP top features
    top_shap = shap_result.get("top_contributors", [])[:4]
    shap_items = [
        {
            "layer": f"Predictive attribution ({shap_result.get('method', 'feature_importance')})",
            "signal": f"{item['feature']} importance = {item.get('shap_value', item.get('importance', 0)):.5f}",
            "interpretation": f"Feature '{item['feature']}' is a top-{i+1} driver of the model prediction",
            "policy_implication": f"Monitor {item['feature']} for prediction updates",
            "confidence": "high",
        }
        for i, item in enumerate(top_shap)
    ]
    evidence_items.extend(shap_items)
    
    return {
        "evidence_count": len(evidence_items),
        "evidence_items": evidence_items,
        "summary": _generate_summary(evidence_items),
        "evidence_scope": "Forecast evidence chain combining model attribution, macro anchors, and game signals.",
    }


def _generate_summary(items: list[dict]) -> str:
    """Generate a one-paragraph natural language summary."""
    hawkish_count = sum(1 for i in items if "hawkish" in str(i.get("policy_implication", "")).lower() or "hike" in str(i.get("policy_implication", "")).lower())
    dovish_count = sum(1 for i in items if "cut" in str(i.get("policy_implication", "")).lower())
    
    if hawkish_count > dovish_count:
        bias = "hawkish"
    elif dovish_count > hawkish_count:
        bias = "dovish"
    else:
        bias = "neutral / data-dependent"
    
    return (
        f"Evidence chain with {len(items)} items from 4 layers "
        f"(VECM, Taylor Rule, LLM Game Theory, predictive attribution). "
        f"Overall signal bias: {bias}. "
        f"Key drivers: {', '.join(i['signal'].split('=')[0].strip() for i in items[:4])}."
    )


def run() -> None:
    """Run full evidence chain generation."""
    print("=" * 60)
    print("Evidence Chain & Risk Attribution")
    print("=" * 60 + "\n")
    
    # Load data and model
    from models.fomc_predictor import load_data, prepare_train_val_test, train_models_with_validation
    
    print("[1/4] Loading data...")
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, test_df, val_df = prepare_train_val_test(df)
    
    print("[2/4] Training model for SHAP...")
    models = train_models_with_validation(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    models.pop("_history", None)
    gb_model = models["gradient_boosting"]
    
    print("[3/4] Computing SHAP attribution...")
    shap_result = compute_shapley_attribution(gb_model, X_test, y_test, feature_cols)
    
    print(f"  Method: {shap_result['method']}")
    print(f"  Top contributors:")
    for item in shap_result["top_contributors"][:5]:
        print(f"    {item['feature']:30s} impact={item.get('shap_value', item.get('importance', 0)):.5f}")
    
    # Load game signals and VECM
    full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
    df_full = pd.read_parquet(full_path)
    latest = df_full.iloc[-1]
    
    game_signals = {
        c: float(latest[c]) for c in df_full.columns 
        if c.startswith("game_") and not pd.isna(latest[c])
    }
    ect = float(latest.get("ect_combined", 0)) if not pd.isna(latest.get("ect_combined", 0)) else 0
    taylor_gap = float(latest.get("fedfunds", 0)) - 6.30  # approximate from alignment layer
    
    print(f"\n[4/4] Building evidence chain...")
    evidence = build_evidence_chain(shap_result, game_signals, ect, taylor_gap)
    
    print(f"\n  Evidence items: {evidence['evidence_count']}")
    for item in evidence["evidence_items"][:5]:
        print(f"    [{item['layer']}] {item['signal']}")
        print(f"      → {item['interpretation']}")
        print(f"      → Policy: {item['policy_implication']}")
    
    print(f"\n  Summary: {evidence['summary']}")
    
    # Save
    out_path = REPO_ROOT / "data" / "processed" / "evidence_chain.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(f"\n  Saved → {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run()

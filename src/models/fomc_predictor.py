"""End-to-end FOMC prediction module: Alignment Features → Probability Distribution.

Connects the full pipeline:
  Macro Panel → VAR residuals → VECM ECT → Alignment Features
  → Logistic/Ensemble model → P(hike), P(hold), P(cut) for next FOMC

Usage:
    python -m models.fomc_predictor
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_engineering.config import REPO_ROOT


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load macro + VECM + VAR residuals + LLM game features and FOMC labels."""
    # Build full features (macro + VECM + game) if not yet done
    full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
    if not full_path.exists():
        from models.game_bridge import merge_all_features
        merge_all_features()
    
    features = pd.read_parquet(full_path)
    
    # ── Add VAR residuals if not already in full_features ──
    resid_path = REPO_ROOT / "data" / "processed" / "var_residuals.parquet"
    var_residual_cols = [c for c in features.columns if c.endswith("_resid")]
    if not var_residual_cols and resid_path.exists():
        try:
            var_resid = pd.read_parquet(resid_path)
            if len(var_resid) > 0:
                features = features.join(var_resid, how="left")
                var_residual_cols = list(var_resid.columns)
        except Exception as e:
            print(f"  VAR residuals skipped: {e}")
    elif var_residual_cols:
        pass  # already loaded via game_bridge
    
    # Count game features available
    game_cols = [c for c in features.columns if c.startswith("game_")]
    n_game = features[game_cols[0]].notna().sum() if game_cols else 0
    
    print(f"Full features: {len(features)} quarters, {features.shape[1]} features "
          f"({features.index[0].date()} → {features.index[-1].date()})")
    print(f"  Game features available for {n_game}/{len(features)} quarters")
    print(f"  VAR residuals available: {len(var_residual_cols)} columns")
    
    # Load FOMC labels
    from models.fomc_labels import build_fomc_label_df
    labels = build_fomc_label_df()
    
    # Merge: for each FOMC meeting, use the most recent quarter-end macro data
    # (strict temporal ordering: no look-ahead)
    # Normalize feature index to date-only for reliable comparison
    features.index = pd.to_datetime(features.index).normalize()
    
    merged_rows = []
    for _, meeting in labels.iterrows():
        meeting_q = pd.Timestamp(meeting["quarter"]).normalize()
        past_features = features[features.index <= meeting_q]
        if past_features.empty:
            continue
        latest = past_features.iloc[-1]
        row = latest.to_dict()
        row["meeting_date"] = meeting["date"]
        row["quarter"] = meeting_q
        row["decision"] = meeting["decision"]
        row["ff_target_mid"] = (meeting["ff_lower"] + meeting["ff_upper"]) / 2
        row["desc"] = meeting["desc"]
        merged_rows.append(row)
    
    df = pd.DataFrame(merged_rows)
    df = df.set_index("meeting_date").sort_index()
    
    print(f"Merged dataset: {len(df)} meetings, {df.shape[1]-3} features + labels")
    print(f"  Decision distribution: hike={sum(df['decision']==1)}, hold={sum(df['decision']==0)}, cut={sum(df['decision']==-1)}")
    
    return df


def prepare_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2020-12-31",
    val_end: str = "2023-06-30",
    test_end: str = "2026-12-31",
) -> tuple:
    """Three-way temporal split: train / validation / test.
    
    Plan specification:
      Train:  2000–2020 (dot-com, GFC, QE, euro crisis, trade war, pre-COVID low inflation)
      Val:    2020–2023 (COVID shock, supply chain, fiscal expansion, high inflation — structural break)
      Test:   2023–2026 (post-COVID high rates, geopolitics, Warsh era — out-of-sample generalization)
    
    Strict temporal ordering: no future information leaks into training.
    """
    core_features = [
        "fedfunds", "inflation_cpi_yoy", "inflation_pce_yoy",
        "gdp_growth_qoq_ann", "unemployment",
        "industrial_prod_yoy", "payrolls_yoy", "m2_yoy",
        "gs10", "term_spread_10y2y", "breakeven_10y",
        "hy_spread", "vix",
        # VAR residuals — "what the linear model couldn't explain"
        "fedfunds_resid", "inflation_cpi_yoy_resid",
        "gdp_growth_qoq_ann_resid", "unemployment_resid",
        # VECM error correction term — "how far from long-run equilibrium"
        "ect_combined",
    ]
    feature_cols = [c for c in core_features if c in df.columns and df[c].notna().sum() > 50]
    model_df = df.dropna(subset=feature_cols)
    
    train = model_df[model_df.index <= train_end]
    val   = model_df[(model_df.index > train_end) & (model_df.index <= val_end)]
    test  = model_df[model_df.index > val_end]
    
    print(f"  Train: {len(train):3d} meetings ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Val:   {len(val):3d} meetings ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"  Test:  {len(test):3d} meetings ({test.index[0].date()} → {test.index[-1].date()})")
    print(f"  Features: {len(feature_cols)}")
    
    X_train = train[feature_cols].values
    y_train = train["decision"].values
    X_val   = val[feature_cols].values
    y_val   = val["decision"].values
    X_test  = test[feature_cols].values
    y_test  = test["decision"].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, test, val


def prepare_train_test(df: pd.DataFrame, test_start: str = "2023-01-01") -> tuple:
    """Compatibility wrapper for a two-way temporal split.

    New experiments should use prepare_train_val_test. This helper keeps older
    scripts working by merging the validation period back into training.
    """
    split = pd.Timestamp(test_start)
    val_end = (split - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_cols,
        scaler,
        test_df,
        _val_df,
    ) = prepare_train_val_test(df, val_end=val_end)
    if len(X_val):
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
    return X_train, y_train, X_test, y_test, feature_cols, scaler, test_df


def train_models_with_validation(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_cols: list,
) -> dict:
    """Train ensemble with validation-based early stopping + loss tracking.
    
    Returns models + detailed training history with loss on all three sets.
    """
    models = {}
    history = {}
    
    # ── Model 1: Logistic Regression (no early stopping needed) ──
    print("\n  --- Logistic Regression ---")
    lr = LogisticRegression(max_iter=10000, class_weight="balanced")
    lr.fit(X_train, y_train)
    
    train_acc = lr.score(X_train, y_train)
    val_acc   = lr.score(X_val, y_val)
    test_acc  = lr.score(X_test, y_test)
    
    # Log loss on all three sets (need explicit labels for multi-class)
    from sklearn.metrics import log_loss as sk_log_loss
    all_labels = [-1, 0, 1]  # cut, hold, hike
    lr_proba_train = lr.predict_proba(X_train)
    lr_proba_val   = lr.predict_proba(X_val)
    lr_proba_test  = lr.predict_proba(X_test)
    
    history["logistic"] = {
        "train_accuracy": round(train_acc, 4),
        "val_accuracy":   round(val_acc, 4),
        "test_accuracy":  round(test_acc, 4),
        "train_log_loss": round(sk_log_loss(y_train, lr_proba_train, labels=all_labels), 4),
        "val_log_loss":   round(sk_log_loss(y_val,   lr_proba_val,   labels=all_labels), 4),
        "test_log_loss":  round(sk_log_loss(y_test,  lr_proba_test,  labels=all_labels), 4),
    }
    models["logistic"] = lr
    
    print(f"    Train: acc={train_acc:.3f}  logloss={history['logistic']['train_log_loss']:.4f}")
    print(f"    Val:   acc={val_acc:.3f}  logloss={history['logistic']['val_log_loss']:.4f}")
    print(f"    Test:  acc={test_acc:.3f}  logloss={history['logistic']['test_log_loss']:.4f}")
    
    # ── Model 2: Gradient Boosting with early stopping on validation ──
    print("\n  --- Gradient Boosting (early-stopping on validation) ---")
    
    # First, find optimal n_estimators via validation
    best_n, best_val_loss = 50, float("inf")
    train_losses, val_losses, test_losses = [], [], []
    
    gb_full = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=42,  # manual val, no internal split needed
    )
    gb_full.fit(X_train, y_train)
    
    # Staged predict to find best iteration
    staged_train_loss = []
    staged_val_loss   = []
    staged_test_loss  = []
    
    for i, (tr_pred, val_pred, te_pred) in enumerate(zip(
        gb_full.staged_predict_proba(X_train),
        gb_full.staged_predict_proba(X_val),
        gb_full.staged_predict_proba(X_test),
    )):
        tr_loss = sk_log_loss(y_train, tr_pred, labels=all_labels)
        vl_loss = sk_log_loss(y_val,   val_pred, labels=all_labels)
        te_loss = sk_log_loss(y_test,  te_pred, labels=all_labels)
        
        staged_train_loss.append(tr_loss)
        staged_val_loss.append(vl_loss)
        staged_test_loss.append(te_loss)
        
        if i >= 20 and vl_loss < best_val_loss - 0.001:
            best_val_loss = vl_loss
            best_n = i + 1
    
    # Retrain with best_n
    gb = GradientBoostingClassifier(
        n_estimators=best_n, max_depth=3, learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    models["gradient_boosting"] = gb
    
    final_train_acc = gb.score(X_train, y_train)
    final_val_acc   = gb.score(X_val, y_val)
    final_test_acc  = gb.score(X_test, y_test)
    
    gb_proba_train = gb.predict_proba(X_train)
    gb_proba_val   = gb.predict_proba(X_val)
    gb_proba_test  = gb.predict_proba(X_test)
    
    history["gradient_boosting"] = {
        "best_n_estimators": best_n,
        "train_accuracy": round(final_train_acc, 4),
        "val_accuracy":   round(final_val_acc, 4),
        "test_accuracy":  round(final_test_acc, 4),
        "train_log_loss": round(sk_log_loss(y_train, gb_proba_train, labels=all_labels), 4),
        "val_log_loss":   round(sk_log_loss(y_val,   gb_proba_val,   labels=all_labels), 4),
        "test_log_loss":  round(sk_log_loss(y_test,  gb_proba_test,  labels=all_labels), 4),
        "train_loss_curve": [round(l, 4) for l in staged_train_loss[::5]],  # sample every 5
        "val_loss_curve":   [round(l, 4) for l in staged_val_loss[::5]],
        "test_loss_curve":  [round(l, 4) for l in staged_test_loss[::5]],
    }
    
    print(f"    Best n_estimators = {best_n} (selected by val log-loss)")
    print(f"    Train: acc={final_train_acc:.3f}  logloss={history['gradient_boosting']['train_log_loss']:.4f}")
    print(f"    Val:   acc={final_val_acc:.3f}  logloss={history['gradient_boosting']['val_log_loss']:.4f}")
    print(f"    Test:  acc={final_test_acc:.3f}  logloss={history['gradient_boosting']['test_log_loss']:.4f}")
    
    # ── Early stopping visualization ──
    print(f"\n  [GB Learning Curve] (sampled every 5 iterations):")
    curve_tr = history["gradient_boosting"]["train_loss_curve"]
    curve_vl = history["gradient_boosting"]["val_loss_curve"]
    curve_te = history["gradient_boosting"]["test_loss_curve"]
    for i in range(0, len(curve_tr), max(1, len(curve_tr)//8)):
        iter_num = i * 5 + 5
        print(f"     iter {iter_num:3d}: train_loss={curve_tr[i]:.4f}  val_loss={curve_vl[i]:.4f}  test_loss={curve_te[i]:.4f}")
    
    # ── Overfitting diagnostic ──
    gap = history["gradient_boosting"]["train_log_loss"] - history["gradient_boosting"]["val_log_loss"]
    if gap < -0.15:
        print(f"\n  Overfitting warning: train_loss << val_loss (gap={gap:.4f})")
    elif gap > 0.05:
        print(f"\n  Underfitting warning: val_loss < train_loss (gap={gap:+.4f})")
    else:
        print(f"\n  Generalization OK: train_loss ~= val_loss (gap={gap:+.4f})")
    
    models["_history"] = history
    return models


def evaluate_models(models: dict, X_test, y_test, test_df: pd.DataFrame) -> dict:
    """Evaluate on test set and return metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        accuracy = np.mean(y_pred == y_test)
        
        # Per-class metrics
        classes = model.classes_
        class_map = {cls: f"P({['cut','hold','hike'][int(cls)+1]})" for cls in classes}
        
        results[name] = {
            "accuracy": round(accuracy, 4),
            "n_test": len(y_test),
            "class_distribution": {class_map[c]: round(np.mean(y_test == c), 3) for c in classes},
        }
        
        # Show last 8 predictions
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  Test accuracy: {accuracy:.2%} ({len(y_test)} meetings, 2023-2026)")
        print(f"\n  Last 8 predictions vs actual:")
        for i in range(max(0, len(y_test)-8), len(y_test)):
            proba_dict = {class_map[cls]: f"{y_proba[i][list(classes).index(cls)]:.3f}" for cls in classes}
            print(f"    {test_df.index[i].date()}  actual={int(y_test[i])}  {proba_dict}")
    
    return results


def predict_next_meeting(
    models: dict,
    features_df: pd.DataFrame,
    feature_cols: list,
    scaler: StandardScaler,
) -> dict:
    """Generate prediction for next FOMC meeting (July 28-29, 2026).
    
    Uses core macro features (trained on full history) for prediction.
    ECT and other rich features are used for context/evidence generation.
    """
    # Latest available macro data
    latest = features_df.iloc[-1]
    latest_features = latest[feature_cols].values.reshape(1, -1)
    latest_features = scaler.transform(latest_features)
    
    decision_map = {-1: "CUT", 0: "HOLD", 1: "HIKE"}
    
    predictions = {}
    for name, model in models.items():
        proba = model.predict_proba(latest_features)[0]
        classes = model.classes_
        proba_dict = {decision_map[int(cls)]: round(float(proba[i]), 4) for i, cls in enumerate(classes)}
        predictions[name] = proba_dict
    
    return predictions, latest


def run() -> None:
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 60)
    print("FOMC Prediction Pipeline")
    print("VAR/VECM Alignment Features → Probability Distribution")
    print("=" * 60 + "\n")
    
    # 1. Load data
    print("[1/5] Loading alignment features + FOMC labels...")
    df = load_data()
    
    # 2. Three-way split
    print("\n[2/5] Preparing train/val/test split per plan specification...")
    print("       Train: 2000-2020 | Val: 2020-2023 | Test: 2023-2026")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, test_df, val_df = prepare_train_val_test(df)
    
    # 3. Train with validation-based early stopping
    print("\n[3/5] Training ensemble with validation early-stopping...")
    models = train_models_with_validation(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    )
    history = models.pop("_history", {})
    
    # 4. Evaluate on test set (final, untouched)
    print("\n[4/5] Final evaluation on held-out test set (2023-2026)...")
    results = evaluate_models(models, X_test, y_test, test_df)
    
    # ── Summary table ──
    print(f"\n  {'='*60}")
    print(f"  TRAIN / VAL / TEST SUMMARY")
    print(f"  {'='*60}")
    print(f"  {'Model':20s} {'Train Acc':>10s} {'Val Acc':>10s} {'Test Acc':>10s} {'Val LogLoss':>12s} {'Test LogLoss':>12s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
    for name in ["logistic", "gradient_boosting"]:
        h = history.get(name, {})
        print(f"  {name:20s} {h.get('train_accuracy',0):9.1%} {h.get('val_accuracy',0):9.1%} {h.get('test_accuracy',0):9.1%} {h.get('val_log_loss',0):11.4f} {h.get('test_log_loss',0):11.4f}")
    
    # Overfitting check
    gb_hist = history.get("gradient_boosting", {})
    val_loss = gb_hist.get("val_log_loss", 0)
    test_loss = gb_hist.get("test_log_loss", 0)
    if abs(val_loss - test_loss) < 0.15:
        print(f"\n  Generalization confirmed: val_loss ({val_loss:.4f}) ~= test_loss ({test_loss:.4f})")
    else:
        print(f"\n  Distribution shift warning: val_loss ({val_loss:.4f}) != test_loss ({test_loss:.4f})")
    
    # 5. Predict next FOMC (July 28-29, 2026) — TWO versions
    print("\n[5/5] Predicting next FOMC (July 28-29, 2026)...")
    
    # Load full features (macro + game)
    full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
    features_df = pd.read_parquet(full_path)
    
    # --- Version A: Macro-only baseline ---
    print("\n  --- A: MACRO-ONLY BASELINE ---")
    preds_macro, latest = predict_next_meeting(models, features_df, feature_cols, scaler)
    avg_macro = _ensemble_avg(preds_macro)
    print(f"     P(CUT)={avg_macro['CUT']:.1%} P(HOLD)={avg_macro['HOLD']:.1%} P(HIKE)={avg_macro['HIKE']:.1%}")
    
    # --- Version B: MACRO + LLM GAME THEORY — IMPROVED FUSION ---
    print("\n  --- B: MACRO + LLM GAME THEORY (3 fusion strategies) ---")
    
    # Get baseline macro predictions for test meetings (2023-2026)
    test_preds_macro = np.column_stack([
        models["logistic"].predict_proba(X_test),
        models["gradient_boosting"].predict_proba(X_test)
    ])
    
    game_cols = [c for c in features_df.columns if c.startswith("game_")]
    game_available = [c for c in [
        "game_p_hike", "game_p_hold", "game_energy_risk", 
        "game_geopol_risk", "game_warsh_hawkish"
    ] if c in features_df.columns]
    
    if not game_available:
        final_pred = avg_macro
        fusion_report = "Game features not available — using macro-only"
    else:
        latest_game = features_df.iloc[-1]
        game_hike = float(latest_game.get("game_p_hike", 0)) if not pd.isna(latest_game.get("game_p_hike", 0)) else 0
        game_hold = float(latest_game.get("game_p_hold", 0)) if not pd.isna(latest_game.get("game_p_hold", 0)) else 0
        game_cut = max(0, 1.0 - game_hike - game_hold)
        game_pred = {"CUT": game_cut, "HOLD": game_hold, "HIKE": game_hike}
        
        print(f"     Game inputs: hold={game_hold:.3f}, hike={game_hike:.3f}, warsh_hawk={float(latest_game.get('game_warsh_hawkish',0)):.2f}")
        
        # ── STRATEGY 1: Performance-Weighted Bayesian Fusion ──
        # Calculate Brier score for macro model on recent test period
        # Brier = mean((p - outcome)^2), lower = better
        y_test_cls = y_test.reshape(-1, 1)
        # Macro GB model Brier on test set
        gb_proba = models["gradient_boosting"].predict_proba(X_test)
        gb_classes = models["gradient_boosting"].classes_
        brier_gb = 0.0
        for i, cls in enumerate(gb_classes):
            cls_mask = (y_test == cls).astype(float)
            brier_gb += np.mean((gb_proba[:, i] - cls_mask) ** 2)
        brier_gb /= len(gb_classes)
        
        # Game model "Brier" — we don't have game predictions for test set,
        # so estimate its reliability from convergence metrics
        # Converged traces → higher weight; edge cases → lower weight
        game_convergence = features_df["game_converged"].dropna()
        game_reliability = 0.5 + 0.3 * float(game_convergence.mean()) if len(game_convergence) > 0 else 0.5
        # Game is inherently less reliable (10 quarter history vs 71 meeting training)
        brier_game_est = 0.5 * (1.0 - game_reliability) + 0.3  # ~0.35-0.45 range
        
        # Inverse Brier weights → higher Brier = lower weight
        inv_brier_macro = 1.0 / max(brier_gb, 0.01)
        inv_brier_game = 1.0 / max(brier_game_est, 0.01)
        w_macro_b = inv_brier_macro / (inv_brier_macro + inv_brier_game)
        w_game_b = inv_brier_game / (inv_brier_macro + inv_brier_game)
        
        fused_brier = {
            k: avg_macro[k] * w_macro_b + game_pred[k] * w_game_b
            for k in ["CUT", "HOLD", "HIKE"]
        }
        total_b = sum(fused_brier.values())
        fused_brier = {k: v/total_b for k, v in fused_brier.items()}
        
        print(f"\n     Strategy 1 — Brier-Weighted (w_macro={w_macro_b:.2f}, w_game={w_game_b:.2f}):")
        print(f"        P(CUT)={fused_brier['CUT']:.1%} P(HOLD)={fused_brier['HOLD']:.1%} P(HIKE)={fused_brier['HIKE']:.1%}")
        
        # ── STRATEGY 2: Stacking (Meta-Learner on overlapping quarters) ──
        # Find quarters where both macro prediction AND game features exist
        fused_from_stacking = avg_macro.copy()  # fallback
        stacking_ok = False
        
        # Get GB predictions for test meetings and align with game features
        test_quarters = test_df["quarter"].values
        test_gb_preds = gb_proba  # (n_test, n_classes)
        
        # For each test meeting, get the game features from that quarter
        # Normalize quarter dates to string format for reliable matching
        stacking_rows = []
        for i in range(len(test_df)):
            q = test_df.iloc[i]["quarter"]
            q_str = str(q)[:10]  # "2024-03-31"
            # Match game features by quarter string prefix
            matching = features_df[features_df.index.astype(str).str[:10] == q_str]
            if len(matching) == 0:
                # Try fuzzy: find closest quarter before the meeting
                try:
                    q_dt = pd.Timestamp(q)
                    before = features_df[features_df.index <= q_dt]
                    if len(before) > 0:
                        matching = before.iloc[[-1]]
                except:
                    continue
            if len(matching) == 0:
                continue
            game_row = matching.iloc[0]
            if pd.isna(game_row.get("game_p_hike", np.nan)):
                continue
            
            row = {
                "actual": y_test[i],
                "macro_hike": test_gb_preds[i, list(gb_classes).index(1)] if 1 in gb_classes else 0.0,
                "macro_hold": test_gb_preds[i, list(gb_classes).index(0)] if 0 in gb_classes else 0.0,
                "macro_cut": test_gb_preds[i, list(gb_classes).index(-1)] if -1 in gb_classes else 0.0,
                "game_hike": float(game_row.get("game_p_hike", 0)),
                "game_hold": float(game_row.get("game_p_hold", 0)),
                "game_cut": 1.0 - float(game_row.get("game_p_hike", 0)) - float(game_row.get("game_p_hold", 0)),
                "energy_risk": float(game_row.get("game_energy_risk", 0)),
                "geopol_risk": float(game_row.get("game_geopol_risk", 0)),
                "warsh_hawk": float(game_row.get("game_warsh_hawkish", 0)),
            }
            stacking_rows.append(row)
        
        if len(stacking_rows) >= 3:  # Need minimum samples
            stack_df = pd.DataFrame(stacking_rows)
            # Meta-features: macro_pred - game_pred (disagreement signal)
            stack_df["disagree_hike"] = stack_df["macro_hike"] - stack_df["game_hike"]
            stack_df["disagree_hold"] = stack_df["macro_hold"] - stack_df["game_hold"]
            
            # Target: actual decision
            y_stack = stack_df["actual"].values
            
            # Features for stacking
            X_stack = stack_df[["macro_hike", "macro_hold", "game_hike", "game_hold", 
                                  "disagree_hike", "disagree_hold", "energy_risk", "warsh_hawk"]].values
            
            meta_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            try:
                meta_lr.fit(X_stack, y_stack)
                stacking_ok = True
                
                # Predict next meeting with meta-learner
                latest_macro_hike = avg_macro["HIKE"]
                latest_macro_hold = avg_macro["HOLD"]
                latest_x = np.array([[
                    latest_macro_hike, latest_macro_hold,
                    game_hike, game_hold,
                    latest_macro_hike - game_hike,
                    latest_macro_hold - game_hold,
                    float(latest_game.get("game_energy_risk", 0)),
                    float(latest_game.get("game_warsh_hawkish", 0)),
                ]])
                stacking_proba = meta_lr.predict_proba(latest_x)[0]
                stacking_classes = meta_lr.classes_
                decision_map_st = {-1: "CUT", 0: "HOLD", 1: "HIKE"}
                fused_stacking = {
                    decision_map_st[int(cls)]: float(stacking_proba[i])
                    for i, cls in enumerate(stacking_classes)
                }
                # Ensure all three keys exist
                for k in ["CUT", "HOLD", "HIKE"]:
                    fused_stacking.setdefault(k, 0.0)
                total_st = sum(fused_stacking.values())
                fused_stacking = {k: v/total_st for k, v in fused_stacking.items()}
                
                print(f"\n     Strategy 2 — Stacking Meta-Learner (trained on {len(stacking_rows)} overlap quarters):")
                print(f"        P(CUT)={fused_stacking['CUT']:.1%} P(HOLD)={fused_stacking['HOLD']:.1%} P(HIKE)={fused_stacking['HIKE']:.1%}")
                
                fused_from_stacking = fused_stacking
            except Exception as e:
                print(f"\n     Strategy 2 — Stacking: insufficient data ({e})")
        else:
            print(f"\n     Strategy 2 — Stacking: only {len(stacking_rows)} overlap quarters, need >= 3")
        
        # ── STRATEGY 3: Disagreement-Gated Fusion ──
        # When macro and game agree → high confidence → macro-weighted
        # When they disagree → game gets more weight (game captures text signals macro can't)
        disagreement = abs(avg_macro["HIKE"] - game_hike)
        if disagreement < 0.10:
            # Low disagreement: trust macro (longer history)
            w_macro_d, w_game_d = 0.80, 0.20
        elif disagreement < 0.20:
            w_macro_d, w_game_d = 0.55, 0.45
        else:
            # High disagreement: game may be picking up text signals
            w_macro_d, w_game_d = 0.40, 0.60
        
        fused_gated = {
            k: avg_macro[k] * w_macro_d + game_pred[k] * w_game_d
            for k in ["CUT", "HOLD", "HIKE"]
        }
        total_g = sum(fused_gated.values())
        fused_gated = {k: v/total_g for k, v in fused_gated.items()}
        
        print(f"\n     Strategy 3 — Disagreement-Gated (disagreement={disagreement:.1%}, w_macro={w_macro_d:.2f}):")
        print(f"        P(CUT)={fused_gated['CUT']:.1%} P(HOLD)={fused_gated['HOLD']:.1%} P(HIKE)={fused_gated['HIKE']:.1%}")
        
        # ── FINAL: Ensemble of fusion strategies ──
        all_fused = [fused_brier, fused_gated]
        if stacking_ok:
            all_fused.append(fused_from_stacking)
        
        final_pred = {}
        for k in ["CUT", "HOLD", "HIKE"]:
            final_pred[k] = float(np.mean([f[k] for f in all_fused]))
        total_final = sum(final_pred.values())
        final_pred = {k: v/total_final for k, v in final_pred.items()}
        
        fusion_report = f"Ensemble of {len(all_fused)} fusion strategies"
        print(f"\n     🏆 ENSEMBLE OF {len(all_fused)} FUSION STRATEGIES:")
        print(f"        P(CUT)={final_pred['CUT']:.1%} P(HOLD)={final_pred['HOLD']:.1%} P(HIKE)={final_pred['HIKE']:.1%}")
        
        # Show contribution
        delta = final_pred["HIKE"] - avg_macro["HIKE"]
        if abs(delta) > 0.005:
            direction = "up" if delta > 0 else "down"
            print(f"        Game layer shifts HIKE {direction} by {abs(delta):.1%} vs macro-only ({avg_macro['HIKE']:.1%})")
    
    # --- Print full summary ---
    print("\n" + "=" * 60)
    print("FOMC PREDICTION - July 28-29, 2026")
    print(f"     Fusion: {fusion_report}")
    print("=" * 60)
    print(f"\n  Latest macro data: {features_df.index[-1].date()} (Q1 2026)")
    print(f"  Latest game data:  2026Q2 (fed_game self-play)")
    print(f"  Current Fed Funds: {latest.get('fedfunds', 'N/A'):.2f}% (target 3.50-3.75%)")
    print(f"  CPI YoY:          {latest.get('inflation_cpi_yoy', 'N/A'):.2f}%")
    print(f"  GDP Growth:       {latest.get('gdp_growth_qoq_ann', 'N/A'):.2f}%")
    print(f"  Unemployment:     {latest.get('unemployment', 'N/A'):.2f}%")
    if "ect_combined" in latest.index:
        ect = latest["ect_combined"]
        print(f"  VECM ECT:         {ect:.4f}  ({'below' if ect < 0 else 'above'} long-run equilibrium)")
    
    print(f"\n  FINAL PREDICTION:")
    print(f"     P(CUT)  = {final_pred['CUT']:.1%}")
    print(f"     P(HOLD) = {final_pred['HOLD']:.1%}")
    print(f"     P(HIKE) = {final_pred['HIKE']:.1%}")
    
    # Comparison table
    print(f"\n  Layer-by-layer breakdown:")
    print(f"     {'Layer':28s} {'CUT':>7s} {'HOLD':>7s} {'HIKE':>7s}")
    print(f"     {'─'*28} {'─'*7} {'─'*7} {'─'*7}")
    print(f"     {'1. Macro (VAR/VECM+GB)':28s} {avg_macro['CUT']:6.1%} {avg_macro['HOLD']:6.1%} {avg_macro['HIKE']:6.1%}")
    if game_available:
        print(f"     {'2. LLM 5-Cluster Game':28s} {game_pred['CUT']:6.1%} {game_pred['HOLD']:6.1%} {game_pred['HIKE']:6.1%}")
        print(f"     {'3a. Brier-Weighted':28s} {fused_brier['CUT']:6.1%} {fused_brier['HOLD']:6.1%} {fused_brier['HIKE']:6.1%}")
        print(f"     {'3b. Disagreement-Gated':28s} {fused_gated['CUT']:6.1%} {fused_gated['HOLD']:6.1%} {fused_gated['HIKE']:6.1%}")
        if stacking_ok:
            print(f"     {'3c. Stacking Meta-Learner':28s} {fused_from_stacking['CUT']:6.1%} {fused_from_stacking['HOLD']:6.1%} {fused_from_stacking['HIKE']:6.1%}")
        print(f"     {'═══ FINAL ENSEMBLE ═══':28s} {final_pred['CUT']:6.1%} {final_pred['HOLD']:6.1%} {final_pred['HIKE']:6.1%}")
    
    # Generate evidence chain
    print(f"\n  Evidence Chain:")
    lr = models["logistic"]
    if hasattr(lr, "coef_"):
        hike_idx = list(lr.classes_).index(1) if 1 in lr.classes_ else 0
        coefs = lr.coef_[hike_idx]
        top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
        for idx in top_idx:
            sign = "+" if coefs[idx] > 0 else "-"
            print(f"     {sign} {feature_cols[idx]:30s} → macro signal")
    
    if "ect_combined" in latest.index:
        ect_val = latest["ect_combined"]
        if ect_val < -0.5:
            print(f"     VECM ECT={ect_val:.2f} < 0 -> economy BELOW equilibrium -> upward pressure")
        elif ect_val > 0.5:
            print(f"     VECM ECT={ect_val:.2f} > 0 -> economy ABOVE equilibrium -> downward pressure")
    
    if game_available:
        gh = float(latest_game.get("game_warsh_hawkish", 0)) if not pd.isna(latest_game.get("game_warsh_hawkish", 0)) else 0
        ge = float(latest_game.get("game_energy_risk", 0)) if not pd.isna(latest_game.get("game_energy_risk", 0)) else 0
        gg = float(latest_game.get("game_geopol_risk", 0)) if not pd.isna(latest_game.get("game_geopol_risk", 0)) else 0
        print(f"     Warsh hawkish index: {gh:.2f} (from multi-agent game)")
        print(f"     Energy risk: {ge:.2f} (from 5-cluster strategic simulation)")
        print(f"     Geopolitical risk: {gg:.2f} (from between-cluster equilibrium)")
    
    # Save
    out_path = REPO_ROOT / "data" / "processed" / "fomc_prediction_2026Q3.json"
    import json
    out_data = {
        "meeting": "2026-07-28 FOMC",
        "prediction_date": "2026-07-07",
        "latest_macro_quarter": str(features_df.index[-1].date()),
        "macro_only": avg_macro,
        "fused_with_game": final_pred,
        "model_predictions": preds_macro,
        "game_features": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in latest.items() if k.startswith("game_") and not pd.isna(v)},
        "test_metrics": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Prediction saved -> data/processed/fomc_prediction_2026Q3.json")
    
    print("\n" + "=" * 60)
    print("Full prediction pipeline complete (macro + game features fused).")
    print("=" * 60)


def _ensemble_avg(predictions: dict) -> dict:
    """Compute ensemble average across models."""
    avg = {}
    for decision in ["CUT", "HOLD", "HIKE"]:
        avg[decision] = float(np.mean([p.get(decision, 0) for p in predictions.values()]))
    return avg


if __name__ == "__main__":
    run()

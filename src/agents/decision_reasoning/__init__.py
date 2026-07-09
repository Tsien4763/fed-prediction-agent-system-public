"""Agent 3: Decision & Reasoning — multi-model prediction with VAR/VECM alignment.

Responsible for:
  - VAR model: linear macro dynamics + residual extraction
  - VECM model: long-run cointegration + Error Correction Terms
  - Residual TFT: static covariates -> variable selection -> LSTM
    -> interpretable attention -> VAR/VECM cross-attention -> quantiles
  - GB/Logistic: small-sample robust classifiers
  - Three-strategy fusion: Brier-weighted + Disagreement-gated + Stacking
  - Train/val/test temporal split with early stopping

Entry points:
  from agents.decision_reasoning import predict_fomc, train_models, evaluate

Implementation:
  Core logic in models/fomc_predictor.py (prediction pipeline)
  VAR in models/var_model.py
  VECM in models/vecm_model.py
  TFT in models/tft_model.py
  Alignment layer in models/alignment_layer.py
"""
from typing import Any

from agents.runtime_support import append_audit


class DecisionReasoningAgent:
    """Agent boundary for macro/semantic decision feature preparation."""

    name = "decision_reasoning"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        scores = [
            float(item.get("score", {}).get("hawkish_dovish_score", 0.0))
            for item in state.get("semantic_signals", [])
            if isinstance(item.get("score"), dict)
        ]
        mean_hawkish = sum(scores) / len(scores) if scores else 0.0
        requested_prediction = state.get("decision_context", {}).get("fed_prediction") or state.get("fed_prediction", {})
        decision_context = {
            "status": "prepared",
            "facade": __name__,
            "semantic_signal_count": len(scores),
            "mean_hawkish_dovish_score": round(mean_hawkish, 6),
            "fed_prediction": requested_prediction,
            "model_entrypoints": ["predict_fomc", "train_models", "evaluate", "split_data"],
            "note": (
                "DecisionReasoningAgent prepared decision features. Full model execution is handled "
                "by this facade's model entrypoints and fed_game CLI commands."
            ),
        }
        state["decision_context"] = decision_context
        return append_audit(state, self.name, decision_context)


def predict_fomc(*args, **kwargs):
    from models.fomc_predictor import run

    return run(*args, **kwargs)


def train_models(*args, **kwargs):
    from models.fomc_predictor import train_models_with_validation

    return train_models_with_validation(*args, **kwargs)


def evaluate(*args, **kwargs):
    from models.fomc_predictor import evaluate_models

    return evaluate_models(*args, **kwargs)


def split_data(*args, **kwargs):
    from models.fomc_predictor import prepare_train_val_test

    return prepare_train_val_test(*args, **kwargs)


def build_var(*args, **kwargs):
    from models.var_model import build_var as _build_var

    return _build_var(*args, **kwargs)


def build_vecm(*args, **kwargs):
    from models.vecm_model import build_vecm as _build_vecm

    return _build_vecm(*args, **kwargs)


def build_varx_features(*args, **kwargs):
    from models.alignment_layer import build_varx_features as _build_varx_features

    return _build_varx_features(*args, **kwargs)


def compute_economics_penalty(*args, **kwargs):
    from models.alignment_layer import compute_economics_penalty as _compute_economics_penalty

    return _compute_economics_penalty(*args, **kwargs)


def compute_taylor_rule_rate(*args, **kwargs):
    from models.alignment_layer import compute_taylor_rule_rate as _compute_taylor_rule_rate

    return _compute_taylor_rule_rate(*args, **kwargs)

__all__ = [
    "DecisionReasoningAgent",
    "predict_fomc",
    "train_models",
    "evaluate",
    "split_data",
    "build_var",
    "build_vecm",
    "build_varx_features",
    "compute_economics_penalty",
    "compute_taylor_rule_rate",
]

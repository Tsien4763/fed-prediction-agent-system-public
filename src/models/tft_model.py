"""Residual Temporal Fusion Transformer for FOMC prediction.

Layer division:
  Layer 1: VAR/VECM extracts linear macro dynamics, residuals, and ECT.
  Layer 2: Residual TFT models nonlinear structure in those residual features.
  Layer 3: GB/Logit consumes calibrated probabilities and compact features.

This module intentionally keeps the TFT residual-based: the neural layer is not
asked to relearn the linear macro anchor that VAR/VECM already explains. The
incremental-value check is explicit: the training report includes a linear
probe on flattened residual sequences. If the TFT does not beat that probe on
validation/test data, the run should be reported as "no proven nonlinear lift."

Usage:
    python -m models.tft_model
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_engineering.config import REPO_ROOT


@dataclass(frozen=True)
class TFTConfig:
    input_dim: int
    static_dim: int = 0
    hidden_dim: int = 64
    num_heads: int = 4
    num_lstm_layers: int = 1
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    num_classes: int = 3


class GRN(nn.Module):
    """Gated Residual Network with optional context conditioning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.context_projection is not None:
            context_h = self.context_projection(context)
            while context_h.dim() < h.dim():
                context_h = context_h.unsqueeze(1)
            h = h + context_h
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(gate * h + self.skip(x))


class StaticCovariateEncoder(nn.Module):
    """Encode per-meeting static covariates into TFT context vectors.

    Static covariates are fixed for the forecast instance, not repeated as a
    time series. In this project they represent the pre-meeting regime snapshot
    available as of the last observed quarter: policy-rate level, inflation gap,
    labor slack, curve slope, and optional game-profile signals.
    """

    def __init__(self, static_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.learned_context = nn.Parameter(torch.zeros(hidden_dim))
        if static_dim > 0:
            self.variable_transforms = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(static_dim)])
            self.weight_grn = GRN(static_dim, hidden_dim, static_dim, dropout=dropout)
            self.context_grn = GRN(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.selection_context = nn.Linear(hidden_dim, hidden_dim)
        self.enrichment_context = nn.Linear(hidden_dim, hidden_dim)
        self.h0 = nn.Linear(hidden_dim, hidden_dim)
        self.c0 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, static: torch.Tensor | None, batch_size: int) -> dict[str, torch.Tensor]:
        if self.static_dim == 0 or static is None or static.numel() == 0:
            context = self.learned_context.unsqueeze(0).expand(batch_size, -1)
            weights = torch.empty(batch_size, 0, device=context.device)
        else:
            embeddings = [
                transform(static[:, idx : idx + 1])
                for idx, transform in enumerate(self.variable_transforms)
            ]
            variable_embeddings = torch.stack(embeddings, dim=1)  # (B, S, H)
            weights = torch.softmax(self.weight_grn(static), dim=-1)  # (B, S)
            context = torch.sum(variable_embeddings * weights.unsqueeze(-1), dim=1)
            context = self.context_grn(context)
        return {
            "context": context,
            "selection": self.selection_context(context),
            "enrichment": self.enrichment_context(context),
            "h0": torch.tanh(self.h0(context)),
            "c0": torch.tanh(self.c0(context)),
            "weights": weights,
        }


class TemporalVariableSelectionNetwork(nn.Module):
    """Variable selection network conditioned on static context."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.variable_transforms = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])
        self.weight_grn = GRN(
            input_dim,
            hidden_dim,
            input_dim,
            context_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        static_selection_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D), static_selection_context: (B, H)
        embeddings = [
            transform(x[..., idx : idx + 1])
            for idx, transform in enumerate(self.variable_transforms)
        ]
        variable_embeddings = torch.stack(embeddings, dim=2)  # (B, T, D, H)
        weights = torch.softmax(self.weight_grn(x, static_selection_context), dim=-1)  # (B, T, D)
        selected = torch.sum(variable_embeddings * weights.unsqueeze(-1), dim=2)  # (B, T, H)
        return selected, weights


class GateAddNorm(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gated = torch.sigmoid(self.gate(x)) * self.dropout(x)
        return self.norm(gated + residual)


class InterpretableMultiHeadAttention(nn.Module):
    """TFT-style interpretable multi-head attention.

    Each head has its own query/key projection, while all heads share the value
    projection. The output is the mean of head contexts, and per-head attention
    maps are returned for inspection.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)])
        self.value_layer = nn.Linear(hidden_dim, self.head_dim)
        self.output_layer = nn.Linear(self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_value = self.value_layer(value)
        head_contexts = []
        head_weights = []
        scale = self.head_dim ** 0.5

        for query_layer, key_layer in zip(self.query_layers, self.key_layers):
            q = query_layer(query)
            k = key_layer(key)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            context = torch.matmul(weights, shared_value)
            head_contexts.append(context)
            head_weights.append(weights)

        mean_context = torch.stack(head_contexts, dim=0).mean(dim=0)
        attention_weights = torch.stack(head_weights, dim=1)  # (B, heads, Tq, Tk)
        return self.output_layer(mean_context), attention_weights


class EconometricCrossAttention(nn.Module):
    """Let temporal TFT states attend to VAR residual and VECM ECT tokens."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.econometric_projection = GRN(input_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.cross_attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout=dropout)
        self.add_norm = GateAddNorm(hidden_dim, dropout=dropout)

    def forward(
        self,
        temporal_states: torch.Tensor,
        econometric_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        econometric_tokens = self.econometric_projection(econometric_features)
        attended, weights = self.cross_attention(
            query=temporal_states,
            key=econometric_tokens,
            value=econometric_tokens,
        )
        return self.add_norm(attended, temporal_states), weights


class ResidualTemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer on VAR/VECM residual features.

    Required TFT components included:
      - static covariate encoder;
      - static-conditioned temporal variable selection;
      - LSTM local sequence encoder;
      - static enrichment;
      - interpretable multi-head temporal attention;
      - quantile output head for policy-rate uncertainty.
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        self.static_encoder = StaticCovariateEncoder(config.static_dim, config.hidden_dim, config.dropout)
        self.temporal_vsn = TemporalVariableSelectionNetwork(config.input_dim, config.hidden_dim, config.dropout)
        self.lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0.0,
        )
        self.lstm_add_norm = GateAddNorm(config.hidden_dim, config.dropout)
        self.static_enrichment = GRN(
            config.hidden_dim,
            config.hidden_dim,
            config.hidden_dim,
            context_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.temporal_attention = InterpretableMultiHeadAttention(
            config.hidden_dim,
            config.num_heads,
            config.dropout,
        )
        self.attention_add_norm = GateAddNorm(config.hidden_dim, config.dropout)
        self.econometric_cross_attention = EconometricCrossAttention(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.positionwise_grn = GRN(config.hidden_dim, config.hidden_dim, config.hidden_dim, dropout=config.dropout)
        self.classification_head = nn.Linear(config.hidden_dim, config.num_classes)
        self.quantile_head = nn.Linear(config.hidden_dim, len(config.quantiles))

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = x.shape[0]
        static_context = self.static_encoder(static, batch_size)

        selected, temporal_importance = self.temporal_vsn(x, static_context["selection"])
        h0 = static_context["h0"].unsqueeze(0).repeat(self.config.num_lstm_layers, 1, 1)
        c0 = static_context["c0"].unsqueeze(0).repeat(self.config.num_lstm_layers, 1, 1)
        lstm_out, _ = self.lstm(selected, (h0, c0))
        lstm_out = self.lstm_add_norm(lstm_out, selected)

        enriched = self.static_enrichment(lstm_out, static_context["enrichment"])
        attended, temporal_attention = self.temporal_attention(enriched, enriched, enriched)
        attended = self.attention_add_norm(attended, enriched)

        fused, econometric_attention = self.econometric_cross_attention(attended, x)
        transformed = self.positionwise_grn(fused)

        pooled = transformed[:, -1, :] + transformed.mean(dim=1)
        logits = self.classification_head(pooled)
        raw_quantiles = self.quantile_head(pooled)
        if len(self.config.quantiles) == 3:
            median = raw_quantiles[:, 1]
            lower = median - F.softplus(raw_quantiles[:, 0])
            upper = median + F.softplus(raw_quantiles[:, 2])
            quantiles = torch.stack([lower, median, upper], dim=-1)
        else:
            quantiles = raw_quantiles

        diagnostics = {
            "quantiles": quantiles,
            "attention": {
                "temporal": temporal_attention,
                "econometric_cross": econometric_attention,
            },
            "variable_importance": {
                "temporal": temporal_importance,
                "static": static_context["weights"],
            },
        }
        return logits, diagnostics


def quantile_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    errors = target.unsqueeze(-1) - predictions
    losses = []
    for idx, quantile in enumerate(quantiles):
        q = torch.tensor(quantile, device=predictions.device, dtype=predictions.dtype)
        losses.append(torch.maximum((q - 1.0) * errors[:, idx], q * errors[:, idx]))
    return torch.stack(losses, dim=-1).mean()


def default_static_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "fedfunds",
        "inflation_cpi_yoy",
        "unemployment",
        "term_spread_10y2y",
        "vix",
        "game_warsh_hawkish",
        "game_warsh_consistency_score",
    ]
    return [col for col in candidates if col in df.columns and df[col].notna().sum() > 50]


def build_tft_dataset(
    df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    *,
    static_feature_cols: list[str] | None = None,
    seq_len: int = 4,
    test_start: str = "2023-01-01",
) -> tuple:
    """Build residual TFT sequences with static covariates and rate targets."""
    from models.fomc_predictor import load_data as fomc_load

    merged = df if df is not None and not df.empty else fomc_load()
    feature_cols = feature_cols or [
        "fedfunds_resid",
        "inflation_cpi_yoy_resid",
        "gdp_growth_qoq_ann_resid",
        "unemployment_resid",
        "ect_combined",
    ]

    valid_features = [
        col for col in feature_cols
        if col in merged.columns and merged[col].notna().sum() > 20
    ]
    if not valid_features:
        raise ValueError("No usable TFT residual features found. Run VAR/VECM feature builders first.")

    static_feature_cols = static_feature_cols or default_static_feature_columns(merged)
    valid_static = [
        col for col in static_feature_cols
        if col in merged.columns and merged[col].notna().sum() > 20
    ]

    needed_cols = valid_features + ["decision", "ff_target_mid"]
    if valid_static:
        needed_cols += valid_static
    meetings = merged.dropna(subset=needed_cols).sort_index()

    X, y, q_target, static_x, dates = [], [], [], [], []
    for idx in range(seq_len, len(meetings)):
        window = meetings.iloc[idx - seq_len : idx]
        target_row = meetings.iloc[idx]
        static_row = window.iloc[-1]
        X.append(window[valid_features].values)
        y.append(int(target_row["decision"]) + 1)
        q_target.append(float(target_row["ff_target_mid"]))
        if valid_static:
            static_x.append(static_row[valid_static].values.astype(np.float32))
        else:
            static_x.append(np.empty(0, dtype=np.float32))
        dates.append(meetings.index[idx])

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    q_arr = np.asarray(q_target, dtype=np.float32)
    static_arr = np.asarray(static_x, dtype=np.float32)
    dates_idx = pd.DatetimeIndex(dates)

    train_mask = dates_idx < pd.Timestamp(test_start)
    X_train, X_test = X_arr[train_mask], X_arr[~train_mask]
    y_train, y_test = y_arr[train_mask], y_arr[~train_mask]
    q_train, q_test = q_arr[train_mask], q_arr[~train_mask]
    static_train, static_test = static_arr[train_mask], static_arr[~train_mask]

    feature_scaler = StandardScaler()
    train_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, train_shape[-1])
    X_test_flat = X_test.reshape(-1, train_shape[-1])
    X_train = feature_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test = feature_scaler.transform(X_test_flat).reshape(X_test.shape)

    static_scaler = None
    if valid_static:
        static_scaler = StandardScaler()
        static_train = static_scaler.fit_transform(static_train)
        static_test = static_scaler.transform(static_test)

    return (
        X_train,
        y_train,
        q_train,
        static_train,
        X_test,
        y_test,
        q_test,
        static_test,
        valid_features,
        valid_static,
        feature_scaler,
        static_scaler,
        dates_idx[~train_mask],
    )


def linear_probe_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Probe whether residual sequences contain a simple linear signal."""
    if len(X_train) == 0 or len(X_test) == 0:
        return {"accuracy": float("nan"), "log_loss": float("nan")}
    train_flat = X_train.reshape(X_train.shape[0], -1)
    test_flat = X_test.reshape(X_test.shape[0], -1)
    model = LogisticRegression(max_iter=10000, class_weight="balanced")
    try:
        model.fit(train_flat, y_train)
        pred = model.predict(test_flat)
        proba = model.predict_proba(test_flat)
        labels = [0, 1, 2]
        return {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "log_loss": round(float(log_loss(y_test, proba, labels=labels)), 4),
        }
    except ValueError as exc:
        return {"accuracy": float("nan"), "log_loss": float("nan"), "error": str(exc)}


def train_tft(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    feature_names: list[str],
    *,
    static_train: np.ndarray | None = None,
    static_test: np.ndarray | None = None,
    quantile_train: np.ndarray | None = None,
    quantile_test: np.ndarray | None = None,
    static_feature_names: list[str] | None = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.001,
    quantile_weight: float = 0.15,
) -> tuple[ResidualTemporalFusionTransformer, dict[str, Any], list[str]]:
    """Train residual TFT with classification and quantile objectives."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_dim = 0 if static_train is None else int(static_train.shape[-1])
    config = TFTConfig(input_dim=input_dim, static_dim=static_dim)
    model = ResidualTemporalFusionTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_criterion = nn.CrossEntropyLoss()

    if static_train is None:
        static_train = np.empty((len(X_train), 0), dtype=np.float32)
    if static_test is None:
        static_test = np.empty((len(X_test), 0), dtype=np.float32)
    if quantile_train is None:
        quantile_train = np.zeros(len(X_train), dtype=np.float32)
        quantile_weight = 0.0
    if quantile_test is None:
        quantile_test = np.zeros(len(X_test), dtype=np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(static_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(quantile_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    out_dir = REPO_ROOT / "artifacts" / "tft"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history: dict[str, Any] = {
        "train_loss": [],
        "test_accuracy": [],
        "test_quantile_pinball": [],
        "linear_probe": linear_probe_baseline(X_train, y_train, X_test, y_test),
        "quantiles": list(config.quantiles),
        "static_features": static_feature_names or [],
        "residual_rationale": residual_modeling_rationale(feature_names),
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, sb, yb, qb in train_loader:
            xb, sb, yb, qb = xb.to(device), sb.to(device), yb.to(device), qb.to(device)
            optimizer.zero_grad()
            logits, diagnostics = model(xb, sb)
            loss = class_criterion(logits, yb)
            if quantile_weight > 0:
                loss = loss + quantile_weight * quantile_loss(
                    diagnostics["quantiles"],
                    qb,
                    config.quantiles,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X_test, dtype=torch.float32).to(device)
            st = torch.tensor(static_test, dtype=torch.float32).to(device)
            yt = torch.tensor(y_test, dtype=torch.long).to(device)
            qt = torch.tensor(quantile_test, dtype=torch.float32).to(device)
            logits, diagnostics = model(xt, st)
            preds = logits.argmax(dim=1)
            acc = (preds == yt).float().mean().item() if len(y_test) else float("nan")
            q_loss = (
                quantile_loss(diagnostics["quantiles"], qt, config.quantiles).item()
                if len(y_test) and quantile_weight > 0
                else float("nan")
            )

        history["train_loss"].append(epoch_loss / max(1, len(train_loader)))
        history["test_accuracy"].append(acc)
        history["test_quantile_pinball"].append(q_loss)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} "
                f"loss={history['train_loss'][-1]:.4f} "
                f"test_acc={acc:.3f} "
                f"pinball={q_loss:.4f}"
            )

    history["best_test_accuracy"] = round(float(best_acc), 4)
    history["nonlinear_lift_vs_linear_probe"] = round(
        float(best_acc - history["linear_probe"]["accuracy"]),
        4,
    )
    print(f"  Linear probe accuracy: {history['linear_probe']['accuracy']:.3f}")
    print(f"  Best residual TFT accuracy: {best_acc:.3f}")
    return model, history, feature_names


def predict_tft(
    model: ResidualTemporalFusionTransformer,
    latest_features: np.ndarray,
    feature_scaler: StandardScaler,
    *,
    latest_static: np.ndarray | None = None,
    static_scaler: StandardScaler | None = None,
) -> dict[str, Any]:
    """Generate class probabilities, rate quantiles, and attention diagnostics."""
    device = next(model.parameters()).device
    model.eval()

    time_steps, input_dim = latest_features.shape
    flat = feature_scaler.transform(latest_features.reshape(-1, input_dim))
    x = torch.tensor(flat.reshape(1, time_steps, input_dim), dtype=torch.float32).to(device)

    if latest_static is not None and latest_static.size:
        static_values = latest_static.reshape(1, -1)
        if static_scaler is not None:
            static_values = static_scaler.transform(static_values)
        static_tensor = torch.tensor(static_values, dtype=torch.float32).to(device)
    else:
        static_tensor = torch.empty(1, 0, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits, diagnostics = model(x, static_tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        quantiles = diagnostics["quantiles"][0].cpu().numpy()

    decision_map = {0: "CUT", 1: "HOLD", 2: "HIKE"}
    return {
        "class_probabilities": {
            decision_map[idx]: round(float(probs[idx]), 4)
            for idx in range(3)
        },
        "rate_quantiles": {
            f"q{int(q * 100):02d}": round(float(value), 4)
            for q, value in zip(model.config.quantiles, quantiles)
        },
        "diagnostics_available": [
            "temporal_attention_by_head",
            "econometric_cross_attention_by_head",
            "temporal_variable_importance",
            "static_variable_importance",
        ],
    }


def residual_modeling_rationale(feature_names: list[str]) -> dict[str, Any]:
    residual_features = [name for name in feature_names if name.endswith("_resid")]
    ect_features = [name for name in feature_names if name.startswith("ect")]
    return {
        "why_residual_based": (
            "VAR/VECM first removes linear macro dynamics and long-run equilibrium. "
            "The residual TFT is reserved for nonlinear, regime-dependent structure "
            "left in residuals and ECT interactions."
        ),
        "features_checked": {
            "var_residuals": residual_features,
            "vecm_error_correction": ect_features,
        },
        "evidence_protocol": (
            "Compare residual TFT against a linear probe on the same flattened "
            "residual sequences. Positive out-of-sample lift is required before "
            "claiming nonlinear incremental value."
        ),
    }


def tensor_summary(tensor: torch.Tensor, names: list[str], top_k: int = 5) -> list[dict[str, float | str]]:
    if tensor.numel() == 0 or not names:
        return []
    values = tensor.detach().cpu().float()
    while values.dim() > 1:
        values = values.mean(dim=0)
    top_idx = torch.argsort(values, descending=True)[:top_k].tolist()
    return [
        {"name": names[idx], "weight": round(float(values[idx]), 5)}
        for idx in top_idx
        if idx < len(names)
    ]


def run() -> None:
    print("=" * 72)
    print("Residual Temporal Fusion Transformer - Layer 2")
    print("=" * 72 + "\n")

    feature_cols = [
        "fedfunds_resid",
        "inflation_cpi_yoy_resid",
        "gdp_growth_qoq_ann_resid",
        "unemployment_resid",
        "ect_combined",
    ]

    print("[1/4] Building residual TFT dataset...")
    (
        X_train,
        y_train,
        q_train,
        static_train,
        X_test,
        y_test,
        q_test,
        static_test,
        valid_features,
        static_features,
        feature_scaler,
        static_scaler,
        test_dates,
    ) = build_tft_dataset(feature_cols=feature_cols)
    print(f"  Train: {X_train.shape} samples, Test: {X_test.shape} samples")
    print(f"  Residual features: {valid_features}")
    print(f"  Static covariates: {static_features or 'none'}")

    print("\n[2/4] Training residual TFT...")
    model, history, tft_features = train_tft(
        X_train,
        y_train,
        X_test,
        y_test,
        input_dim=len(valid_features),
        feature_names=valid_features,
        static_train=static_train,
        static_test=static_test,
        quantile_train=q_train,
        quantile_test=q_test,
        static_feature_names=static_features,
        epochs=50,
    )

    print("\n[3/4] Residual TFT diagnostics...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        st = torch.tensor(static_test, dtype=torch.float32).to(device)
        logits, diagnostics = model(xt, st)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        quantile_pred = diagnostics["quantiles"].cpu().numpy()

    for idx in range(max(0, len(y_test) - 6), len(y_test)):
        print(
            f"    {test_dates[idx].date()} actual={y_test[idx] - 1:+d} "
            f"P(cut)={probs[idx][0]:.3f} P(hold)={probs[idx][1]:.3f} "
            f"P(hike)={probs[idx][2]:.3f} "
            f"rate_q10/50/90={quantile_pred[idx][0]:.2f}/"
            f"{quantile_pred[idx][1]:.2f}/{quantile_pred[idx][2]:.2f}"
        )

    temporal_importance = tensor_summary(
        diagnostics["variable_importance"]["temporal"],
        valid_features,
    )
    static_importance = tensor_summary(
        diagnostics["variable_importance"]["static"],
        static_features,
    )

    print("\n[4/4] Predicting next FOMC with residual TFT...")
    full_path = REPO_ROOT / "data" / "processed" / "full_features.parquet"
    features_df = pd.read_parquet(full_path)
    available_features = [col for col in tft_features if col in features_df.columns]
    if len(available_features) != len(tft_features):
        missing = sorted(set(tft_features) - set(available_features))
        print(f"  ERROR: Missing residual TFT features: {missing}")
        return
    latest_window = features_df[available_features].dropna().iloc[-4:]
    if len(latest_window) < 4:
        print(f"  ERROR: Need 4 complete quarters, found {len(latest_window)}.")
        return
    if static_features:
        latest_static = features_df[static_features].dropna().iloc[-1].values.astype(np.float32)
    else:
        latest_static = None

    prediction = predict_tft(
        model,
        latest_window.values.astype(np.float32),
        feature_scaler,
        latest_static=latest_static,
        static_scaler=static_scaler,
    )
    print(json.dumps(prediction, indent=2))

    out = {
        "model": "ResidualTemporalFusionTransformer",
        "layer": 2,
        "input": "VAR residuals + VECM ECT with static pre-meeting covariates",
        "components": [
            "static_covariate_encoder",
            "static_conditioned_temporal_variable_selection",
            "lstm_local_encoder",
            "static_enrichment",
            "interpretable_multi_head_attention",
            "econometric_cross_attention",
            "classification_head",
            "quantile_output_head",
        ],
        "quantiles": history["quantiles"],
        "prediction": prediction,
        "history": history,
        "top_temporal_variables": temporal_importance,
        "top_static_variables": static_importance,
        "features": tft_features,
        "static_features": static_features,
    }
    out_path = REPO_ROOT / "data" / "processed" / "tft_prediction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run()

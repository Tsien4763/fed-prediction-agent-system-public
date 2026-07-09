from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from fed_game.evaluation import evaluate_forecasting_traces
from fed_game.llm import TeacherClient
from models.semantic_pipeline import _keyword_score, _post_chat_completion
from models.var_model import build_var
from models.vecm_model import build_vecm


def test_keyword_semantic_scorer_separates_hawkish_and_dovish_language() -> None:
    hawkish = _keyword_score(
        "Inflation remains elevated. The Committee is strongly committed to price stability "
        "and ongoing increases may be appropriate."
    )
    dovish = _keyword_score(
        "The labor market is softening, downside risk is rising, and policy can be patient "
        "while supporting maximum employment."
    )

    assert hawkish["hawkish_dovish_score"] > 0
    assert hawkish["rate_hike_signal"] > hawkish["rate_cut_signal"]
    assert dovish["hawkish_dovish_score"] < 0
    assert dovish["rate_cut_signal"] > dovish["rate_hike_signal"]


def test_llm_semantic_client_uses_timeout_and_bounded_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from models import semantic_pipeline

    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if len(calls) == 1:
            raise semantic_pipeline.requests.Timeout("temporary timeout")
        return FakeResponse()

    monkeypatch.setattr(semantic_pipeline.requests, "post", fake_post)
    monkeypatch.setattr(semantic_pipeline.time, "sleep", lambda _seconds: None)

    result = _post_chat_completion(
        "https://api.example.test/v1/chat/completions",
        payload={"model": "deepseek-chat"},
        api_key="test-key",
        timeout_seconds=7,
        max_attempts=2,
    )

    assert result["choices"][0]["message"]["content"] == "{}"
    assert len(calls) == 2
    assert calls[0]["timeout"] == 7
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_teacher_client_uses_requests_timeout_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from fed_game import llm

    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if len(calls) == 1:
            raise llm.requests.ConnectionError("temporary connection error")
        return FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    client = TeacherClient(
        base_url="https://api.example.test",
        model="deepseek-chat",
        api_key="test-key",
        timeout_seconds=9,
    )

    assert client.chat_json([{"role": "user", "content": "return JSON"}]) == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["timeout"] == 9
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_var_build_outputs_aligned_residuals_and_fitted_values() -> None:
    rng = np.random.default_rng(7)
    index = pd.period_range("2000Q1", periods=64, freq="Q").to_timestamp()
    base = rng.normal(size=len(index)).cumsum()
    df = pd.DataFrame(
        {
            "fedfunds": base + rng.normal(scale=0.05, size=len(index)),
            "inflation_cpi_yoy": 0.6 * base + rng.normal(scale=0.05, size=len(index)),
            "gdp_growth_qoq_ann": -0.2 * base + rng.normal(scale=0.05, size=len(index)),
            "unemployment": -0.4 * base + rng.normal(scale=0.05, size=len(index)),
        },
        index=index,
    )

    model, residuals, fitted = build_var(
        df,
        ["fedfunds", "inflation_cpi_yoy", "gdp_growth_qoq_ann", "unemployment"],
        maxlags=1,
        ic=None,
    )

    assert model.k_ar == 1
    assert residuals.shape == fitted.shape
    assert list(residuals.columns) == [
        "fedfunds_resid",
        "inflation_cpi_yoy_resid",
        "gdp_growth_qoq_ann_resid",
        "unemployment_resid",
    ]
    assert np.isfinite(residuals.to_numpy()).all()


def test_vecm_build_outputs_error_correction_terms() -> None:
    rng = np.random.default_rng(11)
    index = pd.period_range("2000Q1", periods=80, freq="Q").to_timestamp()
    trend = rng.normal(scale=0.2, size=len(index)).cumsum()
    df = pd.DataFrame(
        {
            "fedfunds": trend + rng.normal(scale=0.01, size=len(index)),
            "inflation_cpi_yoy": 0.8 * trend + rng.normal(scale=0.01, size=len(index)),
            "gdp_growth_qoq_ann": -0.3 * trend + rng.normal(scale=0.01, size=len(index)),
        },
        index=index,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
        _model, ect = build_vecm(
            df,
            ["fedfunds", "inflation_cpi_yoy", "gdp_growth_qoq_ann"],
            maxlags=1,
        )

    assert "ect_combined" in ect.columns
    assert len(ect) == len(df) - 1
    assert np.isfinite(ect["ect_combined"].to_numpy()).all()


def test_forecast_report_contains_calibration_buckets(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    rows = [
        {"quarter": "2024Q1", "fed_prediction": {"hike_25bp": 0.05, "hold": 0.9, "cut_25bp": 0.05}},
        {"quarter": "2024Q3", "fed_prediction": {"hike_25bp": 0.05, "hold": 0.15, "cut_25bp": 0.8}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = evaluate_forecasting_traces(path, calibration_bins=4)

    assert report["calibration"]["bins"] == 4
    assert len(report["calibration"]["top_label"]["buckets"]) == 4
    assert "macro_ece" in report["calibration"]


def test_residual_tft_forward_shapes_and_diagnostics() -> None:
    torch = pytest.importorskip("torch")
    from models.tft_model import ResidualTemporalFusionTransformer, TFTConfig

    config = TFTConfig(input_dim=5, static_dim=3, hidden_dim=8, num_heads=2, dropout=0.0)
    model = ResidualTemporalFusionTransformer(config).eval()
    x = torch.randn(2, 4, 5)
    static = torch.randn(2, 3)

    with torch.no_grad():
        logits, diagnostics = model(x, static)

    assert tuple(logits.shape) == (2, 3)
    assert tuple(diagnostics["quantiles"].shape) == (2, 3)
    assert tuple(diagnostics["attention"]["temporal"].shape) == (2, 2, 4, 4)
    assert tuple(diagnostics["variable_importance"]["temporal"].shape) == (2, 4, 5)
    assert tuple(diagnostics["variable_importance"]["static"].shape) == (2, 3)
    assert torch.all(diagnostics["quantiles"][:, 0] <= diagnostics["quantiles"][:, 1])
    assert torch.all(diagnostics["quantiles"][:, 1] <= diagnostics["quantiles"][:, 2])

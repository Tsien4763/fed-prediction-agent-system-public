# Public Results

This repository keeps large datasets, LoRA adapters, and generated teacher traces out of git. The public tree includes small, reproducible evaluation artifacts so the project is not only an architecture sketch.

## Included Artifacts

| File | Purpose |
| --- | --- |
| `examples/public_eval/fomc_quarter_labels_2000_2026.csv` | Quarter-level FOMC decision label fixture used by the forecasting evaluator |
| `examples/public_eval/fred_macro_sample_2020_2024.csv` | Small FRED sample: FEDFUNDS, UNRATE, CPIAUCSL, GS10 |
| `examples/public_eval/fred_macro_quarterly_2000_2026.csv` | Public quarterly FRED macro fixture for validation baselines |
| `examples/public_eval/baseline_results.json` | Public baseline metrics on the 2020Q1-2023Q4 validation window |
| `examples/public_eval/forecasting_comparison.json` | Agent vs majority, persistence, naive macro logistic, ordered probit, and rolling VAR comparison |
| `examples/public_eval/public_rerun_manifest.json` | Machine-readable rerun manifest for the public validation protocol |
| `examples/public_eval/public_training_debug_manifest.json` | Public self-play -> SFT split -> GRPO reward dry-run manifest |
| `examples/public_eval/event_counterfactual_result.json` | Event-triggered five-cluster counterfactual rerun summary |
| `examples/public_eval/local_adapter_result_card.json` | Local GRPO adapter result card without model weights |

Regenerate the public artifacts:

```powershell
$env:PYTHONPATH = "src"
uv run --extra dev --extra agent python scripts/generate_public_results.py --refresh-fred-sample
```

Validation rerun protocol used for the Agent row:

```powershell
python scripts/run_public_eval_protocol.py
```

## Baseline Evaluation

Validation window: `2020Q1-2023Q4`  
Training prior window: `2000Q1-2019Q4`

| Model | Accuracy | Balanced Accuracy | Macro F1 | Brier | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train-prior majority baseline | 0.700 | 0.333 | 0.275 | 0.582 | 0.976 |
| Previous-quarter persistence baseline | 0.700 | 0.619 | 0.508 | 0.480 | 0.847 |

These baselines are deliberately simple. They set the minimum bar for the agent system: a full agent run should beat persistence on calibration and non-hold recall before being described as a useful forecasting model.

## Forecasting Comparison

Validation fixture: 10 labeled quarters, class mix `hike=7`, `hold=2`, `cut=1`. This is a diagnostic sample, not a production backtest.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Brier | Log Loss | Predicted Classes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train-prior majority | 0.700 | 0.333 | 0.275 | 0.582 | 0.976 | `hike=10` |
| Previous-quarter persistence | 0.700 | 0.619 | 0.508 | 0.480 | 0.847 | `cut=2`, `hold=1`, `hike=7` |
| Naive macro logistic | 0.800 | 0.667 | 0.533 | 0.329 | 0.691 | `cut=2`, `hike=8` |
| Naive macro ordered probit | 0.900 | 0.667 | 0.600 | 0.191 | 0.325 | `hold=3`, `hike=7` |
| Rolling VAR rate-direction | 0.800 | 0.667 | 0.533 | 0.252 | 0.470 | `cut=2`, `hike=8` |
| Legacy self-play agent baseline | 0.700 | 0.571 | 0.468 | 0.478 | 0.760 | `hold=5`, `hike=5` |

The legacy public self-play baseline avoids the previous all-hold collapse on the 2022-2023 validation window. It matches the majority and persistence accuracy but does not beat the naive macro logistic, ordered probit, or rolling VAR baselines. The ordered probit fit converges but reports a Hessian covariance warning in this small sample, so it is best read as a diagnostic baseline. Current self-play is stricter: DeepSeek is required for strategy generation, payoff judgement, and equilibrium auditing, and rule/mock fallback is disabled. The honest v1 result is: the repository demonstrates the full agent pipeline and evaluation contract, while forecasting edge still requires stronger teacher coverage, learned calibration, and validation-selected GRPO.

## Local Adapter Result Card

The public repo does not commit adapter weights. The latest retained local GRPO card is:

| Adapter | Rows | JSON Valid | Reward | Calibration Component | Anti-Hold Component | Predicted Classes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `grpo_from_sft_v2_lr1e4_steps12` | 128 | 1.000 | 0.483 | 0.460 | 0.776 | `hike_25bp=76`, `hold=52` |

This is a training/evaluation card, not a shipped model. Reproducing the full adapter requires local DeepSeek teacher traces and Qwen LoRA training artifacts.

## Training Debug Rerun

`scripts/run_public_training_debug_protocol.py` reruns the public training-data path without publishing generated traces, datasets, or adapter weights:

| Step | Result |
| --- | --- |
| Historical self-play | 106 quarter traces from `2000Q1-2026Q2` |
| Equilibrium distillation | 1,166 rows |
| Compact SFT split | train 880 / val 176 / test 110 |
| GRPO reward dry-run on val | 176 rows scored, weighted reward 0.621 |
| Reward diagnostics | schema 1.0, evidence traceable 1.0, no future leakage 1.0 |
| Adapter training smoke | CUDA LoRA SFT 80 steps completed, then CUDA GRPO 12 steps completed from the SFT adapter |
| Adapter val16 inference | valid JSON 1.000, weighted reward 0.460, forecast accuracy 0.3125, predicted `hold=16` |

This is a debug rerun of the training-data, reward, adapter-training, and adapter-generation path. The smoke adapters are generated locally under ignored `artifacts/`; they verify that the Qwen LoRA SFT and GRPO code paths run on CUDA. The val16 inference result shows the adapter learned the JSON schema but collapsed to hold, so it is not a trained forecasting checkpoint.

## Leakage Guard

The forecasting evaluator checks future-dated evidence and source metadata. Persona `research_cutoff` is treated as audit metadata rather than forecast evidence. A 2024Q1-2026Q2 rule self-play smoke returns `future_leakage_rate = 0.0` after this guard.

## Event-Driven Counterfactual

`scripts/smoke_realtime_pipeline.py` includes an event-level five-cluster stress test. `scripts/run_public_event_counterfactual.py` also runs a public, non-fake bounded self-play rerun and writes `examples/public_eval/event_counterfactual_result.json`.

The public rerun publishes a `p5_game_counterfactual` event at `2026-05-06T13:45:00Z`, maps it to `2026Q2`, applies an energy/geopolitical/liquidity shock, and archives:

- Fed probability delta: `hike_25bp=+0.0063`, `hold=-0.0062`, `cut_25bp=-0.0001`
- Top impacted clusters: `CHN`, `RUS`, `GBR`, `FRA`, then `USA`
- Dominant strategy shift: `trade_or_sanction_pressure_prob` rises by about `+0.32` for the top non-US clusters
- Risk scope: `rolling_prediction_attribution_with_p5_counterfactual`

This addresses the real-time critique as an event-driven rolling layer, not as an intraday label backtest. The public FOMC labels remain quarterly; the event bus can still react to daily or intraday policy shocks and immediately run a five-cluster counterfactual against the active quarter.

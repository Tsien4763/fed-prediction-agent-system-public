# Fed Policy Agent System

Research prototype for continuous Federal Reserve policy forecasting with agent orchestration, macro time-series models, LLM semantic extraction, and evidence-aware training.

The project is intentionally scoped as a reproducible research system, not a production trading model. Local experiments use a small Qwen 0.8B LoRA setup so the full pipeline can run on limited hardware.

## What It Builds

```text
policy and macro data
  -> registry-based bounded BFS policy crawler
  -> semantic feature extraction
  -> Nuwa-style policy persona loading
  -> LangChain Runnable orchestration across five agent facades
  -> multi-agent policy game with LLM payoff and equilibrium auditing
  -> external shock variables
  -> VAR/VECM + residual TFT forecasting and quantile uncertainty
  -> counterfactual self-play stress tests
  -> evidence chain and risk attribution
  -> SFT / GRPO training and validation
```

Core agents:

| Agent | Package | Role |
| --- | --- | --- |
| Data perception | `agents.data_perception` | Crawl policy source registries, collect macro data, normalize documents, and publish update triggers |
| Semantic extraction | `agents.semantic_extraction` | Convert unstructured text into structured policy signals |
| Decision reasoning | `agents.decision_reasoning` | Combine semantic, macro, VAR/VECM, and classifier features |
| Multi-cluster game | `agents.multi_cluster_game` | Simulate role-based policy interaction, policy personas, DeepSeek payoff judging, and equilibrium checks |
| Evidence chain | `agents.evidence_chain` | Attribute forecast changes to data, text, and game signals |

The public `src/agents` packages expose the five job-required agent boundaries
as concrete classes: `DataPerceptionAgent`, `SemanticExtractionAgent`,
`DecisionReasoningAgent`, `MultiClusterGameAgent`, and `EvidenceChainAgent`.
`agents.langchain_runtime` composes those classes as an explicit LangChain
`Runnable` sequence. The deeper strategic loop remains in `fed_game.self_play`,
`fed_game.skills`, `fed_game.clusters`, and `fed_game.persona`.

## Verification

See [RESULTS.md](RESULTS.md) for the public baseline metrics, FOMC label fixture,
FRED macro sample, and local GRPO result card.

| Check | Current status |
| --- | --- |
| LangChain runtime | `agents.langchain_runtime` runs a five-agent `Runnable` sequence over concrete facade Agent classes |
| Static checks | CI runs `ruff check src scripts tests` for high-signal Python errors |
| Unit tests | 38 passing tests plus 1 optional Torch/TFT shape test, covering source monitoring, event buses, realtime closed loop, concrete Agent classes, strict DeepSeek semantic mode, TF-IDF policy-context filtering, persona/game contracts, rewards, temporal splits, forecasting metrics, calibration buckets, VAR/VECM smoke checks, semantic scoring, leakage guards, event-triggered five-cluster counterfactuals, strict DeepSeek self-play contracts, public result artifacts, and LangChain orchestration |
| Hardening smoke | `scripts/smoke_hardening.py` passes without network access or model weights |
| Realtime smoke | `scripts/smoke_realtime_pipeline.py` verifies `publish -> poll -> RollingPredictor -> risk_attribution -> ack` for file, memory, Redis adapter, and Kafka adapter, plus an event-triggered five-cluster counterfactual stress test |

## Repository Layout

```text
src/agents/           Public agent facades and LangChain Runnable runtime
src/data_engineering/ Source monitoring, data collection, context snapshots, and RAG index builders
src/fed_game/         Teacher data, self-play, SFT/GRPO, inference, and evaluation CLI
src/models/           VAR/VECM, residual TFT, forecast models, and predictive attribution
policy_personas/      Runtime persona cards
configs/              Runtime configuration
scripts/              Verification scripts
```

Generated data, adapters, checkpoints, and prediction dumps are ignored by git.

## Quick Start

```powershell
$env:PYTHONPATH = "src"
python -m compileall -q src scripts
uv run --extra dev --extra agent pytest -q
python -m fed_game.cli status
```

Public smoke commands:

```powershell
python -m fed_game.cli generate-balanced-teacher --dry-run
python -m fed_game.cli persona-summary --role-id usa_warsh
python -m fed_game.cli counterfactual --quarter 2024Q2 --scenario high_inflation --override inflation_cpi_yoy=4.5
python -m fed_game.cli infer-adapter --help
python -m fed_game.cli score-grpo-predictions --help
python -m models.identification_diagnostics
uv run --extra agent python -m agents.langchain_runtime
python scripts/smoke_hardening.py
python scripts/smoke_realtime_pipeline.py
python scripts/run_public_event_counterfactual.py
```

After local policy data and self-play traces exist:

```powershell
python -m fed_game.cli self-play --quarter-start 2000Q1 --quarter-end 2026Q2
python -m fed_game.cli prepare-training-data --dapt-limit 2000
python -m fed_game.cli split-training-data
python -m fed_game.cli evaluate-forecasting
```

The pytest suite is intentionally lightweight: it uses fake HTTP pages, fake
Redis/Kafka clients, a fake DeepSeek teacher, and a real LangChain Runnable
sequence so public verification does not need network access, API keys, or
model weights.

Semantic extraction is fail-closed. The runtime first uses a deterministic
TF-IDF policy-context filter to select the most relevant FOMC/policy excerpts
from long documents, then sends only those excerpts to DeepSeek for structured
hawkish-dovish scoring. There is no local semantic scoring fallback: a missing
or failing DeepSeek key raises an error. Public tests use a fake DeepSeek
client, so CI stays offline without weakening the production contract.

The LangChain runtime is installed through the agent extra:

```powershell
uv sync --extra agent
$env:PYTHONPATH = "src"
$env:DEEPSEEK_API_KEY = "<your-key>"
uv run --extra agent python -m agents.langchain_runtime
```

DeepSeek is required inside self-play as the role strategy generator, payoff
judge, and equilibrium auditor:

```powershell
python -m fed_game.cli self-play `
  --quarter-start 2024Q1 `
  --quarter-end 2024Q2 `
  --api-key-file path\to\api-key.txt `
  --teacher-model deepseek-chat
```

Each LLM role proposal is scored by a second DeepSeek payoff-judgement call.
When the cheap stability rule marks a candidate equilibrium, DeepSeek checks
whether any role has a profitable unilateral deviation. Traces record
`payoff_source`, `payoff_reasoning`, `deviation_candidate`, and
`equilibrium_check`.

Self-play is fail-closed: `BestResponseSkill` rule mode is disabled, mock
teacher mode is disabled, malformed LLM JSON raises, and heuristic-only
equilibrium acceptance is not allowed. Public/offline tests use fake DeepSeek
clients to verify the contract without weakening runtime behavior.

Realtime rolling-prediction transports are optional. Public verification runs a
closed-loop smoke over file, memory, Redis-adapter, and Kafka-adapter transports
with in-process fake Redis/Kafka clients. That proves the event contract,
prediction archive, risk attribution, and manual acknowledgement flow. The
forecasting labels and macro evaluation panel are quarterly, but the event
runtime accepts daily or intraday timestamps. A `p5_game_counterfactual`,
`geopolitical_shock`, `macro_shock`, or `policy_shock` event can trigger a
bounded-round five-cluster self-play stress test and archive the resulting Fed
probability delta plus `p5_impact_summary`. The public artifact
`examples/public_eval/event_counterfactual_result.json` is generated from a
2026-05-06 intraday shock event and records the mapped quarter, five-cluster
impact ranking, strategy deltas, belief deltas, and risk-attribution scope. A
real production smoke still requires external Redis/Kafka brokers:

```powershell
uv sync --extra realtime
$env:MAE_CPS_EVENT_BUS = "redis"   # or "kafka"
$env:REDIS_URL = "redis://localhost:6379/0"
# $env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
python -m models.event_pipeline
```

Neural training modules, including the residual TFT and LoRA training scripts,
use the optional training dependencies:

```powershell
uv sync --extra train
$env:PYTHONPATH = "src"
python -m models.tft_model
```

For NVIDIA GPU training on Windows, install a CUDA PyTorch wheel into the
project environment after dependency sync, then invoke the environment Python
directly so the lockfile does not replace it with a CPU wheel:

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 `
  --upgrade --reinstall torch

$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m fed_game.cli train-grpo --help
```

After local data and artifacts are generated, run:

```powershell
python scripts/verify_first_version.py
```

## Notes

- DeepSeek can be used as a teacher, payoff judge, and equilibrium auditor through `--api-key-file`; local keys are not committed.
- Temporal validation is part of the project contract. Test data should not be used for tuning.
- Forecast outputs are research diagnostics and are not investment advice.

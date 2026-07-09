from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PUBLIC_EVAL_DIR = REPO_ROOT / "examples" / "public_eval"
MACRO_FIXTURE = PUBLIC_EVAL_DIR / "fred_macro_quarterly_2000_2026.csv"
LABEL_FIXTURE = PUBLIC_EVAL_DIR / "fomc_quarter_labels_2000_2026.csv"
MANIFEST_PATH = PUBLIC_EVAL_DIR / "public_training_debug_manifest.json"
TRACE_PATH = REPO_ROOT / "artifacts" / "first_version" / "traces" / "rolling_self_play_traces.jsonl"
TRAIN_DIR = REPO_ROOT / "artifacts" / "first_version" / "train"
SPLIT_DIR = REPO_ROOT / "artifacts" / "first_version" / "splits"
RESULTS_DIR = REPO_ROOT / "artifacts" / "first_version" / "results"
SFT_SMOKE_DIR = REPO_ROOT / "artifacts" / "first_version" / "adapters" / "public_sft_smoke"
GRPO_SMOKE_DIR = REPO_ROOT / "artifacts" / "first_version" / "adapters" / "public_grpo_smoke"
INFER_SMOKE_PREDICTIONS = RESULTS_DIR / "public_grpo_smoke_val_predictions.jsonl"
INFER_SMOKE_REPORT = RESULTS_DIR / "public_grpo_smoke_val_report.json"


def main() -> int:
    commands: list[dict[str, Any]] = []
    build_macro_panel()
    build_public_rag_index()
    commands.extend(
        [
            {"step": "build_macro_panel", "status": "completed", "source": relative_path(MACRO_FIXTURE)},
            {"step": "build_public_rag_index", "status": "completed", "source": relative_path(LABEL_FIXTURE)},
        ]
    )

    run_command(
        commands,
        "historical_self_play",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "self-play",
            "--quarter-start",
            "2000Q1",
            "--quarter-end",
            "2026Q2",
            "--max-rounds",
            "1",
            "--stable-rounds-required",
            "1",
            "--strategy-epsilon",
            "1",
            "--deviation-gain-tau",
            "1",
        ],
    )
    run_command(
        commands,
        "prepare_training_data",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "prepare-training-data",
            "--dapt-limit",
            "200",
        ],
    )
    run_command(
        commands,
        "split_training_data",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "split-training-data",
        ],
    )
    reward_file = SPLIT_DIR / "compact_equilibrium_sft.val.jsonl"
    reward_output = RESULTS_DIR / "public_grpo_reward_dry_run.json"
    run_command(
        commands,
        "grpo_reward_dry_run",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "score-grpo-rewards",
            "--train-file",
            str(reward_file),
            "--output-path",
            str(reward_output),
            "--limit",
            "200",
        ],
    )
    adapter_training = maybe_run_adapter_training_smoke(commands)
    adapter_inference = maybe_run_adapter_inference_smoke(commands, adapter_training)

    manifest = build_manifest(commands, adapter_training=adapter_training, adapter_inference=adapter_inference)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "status": manifest["status"]}, indent=2))
    return 0


def build_macro_panel() -> None:
    macro = pd.read_csv(MACRO_FIXTURE)
    idx = pd.PeriodIndex(macro["quarter"], freq="Q").to_timestamp(how="end").normalize()
    panel = macro.set_index(idx).drop(columns=["quarter"])
    panel["gdp_growth_qoq_ann"] = 2.0
    out = REPO_ROOT / "data" / "processed" / "us_macro_panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)


def build_public_rag_index() -> None:
    labels = pd.read_csv(LABEL_FIXTURE)
    macro = pd.read_csv(MACRO_FIXTURE).set_index("quarter")
    rows = []
    for item in labels.to_dict(orient="records"):
        quarter = str(item["quarter"])
        macro_row = macro.loc[quarter].to_dict() if quarter in macro.index else {}
        date = quarter_end_date(quarter)
        decision = item.get("actual_class", "")
        text = (
            f"Public FOMC label for {quarter}: {decision}. "
            f"Meetings: {item.get('meeting_dates', '')}. "
            f"Descriptions: {item.get('meeting_descriptions', '')}. "
            f"Macro snapshot: fedfunds={macro_row.get('fedfunds')}, "
            f"inflation_cpi_yoy={macro_row.get('inflation_cpi_yoy')}, "
            f"unemployment={macro_row.get('unemployment')}, gs10={macro_row.get('gs10')}."
        )
        rows.append(
            {
                "doc_id": f"public_fomc_{quarter}",
                "date": date,
                "country": "USA",
                "actor": "FOMC",
                "strategy_key": "fed_policy_direction",
                "title": f"Public FOMC label and macro context {quarter}",
                "url": "public://examples/public_eval/fomc_quarter_labels_2000_2026.csv",
                "text": text,
            }
        )
    out = REPO_ROOT / "data" / "index" / "policy_context_index.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(commands: list[dict[str, Any]], step: str, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    commands.append(
        {
            "step": step,
            "status": "completed",
            "command": sanitize_text(printable_command(command)),
            "stdout_tail": sanitize_text(completed.stdout[-2000:]),
            "stderr_tail": sanitize_text(completed.stderr[-2000:]),
        }
    )


def run_optional_command(
    commands: list[dict[str, Any]],
    step: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    record = {
        "step": step,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": sanitize_text(printable_command(command)),
        "stdout_tail": sanitize_text(completed.stdout[-2000:]),
        "stderr_tail": sanitize_text(completed.stderr[-2000:]),
    }
    commands.append(record)
    return record


def build_manifest(
    commands: list[dict[str, Any]],
    *,
    adapter_training: dict[str, Any],
    adapter_inference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "public_training_debug_manifest",
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Public rerun of self-play training data preparation, temporal split, and GRPO reward debug.",
        "boundaries": [
            "Does not publish private DeepSeek traces, LoRA adapter weights, or generated training datasets.",
            "Uses rule-fallback self-play and public FRED/FOMC fixtures.",
            "Adapter weight training is not executed unless the train extra and model weights are available locally.",
        ],
        "commands": commands,
        "outputs": {
            "trace_path": relative_path(TRACE_PATH),
            "train_dir": relative_path(TRAIN_DIR),
            "split_dir": relative_path(SPLIT_DIR),
            "reward_report": relative_path(RESULTS_DIR / "public_grpo_reward_dry_run.json"),
        },
        "counts": {
            "trace_rows": count_jsonl(TRACE_PATH),
            "equilibrium_distill_rows": count_jsonl(TRAIN_DIR / "equilibrium_distill.jsonl"),
            "first_version_sft_rows": count_jsonl(TRAIN_DIR / "first_version_sft.jsonl"),
            "compact_equilibrium_sft_rows": count_jsonl(TRAIN_DIR / "compact_equilibrium_sft.jsonl"),
            "compact_train_rows": count_jsonl(SPLIT_DIR / "compact_equilibrium_sft.train.jsonl"),
            "compact_val_rows": count_jsonl(SPLIT_DIR / "compact_equilibrium_sft.val.jsonl"),
            "compact_test_rows": count_jsonl(SPLIT_DIR / "compact_equilibrium_sft.test.jsonl"),
        },
        "temporal_split": read_json(SPLIT_DIR / "temporal_split_report.json"),
        "reward_dry_run": reward_summary(RESULTS_DIR / "public_grpo_reward_dry_run.json"),
        "adapter_training": adapter_training,
        "adapter_inference": adapter_inference,
    }


def training_dependency_status() -> dict[str, Any]:
    try:
        from fed_game.training.common import require_training_deps

        require_training_deps()
        import torch

        return {
            "status": "available",
            "torch": {
                "version": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
    except Exception as exc:
        return {
            "status": "skipped_missing_train_extra",
            "reason": str(exc),
            "rerun_command": (
                "uv sync --extra train; "
                "python -m fed_game.cli train-sft --train-file artifacts/first_version/splits/compact_equilibrium_sft.train.jsonl "
                "--output-dir artifacts/first_version/adapters/public_sft_smoke --max-steps 1"
            ),
        }


def maybe_run_adapter_training_smoke(commands: list[dict[str, Any]]) -> dict[str, Any]:
    dependency_status = training_dependency_status()
    if dependency_status["status"] != "available":
        return dependency_status
    if os.getenv("MAE_CPS_RUN_TRAIN_SMOKE", "").strip() != "1":
        return {
            "status": "available_not_executed_by_public_protocol",
            "reason": (
                "Training dependencies are available, but the public protocol skips model-weight "
                "training unless MAE_CPS_RUN_TRAIN_SMOKE=1 is set."
            ),
            "torch": dependency_status.get("torch", {}),
            "rerun_command": (
                "$env:MAE_CPS_RUN_TRAIN_SMOKE='1'; "
                "python scripts/run_public_training_debug_protocol.py"
            ),
        }

    train_file = SPLIT_DIR / "compact_equilibrium_sft.train.jsonl"
    sft_steps = env_int("MAE_CPS_SFT_SMOKE_STEPS", 1)
    grpo_steps = env_int("MAE_CPS_GRPO_SMOKE_STEPS", 1)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("FED_GAME_TRAIN_SAVE_STEPS", "1000")
    env.setdefault("FED_GAME_TRAIN_LOGGING_STEPS", "1")

    sft = run_optional_command(
        commands,
        "sft_lora_smoke",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "train-sft",
            "--train-file",
            str(train_file),
            "--output-dir",
            str(SFT_SMOKE_DIR),
            "--max-steps",
            str(sft_steps),
        ],
        env=env,
    )
    if sft["status"] != "completed":
        return {
            "status": "sft_smoke_failed",
            "sft": sft,
            "grpo": {"status": "skipped_after_sft_failure"},
            "torch": dependency_status.get("torch", {}),
        }

    grpo = run_optional_command(
        commands,
        "grpo_lora_smoke",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "train-grpo",
            "--train-file",
            str(train_file),
            "--output-dir",
            str(GRPO_SMOKE_DIR),
            "--base-adapter-dir",
            str(SFT_SMOKE_DIR),
            "--max-steps",
            str(grpo_steps),
            "--learning-rate",
            "0.00005",
            "--max-completion-length",
            "64",
        ],
        env=env,
    )
    return {
        "status": "smoke_completed" if grpo["status"] == "completed" else "grpo_smoke_failed",
        "scope": "Public LoRA SFT/GRPO training smoke; adapters are ignored by git.",
        "sft": adapter_dir_summary(SFT_SMOKE_DIR) | {"command_status": sft["status"], "max_steps": sft_steps},
        "grpo": adapter_dir_summary(GRPO_SMOKE_DIR) | {"command_status": grpo["status"], "max_steps": grpo_steps},
        "torch": dependency_status.get("torch", {}),
        "note": "This verifies the training code path; it is not a trained forecasting model.",
    }


def adapter_dir_summary(path: Path) -> dict[str, Any]:
    key_names = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"}
    files = sorted(item.name for item in path.iterdir() if item.is_file() and item.name in key_names) if path.exists() else []
    return {
        "output_dir": relative_path(path),
        "exists": path.exists(),
        "key_files": files,
    }


def maybe_run_adapter_inference_smoke(
    commands: list[dict[str, Any]],
    adapter_training: dict[str, Any],
) -> dict[str, Any]:
    if os.getenv("MAE_CPS_RUN_INFER_SMOKE", "").strip() != "1":
        return {
            "status": "skipped",
            "reason": "Set MAE_CPS_RUN_INFER_SMOKE=1 to run adapter generation and scoring on validation rows.",
        }
    if not GRPO_SMOKE_DIR.exists():
        return {
            "status": "skipped_missing_adapter",
            "reason": "GRPO smoke adapter directory does not exist.",
            "adapter_training_status": adapter_training.get("status"),
        }

    limit = env_int("MAE_CPS_INFER_SMOKE_LIMIT", 16)
    max_new_tokens = env_int("MAE_CPS_INFER_MAX_NEW_TOKENS", 256)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    infer = run_optional_command(
        commands,
        "grpo_adapter_val_inference",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "infer-adapter",
            "--adapter-dir",
            str(GRPO_SMOKE_DIR),
            "--eval-file",
            str(SPLIT_DIR / "compact_equilibrium_sft.val.jsonl"),
            "--output-path",
            str(INFER_SMOKE_PREDICTIONS),
            "--limit",
            str(limit),
            "--max-new-tokens",
            str(max_new_tokens),
            "--batch-size",
            "1",
        ],
        env=env,
    )
    if infer["status"] != "completed":
        return {"status": "inference_failed", "inference": infer}

    score = run_optional_command(
        commands,
        "grpo_adapter_val_scoring",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "score-grpo-predictions",
            "--prediction-file",
            str(INFER_SMOKE_PREDICTIONS),
            "--output-path",
            str(INFER_SMOKE_REPORT),
        ],
        env=env,
    )
    report = read_json(INFER_SMOKE_REPORT)
    return {
        "status": "completed" if score["status"] == "completed" else "scoring_failed",
        "scope": "Validation generation/scoring from the public GRPO smoke adapter.",
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "prediction_file": relative_path(INFER_SMOKE_PREDICTIONS),
        "report_file": relative_path(INFER_SMOKE_REPORT),
        "samples": count_jsonl(INFER_SMOKE_PREDICTIONS),
        "valid_json_rate": report.get("valid_json_rate"),
        "weighted_reward": (report.get("average_components") or {}).get("weighted_total"),
        "predicted_class_counts": report.get("predicted_class_counts"),
        "forecast_row_accuracy": (report.get("forecasting") or {}).get("row_metrics", {}).get("accuracy"),
        "forecast_row_brier": (report.get("forecasting") or {}).get("row_metrics", {}).get("brier_score"),
        "forecast_future_leakage_rate": (report.get("forecasting") or {}).get("future_leakage", {}).get(
            "future_leakage_rate"
        ),
        "note": "This is a smoke adapter evaluation; poor generation quality is expected at this step count.",
    }


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def reward_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "rows_scored": data.get("rows_scored"),
        "batch_hold_collapse_penalty": data.get("batch_hold_collapse_penalty"),
        "predicted_class_counts": data.get("predicted_class_counts"),
        "average_components": data.get("average_components"),
        "issue_counts": data.get("issue_counts"),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return sanitize_json(data)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def quarter_end_date(quarter: str) -> str:
    year = int(quarter[:4])
    q = int(quarter[-1])
    month = q * 3
    day = 31 if month in {3, 12} else 30
    return f"{year}-{month:02d}-{day:02d}"


def printable_command(command: list[str]) -> str:
    parts = []
    for item in command:
        if " " in item:
            parts.append(f'"{item}"')
        else:
            parts.append(item)
    return " ".join(parts)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return sanitize_text(str(path))


def sanitize_text(value: str) -> str:
    root = str(REPO_ROOT)
    python_path = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
    normalized = value.replace(python_path, "python").replace(python_path.replace("\\", "\\\\"), "python")
    return normalized.replace(root, ".").replace(root.replace("\\", "\\\\"), ".")


if __name__ == "__main__":
    raise SystemExit(main())

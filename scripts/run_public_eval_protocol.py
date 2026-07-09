from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MACRO_FIXTURE = REPO_ROOT / "examples" / "public_eval" / "fred_macro_quarterly_2000_2026.csv"
TRACE_PATH = REPO_ROOT / "artifacts" / "first_version" / "traces" / "rolling_self_play_traces.jsonl"
EVAL_PATH = REPO_ROOT / "artifacts" / "first_version" / "results" / "public_val_rule_self_play_macro_eval.json"
COMPARISON_PATH = REPO_ROOT / "examples" / "public_eval" / "forecasting_comparison.json"
MANIFEST_PATH = REPO_ROOT / "examples" / "public_eval" / "public_rerun_manifest.json"


def main() -> int:
    commands: list[dict[str, Any]] = []
    build_macro_panel()
    commands.append({"step": "build_macro_panel", "status": "completed", "source": relative_path(MACRO_FIXTURE)})

    run_command(
        commands,
        "validation_self_play",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "self-play",
            "--quarter-start",
            "2020Q1",
            "--quarter-end",
            "2023Q4",
            "--max-rounds",
            "3",
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
        "forecast_eval",
        [
            PYTHON,
            "-m",
            "fed_game.cli",
            "evaluate-forecasting",
            "--trace-path",
            str(TRACE_PATH),
            "--output-path",
            str(EVAL_PATH),
        ],
    )
    run_command(
        commands,
        "public_result_generation",
        [
            PYTHON,
            "scripts/generate_public_results.py",
            "--trace-path",
            str(TRACE_PATH),
        ],
    )

    manifest = build_manifest(commands)
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


def build_manifest(commands: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    eval_report = json.loads(EVAL_PATH.read_text(encoding="utf-8")) if EVAL_PATH.exists() else {}
    models = comparison.get("models", {})
    metrics = {
        name: report.get("metrics", {})
        for name, report in models.items()
        if name
        in {
            "train_prior_majority",
            "previous_quarter_persistence",
            "naive_macro_logistic",
            "naive_macro_ordered_probit",
            "rolling_var_rate_direction",
            "rule_self_play_agent",
        }
    }
    return {
        "artifact_type": "public_eval_rerun_manifest",
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Public reproducibility protocol for validation forecasting comparison.",
        "boundaries": [
            "Does not publish private DeepSeek traces, LoRA adapter weights, or generated training datasets.",
            "Uses rule-fallback self-play for the public Agent row.",
            "Validation fixture has 10 labeled quarters; metrics are diagnostic, not production claims.",
        ],
        "commands": commands,
        "outputs": {
            "trace_path": relative_path(TRACE_PATH),
            "forecast_eval_path": relative_path(EVAL_PATH),
            "forecasting_comparison_path": relative_path(COMPARISON_PATH),
        },
        "forecast_eval_summary": {
            "evaluated_quarters": eval_report.get("evaluated_quarters"),
            "accuracy": (eval_report.get("forecast_metrics") or {}).get("accuracy"),
            "brier_score": (eval_report.get("forecast_metrics") or {}).get("brier_score"),
            "log_loss": (eval_report.get("forecast_metrics") or {}).get("log_loss"),
            "future_leakage_rate": (eval_report.get("future_leakage") or {}).get("future_leakage_rate"),
        },
        "comparison_metrics": metrics,
    }


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

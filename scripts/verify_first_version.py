from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "artifacts" / "first_version" / "splits" / "temporal_split_report.json"
FORECAST_EVAL_PATH = REPO_ROOT / "artifacts" / "first_version" / "results" / "forecasting_eval.json"
TRAIN_DIR = REPO_ROOT / "artifacts" / "first_version" / "train"
COMPACT_PATH = TRAIN_DIR / "compact_equilibrium_sft.jsonl"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []

    required_train_files = [
        "semantic_sft.jsonl",
        "role_best_response_sft.jsonl",
        "critique_traces_sft.jsonl",
        "evidence_chain_sft.jsonl",
        "first_version_sft.jsonl",
    ]
    train_counts = {name: count_jsonl(TRAIN_DIR / name) for name in required_train_files}
    for name, rows in train_counts.items():
        if rows == 0:
            issues.append(f"{name} has zero rows")

    if not REPORT_PATH.exists():
        issues.append(f"missing temporal split report: {REPORT_PATH}")
        report = None
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    split_counts = {}
    leakage_passed = False
    if report is not None:
        leakage = report.get("leakage_check", {})
        leakage_passed = bool(leakage.get("passed"))
        if not leakage_passed:
            issues.extend(str(item) for item in leakage.get("issues", []))

        datasets = report.get("datasets", {})
        first_version = datasets.get("first_version_sft")
        if not first_version:
            issues.append("first_version_sft missing from temporal split report")
        else:
            for split_name in ["train", "val", "test"]:
                rows = int(first_version["splits"][split_name]["rows"])
                split_counts[split_name] = rows
                if rows <= 0:
                    issues.append(f"first_version_sft {split_name} split has zero rows")
            if int(first_version.get("unassigned", {}).get("rows", 0)):
                issues.append("first_version_sft has unassigned rows")

        for dataset_name in ["equilibrium_distill", "compact_equilibrium_sft"]:
            dataset = datasets.get(dataset_name)
            if not dataset:
                issues.append(f"{dataset_name} missing from temporal split report")
                continue
            for split_name in ["train", "val", "test"]:
                rows = int(dataset["splits"][split_name]["rows"])
                if rows <= 0:
                    issues.append(f"{dataset_name} {split_name} split has zero rows")
            if int(dataset.get("unassigned", {}).get("rows", 0)):
                issues.append(f"{dataset_name} has unassigned rows")

        current_root = str(REPO_ROOT)
        if current_root not in str(report.get("output_dir", "")):
            warnings.append("temporal split report output_dir does not point at this repo copy")

        warnings.extend(str(item) for item in leakage.get("warnings", []))

    compact_evidence_checked = False
    if not COMPACT_PATH.exists():
        issues.append(f"missing compact equilibrium SFT file: {COMPACT_PATH}")
    else:
        with COMPACT_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                assistant = json.loads(row["messages"][2]["content"])
                evidence_chain = assistant.get("evidence_chain")
                compact_evidence_checked = True
                if not isinstance(evidence_chain, list) or not evidence_chain:
                    issues.append("compact_equilibrium_sft assistant is missing evidence_chain")
                else:
                    first = evidence_chain[0]
                    if not isinstance(first, dict) or not first.get("source_id"):
                        issues.append("compact_equilibrium_sft evidence_chain is not traceable")
                break

    forecast_eval = None
    forecasting_summary = {}
    if not FORECAST_EVAL_PATH.exists():
        issues.append(f"missing forecasting eval: {FORECAST_EVAL_PATH}")
    else:
        forecast_eval = json.loads(FORECAST_EVAL_PATH.read_text(encoding="utf-8"))
        forecast_metrics = forecast_eval.get("forecast_metrics", {})
        future_leakage = forecast_eval.get("future_leakage", {})
        convergence = forecast_eval.get("convergence", {})
        calibration = forecast_eval.get("calibration", {})
        split_metrics_report = forecast_eval.get("split_metrics", {})

        evaluated_quarters = int(forecast_eval.get("evaluated_quarters", 0) or 0)
        if evaluated_quarters <= 0:
            issues.append("forecasting eval has zero evaluated quarters")
        for metric_name in [
            "accuracy",
            "direction_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "hold_collapse_index",
            "brier_score",
            "log_loss",
        ]:
            if forecast_metrics.get(metric_name) is None:
                issues.append(f"forecasting eval missing metric: {metric_name}")
        for split_name in ["train", "val", "test"]:
            if split_name not in split_metrics_report:
                issues.append(f"forecasting eval missing {split_name} split metrics")
        if future_leakage.get("future_leakage_rate") not in (0, 0.0):
            issues.append("forecasting eval detected future-dated evidence leakage")
        if "top_label" not in calibration or "one_vs_rest" not in calibration:
            issues.append("forecasting eval missing calibration metrics")
        if convergence.get("convergence_rate") is None or convergence.get("avg_rounds") is None:
            issues.append("forecasting eval missing convergence metrics")

        forecasting_summary = {
            "evaluated_quarters": evaluated_quarters,
            "accuracy": forecast_metrics.get("accuracy"),
            "balanced_accuracy": forecast_metrics.get("balanced_accuracy"),
            "macro_f1": forecast_metrics.get("macro_f1"),
            "non_hold_recall": forecast_metrics.get("non_hold_recall"),
            "hold_collapse_index": forecast_metrics.get("hold_collapse_index"),
            "brier_score": forecast_metrics.get("brier_score"),
            "log_loss": forecast_metrics.get("log_loss"),
            "future_leakage_rate": future_leakage.get("future_leakage_rate"),
            "convergence_rate": convergence.get("convergence_rate"),
            "avg_rounds": convergence.get("avg_rounds"),
        }

    payload = {
        "repo_root": str(REPO_ROOT),
        "train_counts": train_counts,
        "first_version_split_counts": split_counts,
        "leakage_passed": leakage_passed,
        "compact_evidence_checked": compact_evidence_checked,
        "forecasting_eval": forecasting_summary,
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

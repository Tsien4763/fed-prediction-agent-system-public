from __future__ import annotations

import json
import re
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fed_game.config import RuntimeConfig, ensure_dir, ensure_parent, repo_path


REQUESTED_TEMPORAL_RANGES = {
    "train": "2000-2020",
    "val": "2020-2023",
    "test": "2024-2026",
}


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: date
    end: date
    requested_range: str
    note: str = ""


@dataclass(frozen=True)
class DateEvidence:
    value: date
    raw_value: str
    source: str


TEMPORAL_SPLIT_WINDOWS = [
    SplitWindow(
        "train",
        date(2000, 1, 1),
        date(2019, 12, 31),
        REQUESTED_TEMPORAL_RANGES["train"],
        "2020 is assigned to validation to avoid train/validation overlap.",
    ),
    SplitWindow("val", date(2020, 1, 1), date(2023, 12, 31), REQUESTED_TEMPORAL_RANGES["val"]),
    SplitWindow("test", date(2024, 1, 1), date(2026, 12, 31), REQUESTED_TEMPORAL_RANGES["test"]),
]


DEFAULT_SPLIT_FILENAMES = [
    "semantic_sft.jsonl",
    "role_best_response_sft.jsonl",
    "critique_traces_sft.jsonl",
    "evidence_chain_sft.jsonl",
    "equilibrium_distill.jsonl",
    "first_version_sft.jsonl",
    "compact_equilibrium_sft.jsonl",
]


def build_temporal_training_splits(
    config: RuntimeConfig,
    *,
    input_files: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    train_dir = config.paths["train_dir"]
    split_dir = ensure_dir(output_dir or config.paths["artifacts_dir"] / "splits")
    if input_files is None:
        inputs = [train_dir / filename for filename in DEFAULT_SPLIT_FILENAMES]
    else:
        inputs = [repo_path(path) for path in input_files]

    datasets: dict[str, Any] = {}
    for input_path in inputs:
        if not input_path.exists():
            continue
        datasets[input_path.stem] = split_jsonl_temporally(input_path, split_dir)

    report = {
        "requested_ranges": REQUESTED_TEMPORAL_RANGES,
        "effective_windows": [_window_to_dict(window) for window in TEMPORAL_SPLIT_WINDOWS],
        "overlap_resolution": "The requested train/val shorthand overlaps in 2020; 2020 rows are assigned to val.",
        "output_dir": str(split_dir),
        "datasets": datasets,
    }
    report["leakage_check"] = build_leakage_check(datasets)
    report_path = ensure_parent(split_dir / "temporal_split_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def split_jsonl_temporally(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    resolved_input = repo_path(input_path)
    resolved_output = ensure_dir(output_dir)
    dataset_name = resolved_input.stem
    output_paths = {
        window.name: ensure_parent(resolved_output / f"{dataset_name}.{window.name}.jsonl")
        for window in TEMPORAL_SPLIT_WINDOWS
    }
    unassigned_path = ensure_parent(resolved_output / f"{dataset_name}.unassigned.jsonl")

    stats = {window.name: _empty_stats(output_paths[window.name]) for window in TEMPORAL_SPLIT_WINDOWS}
    unassigned = {"path": str(unassigned_path), "rows": 0, "reasons": Counter()}
    total_rows = 0

    with resolved_input.open("r", encoding="utf-8") as source, ExitStack() as stack:
        handles = {name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in output_paths.items()}
        unknown_handle = stack.enter_context(unassigned_path.open("w", encoding="utf-8"))
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            evidence = extract_temporal_evidence(row)
            split_name = split_for_date(evidence.value) if evidence else None
            if split_name is None:
                reason = "missing_temporal_evidence" if evidence is None else "outside_configured_windows"
                row.setdefault("metadata", {})["temporal_unassigned_reason"] = reason
                row["metadata"]["source_line"] = line_no
                unknown_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                unassigned["rows"] += 1
                unassigned["reasons"][reason] += 1
                continue

            metadata = row.setdefault("metadata", {})
            metadata["temporal_split"] = split_name
            metadata["temporal_date"] = evidence.value.isoformat()
            metadata["temporal_source"] = evidence.source
            handles[split_name].write(json.dumps(row, ensure_ascii=False) + "\n")
            _update_stats(stats[split_name], evidence)

    for item in stats.values():
        item["date_sources"] = dict(item["date_sources"])
        item["years"] = dict(sorted(item["years"].items()))
    unassigned["reasons"] = dict(unassigned["reasons"])
    return {
        "input_path": str(resolved_input),
        "total_rows": total_rows,
        "splits": stats,
        "unassigned": unassigned,
    }


def extract_temporal_evidence(row: dict[str, Any]) -> DateEvidence | None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        evidence = _evidence_from_mapping(metadata, "metadata")
        if evidence is not None:
            return evidence

    for idx, message in enumerate(row.get("messages", []) or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        parsed = _parse_json_content(content)
        if isinstance(parsed, dict):
            evidence = _evidence_from_mapping(parsed, f"messages[{idx}].content")
            if evidence is not None:
                return evidence
    return None


def split_for_date(value: date) -> str | None:
    for window in TEMPORAL_SPLIT_WINDOWS:
        if window.start <= value <= window.end:
            return window.name
    return None


def build_leakage_check(datasets: dict[str, Any]) -> dict[str, Any]:
    window_by_name = {window.name: window for window in TEMPORAL_SPLIT_WINDOWS}
    dataset_checks = {}
    global_issues: list[str] = []
    global_warnings: list[str] = []
    for dataset_name, summary in datasets.items():
        issues: list[str] = []
        warnings: list[str] = []
        splits = summary["splits"]
        for split_name, stats in splits.items():
            rows = int(stats["rows"])
            if rows == 0:
                warnings.append(f"{split_name} split has zero rows")
                continue
            window = window_by_name[split_name]
            min_date = date.fromisoformat(stats["min_date"])
            max_date = date.fromisoformat(stats["max_date"])
            if min_date < window.start or max_date > window.end:
                issues.append(
                    f"{split_name} date range {min_date.isoformat()}..{max_date.isoformat()} "
                    f"outside {window.start.isoformat()}..{window.end.isoformat()}"
                )

        if summary["unassigned"]["rows"]:
            warnings.append(f"{summary['unassigned']['rows']} rows were unassigned")

        chronological_issue = _chronological_issue(splits)
        if chronological_issue:
            issues.append(chronological_issue)

        dataset_checks[dataset_name] = {
            "passed": not issues,
            "issues": issues,
            "warnings": warnings,
        }
        global_issues.extend(f"{dataset_name}: {issue}" for issue in issues)
        global_warnings.extend(f"{dataset_name}: {warning}" for warning in warnings)

    return {
        "passed": not global_issues,
        "issues": global_issues,
        "warnings": global_warnings,
        "datasets": dataset_checks,
    }


def _evidence_from_mapping(mapping: dict[str, Any], source_prefix: str) -> DateEvidence | None:
    for key in ["as_of_date", "date", "information_cutoff"]:
        evidence = _parse_date_evidence(mapping.get(key), f"{source_prefix}.{key}")
        if evidence is not None:
            return evidence
    return _parse_quarter_evidence(mapping.get("quarter"), f"{source_prefix}.quarter")


def _parse_date_evidence(value: Any, source: str) -> DateEvidence | None:
    if value is None:
        return None
    raw = str(value).strip()
    match = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", raw)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return DateEvidence(date(year, month, day), raw, source)
    except ValueError:
        return None


def _parse_quarter_evidence(value: Any, source: str) -> DateEvidence | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    match = re.fullmatch(r"(\d{4})Q([1-4])", raw)
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    return DateEvidence(date(year, 1 + (quarter - 1) * 3, 1), raw, source)


def _parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _empty_stats(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": 0,
        "min_date": None,
        "max_date": None,
        "date_sources": Counter(),
        "years": Counter(),
    }


def _update_stats(stats: dict[str, Any], evidence: DateEvidence) -> None:
    value = evidence.value.isoformat()
    stats["rows"] += 1
    stats["min_date"] = value if stats["min_date"] is None else min(stats["min_date"], value)
    stats["max_date"] = value if stats["max_date"] is None else max(stats["max_date"], value)
    stats["date_sources"][evidence.source] += 1
    stats["years"][str(evidence.value.year)] += 1


def _window_to_dict(window: SplitWindow) -> dict[str, Any]:
    return {
        "name": window.name,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "requested_range": window.requested_range,
        "note": window.note,
    }


def _chronological_issue(splits: dict[str, Any]) -> str | None:
    ordered = ["train", "val", "test"]
    previous_name = None
    previous_max = None
    for name in ordered:
        stats = splits[name]
        if stats["rows"] == 0:
            continue
        current_min = date.fromisoformat(stats["min_date"])
        current_max = date.fromisoformat(stats["max_date"])
        if previous_max is not None and current_min <= previous_max:
            return (
                f"{previous_name} max date {previous_max.isoformat()} is not before "
                f"{name} min date {current_min.isoformat()}"
            )
        previous_name = name
        previous_max = current_max
    return None

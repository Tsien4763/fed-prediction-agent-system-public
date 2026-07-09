from __future__ import annotations

from typing import Any

from .config import RuntimeConfig
from .evaluation import evaluate_traces
from .self_play import RollingSelfPlayEngine
from .teacher import generate_balanced_teacher_sft
from .training.prepare import build_compact_equilibrium_sft, combine_sft_data
from .training.splits import build_temporal_training_splits


def run_first_version_workflow(config: RuntimeConfig, *, teacher_limit: int | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"config": config}
    graph = build_langgraph_if_available()
    if graph is None:
        return run_sequential_workflow(config, teacher_limit=teacher_limit)
    compiled = graph.compile()
    return compiled.invoke({"config": config, "teacher_limit": teacher_limit})


def run_sequential_workflow(config: RuntimeConfig, *, teacher_limit: int | None = None) -> dict[str, Any]:
    teacher_report = generate_balanced_teacher_sft(config)
    result = RollingSelfPlayEngine(config).run()
    metrics = evaluate_traces(result.trace_path)
    sft_path = combine_sft_data(config)
    compact_path = build_compact_equilibrium_sft(config)
    split_report = build_temporal_training_splits(config)
    return {
        "teacher_report": teacher_report,
        "equilibrium_distill": str(result.distill_path),
        "first_version_sft": str(sft_path),
        "compact_equilibrium_sft": str(compact_path),
        "trace_path": str(result.trace_path),
        "metrics": metrics,
        "temporal_split_report": split_report["report_path"],
        "leakage_check": split_report["leakage_check"],
    }


def build_langgraph_if_available():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None

    def build_teacher(state: dict[str, Any]) -> dict[str, Any]:
        config: RuntimeConfig = state["config"]
        state["teacher_report"] = generate_balanced_teacher_sft(config)
        return state

    def run_self_play(state: dict[str, Any]) -> dict[str, Any]:
        config: RuntimeConfig = state["config"]
        result = RollingSelfPlayEngine(config).run()
        state["equilibrium_distill"] = str(result.distill_path)
        state["trace_path"] = str(result.trace_path)
        return state

    def evaluate(state: dict[str, Any]) -> dict[str, Any]:
        config: RuntimeConfig = state["config"]
        state["metrics"] = evaluate_traces(state["trace_path"])
        state["first_version_sft"] = str(combine_sft_data(config))
        state["compact_equilibrium_sft"] = str(build_compact_equilibrium_sft(config))
        split_report = build_temporal_training_splits(config)
        state["temporal_split_report"] = split_report["report_path"]
        state["leakage_check"] = split_report["leakage_check"]
        return state

    graph = StateGraph(dict)
    graph.add_node("teacher_distillation", build_teacher)
    graph.add_node("rolling_self_play", run_self_play)
    graph.add_node("evaluate", evaluate)
    graph.add_edge(START, "teacher_distillation")
    graph.add_edge("teacher_distillation", "rolling_self_play")
    graph.add_edge("rolling_self_play", "evaluate")
    graph.add_edge("evaluate", END)
    return graph

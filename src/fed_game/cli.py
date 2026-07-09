from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import default_self_play_trace_path, load_config
from .evaluation import evaluate_forecasting_traces, evaluate_traces
from .self_play import RollingSelfPlayEngine, quarter_range
from .teacher import (
    generate_balanced_teacher_sft,
    generate_critique_sft,
    generate_evidence_chain_sft,
    generate_role_sft,
    generate_semantic_sft,
)
from .workflow import run_first_version_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Fed multi-agent policy game first-version CLI.")
    parser.add_argument("--config", default="configs/first_version.json", help="Runtime config JSON path.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print configured paths and whether they exist.")

    persona = sub.add_parser("persona-summary", help="Print loaded policy persona metadata.")
    persona.add_argument("--role-id", default="usa_warsh", help="Role id configured under personas.")
    persona.add_argument("--skill-path", default=None, help="Optional SKILL.md path to load directly.")

    teacher = sub.add_parser("generate-teacher", help="Generate DeepSeek teacher SFT datasets.")
    teacher.add_argument("--limit", type=int, default=None)
    teacher.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "semantic", "role", "critique", "evidence"],
        help="Teacher datasets to generate. Defaults to all.",
    )
    teacher.add_argument("--skip-semantic-teacher", action="store_true", help="Use heuristic semantic labels.")
    teacher.add_argument("--skip-role-teacher", action="store_true", help="Use heuristic role best-response labels.")
    teacher.add_argument("--skip-critique-teacher", action="store_true", help="Use fallback critique labels.")
    teacher.add_argument("--skip-evidence-teacher", action="store_true", help="Use fallback evidence-chain labels.")
    teacher.add_argument("--api-key-file", default=None, help="Read DEEPSEEK_API_KEY from a local text file.")
    teacher.add_argument("--teacher-model", default=None, help="Override DEEPSEEK_MODEL for this run.")
    teacher.add_argument("--teacher-base-url", default=None, help="Override DEEPSEEK_BASE_URL for this run.")
    teacher.add_argument("--as-of-start", default=None, help="Earliest as-of date to include, YYYY-MM-DD.")
    teacher.add_argument("--as-of-end", default=None, help="Latest as-of date to include, YYYY-MM-DD.")
    teacher.add_argument("--sample-per-year", type=int, default=None, help="Take the first N contexts per as-of year.")
    teacher.add_argument("--append", action="store_true", help="Append rows to existing teacher files instead of replacing them.")

    balanced_teacher = sub.add_parser("generate-balanced-teacher", help="Generate teacher data from balanced train/val/test time windows.")
    balanced_teacher.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "semantic", "role", "critique", "evidence"],
        help="Teacher datasets to generate. Defaults to all.",
    )
    balanced_teacher.add_argument("--skip-semantic-teacher", action="store_true", help="Use heuristic semantic labels.")
    balanced_teacher.add_argument("--skip-role-teacher", action="store_true", help="Use heuristic role best-response labels.")
    balanced_teacher.add_argument("--skip-critique-teacher", action="store_true", help="Use fallback critique labels.")
    balanced_teacher.add_argument("--skip-evidence-teacher", action="store_true", help="Use fallback evidence-chain labels.")
    balanced_teacher.add_argument("--api-key-file", default=None, help="Read DEEPSEEK_API_KEY from a local text file.")
    balanced_teacher.add_argument("--teacher-model", default=None, help="Override DEEPSEEK_MODEL for this run.")
    balanced_teacher.add_argument("--teacher-base-url", default=None, help="Override DEEPSEEK_BASE_URL for this run.")
    balanced_teacher.add_argument("--dry-run", action="store_true", help="Only report selected context counts; do not call teacher.")

    game = sub.add_parser("self-play", help="Run quarterly rolling self-play and equilibrium distillation.")
    game.add_argument("--quarters", nargs="*", help="Optional quarter IDs, for example 2026Q1 2026Q2.")
    game.add_argument("--quarter-start", default=None, help="First quarter to run, for example 2000Q1.")
    game.add_argument("--quarter-end", default=None, help="Last quarter to run, for example 2026Q2.")
    game.add_argument("--api-key-file", default=None, help="Read DEEPSEEK_API_KEY from a local text file.")
    game.add_argument("--teacher-model", default=None, help="Override DEEPSEEK_MODEL for this run.")
    game.add_argument("--teacher-base-url", default=None, help="Override DEEPSEEK_BASE_URL for this run.")
    _add_self_play_runtime_args(game)

    ev = sub.add_parser("evaluate-traces", help="Evaluate trace convergence metrics.")
    ev.add_argument("--trace-path", default=None)

    fev = sub.add_parser("evaluate-forecasting", help="Evaluate Fed decision forecasting quality on self-play traces.")
    fev.add_argument("--trace-path", default=None)
    fev.add_argument("--output-path", default=None)
    fev.add_argument("--calibration-bins", type=int, default=10)

    cf = sub.add_parser("counterfactual", help="Run factual vs counterfactual self-play and report prediction deltas.")
    cf.add_argument("--quarter", required=True, help="Quarter to simulate, for example 2024Q2.")
    cf.add_argument("--scenario", required=True, help="Scenario name, for example high_inflation.")
    cf.add_argument("--override", action="append", default=[], help="Counterfactual override as key=value. Repeatable.")
    cf.add_argument("--output-path", default=None, help="JSON output path.")
    cf.add_argument("--markdown-path", default=None, help="Markdown report path.")
    cf.add_argument("--max-context-docs", type=int, default=4)
    cf.add_argument("--api-key-file", default=None, help="Read DEEPSEEK_API_KEY from a local text file.")
    cf.add_argument("--teacher-model", default=None, help="Override DEEPSEEK_MODEL for this run.")
    cf.add_argument("--teacher-base-url", default=None, help="Override DEEPSEEK_BASE_URL for this run.")
    _add_self_play_runtime_args(cf)

    wf = sub.add_parser("run-first-version", help="Run balanced teacher generation, self-play, and trace evaluation.")
    wf.add_argument("--teacher-limit", type=int, default=None, help="Deprecated; teacher data now uses balanced temporal windows.")

    prep = sub.add_parser("prepare-training-data", help="Build DAPT corpus and combined SFT data.")
    prep.add_argument("--dapt-limit", type=int, default=2000)
    prep.add_argument("--limit-per-sft-file", type=int, default=None)
    prep.add_argument("--compact-equilibrium-limit", type=int, default=None)

    split = sub.add_parser("split-training-data", help="Build leakage-checked temporal train/val/test JSONL files.")
    split.add_argument("--input-file", action="append", default=None, help="JSONL file to split. Repeatable.")
    split.add_argument("--output-dir", default=None, help="Output directory for split JSONL files and report.")

    sft = sub.add_parser("train-sft", help="Run LoRA SFT on generated chat JSONL.")
    sft.add_argument("--train-file", required=True)
    sft.add_argument("--output-dir", required=True)
    sft.add_argument("--max-steps", type=int, default=-1)
    sft.add_argument("--resume-from-checkpoint", default=None)

    dapt = sub.add_parser("train-dapt", help="Run domain continued pretraining.")
    dapt.add_argument("--text-file", required=True)
    dapt.add_argument("--output-dir", required=True)
    dapt.add_argument("--max-steps", type=int, default=-1)

    grpo = sub.add_parser("train-grpo", help="Run GRPO prototype on equilibrium prompts.")
    grpo.add_argument("--train-file", required=True)
    grpo.add_argument("--output-dir", required=True)
    grpo.add_argument("--max-steps", type=int, default=-1)
    grpo.add_argument("--base-adapter-dir", default=None, help="Optional SFT adapter to continue with GRPO.")
    grpo.add_argument("--learning-rate", type=float, default=None, help="Override GRPO learning rate for val-tuned runs.")
    grpo.add_argument("--max-completion-length", type=int, default=None, help="Override GRPO max completion length.")

    grpo_score = sub.add_parser("score-grpo-rewards", help="Dry-run GRPO reward components on a JSONL training file.")
    grpo_score.add_argument("--train-file", required=True)
    grpo_score.add_argument("--output-path", default=None)
    grpo_score.add_argument("--limit", type=int, default=None)

    pred_score = sub.add_parser("score-grpo-predictions", help="Score adapter prediction JSONL with GRPO reward components.")
    pred_score.add_argument("--prediction-file", required=True)
    pred_score.add_argument("--output-path", required=True)

    infer = sub.add_parser("infer-adapter", help="Run first-version adapter inference and save results.")
    infer.add_argument("--adapter-dir", required=True)
    infer.add_argument("--eval-file", required=True)
    infer.add_argument("--output-path", required=True)
    infer.add_argument("--limit", type=int, default=8)
    infer.add_argument("--max-new-tokens", type=int, default=256)
    infer.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "status":
        payload = {
            "base_model": config.base_model,
            "teacher_model": config.teacher_model,
            "teacher_base_url": config.teacher_base_url,
            "teacher_is_mock": not bool(config.teacher_api_key) and config.allow_mock_teacher,
            "paths": {key: {"path": str(path), "exists": path.exists()} for key, path in config.paths.items()},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "persona-summary":
        from .persona import load_configured_personas, load_policy_persona

        if args.skill_path:
            persona_obj = load_policy_persona(args.skill_path)
        else:
            personas = load_configured_personas(config)
            if args.role_id not in personas:
                raise SystemExit(f"No persona configured for role_id={args.role_id}")
            persona_obj = personas[args.role_id]
        print(
            json.dumps(
                {
                    "persona_id": persona_obj.persona_id,
                    "role_id": persona_obj.role_id,
                    "name": persona_obj.name,
                    "skill_path": persona_obj.skill_path,
                    "research_cutoff": persona_obj.research_cutoff,
                    "nuwa_dimensions": sorted(persona_obj.nuwa_dimensions),
                    "mental_models": [item.name for item in persona_obj.mental_models],
                    "decision_heuristics": len(persona_obj.decision_heuristics),
                    "evidence_source_count": len(persona_obj.evidence_sources),
                    "honest_boundaries": persona_obj.honest_boundaries,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "generate-teacher":
        _apply_teacher_env(args)
        if args.limit is not None and args.sample_per_year is not None:
            raise SystemExit("--limit cannot be combined with --sample-per-year; use temporal sampling without a head limit.")
        if args.limit is None and not (args.as_of_start or args.as_of_end or args.sample_per_year):
            raise SystemExit(
                "generate-teacher now requires temporal sampling flags or an explicit --limit. "
                "Prefer generate-balanced-teacher for train/val/test-balanced teacher data."
            )
        tasks = set(args.tasks)
        if "all" in tasks:
            tasks = {"semantic", "role", "critique", "evidence"}
        outputs = {}
        teacher_kwargs = {
            "limit": args.limit,
            "as_of_start": args.as_of_start,
            "as_of_end": args.as_of_end,
            "sample_per_year": args.sample_per_year,
            "append": args.append,
        }
        if "semantic" in tasks:
            outputs["semantic_sft"] = str(
                generate_semantic_sft(config, use_teacher=not args.skip_semantic_teacher, **teacher_kwargs)
            )
        if "role" in tasks:
            outputs["role_sft"] = str(generate_role_sft(config, use_teacher=not args.skip_role_teacher, **teacher_kwargs))
        if "critique" in tasks:
            outputs["critique_sft"] = str(
                generate_critique_sft(config, use_teacher=not args.skip_critique_teacher, **teacher_kwargs)
            )
        if "evidence" in tasks:
            outputs["evidence_chain_sft"] = str(
                generate_evidence_chain_sft(config, use_teacher=not args.skip_evidence_teacher, **teacher_kwargs)
            )
        print(
            json.dumps(
                {
                    **outputs,
                    "teacher_model": config.teacher_model,
                    "teacher_base_url": config.teacher_base_url,
                    "teacher_is_mock": not bool(config.teacher_api_key) and config.allow_mock_teacher,
                    "as_of_start": args.as_of_start,
                    "as_of_end": args.as_of_end,
                    "sample_per_year": args.sample_per_year,
                    "append": args.append,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "generate-balanced-teacher":
        _apply_teacher_env(args)
        tasks = set(args.tasks)
        if "all" in tasks:
            tasks = {"semantic", "role", "critique", "evidence"}
        outputs = generate_balanced_teacher_sft(
            config,
            tasks=tasks,
            use_semantic_teacher=not args.skip_semantic_teacher,
            use_role_teacher=not args.skip_role_teacher,
            use_critique_teacher=not args.skip_critique_teacher,
            use_evidence_teacher=not args.skip_evidence_teacher,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    **outputs,
                    "teacher_model": config.teacher_model,
                    "teacher_base_url": config.teacher_base_url,
                    "teacher_is_mock": not bool(config.teacher_api_key) and config.allow_mock_teacher,
                    "tasks": sorted(tasks),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "self-play":
        _apply_teacher_env(args)
        _apply_self_play_runtime_args(config, args)
        quarters = _resolve_quarters(args)
        result = RollingSelfPlayEngine(config).run(quarters=quarters)
        print(
            json.dumps(
                {
                    "traces": len(result.traces),
                    "first_quarter": result.traces[0].quarter if result.traces else None,
                    "last_quarter": result.traces[-1].quarter if result.traces else None,
                    "trace_path": str(result.trace_path),
                    "equilibrium_distill": str(result.distill_path),
                },
                indent=2,
            )
        )
        return

    if args.command == "evaluate-traces":
        trace_path = Path(args.trace_path) if args.trace_path else default_self_play_trace_path(config)
        print(json.dumps(evaluate_traces(trace_path), indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-forecasting":
        trace_path = Path(args.trace_path) if args.trace_path else default_self_play_trace_path(config)
        result = evaluate_forecasting_traces(trace_path, calibration_bins=args.calibration_bins)
        output_path = Path(args.output_path) if args.output_path else config.paths["artifacts_dir"] / "results" / "forecasting_eval.json"
        output_path = output_path if output_path.is_absolute() else Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "output_path": str(output_path),
            "trace_count": result["trace_count"],
            "evaluated_quarters": result["evaluated_quarters"],
            "accuracy": result["forecast_metrics"]["accuracy"],
            "brier_score": result["forecast_metrics"]["brier_score"],
            "log_loss": result["forecast_metrics"]["log_loss"],
            "future_leakage_rate": result["future_leakage"]["future_leakage_rate"],
            "convergence_rate": result["convergence"]["convergence_rate"],
            "avg_rounds": result["convergence"]["avg_rounds"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "counterfactual":
        _apply_teacher_env(args)
        _apply_self_play_runtime_args(config, args)
        from .counterfactual import parse_override_assignments, run_counterfactual, save_counterfactual_outputs

        overrides = parse_override_assignments(args.override)
        result = run_counterfactual(
            config,
            quarter=args.quarter,
            scenario_name=args.scenario,
            overrides=overrides,
            max_context_docs=args.max_context_docs,
        )
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.scenario)
        default_json = config.paths["artifacts_dir"] / "results" / f"counterfactual_{args.quarter}_{safe_name}.json"
        json_path = Path(args.output_path) if args.output_path else default_json
        markdown_path = (
            Path(args.markdown_path)
            if args.markdown_path
            else json_path.with_suffix(".md")
        )
        outputs = save_counterfactual_outputs(result, json_path=json_path, markdown_path=markdown_path)
        summary = {
            **outputs,
            "quarter": result.quarter,
            "scenario": result.scenario_name,
            "delta": result.delta,
            "top_strategy_delta": result.strategy_delta[:3],
            "top_belief_delta": result.belief_delta[:3],
            "scenario_scope": result.scenario_scope,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "run-first-version":
        print(json.dumps(run_first_version_workflow(config, teacher_limit=args.teacher_limit), indent=2, ensure_ascii=False))
        return

    if args.command == "prepare-training-data":
        from .training.prepare import build_compact_equilibrium_sft, build_dapt_corpus, combine_sft_data

        try:
            dapt_path = build_dapt_corpus(config, limit=args.dapt_limit)
            sft_path = combine_sft_data(config, limit_per_file=args.limit_per_sft_file)
            compact_path = build_compact_equilibrium_sft(config, limit=args.compact_equilibrium_limit)
        except RuntimeError as exc:
            parser.exit(2, f"error: {exc}\n")
        print(
            json.dumps(
                {
                    "dapt_corpus": str(dapt_path),
                    "first_version_sft": str(sft_path),
                    "compact_equilibrium_sft": str(compact_path),
                },
                indent=2,
            )
        )
        return

    if args.command == "split-training-data":
        from .training.splits import build_temporal_training_splits

        report = build_temporal_training_splits(
            config,
            input_files=[Path(path) for path in args.input_file] if args.input_file else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.command == "train-sft":
        from .training.sft import train_sft

        train_sft(
            config,
            train_file=args.train_file,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        return

    if args.command == "train-dapt":
        from .training.dapt import train_dapt

        train_dapt(config, text_file=args.text_file, output_dir=args.output_dir, max_steps=args.max_steps)
        return

    if args.command == "train-grpo":
        from .training.grpo import train_grpo

        train_grpo(
            config,
            train_file=args.train_file,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            base_adapter_dir=args.base_adapter_dir,
            learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length,
        )
        return

    if args.command == "score-grpo-rewards":
        from .training.rewards import score_reward_file

        output_path = args.output_path
        if output_path is None:
            output_path = config.paths["artifacts_dir"] / "results" / "grpo_reward_dry_run.json"
        report = score_reward_file(args.train_file, limit=args.limit, output_path=output_path)
        summary = {
            "output_path": str(output_path),
            "rows_scored": report["rows_scored"],
            "average_weighted_total": report["average_components"].get("weighted_total"),
            "average_components": report["average_components"],
            "issue_counts": report["issue_counts"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "score-grpo-predictions":
        from .training.rewards import score_prediction_file

        report = score_prediction_file(args.prediction_file, output_path=args.output_path)
        summary = {
            "output_path": str(args.output_path),
            "rows_scored": report["rows_scored"],
            "valid_json_rate": report["valid_json_rate"],
            "average_weighted_total": report["average_components"].get("weighted_total"),
            "average_components": report["average_components"],
            "predicted_class_counts": report["predicted_class_counts"],
            "forecast_row_accuracy": report["forecasting"]["row_metrics"].get("accuracy"),
            "forecast_row_brier": report["forecasting"]["row_metrics"].get("brier_score"),
            "forecast_row_log_loss": report["forecasting"]["row_metrics"].get("log_loss"),
            "forecast_quarter_accuracy": report["forecasting"]["quarter_mean_metrics"].get("accuracy"),
            "forecast_future_leakage_rate": report["forecasting"]["future_leakage"].get("future_leakage_rate"),
            "issue_counts": report["issue_counts"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "infer-adapter":
        from .inference import run_adapter_inference

        result = run_adapter_inference(
            config,
            adapter_dir=args.adapter_dir,
            eval_file=args.eval_file,
            output_path=args.output_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return


def _apply_teacher_env(args: argparse.Namespace) -> None:
    if getattr(args, "api_key_file", None):
        key_path = Path(args.api_key_file).expanduser()
        os.environ["DEEPSEEK_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
    if getattr(args, "teacher_model", None):
        os.environ["DEEPSEEK_MODEL"] = str(args.teacher_model)
    if getattr(args, "teacher_base_url", None):
        os.environ["DEEPSEEK_BASE_URL"] = str(args.teacher_base_url)


def _add_self_play_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-rounds", type=int, default=None, help="Override self_play.max_rounds for this run.")
    parser.add_argument(
        "--stable-rounds-required",
        type=int,
        default=None,
        help="Override self_play.stable_rounds_required for this run.",
    )
    parser.add_argument(
        "--strategy-epsilon",
        type=float,
        default=None,
        help="Override self_play.strategy_epsilon for this run.",
    )
    parser.add_argument(
        "--deviation-gain-tau",
        type=float,
        default=None,
        help="Override self_play.deviation_gain_tau for this run.",
    )


def _apply_self_play_runtime_args(config, args: argparse.Namespace) -> None:
    self_play = config.raw.setdefault("self_play", {})
    updates = {
        "max_rounds": getattr(args, "max_rounds", None),
        "stable_rounds_required": getattr(args, "stable_rounds_required", None),
        "strategy_epsilon": getattr(args, "strategy_epsilon", None),
        "deviation_gain_tau": getattr(args, "deviation_gain_tau", None),
    }
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, (int, float)) and value <= 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be positive.")
        self_play[key] = value


def _resolve_quarters(args: argparse.Namespace) -> list[str] | None:
    has_range = bool(args.quarter_start or args.quarter_end)
    if has_range and args.quarters:
        raise SystemExit("--quarters cannot be combined with --quarter-start/--quarter-end")
    if not has_range:
        return args.quarters
    if not args.quarter_start or not args.quarter_end:
        raise SystemExit("--quarter-start and --quarter-end must be provided together")
    return quarter_range(args.quarter_start, args.quarter_end)


if __name__ == "__main__":
    main()

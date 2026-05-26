"""CLI entry point for the single-run and batch experimental harnesses."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_ALL_CONDITIONS = [
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
]
_ALL_GOAL_TYPES = [
    "research_synthesis", "financial_analysis", "sequential_reasoning",
    "code_analysis", "adversarial_reliability",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m mas.eval.cli",
        description = "MAS-OS experimental harness CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    run_p = sub.add_parser(
        "run",
        help        = "Run one goal under one experimental condition.",
        description = (
            "Execute a single goal instance under a specified experimental "
            "condition and emit one RunLog entry (EXPERIMENTS.md §4)."
        ),
    )
    run_p.add_argument(
        "--condition", "-c",
        required = True,
        choices  = _ALL_CONDITIONS,
        metavar  = "CONDITION",
        help     = "Experimental condition.",
    )
    run_p.add_argument(
        "--goal-id", "-g",
        required = True,
        dest     = "goal_id",
        help     = "Goal instance ID (e.g. 'research_synth_001').",
    )
    run_p.add_argument(
        "--seed", "-s",
        type    = int,
        default = 42,
        help    = "Integer random seed (default: 42).",
    )
    _add_model_args(run_p)
    run_p.add_argument(
        "--output-path", "-o",
        dest    = "output_path",
        default = None,
        help    = (
            "Path to runs.jsonl.  "
            "Overrides MAS_RUNS_LOG env var.  Defaults to ./runs.jsonl."
        ),
    )
    run_p.add_argument(
        "--no-write",
        action  = "store_true",
        default = False,
        help    = "Print RunLog to stdout instead of appending to runs.jsonl.",
    )
    run_p.add_argument(
        "--no-warmup",
        action  = "store_true",
        default = False,
        help    = "Skip model pre-warming.",
    )
    run_p.add_argument(
        "--judge-backend",
        dest    = "judge_backend",
        default = "none",
        choices = ["none", "anthropic", "ollama"],
        help    = (
            "LLM judge backend for semantic_judge rubric checks.  "
            "'anthropic' calls Claude Sonnet via ANTHROPIC_API_KEY; "
            "'ollama' calls a local Ollama model (default qwen2.5:14b, "
            "see DEV-019 for calibration requirement); "
            "'none' stubs all semantic_judge checks as True (default)."
        ),
    )
    run_p.add_argument(
        "--backend",
        dest    = "backend",
        default = "ollama",
        choices = ["ollama", "anthropic"],
        help    = (
            "Inference backend for LLM/SLM tiers.  "
            "'ollama' uses local Ollama (default); "
            "'anthropic' uses Claude Sonnet via ANTHROPIC_API_KEY "
            "(frontier replication, EXPERIMENTS.md §6)."
        ),
    )

    logs_p = sub.add_parser(
        "logs",
        help        = "Audit, clean, and summarise runs.jsonl.",
        description = "Utilities for validating and exploring the run log.",
    )
    logs_sub = logs_p.add_subparsers(dest="logs_command", metavar="LOGS_COMMAND")
    logs_sub.required = True

    for _lname, _lhelp in [
        ("validate", "Validate every line against the EXPERIMENTS.md §4 schema."),
        ("dedupe",   "Remove duplicate (goal_id, condition, seed) entries."),
        ("summary",  "Print aggregate statistics grouped by condition and goal type."),
    ]:
        _lp = logs_sub.add_parser(_lname, help=_lhelp)
        _lp.add_argument(
            "--path", "-p",
            default = None,
            help    = "Path to runs.jsonl (default: MAS_RUNS_LOG or ./runs.jsonl).",
        )
    # dedupe-specific flag
    logs_sub.choices["dedupe"].add_argument(
        "--strategy",
        default = "latest",
        choices = ["latest", "earliest", "successful"],
        help    = "Which duplicate to keep (default: latest).",
    )

    batch_p = sub.add_parser(
        "batch",
        help        = "Run the full experimental matrix (300 runs).",
        description = (
            "Orchestrate multiple goal × condition × seed runs with "
            "resumability, parallelism control, and progress reporting."
        ),
    )
    batch_p.add_argument(
        "--conditions",
        default = "all",
        help    = (
            "Comma-separated conditions to run, or 'all' (default).  "
            "Valid: FULL_FRAMEWORK, BASELINE_LANGGRAPH."
        ),
    )
    batch_p.add_argument(
        "--goal-types",
        dest    = "goal_types",
        default = "all",
        help    = "Comma-separated goal types to include, or 'all' (default).",
    )
    batch_p.add_argument(
        "--instances",
        default = "all",
        help    = "Comma-separated goal IDs, or 'all' (default).",
    )
    batch_p.add_argument(
        "--seeds-per-instance",
        dest    = "seeds_per_instance",
        type    = int,
        default = 1,
        help    = "Number of seeds per instance (default: 1, i.e. seed=0 only).",
    )
    batch_p.add_argument(
        "--parallel",
        type    = int,
        default = 1,
        help    = "Number of parallel workers (default: 1; max enforced at 2).",
    )
    batch_p.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = "List runs that would execute without actually running them.",
    )
    batch_p.add_argument(
        "--max-runs",
        dest    = "max_runs",
        type    = int,
        default = None,
        help    = "Stop after this many runs (useful for smoke tests).",
    )
    batch_p.add_argument(
        "--resume",
        action  = "store_true",
        default = False,
        help    = "Skip triples already recorded in runs_completed.txt.",
    )
    batch_p.add_argument(
        "--skip-preflight",
        dest    = "skip_preflight",
        action  = "store_true",
        default = False,
        help    = "Bypass the pre-flight goal-suite validation.",
    )
    _add_model_args(batch_p)
    batch_p.add_argument(
        "--output-path", "-o",
        dest    = "output_path",
        default = None,
        help    = "Path to runs.jsonl (default: ./runs.jsonl).",
    )
    batch_p.add_argument(
        "--no-warmup",
        action  = "store_true",
        default = False,
        help    = "Skip model pre-warming.",
    )
    batch_p.add_argument(
        "--judge-backend",
        dest    = "judge_backend",
        default = "none",
        choices = ["none", "anthropic", "ollama"],
        help    = (
            "LLM judge backend for semantic_judge rubric checks.  "
            "'anthropic' calls Claude Sonnet via ANTHROPIC_API_KEY; "
            "'ollama' calls a local Ollama model (default qwen2.5:14b, "
            "see DEV-019 for calibration requirement); "
            "'none' stubs all semantic_judge checks as True (default)."
        ),
    )
    batch_p.add_argument(
        "--backend",
        dest    = "backend",
        default = "ollama",
        choices = ["ollama", "anthropic"],
        help    = (
            "Inference backend for LLM/SLM tiers.  "
            "'ollama' uses local Ollama (default); "
            "'anthropic' uses Claude Sonnet via ANTHROPIC_API_KEY "
            "(frontier replication, EXPERIMENTS.md §6)."
        ),
    )

    rob_p = sub.add_parser(
        "robustness",
        help        = "Run pre-registered robustness checks (EXPERIMENTS.md §7.5).",
        description = (
            "Execute one or all of the four robustness checks: "
            "temperature sweep, seed sweep, drop-adversarial sensitivity, "
            "and threshold sensitivity."
        ),
    )
    rob_sub = rob_p.add_subparsers(dest="robustness_command", metavar="CHECK")
    rob_sub.required = True

    for _rname, _rhelp in [
        ("temperature",     "Temperature sweep: re-run 50 instances at T∈{0.0, 0.7}."),
        ("seed",            "Seed sweep: re-run 50 instances at seed and seed+1000."),
        ("drop-adversarial","Drop-adversarial: recompute analysis excluding adversarial_reliability."),
        ("threshold",       "Threshold sweep: re-run 50 instances at τ∈{0.5, 0.7, 0.9}."),
        ("all",             "Run all four robustness checks sequentially."),
    ]:
        _rp = rob_sub.add_parser(_rname, help=_rhelp)
        _rp.add_argument(
            "--runs-path", "-r",
            dest    = "runs_path",
            default = None,
            help    = "Path to runs.jsonl (default: MAS_RUNS_LOG or ./runs.jsonl).",
        )
        _rp.add_argument(
            "--log-dir",
            dest    = "log_dir",
            default = ".",
            help    = "Directory for robustness JSONL logs (default: .).",
        )
        _rp.add_argument(
            "--results-md",
            dest    = "results_md",
            default = "RESULTS.md",
            help    = "Path to RESULTS.md to append results to (default: RESULTS.md).",
        )
        _rp.add_argument(
            "--n-instances",
            dest    = "n_instances",
            type    = int,
            default = 50,
            help    = "Number of instances to sample per sweep (default: 50).",
        )
        if _rname in ("temperature", "seed", "threshold", "all"):
            _rp.add_argument(
                "--no-warmup",
                action  = "store_true",
                default = False,
                help    = "Skip model pre-warming.",
            )

    demo_p = sub.add_parser(
        "demo",
        help        = "Run an interactive terminal demo.",
        description = (
            "Display a real-time, colour-coded side-by-side comparison of "
            "FULL_FRAMEWORK vs a baseline condition.  "
            "Designed for 60–90 second screen-recording demos."
        ),
    )
    demo_sub = demo_p.add_subparsers(dest="demo_command", metavar="MODE")
    demo_sub.required = True

    for _dname, _dhelp in [
        ("cascade",  "FULL vs BASELINE — System-0 catches corruption, baseline cascades."),
        ("budget",   "FULL vs BASELINE — token-cost comparison on a multi-step goal."),
        ("escalate", "FULL only — System-3 P2 predicate fires, VerifierRetryAgent escalates."),
        ("all",      "Run cascade → budget → escalate sequentially."),
    ]:
        _dp = demo_sub.add_parser(_dname, help=_dhelp)
        _dp.add_argument(
            "--speed",
            type    = float,
            default = 1.0,
            metavar = "N",
            help    = (
                "Playback speed multiplier (1.0 = real time, 2.0 = 2× faster, "
                "0.5 = half speed).  Default: 1.0."
            ),
        )
        _dp.add_argument(
            "--live",
            action  = "store_true",
            default = False,
            help    = (
                "Execute real Ollama inference instead of replaying scripted events.  "
                "Requires Ollama running with phi3:mini and llama3.1:8b pulled."
            ),
        )
    # cascade-only: override goal-id
    demo_sub.choices["cascade"].add_argument(
        "--goal-id",
        dest    = "goal_id",
        default = None,
        help    = "Override the default adversarial goal instance (cascade mode only).",
    )

    fail_p = sub.add_parser(
        "failures",
        help        = "Categorise failure modes in runs.jsonl and update RESULTS.md.",
        description = (
            "Post-hoc failure mode analysis.  Reads runs.jsonl, categorises "
            "every non-successful run into seven mutually exclusive categories, "
            "and appends a Failure Mode Analysis section to RESULTS.md."
        ),
    )
    fail_p.add_argument(
        "--runs-path", "-r",
        dest    = "runs_path",
        default = None,
        help    = "Path to runs.jsonl (default: MAS_RUNS_LOG or ./runs.jsonl).",
    )
    fail_p.add_argument(
        "--output", "-o",
        dest    = "output",
        default = "RESULTS.md",
        help    = "RESULTS.md path to append the section to (default: RESULTS.md).",
    )
    fail_p.add_argument(
        "--no-append",
        action  = "store_true",
        default = False,
        help    = "Print the section to stdout instead of appending to RESULTS.md.",
    )

    results_p = sub.add_parser(
        "results",
        help        = "Generate RESULTS.md from runs.jsonl.",
        description = (
            "Read runs.jsonl, run all pre-registered analyses, optionally "
            "generate figures, and write RESULTS.md.  "
            "Do not hand-edit RESULTS.md — regenerate instead."
        ),
    )
    results_p.add_argument(
        "--runs-path", "-r",
        dest    = "runs_path",
        default = None,
        help    = (
            "Path to runs.jsonl.  "
            "Overrides MAS_RUNS_LOG env var.  Defaults to ./runs.jsonl."
        ),
    )
    results_p.add_argument(
        "--output", "-o",
        dest    = "output",
        default = "RESULTS.md",
        help    = "Output path for the generated Markdown (default: RESULTS.md).",
    )
    results_p.add_argument(
        "--no-figures",
        dest    = "no_figures",
        action  = "store_true",
        default = False,
        help    = "Skip figure generation (useful if figures are already up to date).",
    )

    regrade_p = sub.add_parser(
        "regrade",
        help        = "Re-grade existing runs.jsonl with a real judge backend.",
        description = (
            "Re-score every applicable run in runs.jsonl against the existing "
            "rubric using a real judge (OllamaJudge or AnthropicJudge).  "
            "No agent inference is repeated — operates post-hoc on final_output. "
            "Updates rubric_scores, ground_truth_match, grading_*, and "
            "cascade_failure in place.  Writes a backup and an audit log "
            "by default.  See DEV-019."
        ),
    )
    regrade_p.add_argument(
        "--runs", "-r",
        dest    = "runs_path",
        default = None,
        help    = "Path to runs.jsonl (default: MAS_RUNS_LOG or ./runs.jsonl).",
    )
    regrade_p.add_argument(
        "--judge-backend",
        dest    = "judge_backend",
        required= True,
        choices = ["ollama", "anthropic"],
        help    = (
            "Real judge backend.  'ollama' uses a local model "
            "(default qwen2.5:14b, see DEV-019); 'anthropic' uses Claude "
            "Sonnet (requires ANTHROPIC_API_KEY)."
        ),
    )
    regrade_p.add_argument(
        "--judge-model",
        dest    = "judge_model",
        default = None,
        help    = (
            "Override the judge model.  Default: 'qwen2.5:14b' for ollama, "
            "'claude-sonnet-4-5' for anthropic."
        ),
    )
    regrade_p.add_argument(
        "--ollama-url",
        dest    = "ollama_url",
        default = "http://localhost:11434",
        help    = "Ollama base URL when --judge-backend=ollama.",
    )
    regrade_p.add_argument(
        "--conditions",
        dest    = "filter_conditions",
        default = None,
        help    = (
            "Comma-separated list of conditions to regrade "
            "(e.g. 'FULL_FRAMEWORK,BASELINE_LANGGRAPH').  Default: all."
        ),
    )
    regrade_p.add_argument(
        "--goal-types",
        dest    = "filter_goal_types",
        default = None,
        help    = (
            "Comma-separated list of goal_types to regrade "
            "(e.g. 'adversarial_reliability,research_synthesis').  Default: all."
        ),
    )
    regrade_p.add_argument(
        "--goal-ids",
        dest    = "filter_goal_ids",
        default = None,
        help    = "Comma-separated list of specific goal_ids to regrade.  Default: all.",
    )
    regrade_p.add_argument(
        "--force",
        action  = "store_true",
        default = False,
        help    = (
            "Re-grade runs even when grading_api_calls > 0 "
            "(i.e. they were already graded by a real backend).  "
            "Default: skip already-graded runs."
        ),
    )
    regrade_p.add_argument(
        "--judge-all",
        dest    = "judge_all",
        action  = "store_true",
        default = False,
        help    = (
            "DEV-020 OVERRIDE: invoke the judge on EVERY eligible run, "
            "replacing the authored rubric (substring/numeric/...) with a "
            "single judge verdict against a synthetic criterion built from "
            "goal_text + ground_truth.  Original rubric_scores are preserved "
            "under 'rubric_scores_pre_judge_all'.  Methodologically heavy: "
            "results derived under --judge-all are EXPLORATORY ONLY."
        ),
    )
    regrade_p.add_argument(
        "--no-backup",
        action  = "store_true",
        default = False,
        help    = (
            "Skip writing <runs.jsonl>.bak.<ts> before replacement. "
            "NOT recommended — regrade is destructive without a backup."
        ),
    )
    regrade_p.add_argument(
        "--dry-run",
        action  = "store_true",
        default = False,
        help    = (
            "Compute the regrade in memory and print the summary, but do "
            "NOT write the result file or the audit log."
        ),
    )
    regrade_p.add_argument(
        "--audit-log",
        dest    = "audit_log_path",
        default = None,
        help    = (
            "Override the audit-log path "
            "(default: <runs.jsonl>.regrade-audit.jsonl)."
        ),
    )
    regrade_p.add_argument(
        "--json",
        action  = "store_true",
        default = False,
        help    = "Print the result summary as JSON instead of human-readable text.",
    )

    return parser


def _add_model_args(p: argparse.ArgumentParser) -> None:
    """Add shared model/URL arguments to a subparser."""
    p.add_argument(
        "--model-slm",
        dest    = "model_slm",
        default = "phi3:mini",
        help    = "System-1 SLM model name (default: phi3:mini).",
    )
    p.add_argument(
        "--model-llm",
        dest    = "model_llm",
        default = "llama3.1:8b",
        help    = "System-2 LLM model name (default: llama3.1:8b).",
    )
    p.add_argument(
        "--ollama-url",
        dest    = "ollama_url",
        default = "http://localhost:11434",
        help    = "Ollama base URL (default: http://localhost:11434).",
    )
    p.add_argument(
        "--tau",
        type    = float,
        default = 0.72,
        help    = "System-0 confidence threshold (default: 0.72).",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(levelname)s %(name)s: %(message)s",
    )

    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "batch":
        _cmd_batch(args)
    elif args.command == "logs":
        _cmd_logs(args)
    elif args.command == "results":
        _cmd_results(args)
    elif args.command == "robustness":
        _cmd_robustness(args)
    elif args.command == "failures":
        _cmd_failures(args)
    elif args.command == "demo":
        _cmd_demo(args)
    elif args.command == "regrade":
        _cmd_regrade(args)


def _cmd_demo(args: argparse.Namespace) -> None:
    """Execute the 'demo' subcommand — interactive terminal demo."""
    from mas.eval.demo import run_demo  # noqa: PLC0415

    mode      = args.demo_command
    speed     = getattr(args, "speed", 1.0)
    live_mode = getattr(args, "live", False)
    goal_id   = getattr(args, "goal_id", None)

    try:
        run_demo(
            mode      = mode,
            speed     = speed,
            live_mode = live_mode,
            goal_id   = goal_id,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted.", file=sys.stderr)
        sys.exit(0)


def _cmd_failures(args: argparse.Namespace) -> None:
    """Execute the 'failures' subcommand — failure mode analysis."""
    from mas.eval.failure_analysis import (   # noqa: PLC0415
        categorize_failures,
        failure_section_markdown,
    )

    runs_path = (
        getattr(args, "runs_path", None)
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    runs_path_obj = Path(runs_path)
    if not runs_path_obj.exists():
        print(f"ERROR: runs.jsonl not found at {runs_path}", file=sys.stderr)
        sys.exit(1)

    # Print JSON categorization to stdout always
    try:
        cats = categorize_failures(runs_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(cats, indent=2))

    # Build markdown section
    try:
        section = failure_section_markdown(runs_path)
    except Exception as exc:
        print(f"ERROR generating markdown: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.no_append:
        print("\n" + section)
    else:
        output_path = Path(args.output)
        _append_failure_section(output_path, section)
        print(f"\nAppended Failure Mode Analysis section to {output_path}")


def _append_failure_section(results_md: Path, section: str) -> None:
    """Append (or replace) the Failure Mode Analysis section in *results_md*.

Replaces the existing section if present; appends at end otherwise."""
    heading  = "### Failure Mode Analysis"
    existing = results_md.read_text(encoding="utf-8") if results_md.exists() else ""

    if heading in existing:
        # Replace from heading to next ### heading (or EOF)
        lines    = existing.splitlines(keepends=True)
        out      = []
        skip     = False
        replaced = False
        for line in lines:
            stripped = line.rstrip()
            if stripped == heading and not replaced:
                skip = True
                out.append(section.rstrip() + "\n\n")
                replaced = True
                continue
            if skip:
                if stripped.startswith("###") and stripped != heading:
                    skip = False
                else:
                    continue
            out.append(line)
        existing = "".join(out)
    else:
        existing = existing.rstrip() + "\n\n" + section.rstrip() + "\n"

    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text(existing, encoding="utf-8")


def _cmd_robustness(args: argparse.Namespace) -> None:
    """Execute the 'robustness' subcommand — pre-registered robustness checks."""
    from mas.eval.robustness import (              # noqa: PLC0415
        temperature_sweep,
        seed_sweep,
        drop_adversarial_sensitivity,
        threshold_sensitivity,
        append_robustness_section,
    )

    runs_path = (
        getattr(args, "runs_path", None)
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    log_dir    = getattr(args, "log_dir",    ".")
    results_md = getattr(args, "results_md", "RESULTS.md")
    n_inst     = getattr(args, "n_instances", 50)
    check      = args.robustness_command

    def _run_temperature() -> None:
        print("Running temperature sweep …")
        result = temperature_sweep(n_instances=n_inst, log_dir=log_dir)
        append_robustness_section(results_md, "Temperature Sweep", result)
        print(json.dumps(result, indent=2))

    def _run_seed() -> None:
        print("Running seed sweep …")
        result = seed_sweep(n_instances=n_inst, log_dir=log_dir)
        append_robustness_section(results_md, "Seed Sweep", result)
        print(json.dumps(result, indent=2))

    def _run_drop_adversarial() -> None:
        from pathlib import Path as _Path  # noqa: PLC0415
        rp = _Path(runs_path)
        if not rp.exists():
            print(f"ERROR: runs.jsonl not found at {runs_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Running drop-adversarial sensitivity on {runs_path} …")
        result = drop_adversarial_sensitivity(runs_path)
        append_robustness_section(results_md, "Drop-Adversarial Sensitivity", result)
        print(json.dumps(result, indent=2))

    def _run_threshold() -> None:
        print("Running threshold sensitivity …")
        result = threshold_sensitivity(n_instances=n_inst, log_dir=log_dir)
        append_robustness_section(results_md, "Threshold Sensitivity", result)
        print(json.dumps(result, indent=2))

    dispatch = {
        "temperature":      _run_temperature,
        "seed":             _run_seed,
        "drop-adversarial": _run_drop_adversarial,
        "threshold":        _run_threshold,
    }

    if check == "all":
        for fn in dispatch.values():
            try:
                fn()
            except Exception as exc:
                print(f"ERROR in robustness check: {exc}", file=sys.stderr)
                log.exception("Robustness check failed")
    else:
        try:
            dispatch[check]()
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            log.exception("Robustness check failed")
            sys.exit(1)


def _cmd_regrade(args: argparse.Namespace) -> None:
    """Execute the 'regrade' subcommand — re-grade runs.jsonl with a real judge."""
    from mas.eval.regrade import (  # noqa: PLC0415
        format_summary, regrade, result_to_dict,
    )

    runs_path = (
        args.runs_path
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    runs_path_obj = Path(runs_path)
    if not runs_path_obj.exists():
        print(f"ERROR: runs.jsonl not found at {runs_path}", file=sys.stderr)
        sys.exit(1)

    def _split_csv(value: str | None) -> list[str] | None:
        if not value:
            return None
        items = [v.strip() for v in value.split(",") if v.strip()]
        return items or None

    try:
        result = regrade(
            runs_path        = runs_path_obj,
            judge_backend    = args.judge_backend,
            judge_model      = args.judge_model,
            ollama_url       = args.ollama_url,
            filter_conditions= _split_csv(args.filter_conditions),
            filter_goal_types= _split_csv(args.filter_goal_types),
            filter_goal_ids  = _split_csv(args.filter_goal_ids),
            force            = args.force,
            backup           = not args.no_backup,
            dry_run          = args.dry_run,
            audit_log_path   = args.audit_log_path,
            judge_all        = args.judge_all,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nregrade interrupted (no file changes have been written).",
              file=sys.stderr)
        sys.exit(130)

    if args.json:
        print(json.dumps(result_to_dict(result), indent=2))
    else:
        print(format_summary(result))


def _cmd_results(args: argparse.Namespace) -> None:
    """Execute the 'results' subcommand — generate RESULTS.md."""
    from mas.eval.results_generator import generate  # noqa: PLC0415

    runs_path = (
        args.runs_path
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    runs_path_obj = Path(runs_path)
    if not runs_path_obj.exists():
        print(f"ERROR: runs.jsonl not found at {runs_path}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    generate_figures = not args.no_figures

    print(f"Generating {output} from {runs_path} ...")
    generate(runs_path, output, generate_figures=generate_figures)
    print(f"Done: {output}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' subcommand."""

    # Test hook: if _MAS_HARNESS_MOCK_LOG is set, skip Ollama entirely.
    mock_log_json = os.environ.get("_MAS_HARNESS_MOCK_LOG")
    if mock_log_json:
        try:
            run_log = json.loads(mock_log_json)
            log.info("Using mock RunLog from _MAS_HARNESS_MOCK_LOG.")
        except json.JSONDecodeError as exc:
            print(f"ERROR: _MAS_HARNESS_MOCK_LOG is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        from mas.eval.harness import run_one  # noqa: PLC0415
        model_overrides = {
            "slm":           args.model_slm,
            "llm":           args.model_llm,
            "ollama_url":    args.ollama_url,
            "tau":           args.tau,
            "judge_backend": args.judge_backend,
            "backend":       getattr(args, "backend", "ollama"),
        }
        try:
            run_log = run_one(
                goal_id         = args.goal_id,
                condition       = args.condition,
                seed            = args.seed,
                model_overrides = model_overrides,
                _skip_warmup    = args.no_warmup,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            log.exception("run_one() failed")
            sys.exit(1)

    run_log_json = json.dumps(run_log, indent=None)

    if args.no_write:
        print(run_log_json)
        log.info(
            "Run complete [%s | %s | seed=%s] — printed to stdout (--no-write).",
            args.goal_id, args.condition, args.seed,
        )
    else:
        log_path_str = (
            args.output_path
            or os.environ.get("MAS_RUNS_LOG")
            or "./runs.jsonl"
        )
        log_path = Path(log_path_str)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(run_log_json + "\n")
                fh.flush()
        except Exception as exc:
            print(f"ERROR: Failed to write to {log_path}: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Run complete → appended to {log_path}")
        log.info(
            "Run complete [%s | %s | seed=%s] → %s",
            args.goal_id, args.condition, args.seed, log_path,
        )


def _cmd_batch(args: argparse.Namespace) -> None:
    """Execute the 'batch' subcommand."""
    from mas.eval.batch import build_run_queue, run_batch  # noqa: PLC0415

    # Resolve conditions
    if args.conditions.strip().lower() == "all":
        conditions = list(_ALL_CONDITIONS)
    else:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        invalid = [c for c in conditions if c not in _ALL_CONDITIONS]
        if invalid:
            print(f"ERROR: Unknown condition(s): {invalid}", file=sys.stderr)
            sys.exit(1)

    # Resolve goal_types
    if args.goal_types.strip().lower() == "all":
        goal_types = None   # load_all_goals returns all
    else:
        goal_types = [t.strip() for t in args.goal_types.split(",") if t.strip()]

    # Resolve instances
    if args.instances.strip().lower() == "all":
        instance_ids = None  # load from disk
    else:
        instance_ids = [i.strip() for i in args.instances.split(",") if i.strip()]

    # Resolve paths
    output_path_str = (
        args.output_path
        or os.environ.get("MAS_RUNS_LOG")
        or "./runs.jsonl"
    )
    runs_jsonl     = Path(output_path_str)
    completed_path = runs_jsonl.parent / "runs_completed.txt"

    model_overrides = {
        "slm":           args.model_slm,
        "llm":           args.model_llm,
        "ollama_url":    args.ollama_url,
        "tau":           args.tau,
        "judge_backend": args.judge_backend,
        "backend":       getattr(args, "backend", "ollama"),
    }

    # Build queue
    try:
        queue = build_run_queue(
            conditions         = conditions,
            goal_types         = goal_types,
            instance_ids       = instance_ids,
            seeds_per_instance = args.seeds_per_instance,
        )
    except Exception as exc:
        print(f"ERROR: Failed to build run queue: {exc}", file=sys.stderr)
        sys.exit(1)

    if not queue:
        print("No runs to execute (queue is empty after filtering).")
        sys.exit(0)

    total = len(queue)
    capped = min(total, args.max_runs) if args.max_runs else total
    if not args.dry_run:
        print(
            f"Batch: {total} runs queued"
            + (f", capped at {args.max_runs}" if args.max_runs else "")
            + f" → {runs_jsonl}"
        )

    # Run
    run_batch(
        queue            = queue,
        runs_jsonl       = runs_jsonl,
        completed_path   = completed_path,
        model_overrides  = model_overrides,
        parallel         = args.parallel,
        max_runs         = args.max_runs,
        dry_run          = args.dry_run,
        resume           = args.resume,
        skip_warmup      = args.no_warmup,
        skip_preflight   = args.skip_preflight,
    )


def _resolve_log_path(args: argparse.Namespace) -> Path:
    """Return the runs.jsonl path for a logs sub-subcommand."""
    p = getattr(args, "path", None)
    return Path(p or os.environ.get("MAS_RUNS_LOG") or "./runs.jsonl")


def _cmd_logs(args: argparse.Namespace) -> None:
    """Dispatch to validate / dedupe / summary."""
    if args.logs_command == "validate":
        _cmd_logs_validate(args)
    elif args.logs_command == "dedupe":
        _cmd_logs_dedupe(args)
    elif args.logs_command == "summary":
        _cmd_logs_summary(args)


def _cmd_logs_validate(args: argparse.Namespace) -> None:
    """Stream runs.jsonl and report JSON + schema errors and duplicates."""
    from mas.eval.log_utils import validate_runs_jsonl  # noqa: PLC0415

    log_path = _resolve_log_path(args)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    report = validate_runs_jsonl(log_path)
    print(report.to_text())
    sys.exit(0 if report.ok else 1)


def _cmd_logs_dedupe(args: argparse.Namespace) -> None:
    """Remove duplicate triples from runs.jsonl using the chosen strategy."""
    from mas.eval.log_utils import deduplicate_runs  # noqa: PLC0415

    log_path = _resolve_log_path(args)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    removed = deduplicate_runs(log_path, strategy=args.strategy)
    if removed == 0:
        print("No duplicates found — file unchanged.")
    else:
        print(f"Removed {removed} duplicate entrie(s) using strategy '{args.strategy}'.")


def _cmd_logs_summary(args: argparse.Namespace) -> None:
    """Print aggregate statistics from runs.jsonl."""
    from mas.eval.log_utils import runs_summary, format_summary  # noqa: PLC0415

    log_path = _resolve_log_path(args)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    summary = runs_summary(log_path)
    print(format_summary(summary))


if __name__ == "__main__":
    main()

"""Single-run experimental harness."""
from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

from mas.eval.cascade_detector import detect_cascade
from mas.eval.goal_loader import GoalSpec, load_goal

log = logging.getLogger(__name__)


RunLog = dict[str, Any]

_VALID_CONDITIONS = frozenset({
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
})


_RUNLOG_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RunLog",
    "type": "object",
    "required": [
        "run_id", "goal_id", "goal_type",
        "pipeline_depth_k", "condition", "routing_skipped", "seed",
        "model_slm", "model_llm", "started_at", "completed_at",
        "latency_s", "success", "cascade_failure",
        "tokens", "s0", "escalations", "per_step",
        "final_output", "ground_truth", "rubric_scores",
        "ground_truth_match", "voc_trace", "crash_reason",
        "grading_api_calls", "grading_cost_usd",
        "backend", "api_cost_usd",
    ],
    "additionalProperties": True,
    "properties": {
        "run_id":             {"type": "string", "minLength": 1},
        "goal_id":            {"type": "string", "minLength": 1},
        "goal_type":          {"type": "string", "minLength": 1},
        "pipeline_depth_k": {"type": "integer", "minimum": 0},
        "condition": {
            "type": "string",
            "enum": list(_VALID_CONDITIONS),
        },
        "routing_skipped": {"type": "boolean"},
        "seed":            {"type": "integer"},
        "model_slm":       {"type": "string"},
        "model_llm":       {"type": "string"},
        "started_at":      {"type": "string"},
        "completed_at":    {"type": "string"},
        "latency_s":       {"type": "number", "minimum": 0},
        "success":         {"type": "integer", "enum": [0, 1]},
        "cascade_failure": {"type": "integer", "enum": [0, 1]},
        "tokens": {
            "type": "object",
            "required": ["slm", "llm", "total"],
            "properties": {
                "slm":   {"type": "integer", "minimum": 0},
                "llm":   {"type": "integer", "minimum": 0},
                "total": {"type": "integer", "minimum": 0},
            },
        },
        "s0": {
            "type": "object",
            "required": ["catches", "repairs", "would_have_failed"],
            "properties": {
                "catches":          {"type": "integer", "minimum": 0},
                "repairs":          {"type": "integer", "minimum": 0},
                "would_have_failed": {"type": "integer", "minimum": 0},
            },
        },
        "escalations": {
            "type": "object",
            "required": ["P1", "P2", "P3", "P4"],
            "properties": {
                "P1": {"type": "integer", "minimum": 0},
                "P2": {"type": "integer", "minimum": 0},
                "P3": {"type": "integer", "minimum": 0},
                "P4": {"type": "integer", "minimum": 0},
            },
        },
        # FIX 5 (P2): Full per_step item subschema enforcing EXPERIMENTS.md §4 fields.
        "per_step": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "step_id", "agent_id", "tier", "raw_output",
                    "schema_valid", "would_have_been_invalid",
                    "downstream_consumed", "tokens_used", "latency_ms",
                    "adversarial_corruption_applied",
                    "adversarial_injection_status",
                ],
                "additionalProperties": True,
                "properties": {
                    "step_id":   {"type": "integer", "minimum": 1},
                    "agent_id":  {"type": "string"},
                    "tier":      {"type": "integer", "enum": [1, 2]},
                    "raw_output": {"type": "string"},
                    "schema_valid": {"type": "boolean"},
                    "would_have_been_invalid": {"type": "boolean"},
                    "downstream_consumed": {"type": "boolean"},
                    "tokens_used": {"type": "integer", "minimum": 0},
                    "latency_ms":  {"type": "integer", "minimum": 0},
                    "adversarial_corruption_applied": {
                        "oneOf": [{"type": "null"}, {"type": "string"}],
                    },
                    "adversarial_injection_status": {
                        "oneOf": [{"type": "null"}, {"type": "string"}],
                    },
                },
            },
        },
        "final_output":       {"type": "string"},
        "ground_truth":       {"type": "string"},
        "rubric_scores":      {"type": "object"},
        "ground_truth_match": {"type": "boolean"},
        "voc_trace":          {"type": "array"},
        "crash_reason": {
            "oneOf": [{"type": "null"}, {"type": "string"}],
        },
        "grading_api_calls": {"type": "integer", "minimum": 0},
        "grading_cost_usd":  {"type": "number",  "minimum": 0.0},
        "backend":           {"type": "string"},
        "api_cost_usd":      {"type": "number",  "minimum": 0.0},
    },
}

_RUNLOG_VALIDATOR = Draft7Validator(_RUNLOG_SCHEMA)


def _make_kernel(
    condition:            str,
    slm_model:            str,
    llm_model:            str,
    ollama_url:           str,
    tau:                  float,
    _llm_client:          Any = None,
    temperature_override: Any = None,
) -> Any:
    """Build and return a MASKernel for the given condition."""
    from mas.kernel import create_kernel  # noqa: PLC0415
    return create_kernel(
        slm_model            = slm_model,
        llm_model            = llm_model,
        ollama_url           = ollama_url,
        tau                  = tau,
        condition            = condition,
        _llm_client          = _llm_client,
        temperature_override = temperature_override,
    )


def prewarm(slm: str, llm: str, ollama_url: str) -> None:
    """Send a 1-token generation to each model to ensure they are loaded."""
    from mas.core.ollama_client import OllamaClient  # noqa: PLC0415
    client = OllamaClient(ollama_url)
    for model in (slm, llm):
        try:
            client.generate(
                model      = model,
                prompt     = "ping",
                force_json = False,
                max_tokens = 1,
            )
        except Exception as exc:
            log.warning("Pre-warm failed for model %s: %s", model, exc)


def _extract_from_output(final_output: str, field: str) -> str:
    """Extract *field* from *final_output*, treating it as JSON if possible.

Falls back to the raw string for 'answer', 'raw', or on parse failure."""
    if not final_output:
        return ""
    try:
        parsed = json.loads(final_output)
        if isinstance(parsed, dict) and field in parsed:
            return str(parsed[field])
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: treat the whole output as the field value
    return final_output


def score_rubric(
    rubric:       list[dict[str, Any]],
    final_output: str,
    judge_client: Any | None = None,
) -> dict[str, bool]:
    """Score rubric checks against the agent's final_output.

Supported check_types:
    substring      — must_contain items appear in extracted field text AND
                     must_not_contain items do NOT appear in the field
                     (both case-insensitive)
    exact_match    — extracted field matches expected exactly (case-insensitive)
    numeric        — extracted field is numeric and within [min, max]
    list_subset    — required_items all appear in extracted field text
    semantic_judge — delegates to judge_client.judge(text, criteria) when
                     judge_client is provided; otherwise logs a warning and
                     returns True (stub behaviour, preserves unit-test compat)"""
    scores: dict[str, bool] = {}
    for check in rubric:
        check_id   = check.get("check_id", f"check_{len(scores)}")
        check_type = check.get("check_type", "")
        params     = check.get("check_params", {})
        field      = params.get("field", "answer")
        text       = _extract_from_output(final_output, field)

        try:
            if check_type == "substring":
                must_contain     = params.get("must_contain", [])
                must_not_contain = params.get("must_not_contain", [])
                # Convert text to lower-case once for efficiency
                text_lc = text.lower()
                contains_all = all(
                    kw.lower() in text_lc for kw in must_contain
                )
                contains_none = all(
                    nw.lower() not in text_lc for nw in must_not_contain
                )
                scores[check_id] = contains_all and contains_none

            elif check_type == "exact_match":
                expected         = str(params.get("expected", ""))
                scores[check_id] = text.strip().lower() == expected.strip().lower()

            elif check_type == "numeric":
                try:
                    value  = float(text)
                    ok     = True
                    mn     = params.get("min")
                    mx     = params.get("max")
                    if mn is not None:
                        ok = ok and value >= float(mn)
                    if mx is not None:
                        ok = ok and value <= float(mx)
                    scores[check_id] = ok
                except (ValueError, TypeError):
                    scores[check_id] = False

            elif check_type == "list_subset":
                required         = params.get("required_items", [])
                scores[check_id] = all(
                    item.lower() in text.lower() for item in required
                )

            elif check_type == "semantic_judge":
                if judge_client is not None:
                    criteria = params.get("criteria", "")
                    scores[check_id] = judge_client.judge(text, criteria)
                else:
                    log.warning(
                        "Rubric check '%s' uses semantic_judge but no "
                        "judge_client was provided; defaulting to True.",
                        check_id,
                    )
                    scores[check_id] = True

            else:
                log.warning("Unknown rubric check_type '%s' for '%s'; scoring False.",
                            check_type, check_id)
                scores[check_id] = False

        except Exception as exc:
            log.error("Error scoring rubric check '%s': %s", check_id, exc)
            scores[check_id] = False

    return scores


def _validate_run_log(run_log: RunLog) -> None:
    """Validate *run_log* against the EXPERIMENTS.md §4 JSON Schema."""
    errors = list(_RUNLOG_VALIDATOR.iter_errors(run_log))
    if errors:
        msgs = "; ".join(e.message for e in errors[:3])
        raise jsonschema.ValidationError(
            f"RunLog fails schema validation: {msgs}"
        )


def _build_crash_run_log(
    goal:         GoalSpec,
    condition:    str,
    seed:         int,
    slm:          str,
    llm:          str,
    crash_reason: str | None,
) -> RunLog:
    """Build a minimal valid crash RunLog when the kernel could not write one."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "run_id":             str(uuid.uuid4()),
        "goal_id":            goal.id,
        "goal_type":          goal.goal_type,
        "pipeline_depth_k":   goal.pipeline_depth_k,
        "condition":          condition,
        "routing_skipped":    condition == "BASELINE_LANGGRAPH",
        "seed":               seed,
        "model_slm":          slm,
        "model_llm":          llm,
        "started_at":         now,
        "completed_at":       now,
        "latency_s":          0.0,
        "success":            0,
        "cascade_failure":    None,   # filled by run_one()
        "tokens":             {"slm": 0, "llm": 0, "total": 0},
        "s0":                 {"catches": 0, "repairs": 0, "would_have_failed": 0},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           [],
        "final_output":       "",
        "ground_truth":       goal.ground_truth,
        "rubric_scores":      {},
        "ground_truth_match": None,   # filled by run_one()
        "voc_trace":          [],
        "crash_reason":       crash_reason or "Unknown crash",
        "grading_api_calls":  0,
        "grading_cost_usd":   0.0,
    }


def _run_framework(
    goal:                 GoalSpec,
    condition:            str,
    seed:                 int,
    slm:                  str,
    llm:                  str,
    ollama_url:           str,
    tau:                  float,
    skip_warmup:          bool,
    _llm_client:          Any = None,
    temperature_override: Any = None,
) -> RunLog:
    """Run *goal* under a framework condition using the MASKernel.

Redirects the kernel's internal run-log write to a temp file,
reads it back, and returns the (incomplete) RunLog dict for the
harness to augment."""
    if not skip_warmup and _llm_client is None:
        prewarm(slm, llm, ollama_url)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jsonl")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    original_log = os.environ.get("MAS_RUNS_LOG")
    os.environ["MAS_RUNS_LOG"] = str(tmp_path)

    kernel_crash: str | None = None
    try:
        kernel = _make_kernel(
            condition, slm, llm, ollama_url, tau,
            _llm_client=_llm_client,
            temperature_override=temperature_override,
        )
        kernel.run(
            goal.goal_text,
            adversarial_inject  = (goal.goal_type == "adversarial_reliability"),
            corruption_pattern  = goal.corruption_pattern or "",
            corruption_seed     = seed,
        )
    except Exception as exc:
        kernel_crash = f"{type(exc).__name__}: {exc}"
        log.exception("Kernel init/run raised for goal %s condition %s",
                      goal.id, condition)
    finally:
        if original_log is None:
            os.environ.pop("MAS_RUNS_LOG", None)
        else:
            os.environ["MAS_RUNS_LOG"] = original_log

    # Read back what the kernel wrote.
    run_log: RunLog | None = None
    try:
        text  = tmp_path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            run_log = json.loads(lines[-1])
    except Exception as read_exc:
        log.error("Failed to read kernel run_log from %s: %s", tmp_path, read_exc)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

    if run_log is None:
        # Kernel never wrote — build a minimal crash entry.
        run_log = _build_crash_run_log(
            goal, condition, seed, slm, llm,
            kernel_crash or "Kernel wrote no run_log entry",
        )
    elif kernel_crash:
        # Kernel crashed at init (before it could set crash_reason itself).
        run_log.setdefault("crash_reason", kernel_crash)
        run_log["success"] = 0

    return run_log


def _run_baseline(
    goal:                 GoalSpec,
    seed:                 int,
    slm:                  str,
    llm:                  str,
    ollama_url:           str,
    skip_warmup:          bool,
    temperature_override: Any = None,
) -> RunLog:
    """Run *goal* under the BASELINE_LANGGRAPH condition.

Uses mas/eval/baselines/langgraph_baseline.py's run_goal() and
redirects its internal write to a temp path."""
    if not skip_warmup:
        prewarm(slm, llm, ollama_url)

    from mas.core.ollama_client import OllamaClient              # noqa: PLC0415
    from mas.tools.base import ToolRegistry                      # noqa: PLC0415
    from mas.eval.baselines.langgraph_baseline import (          # noqa: PLC0415
        GoalSpec as BaselineGoalSpec,
        run_goal,
    )

    ollama_client = OllamaClient(ollama_url)
    repo_root = Path(__file__).resolve().parent.parent.parent
    ws = repo_root / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry(ws)

    baseline_goal = BaselineGoalSpec(
        id                 = goal.id,
        goal_type          = goal.goal_type,
        goal_text          = goal.goal_text,
        inputs             = goal.inputs,
        ground_truth       = goal.ground_truth,
        rubric             = goal.rubric,       # type: ignore[arg-type]
        pipeline_depth_k   = goal.pipeline_depth_k,
        corruption_pattern = goal.corruption_pattern,
    )

    # Redirect baseline's internal write to a temp file.
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jsonl")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        run_log = run_goal(
            baseline_goal,
            seed                 = seed,
            ollama_client        = ollama_client,
            tool_registry        = tools,
            model                = llm,
            runs_log_path        = str(tmp_path),
            adversarial_inject   = (goal.goal_type == "adversarial_reliability"),
            temperature_override = temperature_override,
        )
    except Exception as exc:
        crash_reason = f"{type(exc).__name__}: {exc}"
        log.exception("Baseline run raised for goal %s", goal.id)
        run_log = _build_crash_run_log(goal, "BASELINE_LANGGRAPH", seed, slm, llm, crash_reason)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

    return run_log


def run_one(
    goal_id:        str,
    condition:      str,
    seed:           int,
    model_overrides: dict[str, Any] | None = None,
    *,
    _skip_warmup:   bool          = False,
    _goals_dir:     Path | None   = None,
) -> RunLog:
    """Run ONE goal under ONE condition and return a validated RunLog.

This is the single-unit execution function.  It does NOT write to
runs.jsonl — the CLI (mas/eval/cli.py) handles the final append."""
    if condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Valid conditions: {sorted(_VALID_CONDITIONS)}"
        )

    # Load and validate the goal.
    goal = load_goal(goal_id, goals_dir=_goals_dir)

    # Fix random seeds for reproducibility.
    # due to floating-point reordering in parallel GPU kernels.
    random.seed(seed)
    try:
        import numpy as np          # noqa: PLC0415
        np.random.seed(seed)
    except ImportError:
        pass

    # Resolve model names, backend, judge backend, and temperature override.
    overrides     = model_overrides or {}
    slm           = str(overrides.get("slm",           "phi3:mini"))
    llm           = str(overrides.get("llm",           "llama3.1:8b"))
    ollama_url    = str(overrides.get("ollama_url",    "http://localhost:11434"))
    tau           = float(overrides.get("tau",         0.72))
    judge_backend = str(overrides.get("judge_backend", "none"))
    backend       = str(overrides.get("backend",       "ollama"))
    _raw_temp     = overrides.get("temperature")
    temperature_override: Any = float(_raw_temp) if _raw_temp is not None else None

    # Construct judge client (lazy import to avoid hard dependency on anthropic).
    judge_client: Any = None
    if judge_backend == "anthropic":
        from mas.eval.judge import AnthropicJudge  # noqa: PLC0415
        judge_client = AnthropicJudge()
    elif judge_backend == "ollama":
        from mas.eval.judge import OllamaJudge  # noqa: PLC0415
        judge_client = OllamaJudge(ollama_url=ollama_url)

    # Construct inference client for frontier backend.
    llm_client: Any = None
    if backend == "anthropic":
        import os as _os  # noqa: PLC0415
        api_key = _os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "backend='anthropic' requires ANTHROPIC_API_KEY environment variable."
            )
        from mas.core.anthropic_client import AnthropicClient  # noqa: PLC0415
        llm_client = AnthropicClient(api_key=api_key, default_model=llm)

    # Dispatch to the appropriate runtime.
    if condition == "BASELINE_LANGGRAPH":
        run_log = _run_baseline(
            goal, seed, slm, llm, ollama_url, _skip_warmup,
            temperature_override=temperature_override,
        )
    else:
        run_log = _run_framework(
            goal, condition, seed, slm, llm, ollama_url, tau, _skip_warmup,
            _llm_client=llm_client,
            temperature_override=temperature_override,
        )

    # Fill harness-owned fields (kernel leaves these empty or None).
    run_log["goal_id"]            = goal.id
    run_log["goal_type"]          = goal.goal_type
    run_log["pipeline_depth_k"]   = goal.pipeline_depth_k
    run_log["seed"]               = seed
    run_log["ground_truth"]       = goal.ground_truth

    for step in run_log.get("per_step", []):
        step.setdefault("adversarial_corruption_applied", None)
        step.setdefault("adversarial_injection_status", None)

    # Score rubric (pass judge_client for semantic_judge checks).
    rubric_scores = score_rubric(
        goal.rubric,
        run_log.get("final_output", ""),
        judge_client,
    )
    run_log["rubric_scores"]      = rubric_scores
    run_log["ground_truth_match"] = bool(rubric_scores) and all(rubric_scores.values())

    # Populate grading cost fields (zero when no judge was used).
    run_log["grading_api_calls"] = getattr(judge_client, "total_api_calls", 0)
    run_log["grading_cost_usd"]  = getattr(judge_client, "total_cost_usd",  0.0)

    # Populate inference backend metadata (DEV-008).
    run_log["backend"]       = backend
    run_log["api_cost_usd"]  = getattr(llm_client, "total_cost_usd", 0.0)

    # Cascade detection (requires ground_truth_match to be set first).
    run_log["cascade_failure"] = 1 if detect_cascade(run_log) else 0

    # Schema validation — fail loud before returning.
    _validate_run_log(run_log)

    return run_log

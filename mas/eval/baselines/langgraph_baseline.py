"""BASELINE_LANGGRAPH condition: vanilla LangGraph + Pydantic implementation."""
from __future__ import annotations

import fcntl
import json
import logging
import operator
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from jsonschema import Draft7Validator
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from mas.eval.adversarial_wrapper import corrupt
from mas.eval.baselines.pydantic_schemas import parse_agent_response

log = logging.getLogger(__name__)


# Audit schema — used for would_have_been_invalid computation.
# Intentionally NOT in the schemas.py repair dict so raw jsonschema

_AUDIT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type":    "object",
    "required": ["answer", "confidence", "reasoning"],
    "additionalProperties": True,
    "properties": {
        "answer":     {"type": "string", "maxLength": 6000},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning":  {"type": "string", "maxLength": 4000},
    },
}
_AUDIT_VALIDATOR = Draft7Validator(_AUDIT_SCHEMA)

# Step system prompt — intentionally similar to the framework ReActAgent prompt
# (divergence: no System-0 framing; no spawn() sub-agent instruction).
_STEP_SYS = """\
You are a capable AI assistant solving a multi-step task.

At each step, output EXACTLY ONE of these JSON formats:

Action (use a tool):
{"thought": "...", "action": "tool_name", "action_input": {"key": "value"}}

Final answer for this step:
{"answer": "...", "confidence": 0.85, "reasoning": "...", "complexity": "medium"}

Available tools:
- file_read:    {"path": "<filepath>"}
- file_write:   {"path": "<filepath>", "content": "<text>"}
- file_list:    {"path": "<directory>"}
- python_exec:  {"code": "<python_script>"}
- web_search:   {"query": "<search query>"}
- memory_recall: {"key": "<key>"}
- memory_store: {"key": "<key>", "value": "<value>"}

Return only valid JSON — no markdown, no prose outside the JSON object."""

_MAX_TOOL_ITERS = 3   # max tool call rounds per step


@dataclass
class GoalSpec:
    """One hand-crafted goal instance from the evaluation suite."""
    id:                  str
    goal_type:           str           # research_synthesis | financial_analysis | …
    goal_text:           str
    inputs:              dict[str, Any] = field(default_factory=dict)
    ground_truth:        str           = ""
    rubric:              dict[str, Any] = field(default_factory=dict)
    pipeline_depth_k:    int           = 3
    corruption_pattern:  Optional[str] = None  # adversarial only


class _BaselineState(TypedDict):
    # Goal
    goal_text:          str
    pipeline_depth_k:   int
    # Execution
    step_counter:       int
    done:               bool
    context:            Annotated[list[str], operator.add]        # accumulated answers
    per_step_log:       Annotated[list[dict[str, Any]], operator.add]
    # DAG-traced input consumption (flat list of step_ids consumed; FIX 1)
    step_inputs_used:   Annotated[list[Any], operator.add]
    # Result
    final_output:       str
    # Adversarial
    corruption_pattern:  str   # "" = none
    corruption_seed:     int
    corruption_step:     int   # 0-indexed step where corruption is injected; -1 = none
    adversarial_inject:  bool  # True = actually apply corruption
    # Injected dependencies (not modified by nodes)
    _model:             str
    _ollama:            Any    # OllamaClient | MockOllama
    _tools:             Any    # ToolRegistry | None
    _temperature:       float  # temperature override (FIX 3)


def _step_node(state: _BaselineState) -> dict[str, Any]:
    """Execute one pipeline step: prompt LLM, optionally call tools, audit."""
    step_idx    = state["step_counter"]
    ollama      = state["_ollama"]
    tools       = state["_tools"]
    model       = state["_model"]
    temperature = state.get("_temperature", 0.7)

    # Build context string from previous step summaries
    prev_ctx = "\n".join(
        f"[Step {i+1}]: {s}" for i, s in enumerate(state["context"])
    ) or "No prior steps."

    prompt = (
        f"Goal: {state['goal_text']}\n\n"
        f"This is step {step_idx + 1} of {state['pipeline_depth_k']}.\n\n"
        f"Previous results:\n{prev_ctx}\n\n"
        "Provide the output for this step."
    )

    # Inner ReAct tool loop — max _MAX_TOOL_ITERS iterations.
    tool_calls: list[dict[str, Any]] = []
    step_tokens   = 0
    step_start    = time.perf_counter()
    step_dict: dict[str, Any] = {}
    raw_text       = ""
    pydantic_error: Optional[Exception] = None

    for _iter in range(_MAX_TOOL_ITERS + 1):
        if ollama is None:
            raw_text = '{"answer":"No LLM","confidence":0.0,"reasoning":"no client"}'
            break

        resp = ollama.generate(
            model       = model,
            prompt      = prompt,
            system      = _STEP_SYS,
            force_json  = False,          # CRITICAL: no constrained decoding
            temperature = temperature,
            max_tokens  = 800,
        )
        raw_text     = resp.get("text", "")
        step_tokens += resp.get("tokens_used", 0)

        # Parse output (not validated yet — this is plain text)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = {"answer": raw_text[:500], "confidence": 0.0, "reasoning": ""}

        # Tool action branch
        if "action" in parsed and tools is not None:
            action  = str(parsed.get("action", ""))
            ai      = parsed.get("action_input", {})
            kwargs  = ai if isinstance(ai, dict) else {}
            try:
                tool_result = tools.call(action, **kwargs)
                tr_str      = str(tool_result)[:400]
            except Exception as exc:
                tr_str = f"[tool error: {exc}]"
            tool_calls.append({"tool": action, "result": tr_str})
            prompt += f"\n\nTool result ({action}): {tr_str}\nContinue:"
        else:
            # Final answer for this step
            step_dict = parsed
            break

    # Post-hoc Pydantic validation (not enforced — on failure, propagate raw).
    if step_dict:
        _, pydantic_error = parse_agent_response(step_dict)
        if pydantic_error:
            log.warning(
                "[baseline] Step %d Pydantic error: %s",
                step_idx + 1, pydantic_error,
            )
            # Do NOT retry or escalate — pass through as-is.
    else:
        step_dict = {"answer": raw_text[:500], "confidence": 0.0, "reasoning": ""}

    # Adversarial corruption injection — conditional on adversarial_inject flag.
    corruption_applied: Optional[str] = None
    if (
        state.get("adversarial_inject", False)
        and state["corruption_pattern"]
        and state["corruption_step"] == step_idx
    ):
        corrupted = corrupt(
            step_dict,
            pattern = state["corruption_pattern"],
            schema  = _AUDIT_SCHEMA,
            seed    = state["corruption_seed"],
        )
        if isinstance(corrupted, str):
            # malformed_json: try to parse; on failure keep as-is in a wrapper
            try:
                step_dict = json.loads(corrupted)
            except json.JSONDecodeError:
                step_dict = {"answer": corrupted[:300], "confidence": 0.0, "reasoning": ""}
        else:
            step_dict = corrupted
        corruption_applied = state["corruption_pattern"]
        log.info(
            "[baseline] Corruption '%s' applied at step %d",
            state["corruption_pattern"], step_idx + 1,
        )

    # Cascade audit: raw jsonschema validation WITHOUT repair.
    if isinstance(step_dict, dict):
        audit_errors = list(_AUDIT_VALIDATOR.iter_errors(step_dict))
    else:
        audit_errors = ["non-dict output"]
    would_have_been_invalid = len(audit_errors) > 0

    latency_ms = int((time.perf_counter() - step_start) * 1000)

    per_step_entry: dict[str, Any] = {
        "step_id":                        step_idx + 1,
        "agent_id":                       "baseline_llm",
        "tier":                           2,
        "raw_output":                     json.dumps(step_dict, default=str)[:2000],
        "schema_valid":                   not would_have_been_invalid,
        "would_have_been_invalid":        would_have_been_invalid,
        "downstream_consumed":            False,   # post-processed by run_goal()
        "tokens_used":                    step_tokens,
        "latency_ms":                     latency_ms,
        "predicate_fired":                None,    # baseline has no predicates
        "pydantic_error":                 str(pydantic_error) if pydantic_error else None,
        "tool_calls":                     tool_calls,
        "adversarial_corruption_applied": corruption_applied,
    }

    answer      = str(step_dict.get("answer", ""))
    next_step   = step_idx + 1
    is_final    = next_step >= state["pipeline_depth_k"]

    # DAG-traced input consumption: the baseline uses ALL prior step answers as
    # context (not just last 4), so every prior step_id is consumed by this step.
    # step_idx is 0-based; step_ids are 1-based.
    consumed_step_ids = list(range(1, step_idx + 1))

    return {
        "step_counter":    next_step,
        "done":            is_final,
        "context":         [answer[:400]],
        "per_step_log":    [per_step_entry],
        "step_inputs_used": consumed_step_ids,   # DAG edge tracking (FIX 1)
        "final_output":    answer if is_final else state["final_output"],
    }


def _route(state: _BaselineState) -> str:
    return "end" if state["done"] else "continue"


def _build_baseline_graph():
    """Compile the sequential baseline StateGraph."""
    g = StateGraph(_BaselineState)
    g.add_node("step", _step_node)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", _route, {"continue": "step", "end": END})
    return g.compile()


_COMPILED_GRAPH = _build_baseline_graph()


def run_goal(
    goal:                 GoalSpec,
    seed:                 int          = 0,
    ollama_client:        Any          = None,
    tool_registry:        Any          = None,
    model:                str          = "llama3.1:8b",
    runs_log_path:        Optional[str] = None,
    *,
    adversarial_inject:   bool         = False,
    temperature_override: Any          = None,
) -> dict[str, Any]:
    """Execute *goal* under the BASELINE_LANGGRAPH condition."""
    run_id     = str(uuid.uuid4())
    started_at = time.time()

    # Determine which step to corrupt for adversarial goals.
    # EXPERIMENTS.md §3: "deterministic step-2 corruption" (1-indexed → index 1).
    # Only inject if adversarial_inject=True is explicitly requested.
    corruption_step = (1 if (adversarial_inject and goal.corruption_pattern) else -1)

    initial: dict[str, Any] = {
        "goal_text":          goal.goal_text,
        "pipeline_depth_k":   goal.pipeline_depth_k,
        "step_counter":       0,
        "done":               False,
        "context":            [],
        "per_step_log":       [],
        "step_inputs_used":   [],
        "final_output":       "",
        "corruption_pattern":  goal.corruption_pattern or "",
        "corruption_seed":     seed,
        "corruption_step":     corruption_step,
        "adversarial_inject":  adversarial_inject,
        "_model":             model,
        "_ollama":            ollama_client,
        "_tools":             tool_registry,
        "_temperature":       float(temperature_override) if temperature_override is not None else 0.7,
    }

    crash_reason: Optional[str] = None
    try:
        final = _COMPILED_GRAPH.invoke(initial)
    except Exception as exc:
        crash_reason = f"{type(exc).__name__}: {exc}"
        log.exception("[baseline] Graph execution failed for goal %s", goal.id)
        final = {
            "per_step_log": [],
            "final_output": "",
            "step_counter": 0,
        }

    completed_at = time.time()

    # DAG-traced downstream_consumed (FIX 1 / DEV-013): build consumed_set from
    # the flat list of step_ids that were read as context by any subsequent step.
    per_step: list[dict[str, Any]] = list(final.get("per_step_log", []))
    consumed_set: set = set(final.get("step_inputs_used", []))
    for step in per_step:
        step["downstream_consumed"] = bool(step.get("step_id") in consumed_set)

    # Adversarial injection status (FIX 2 / DEV-014).
    adversarial_injection_status: Optional[str] = None
    if adversarial_inject and goal.corruption_pattern:
        step2_reached = any(s.get("step_id") == 2 for s in per_step)
        if not step2_reached:
            adversarial_injection_status = "scheduled_but_not_reached"

    total_tokens = sum(s.get("tokens_used", 0) for s in per_step)
    success      = 1 if (not crash_reason and final.get("final_output")) else 0

    def _iso(ts: float) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    run_log: dict[str, Any] = {
        "run_id":             run_id,
        "goal_id":            goal.id,
        "goal_type":          goal.goal_type,
        "pipeline_depth_k":   goal.pipeline_depth_k,
        "condition":          "BASELINE_LANGGRAPH",
        "routing_skipped":    True,   # no System-3 routing
        "seed":               seed,
        "model_slm":          "phi3:mini",   # not used; logged for schema parity
        "model_llm":          model,
        "started_at":         _iso(started_at),
        "completed_at":       _iso(completed_at),
        "latency_s":          round(completed_at - started_at, 3),
        "success":            success,
        "cascade_failure":    None,   # set by eval harness after rubric scoring
        "tokens":             {"slm": 0, "llm": total_tokens, "total": total_tokens},
        "s0":                 {"catches": 0, "repairs": 0,
                               "would_have_failed": sum(
                                   1 for s in per_step if s.get("would_have_been_invalid")
                               )},
        "escalations":        {"P1": 0, "P2": 0, "P3": 0, "P4": 0},
        "per_step":           per_step,
        "final_output":                final.get("final_output", ""),
        "ground_truth":                goal.ground_truth,
        "rubric_scores":               {},     # set by eval harness
        "ground_truth_match":          None,   # set by eval harness
        "voc_trace":                   [],
        "crash_reason":                crash_reason,
        "grading_api_calls":           0,   # set by eval harness when judge is enabled
        "grading_cost_usd":            0.0, # set by eval harness when judge is enabled
        "adversarial_injection_status": adversarial_injection_status,
    }

    # fcntl-locked atomic append to runs.jsonl (safe for parallel batch workers).
    log_path = runs_log_path or os.environ.get("MAS_RUNS_LOG", "./runs.jsonl")
    try:
        line = json.dumps(run_log) + "\n"
        with open(log_path, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        log.exception("[baseline] Failed to write run log to %s", log_path)

    return run_log

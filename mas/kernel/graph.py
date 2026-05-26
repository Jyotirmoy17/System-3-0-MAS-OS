"""LangGraph execution graph construction."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from jsonschema import Draft7Validator

from langgraph.graph import END, START, StateGraph

from .budget import BudgetManager
from .registry import AgentRegistry, EscalationReason, Tier
from .scheduler import MAX_ESCALATIONS, Scheduler
from .state import (
    OSState,
    evt_agent_done,
    evt_agent_start,
    evt_error,
    evt_halt,
    evt_plan,
    evt_s0,
    evt_tool,
)
from .validator import System0

log = logging.getLogger(__name__)


def build_graph(
    registry:   AgentRegistry,
    scheduler:  Scheduler,
    system0:    System0,
    budget_mgr: BudgetManager,
    agents:     Dict[str, Any],
):
    """
    Compile and return the LangGraph StateGraph (full System 0/3 architecture).

    All node functions are defined here and closed over the injected
    dependencies.  The returned graph is stateless and thread-safe;
    each .invoke() / .astream() call gets its own isolated state copy.
    """

    def node_plan(state: OSState) -> Dict[str, Any]:
        goal        = state["goal"]
        esc_count   = state["escalation_count"]
        esc_reason  = state["escalation_reason"]
        spent       = state["total_tokens"]

        if budget_mgr.is_exhausted():
            return {
                "done":        True,
                "final_answer": state.get("final_answer") or "Budget exhausted.",
                "agent_events": [evt_halt("budget_exhausted", spent)],
            }

        if esc_count == 0:
            agent_id = scheduler.select_initial(goal)
        else:
            agent_id = scheduler.select_recovery(
                EscalationReason(esc_reason),
                state.get("current_agent_id") or "",
            )

        log.info("[S3] plan: agent=%s esc=%d", agent_id, esc_count)

        return {
            "current_agent_id": agent_id,
            "agent_events":     [evt_plan("dynamic", agent_id, esc_count)],
        }

    def node_run(state: OSState) -> Dict[str, Any]:
        # graph.py → agents.base (for TaskInput) would be fine because
        # agents.base only imports from kernel.registry, but to be safe
        # and explicit we do a local import.
        from mas.agents.base import TaskInput  # noqa: PLC0415

        agent_id = state.get("current_agent_id")
        if not agent_id or agent_id not in agents:
            msg = f"Agent '{agent_id}' not in registry"
            log.error("[run_agent] %s", msg)
            return {
                "escalation_reason": EscalationReason.P3_CAPABILITY.value,
                "escalation_count":  state["escalation_count"] + 1,
                "agent_events":      [evt_error(msg)],
            }

        agent = agents[agent_id]
        spec  = registry.get_spec(agent_id)
        name  = spec.name if spec else agent_id

        # DAG-traced input consumption: record which prior step_ids are read as
        # context for this step.  Messages and per_step_log are in 1-1
        # correspondence (one entry per agent dispatch), so last-4 messages
        # correspond to the last-4 per_step_log entries.
        per_step_log_prev = state.get("per_step_log", [])
        n_prev = len(per_step_log_prev)
        consumed_step_ids: list = [
            per_step_log_prev[i]["step_id"]
            for i in range(max(0, n_prev - 4), n_prev)
            if "step_id" in per_step_log_prev[i]
        ]

        # Build context from the most recent agent messages (last 4)
        context: Dict[str, str] = {}
        for msg in state.get("messages", [])[-4:]:
            if msg.get("role") == "agent" and msg.get("content"):
                context[msg.get("agent_id", "prev")] = msg["content"]

        remaining = max(200, state["budget"] - state["total_tokens"])
        # current step_id = number of steps already validated + 1
        current_step_id = n_prev + 1
        task = TaskInput(
            task_id        = f"{state['session_id']}_{uuid.uuid4().hex[:6]}",
            goal           = state["goal"],
            query          = state["goal"],
            context        = context,
            budget_tokens  = remaining,
            depth          = state["escalation_count"],
        )

        ev_start = evt_agent_start(agent_id, name)
        log.info("[run_agent] running %s (budget_remaining=%d)", agent_id, remaining)

        output = agent.run(task)

        ev_done = evt_agent_done(
            agent_id   = agent_id,
            tokens     = output.tokens_used,
            confidence = output.confidence,
            n_tools    = len(output.tool_calls),
        )

        tool_events = [
            evt_tool(
                tc.get("tool", ""),
                tc.get("success", True),
                str(tc.get("output", ""))[:120],
            )
            for tc in output.tool_calls
        ]

        # Update registry cost profile from real execution
        if output.latency_s > 0:
            registry.update_cost(agent_id, output.tokens_used, output.latency_s)

        return {
            "last_output":      output.to_dict(),
            "total_tokens":     state["total_tokens"] + output.tokens_used,
            "agent_events":     [ev_start, ev_done, *tool_events],
            "messages": [{
                "role":     "agent",
                "agent_id": agent_id,
                "content":  output.answer,
                "tokens":   output.tokens_used,
                "ts":       time.time(),
                "step_id":  current_step_id,   # DAG lineage
            }],
            "step_inputs_used": consumed_step_ids,   # DAG edge tracking
        }

    def node_validate(state: OSState) -> Dict[str, Any]:
        raw = state.get("last_output")
        if not raw:
            return {
                "escalation_reason": EscalationReason.P2_SCHEMA.value,
                "escalation_count":  state["escalation_count"] + 1,
                "agent_events":      [evt_error("node_validate: no last_output")],
                "per_step_log":      [],
            }

        agent_id    = raw.get("agent_id", "unknown")
        task_id     = raw.get("task_id",  "unknown")
        depth       = raw.get("depth",    0)
        current_step = len(state.get("per_step_log", [])) + 1  # 1-indexed

        # Injection at step 2 (DEV-014).  Corruption is applied BEFORE
        # System-0 validation so S0 can catch it (Theorem 3.2 test).
        inject_step       = 2
        adversarial_pattern: str = state.get("adversarial_pattern", "")
        corruption_applied: Optional[str] = None

        if (
            state.get("adversarial_inject", False)
            and adversarial_pattern
            and current_step == inject_step
        ):
            try:
                from mas.eval.adversarial_wrapper import corrupt  # lazy import
                from .schemas import SCHEMAS as _SCHEMAS
                schema_for_corrupt = _SCHEMAS.get("agent_response", {})
                corrupted = corrupt(
                    dict(raw),
                    pattern = adversarial_pattern,
                    schema  = schema_for_corrupt,
                    seed    = int(state.get("adversarial_seed", 0)),
                )
                if isinstance(corrupted, str):
                    # malformed_json: try to parse; on failure use empty dict
                    try:
                        raw = json.loads(corrupted)
                    except json.JSONDecodeError:
                        raw = {"answer": "", "confidence": 0.0, "reasoning": ""}
                else:
                    raw = corrupted
                corruption_applied = adversarial_pattern
                log.info(
                    "[ADV] Corruption '%s' injected at step %d",
                    adversarial_pattern, current_step,
                )
            except Exception as exc:
                log.warning("[ADV] Injection failed: %s", exc)

        v_result = system0.validate_dict(
            data        = raw,
            schema_name = "agent_response",
            agent_id    = agent_id,
            task_id     = task_id,
            depth       = depth,
        )

        # Safe confidence extraction.
        raw_conf = v_result.data.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (TypeError, ValueError):
            confidence = 0.0
        answer  = str(v_result.data.get("answer", ""))
        pre_esc = EscalationReason(raw.get("escalation_reason", "none"))

        # Counterfactual instrumentation: would_have_been_invalid is True iff the
        # RAW output (before System-0 repair) fails schema validation.  This is
        # the correct semantics per EXPERIMENTS.md §4: "whether validation WOULD
        # HAVE failed" = what happens to the unrepaired output without System-0.
        from .schemas import SCHEMAS as _SCHEMAS_AUDIT
        _audit_schema = _SCHEMAS_AUDIT.get("agent_response", {})
        _audit_validator = Draft7Validator(_audit_schema)
        initial_errors              = list(_audit_validator.iter_errors(raw))
        would_have_been_invalid: bool = len(initial_errors) > 0

        # Tier lookup for per-step log (S1=1, S2=2).
        spec     = registry.get_spec(agent_id)
        tier_int = 1 if (spec and spec.tier == Tier.S1) else 2

        predicate = system0.check_predicates(
            answer             = answer,
            confidence         = confidence,
            schema_valid       = v_result.valid,
            pre_set_escalation = pre_esc,
        )

        ev = evt_s0(
            valid     = v_result.valid,
            repaired  = v_result.repaired,
            predicate = predicate.value,
            vs        = system0.Vs(),
        )

        # Per-step log entry (downstream_consumed set post-hoc by MASKernel).
        step_entry: Dict[str, Any] = {
            "step_id":                         current_step,
            "agent_id":                        agent_id,
            "tier":                            tier_int,
            "raw_output":                      json.dumps(raw, default=str)[:2000],
            "schema_valid":                    v_result.valid,
            "would_have_been_invalid":         would_have_been_invalid,
            "downstream_consumed":             False,   # updated post-hoc
            "tokens_used":                     int(raw.get("tokens_used", 0)),
            "latency_ms":                      int(float(raw.get("latency_s", 0)) * 1000),
            "predicate_fired":                 predicate.value if predicate != EscalationReason.NONE else None,
            "escalation_reason":               raw.get("escalation_reason", "none"),
            "escalation_message":              str(raw.get("escalation_message", ""))[:500],
            "adversarial_corruption_applied":  corruption_applied,
            "adversarial_injection_status":    "injected" if corruption_applied else None,
        }

        if predicate == EscalationReason.NONE:
            return {
                "final_answer":     answer,
                "final_confidence": confidence,
                "done":             True,
                "agent_events":     [ev],
                "per_step_log":     [step_entry],
            }

        # VoC gate — only for P1.  P2/P3/P4 are hard structural failures that
        # always escalate.  P1 only escalates if the recovery gain exceeds cost.
        if predicate == EscalationReason.P1_CONFIDENCE:
            recovery_id   = scheduler.select_recovery(predicate, agent_id)
            recovery_spec = registry.get_spec(recovery_id)
            if recovery_spec is not None:
                effective_cost = (
                    recovery_spec.cost_profile.expected_tokens
                    * recovery_spec.cost_profile.cost_per_token
                )
                voc_val = scheduler.voc.compute(confidence, effective_cost)
                if voc_val <= 0.0:
                    log.info(
                        "[VoC] P1 suppressed — conf=%.2f VoC=%.4f — committing answer",
                        confidence, voc_val,
                    )
                    return {
                        "final_answer":     answer,
                        "final_confidence": confidence,
                        "done":             True,
                        "agent_events":     [ev],
                        "per_step_log":     [step_entry],
                    }

        new_count = state["escalation_count"] + 1
        log.info("[S0] predicate=%s count=%d/%d", predicate.value, new_count, MAX_ESCALATIONS)
        return {
            "escalation_reason": predicate.value,
            "escalation_count":  new_count,
            "agent_events":      [ev],
            "per_step_log":      [step_entry],
        }

    def route_after_validate(state: OSState) -> str:
        """Conditional edge: complete or escalate."""
        if state.get("done"):
            return "complete"
        if scheduler.should_halt(
            state["escalation_count"],
            state.get("final_confidence", 0.0),
            state["goal"],
        ):
            # Mark done without a clean answer
            return "complete"
        return "escalate"

    g = StateGraph(OSState)

    g.add_node("s3_plan",     node_plan)
    g.add_node("run_agent",   node_run)
    g.add_node("s0_validate", node_validate)

    g.add_edge(START,        "s3_plan")
    g.add_edge("s3_plan",    "run_agent")
    g.add_edge("run_agent",  "s0_validate")

    g.add_conditional_edges(
        "s0_validate",
        route_after_validate,
        {"escalate": "s3_plan", "complete": END},
    )

    return g.compile()

"""LangGraph execution state and event helpers."""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class OSState(TypedDict):
    """
    The typed state carried between LangGraph nodes.

    Append-only fields (messages, agent_events) accumulate across nodes;
    scalar fields (current_agent_id, done, etc.) are overwritten.
    """

    session_id:        str
    goal:              str
    complexity:        str                                    # "" until classified
    current_agent_id:  Optional[str]
    last_output:       Optional[Dict[str, Any]]
    escalation_reason: str                                    # EscalationReason.value
    escalation_count:  int
    total_tokens:      int
    budget:            int
    messages:          Annotated[List[Dict[str, Any]], operator.add]   # append-only
    agent_events:      Annotated[List[Dict[str, Any]], operator.add]   # append-only
    final_answer:      Optional[str]
    final_confidence:  float
    done:              bool
    per_step_log:      Annotated[List[Dict[str, Any]], operator.add]  # one entry per agent dispatch
    # DAG-traced input consumption: flat list of step_ids consumed as context.
    # node_run appends the step_ids it reads; finalization builds consumed_set.
    step_inputs_used:  Annotated[List[Any], operator.add]   # DAG edge tracking
    # Adversarial injection (Prompt 3.1): set per-run, never changed by nodes.
    adversarial_inject:   bool   # True = apply corruption at injection step
    adversarial_pattern:  str    # corruption pattern name from adversarial_wrapper
    adversarial_seed:     int    # seed forwarded to corrupt()


def make_initial_state(session_id: str, goal: str, budget: int) -> Dict[str, Any]:
    """Return a fresh initial OSState dict for a new goal execution."""
    return {
        "session_id":        session_id,
        "goal":              goal,
        "complexity":        "",
        "current_agent_id":  None,
        "last_output":       None,
        "escalation_reason": "none",
        "escalation_count":  0,
        "total_tokens":      0,
        "budget":            budget,
        "messages":          [],
        "agent_events":      [],
        "final_answer":      None,
        "final_confidence":  0.0,
        "done":              False,
        "per_step_log":         [],
        "step_inputs_used":     [],
        "adversarial_inject":   False,
        "adversarial_pattern":  "",
        "adversarial_seed":     0,
    }


def evt(kind: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a typed event dict with a UTC timestamp."""
    return {"type": kind, "ts": time.time(), **kwargs}


def evt_plan(complexity: str, agent_id: str, esc_count: int) -> Dict[str, Any]:
    return evt(
        "plan",
        complexity=complexity,
        selected_agent=agent_id,
        escalation_count=esc_count,
    )


def evt_agent_start(agent_id: str, agent_name: str) -> Dict[str, Any]:
    return evt("agent_start", agent_id=agent_id, agent_name=agent_name)


def evt_agent_done(
    agent_id:   str,
    tokens:     int,
    confidence: float,
    n_tools:    int,
) -> Dict[str, Any]:
    return evt(
        "agent_done",
        agent_id=agent_id,
        tokens=tokens,
        confidence=round(confidence, 4),
        tool_calls=n_tools,
    )


def evt_tool(tool: str, success: bool, preview: str = "") -> Dict[str, Any]:
    return evt("tool_call", tool=tool, success=success, preview=preview[:120])


def evt_s0(valid: bool, repaired: bool, predicate: str, vs: float) -> Dict[str, Any]:
    return evt(
        "s0_validate",
        valid=valid,
        repaired=repaired,
        predicate=predicate,
        Vs=round(vs, 4),
    )


def evt_halt(reason: str, spent: int) -> Dict[str, Any]:
    return evt("halt", reason=reason, spent=spent)


def evt_error(msg: str) -> Dict[str, Any]:
    return evt("error", msg=str(msg)[:400])

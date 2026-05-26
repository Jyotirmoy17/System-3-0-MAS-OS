"""ReActAgent: Reason → Act → Observe loop."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mas.agents.base import AgentOutput, BaseAgent, TaskInput
from mas.kernel.registry import Capability, Tier
from mas.kernel.validator import System0


# Maps ReAct tool names to S1 capability tags for cheapest_capable() lookup.

_TOOL_CAPABILITY: Dict[str, Capability] = {
    "file_read":     Capability.FILE_READ,
    "file_write":    Capability.FILE_WRITE,
    "file_list":     Capability.FILE_READ,
    "file_csv":      Capability.FILE_READ,
    "python_exec":   Capability.PYTHON_EXEC,
    "web_search":    Capability.WEB_SEARCH,
    "web_fetch":     Capability.WEB_SEARCH,
    "memory_store":  Capability.MEMORY_WRITE,
    "memory_recall": Capability.MEMORY_READ,
    "memory_list":   Capability.MEMORY_READ,
    "shell_exec":    Capability.SHELL_EXEC,
}


def _build_spawn_query(action: str, action_input: Dict[str, Any]) -> str:
    """
    Translate a structured (action, action_input) pair into a plain-English
    query that the target S1 agent's LLM prompt can parse and act on.

    Args:
        action:       Tool name from the ReAct LLM's action field.
        action_input: Arguments dict from the ReAct LLM's action_input field.

    Returns:
        A human-readable task query for the S1 sub-agent.
    """
    ai = action_input
    if action == "python_exec":
        return f"Execute this Python code and return the result:\n{ai.get('code', '')}"
    if action in ("file_read", "file_csv"):
        return f"Read the file at path '{ai.get('path', '')}' and return its contents."
    if action == "file_write":
        return f"Write to file '{ai.get('path', '')}': {str(ai.get('content', ''))[:300]}"
    if action == "file_list":
        return f"List files in directory '{ai.get('path', '.')}'"
    if action == "web_search":
        return str(ai.get("query", str(ai)))
    if action == "web_fetch":
        return f"Fetch the URL '{ai.get('url', '')}' and return the page content."
    if action == "memory_store":
        return f"Remember: key='{ai.get('key', '')}', value='{ai.get('value', '')}'"
    if action == "memory_recall":
        return f"Recall the value for key='{ai.get('key', '')}'"
    if action == "memory_list":
        return "List all stored memory keys."
    if action == "shell_exec":
        return f"Run shell command: {ai.get('command', '')}"
    return f"Tool: {action}\nArguments: {action_input}"


_REACT_SYS = """You are a ReAct agent. Think step-by-step and use tools to find answers.

At each step, output EXACTLY ONE of these JSON formats:

Format A — take an action:
{"thought": "...", "action": "tool_name", "action_input": {"key": "value"}}

Format B — final answer (when done):
{"answer": "...", "confidence": 0.85, "reasoning": "...", "complexity": "hard"}

Tool contracts — use these exact action_input keys (string values unless noted):
- file_read: Reads a file from the workspace. You MUST provide the argument as {"path": "<filepath>"}
- file_list: Lists files under a directory in the workspace. You MUST provide the argument as {"path": "<directory>"} (optional second key "pattern" like "*")
- file_write: Writes text to a file. You MUST provide the argument as {"path": "<filepath>", "content": "<text>"}
- file_csv: Reads the first 10 rows of a CSV to preview the column schema. You MUST provide the argument as {"path": "<filepath>"}. CRITICAL: Do NOT use this for full data analysis because it truncates data. To analyze full datasets, you MUST use python_exec to write a script that reads the entire file.
- python_exec: Executes Python code. You MUST provide the argument as {"code": "<python_script>"}.
  CRITICAL: A 'WORKSPACE' pathlib.Path object is pre-loaded. To read a file inside your script, you MUST use this exact pattern:
  text = (WORKSPACE / 'data' / 'filename.csv').read_text()
"""

_FINAL_SYS = """You reached the maximum step limit. Synthesise the best possible answer from the history provided.
Return ONLY JSON: {"answer": "...", "confidence": 0.5, "reasoning": "...", "complexity": "hard"}"""

# Module-level System-0 instance used to validate every S1 spawn result
# before its answer re-enters the ReAct observation history.
# confidence_threshold=0.0 suppresses P1 (we never escalate on spawn results).
_S0: System0 = System0(confidence_threshold=0.0)


class ReActAgent(BaseAgent):
    """S2 ReAct agent.  Dispatches all tool calls through S1 sub-agents."""

    MAX_STEPS = 8

    def run_internal(self, task: TaskInput) -> AgentOutput:
        """Execute the Reason→Act→Observe loop, delegating actions to S1 agents."""
        history: List[str] = []
        if task.context:
            history.append(self.build_context_str(task.context))

        total_tokens    = 0
        tool_calls:      List[Dict[str, Any]] = []
        sub_agents_used: List[str]            = []

        for _step in range(self.MAX_STEPS):
            prompt = "Task: " + task.query + "\n\nHistory:\n" + "\n".join(history)
            d, tokens = self.llm_json(prompt, system=_REACT_SYS, max_tokens=1000)
            total_tokens += tokens

            if "answer" in d:
                return AgentOutput(
                    agent_id        = self.agent_id,
                    task_id         = task.task_id,
                    answer          = str(d["answer"]),
                    confidence      = float(d.get("confidence", 0.85)),
                    reasoning       = str(d.get("reasoning", "")),
                    complexity      = str(d.get("complexity", "hard")),
                    tokens_used     = total_tokens,
                    tool_calls      = tool_calls,
                    sub_agents_used = sub_agents_used,
                )

            thought = d.get("thought", "").strip()
            action  = d.get("action",  "").strip()
            ai      = d.get("action_input", {})

            if thought:
                history.append(f"Thought: {thought}")

            if not action:
                continue

            kwargs = ai if isinstance(ai, dict) else {"query": str(ai)}

            # Find the cheapest S1 agent that owns this tool (via Capability tag)
            cap     = _TOOL_CAPABILITY.get(action)
            s1_spec = (
                self.registry.cheapest_capable({cap}, tier=Tier.S1)
                if cap is not None and self.registry is not None
                else None
            )

            if s1_spec is not None:
                spawn_query = _build_spawn_query(action, kwargs)
                sub_out     = self.spawn(s1_spec.agent_id, task, spawn_query, budget=400)

                if sub_out is not None:
                    total_tokens += sub_out.tokens_used
                    sub_agents_used.append(s1_spec.agent_id)
                    # System-0: validate sub-agent output before it enters the loop
                    vr = _S0.validate_dict(
                        sub_out.to_dict(), "agent_response",
                        agent_id=sub_out.agent_id, task_id=sub_out.task_id,
                    )
                    if vr.valid:
                        last_obs = (vr.data.get("answer") or str(vr.data))[:500]
                    else:
                        last_obs = (
                            "[S0 invalid: "
                            + (vr.errors[0] if vr.errors else "unknown")[:80]
                            + "]"
                        )
                    success = True
                else:
                    last_obs = f"[spawn({s1_spec.agent_id}) returned None]"
                    success  = False

                via = s1_spec.agent_id
            else:
                last_obs = f"[No S1 agent found for tool '{action}']"
                success  = False
                via      = "none"

            history.append(f"Action: {action}({kwargs})")
            history.append(f"Observation: {last_obs}")
            tool_calls.append({
                "tool":    action,
                "via":     via,
                "input":   str(ai)[:120],
                "output":  last_obs[:200],
                "success": success,
            })

        d, tokens = self.llm_json(
            "Task: " + task.query + "\n\nHistory:\n" + "\n".join(history[-8:]),
            system     = _FINAL_SYS,
            max_tokens = 600,
        )
        total_tokens += tokens

        return AgentOutput(
            agent_id        = self.agent_id,
            task_id         = task.task_id,
            answer          = str(d.get("answer", "Could not complete task within step limit.")),
            confidence      = float(d.get("confidence", 0.55)),
            reasoning       = "\n".join(history[-4:]),
            complexity      = "hard",
            tokens_used     = total_tokens,
            tool_calls      = tool_calls,
            sub_agents_used = sub_agents_used,
        )

"""MemoryAgent: persistent KV store across sessions."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_STORE_WORDS  = frozenset({"remember", "store", "save", "memorize", "memorise", "note"})
_RECALL_WORDS = frozenset({"recall", "retrieve", "what did", "remember what", "fetch"})


class MemoryAgent(BaseAgent):
    """
    S1 agent for reading and writing persistent memory.

    No LLM call needed — operations are deterministic.
    Confidence is always high because we either stored or retrieved successfully.
    """

    def run_internal(self, task: TaskInput) -> AgentOutput:
        q_words  = set(task.query.lower().split())
        tc: List[Dict[str, Any]] = []

        if q_words & _STORE_WORDS:
            key   = task.context.get("memory_key",
                                     f"note_{int(time.time())}")
            value = task.context.get("memory_value", task.query)
            r     = self.use_tool_raw("memory_store", key=key, value=value)
            tc.append({"tool": "memory_store", "key": key, "success": r.success})
            answer = (f"Stored '{key}' = {str(value)[:100]}"
                      if r.success else f"Store failed: {r.error}")
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = answer,
                confidence  = 0.95 if r.success else 0.0,
                reasoning   = "Memory write operation",
                complexity  = "easy",
                tokens_used = 0,
                tool_calls  = tc,
            )

        if q_words & _RECALL_WORDS:
            key = task.context.get("memory_key", "")
            r   = self.use_tool_raw("memory_recall", key=key)
            tc.append({"tool": "memory_recall", "key": key, "success": r.success})
            answer = (r.output if r.success
                      else f"Nothing found for key: '{key}'")
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = answer,
                confidence  = 0.90 if r.success else 0.20,
                reasoning   = "Memory read operation",
                complexity  = "easy",
                tokens_used = 0,
                tool_calls  = tc,
            )

        r  = self.use_tool_raw("memory_list")
        tc.append({"tool": "memory_list", "success": r.success})
        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = f"Stored memory keys: {r.output}",
            confidence  = 0.90,
            reasoning   = "Memory list operation",
            complexity  = "easy",
            tokens_used = 0,
            tool_calls  = tc,
        )

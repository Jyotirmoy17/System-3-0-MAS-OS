"""WebAgent: real web search via DuckDuckGo."""
from __future__ import annotations

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_SYS = (
    "You synthesise real web search results into a precise answer. "
    "If the results do not contain enough information, say so clearly. "
    "JSON only:\n"
    '{"answer":"<answer>","confidence":<0-1>,"reasoning":"<brief>","complexity":"medium"}'
)


class WebAgent(BaseAgent):
    """S1 agent that performs live DuckDuckGo searches."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        # Execute real web search
        r  = self.use_tool_raw("web_search", query=task.query, max_results=4)
        tc = [{"tool": "web_search", "query": task.query,
                "success": r.success, "output": str(r)[:200]}]

        ctx = self.build_context_str(task.context)
        d, tokens = self.llm_json(
            f"Question: {task.query}\n\n"
            f"Web search results:\n{str(r)[:1800]}"
            f"{ctx}\n\n"
            "Answer based on the search results:",
            system     = _SYS,
            max_tokens = 600,
        )

        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = str(d.get("answer") or "No relevant results found."),
            confidence  = float(d.get("confidence", 0.70)),
            reasoning   = d.get("reasoning", "Based on web search results."),
            complexity  = "medium",
            tokens_used = tokens,
            tool_calls  = tc,
        )

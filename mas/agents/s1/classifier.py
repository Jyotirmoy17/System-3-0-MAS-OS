"""ClassifierAgent: fast complexity estimation."""
from __future__ import annotations

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_SYS = (
    "Classify the complexity of this task. Be conservative — prefer 'hard' "
    "if multi-step reasoning or tool use is needed. JSON only:\n"
    "{\n"
    '  "complexity": "<easy|medium|hard>",\n'
    '  "confidence": <0-1>,\n'
    '  "reasoning": "<one sentence>",\n'
    '  "answer": "<one sentence summary of the task>"\n'
    "}"
)


class ClassifierAgent(BaseAgent):
    """
    S1 agent that estimates task complexity for System-3's planning step.

    Returns a well-structured AgentOutput with complexity in the
    top-level field so the scheduler can read it directly.
    """

    def run_internal(self, task: TaskInput) -> AgentOutput:
        if self.ollama is None:
            # No LLM: use heuristic from the scheduler
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = "medium complexity (no LLM available)",
                confidence  = 0.50,
                reasoning   = "Default — no LLM client attached",
                complexity  = "medium",
                tokens_used = 0,
            )

        d, tokens = self.llm_json(
            f"Classify: {task.query}",
            system     = _SYS,
            max_tokens = 200,
        )

        complexity = d.get("complexity", "medium")
        if complexity not in ("easy", "medium", "hard"):
            complexity = "medium"

        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = str(d.get("answer") or f"{complexity} complexity"),
            confidence  = float(d.get("confidence", 0.70)),
            reasoning   = d.get("reasoning", ""),
            complexity  = complexity,
            tokens_used = tokens,
        )

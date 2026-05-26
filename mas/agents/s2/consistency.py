"""SelfConsistencyAgent: K independent samples, majority vote."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from mas.agents.base import AgentOutput, BaseAgent, TaskInput

_SYS = (
    "Answer the question carefully and completely. JSON only:\n"
    '{"answer":"<answer>","confidence":<0-1>,"reasoning":"<explanation>","complexity":"hard"}'
)

K = 3  # number of independent samples


def _normalise(text: str) -> str:
    """Collapse whitespace and strip punctuation for voting."""
    t = str(text).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _majority_vote(answers: List[str]) -> Tuple[str, int]:
    """Return (winning_answer, vote_count) using normalised comparison."""
    if not answers:
        return "", 0
    normed  = [_normalise(a) for a in answers]
    winner, count = Counter(normed).most_common(1)[0]
    for original, norm in zip(answers, normed):
        if norm == winner:
            return original, count
    return answers[0], 1


class SelfConsistencyAgent(BaseAgent):
    """S2 agent that reduces uncertainty through K-sample majority voting."""

    def run_internal(self, task: TaskInput) -> AgentOutput:
        ctx          = self.build_context_str(task.context)
        prompt       = f"Goal: {task.goal}\nQuestion: {task.query}{ctx}"
        samples:     List[Dict[str, Any]] = []
        total_tokens = 0

        _kernel   = getattr(self, "kernel", None)
        _override = getattr(_kernel, "temperature_override", None) if _kernel else None
        base_temp = _override if _override is not None else self.spec.model_temp

        for i in range(K):
            # Increase temperature on each sample for answer diversity
            temp = base_temp + i * 0.15
            if self.ollama is None:
                break

            r = self.ollama.generate_json(
                model       = self.spec.model_name,
                prompt      = prompt,
                system      = _SYS,
                temperature = min(temp, 0.90),
                max_tokens  = 700,
            )
            total_tokens += r.get("tokens_used", 0)
            parsed = r.get("parsed") or {}
            if parsed:
                samples.append(parsed)

        if not samples:
            return AgentOutput(
                agent_id    = self.agent_id,
                task_id     = task.task_id,
                answer      = "",
                confidence  = 0.0,
                reasoning   = "No samples produced (LLM unavailable)",
                complexity  = "hard",
                tokens_used = total_tokens,
            )

        answers    = [s.get("answer", "") for s in samples]
        winner, n  = _majority_vote(answers)
        confidence = min(0.95, n / K + 0.05)

        # Use the full sample (reasoning included) that produced the winner
        best = next(
            (s for s in samples if _normalise(s.get("answer", "")) == _normalise(winner)),
            samples[0],
        )

        return AgentOutput(
            agent_id    = self.agent_id,
            task_id     = task.task_id,
            answer      = winner,
            confidence  = confidence,
            reasoning   = best.get("reasoning", f"Majority vote {n}/{K}"),
            complexity  = "hard",
            tokens_used = total_tokens,
        )

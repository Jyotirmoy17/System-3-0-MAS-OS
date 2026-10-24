"""Budget management and Value-of-Computation calculations."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BudgetManager:
    """
    Tracks cumulative token spend against a global budget Ω.

    Thread-safe — multiple agents may call record() concurrently.
    """

    def __init__(self, omega: int = 50_000) -> None:
        self.omega:  int              = omega
        self._spent: int              = 0
        self._log:   List[Dict[str, Any]] = []
        self._lock:  threading.Lock   = threading.Lock()


    def record(self, agent_id: str, tokens: int, escalated: bool = False) -> None:
        """Record token expenditure for one agent call."""
        with self._lock:
            self._spent += max(0, tokens)
            self._log.append({
                "agent_id":  agent_id,
                "tokens":    tokens,
                "escalated": escalated,
                "ts":        time.time(),
                "cumulative": self._spent,
            })

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    def remaining(self) -> int:
        """Remaining token budget."""
        with self._lock:
            return max(0, self.omega - self._spent)

    def is_exhausted(self) -> bool:
        return self.remaining() == 0

    def utilisation(self) -> float:
        with self._lock:
            return self._spent / max(self.omega, 1)


    def allocate_step_budget(
        self,
        n_steps:    int,
        complexity: str,
        floor:      int = 100,
    ) -> List[int]:
        """
        Divide remaining budget evenly across *n_steps* with a
        complexity-based weight.  Hard tasks get more per step.
        """
        weight = {"easy": 0.5, "medium": 0.75, "hard": 1.0}.get(complexity, 0.75)
        avail  = int(self.remaining() * weight)
        per    = max(floor, avail // max(n_steps, 1))
        cap    = min(per, 4000)
        return [cap] * n_steps


    def summary(self) -> Dict[str, Any]:
        with self._lock:
            log_snapshot = list(self._log)
            spent = self._spent
        escalated = sum(1 for e in log_snapshot if e["escalated"])
        return {
            "omega":            self.omega,
            "spent":            spent,
            "remaining":        max(0, self.omega - spent),
            "utilisation_pct":  round(spent / max(self.omega, 1) * 100, 2),
            "call_count":       len(log_snapshot),
            "escalated_calls":  escalated,
        }

    def reset(self) -> None:
        with self._lock:
            self._spent = 0
            self._log.clear()


class VoCCalculator:
    """
    Computes VoC(s_t) = E[ΔV(s_{t+1}) | s_t] − Cost(E_t).

    Positive VoC → worth another reasoning step.
    Zero or negative → stop (optimal T* from Definition 3.7, Eq. 3.13).

    Parameters:
        alpha         reward for a correct answer (quality weight)
        lambda_cost   cost sensitivity (Lagrange multiplier λ)
        llm_cost_mult relative cost of LLM vs SLM (default 8×)
    """

    def __init__(
        self,
        alpha:       float = 1.0,
        lambda_cost: float = 5e-5,
    ) -> None:
        self.alpha       = alpha
        self.lambda_cost = lambda_cost

    def compute(
        self,
        confidence:         float,
        effective_cost:     float = 9600.0,
    ) -> float:
        """
        VoC(s_t) = (1 - conf) * alpha  −  effective_cost * λ

        effective_cost must be pre-weighted by the caller:
            spec.cost_profile.expected_tokens * spec.cost_profile.cost_per_token

        Calibration: lambda_cost=5e-5 puts the break-even for
        consistency_agent (cost_weight=9600) at confidence ≈ 0.52.
        Agents with conf > 0.52 are not worth a K=3 S2 escalation.
        """
        conf_clamped  = max(0.0, min(1.0, confidence))
        expected_gain = (1.0 - conf_clamped) * self.alpha
        expected_cost = effective_cost * self.lambda_cost
        return round(expected_gain - expected_cost, 8)

    def should_continue(
        self,
        confidence:     float,
        effective_cost: float = 9600.0,
    ) -> bool:
        """Return True when an additional reasoning step has positive VoC."""
        return self.compute(confidence, effective_cost) > 0.0

    def jer(self, delta_c: float, epsilon: float) -> float:
        """
        Joint Efficiency Ratio:  JER = ΔC / ε
        Must exceed alpha / lambda for routing to yield positive utility.
        (Thesis Lemma 3.1)
        """
        if epsilon <= 0.0:
            return float("inf")
        return delta_c / epsilon

"""System-3 — the meta-controller / scheduler."""
from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .budget import BudgetManager, VoCCalculator
from .registry import AgentRegistry, Capability, EscalationReason, Tier

log = logging.getLogger(__name__)

MAX_ESCALATIONS: int = 3
FEATURE_DIM: int = 64  # Dimensionality of the task feature vector


class LinUCBRouter:
    """VoC-UCB routing: linear reward model with exploration bonus minus cost."""
    def __init__(self, agent_ids: List[str], alpha_ucb: float = 1.0) -> None:
        self.alpha_ucb = alpha_ucb
        self.d = FEATURE_DIM
        
        self.A: Dict[str, np.ndarray] = {aid: np.eye(self.d) for aid in agent_ids}
        self.A_inv: Dict[str, np.ndarray] = {aid: np.eye(self.d) for aid in agent_ids}
        self.b: Dict[str, np.ndarray] = {aid: np.zeros((self.d, 1)) for aid in agent_ids}
        
    def add_agent(self, agent_id: str) -> None:
        """Dynamically add new agents from the registry."""
        if agent_id not in self.A:
            self.A[agent_id] = np.eye(self.d)
            self.A_inv[agent_id] = np.eye(self.d)
            self.b[agent_id] = np.zeros((self.d, 1))

    def get_ucb_and_sigma(self, agent_id: str, x_t: np.ndarray, lambda_cost: float, cost_k: float) -> Tuple[float, float]:
        """
        Calculates the UCB score and the Reservation Value (sigma_k).
        UCB = (theta^T * x_t) + alpha * sqrt(x_t^T * A_inv * x_t) - lambda * C_k
        Sigma = (theta^T * x_t) - lambda * C_k
        """
        A_inv = self.A_inv[agent_id]
        theta_k = A_inv @ self.b[agent_id]
        
        expected_quality = float((theta_k.T @ x_t).flat[0])

        width = float(np.sqrt((x_t.T @ A_inv @ x_t).flat[0]))
        
        cost_penalty = lambda_cost * cost_k
        
        ucb_score = expected_quality + (self.alpha_ucb * width) - cost_penalty
        sigma_k = expected_quality - cost_penalty
        
        return ucb_score, sigma_k

    def update(self, agent_id: str, x_t: np.ndarray, reward: float) -> None:
        """Update the matrices after an execution step."""
        self.A[agent_id] += x_t @ x_t.T
        self.b[agent_id] += reward * x_t
        self.A_inv[agent_id] = np.linalg.inv(self.A[agent_id])


class Scheduler:
    """System-3 scheduler: VoC-UCB routing and Pandora's Box stopping."""

    def __init__(
        self,
        registry:    AgentRegistry,
        budget_mgr:  BudgetManager,
        lambda_cost: float = 5e-5,
    ) -> None:
        self.registry   = registry
        self.budget_mgr = budget_mgr
        self.lambda_cost = lambda_cost
        self.voc        = VoCCalculator(lambda_cost=lambda_cost)
        
        self.q_hat = 0.85 

        agent_ids = list(self.registry._specs.keys())
        self.router = LinUCBRouter(agent_ids=agent_ids, alpha_ucb=1.0)


    def _extract_features(self, goal: str) -> np.ndarray:
        """Hash-projection feature vector for VoC-UCB routing."""
        x_t = np.zeros((FEATURE_DIM, 1))
        words = goal.lower().split()
        for i, word in enumerate(words):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % FEATURE_DIM
            x_t[idx, 0] += 1.0
        
        norm = np.linalg.norm(x_t)
        if norm > 0:
            x_t = x_t / norm
        return x_t


    def select_initial(self, goal: str) -> str:
        """
        Selects the optimal agent using VoC-UCB.
        Automatically balances the low cost of S1 agents against the
        higher expected accuracy of S2 agents based on online learning,
        completely eliminating the need for heuristic complexity classification.

        Cold-start guard: when the router has zero reward observations every
        agent receives UCB = 1.0 - lambda*C_k, reducing selection to pure cost
        minimisation.  The cheapest S1 agent (memory_agent) always wins,
        producing no-op lookups and rendering System-0 untestable.  Until at
        least one observation exists, fall back to the cheapest capable S2
        agent.  In the single-run experimental harness this guard is always
        active; in a persistent deployment the router transitions to learned
        behaviour after the first successful interaction.  Documented: DEV-007.
        """
        has_observations = any(b.any() for b in self.router.b.values())
        if not has_observations:
            log.info("[S3] Cold-start (no prior observations): S2 fallback routing")
            return self._best_s2_fallback()

        x_t = self._extract_features(goal)
        
        best_agent = None
        max_ucb = -float('inf')

        for agent_id, spec in self.registry._specs.items():
            self.router.add_agent(agent_id) # Ensure dynamic registry agents are tracked
            
            # C_k calculation from CostProfile
            cost_k = spec.cost_profile.expected_tokens * spec.cost_profile.cost_per_token
            
            ucb, _ = self.router.get_ucb_and_sigma(agent_id, x_t, self.lambda_cost, cost_k)
            
            if ucb > max_ucb:
                max_ucb = ucb
                best_agent = agent_id

        # Fallback to S2 sentinel if registry is somehow empty
        if not best_agent:
            return self._best_s2_fallback()
            
        log.info("[S3] VoC-UCB selected %s (UCB: %.4f)", best_agent, max_ucb)
        return best_agent

    def update_router(self, agent_id: str, goal: str, success: bool, confidence: float) -> None:
        """LinUCB feedback loop (Theorem 3.6, sublinear regret). Called after execution validates output."""
        x_t = self._extract_features(goal)
        reward = 1.0 if success else (confidence - 1.0)
        self.router.update(agent_id, x_t, reward)


    def select_recovery(
        self,
        reason:          EscalationReason,
        current_agent_id: str,
    ) -> str:
        """
        Select a recovery agent based on the triggered escalation predicate.
        P1 (low confidence)  → self_consistency   (K=3 majority vote)
        P2 (schema invalid)  → verifier_retry     (generate-verify-retry)
        P3 / P4 (capability / context) → decomposition (Least-to-Most)
        """
        strategy_map: Dict[EscalationReason, str] = {
            EscalationReason.P1_CONFIDENCE: "self_consistency",
            EscalationReason.P2_SCHEMA:     "verifier_retry",
            EscalationReason.P3_CAPABILITY: "decomposition",
            EscalationReason.P4_CONTEXT:    "decomposition",
        }
        target_strategy = strategy_map.get(reason, "decomposition")

        spec = self.registry.find_by_strategy(target_strategy, Tier.S2)
        if spec is not None:
            log.info(
                "[S3] Escalation %s: %s → %s",
                reason.value, current_agent_id, spec.agent_id,
            )
            return spec.agent_id

        return self._best_s2_fallback()


    def should_halt(self, escalation_count: int, current_confidence: float, goal: str) -> bool:
        """
        VoC optimal stopping (Definition 3.7) with Pandora's Box routing (Theorem 3.5).
        Evaluates whether expanding the search graph provides positive expected net value.
        """
        # Hard limits (budget boundedness)
        if self.budget_mgr.is_exhausted():
            log.warning("[S3] Halt: budget exhausted")
            return True
        if escalation_count >= MAX_ESCALATIONS:
            log.warning("[S3] Halt: max escalations (%d) reached", MAX_ESCALATIONS)
            return True

        # Optimal Stopping Evaluation
        x_t = self._extract_features(goal)
        max_remaining_sigma = -float('inf')

        for agent_id, spec in self.registry._specs.items():
            cost_k = spec.cost_profile.expected_tokens * spec.cost_profile.cost_per_token
            _, sigma_k = self.router.get_ucb_and_sigma(agent_id, x_t, self.lambda_cost, cost_k)
            if sigma_k > max_remaining_sigma:
                max_remaining_sigma = sigma_k

        # If our current quality (confidence) exceeds the highest possible 
        if current_confidence >= max_remaining_sigma:
            log.info("[S3] Halt: Optimal stopping condition met (V_current >= max sigma)")
            return True

        return False


    def _best_s2_fallback(self) -> str:
        """Cold-start S2 sentinel: cheapest registered S2 by CostProfile (DEV-007)."""
        s2 = self.registry.all_in_tier(Tier.S2)
        return s2[0].agent_id if s2 else "react_agent"

"""The Agent Registry: R → AgentID → AgentSpec."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Type

log = logging.getLogger(__name__)


class Tier(str, Enum):
    """Agent capability tier."""
    S1 = "S1"  # Fast SLM agents — cheap, narrow, specialist
    S2 = "S2"  # Deep LLM agents — expensive, broad, deliberate


class Capability(str, Enum):
    """Semantic capability taxonomy.  Ti in the agent 6-tuple."""
    FACTUAL_QA      = "factual_qa"
    MATH_ARITHMETIC = "math_arithmetic"
    MATH_SYMBOLIC   = "math_symbolic"
    CODE_ANALYSIS   = "code_analysis"
    CODE_EXECUTION  = "code_execution"
    REASONING       = "reasoning"
    DECOMPOSITION   = "decomposition"
    RESEARCH        = "research"
    SUMMARISATION   = "summarisation"
    PLANNING        = "planning"
    VERIFICATION    = "verification"
    WEB_SEARCH      = "web_search"
    FILE_READ       = "file_read"
    FILE_WRITE      = "file_write"
    MEMORY_READ     = "memory_read"
    MEMORY_WRITE    = "memory_write"
    SHELL_EXEC      = "shell_exec"
    PYTHON_EXEC     = "python_exec"
    DATA_ANALYSIS   = "data_analysis"
    SUB_AGENT_SPAWN = "sub_agent_spawn"
    REACT_LOOP      = "react_loop"


class EscalationReason(str, Enum):
    """Four formal escalation predicates (Definition E.1) plus sentinel."""
    P1_CONFIDENCE = "P1_confidence_below_threshold"
    P2_SCHEMA     = "P2_schema_invalid_unrepaired"
    P3_CAPABILITY = "P3_agent_declared_incapable"
    P4_CONTEXT    = "P4_context_exceeds_limit"
    BUDGET_HALT   = "budget_exhausted"
    NONE          = "none"


@dataclass
class CostProfile:
    """
Ci — runtime cost profile, updated via EMA from real execution telemetry.
    """

    expected_tokens: int   = 400
    latency_p50_s:  float = 5.0
    latency_p95_s:  float = 15.0
    cost_per_token: float = 1.0
    alpha:          float = field(default=0.3, repr=False)  # EMA rate (thesis §5.4.1)

    @property
    def expected_cost(self) -> float:
        """Expected cost in token-equivalent units."""
        return self.expected_tokens * self.cost_per_token

    def update(self, actual_tokens: int, actual_latency_s: float) -> None:
        """EMA update from a single observed execution."""
        a = self.alpha
        self.expected_tokens = int(a * actual_tokens + (1.0 - a) * self.expected_tokens)
        self.latency_p50_s   = a * actual_latency_s + (1.0 - a) * self.latency_p50_s


@dataclass
class AgentSpec:
    """
AgentSpec = <Mi, Gi, Ki, Vi, Ci, Ti>

    Complete 6-tuple specification of one agent. Registered in AgentRegistry.
    """

    # Identity
    agent_id:    str
    name:        str
    tier:        Tier
    description: str = ""

    # Mi — model
    model_name:  str   = "phi3:mini"
    model_temp:  float = 0.1

    # Gi — schema contracts
    input_schema:          str = "task_input"
    output_schema:         str = "agent_response"
    context_window_tokens: int = 4096

    # Vi — verifier
    verifier: Optional[str] = None

    # Ci — cost profile (mutable, updated from telemetry)
    cost_profile: CostProfile = field(default_factory=CostProfile)

    # Ti — capabilities and tools
    capabilities: FrozenSet[Capability] = field(default_factory=frozenset)
    tools:        List[str]             = field(default_factory=list)

    # Strategy / spawning
    strategy:  str  = "direct"
    can_spawn: bool = False

    def can_handle(self, required: Set[Capability]) -> bool:
        """Return True if this agent covers all required capabilities."""
        return required.issubset(self.capabilities)

    def __str__(self) -> str:
        caps = ", ".join(sorted(c.value for c in self.capabilities))
        return f"[{self.tier.value}] {self.name} | {self.model_name} | {caps}"

    def __hash__(self) -> int:
        return hash(self.agent_id)


class AgentRegistry:
    """
R: AgentID → AgentSpec. The OS process table. Thread-safe. Mutable at runtime.
    """

    def __init__(self) -> None:
        self._specs:   Dict[str, AgentSpec]     = {}
        self._classes: Dict[str, Type[Any]]     = {}
        self._lock:    threading.RLock           = threading.RLock()


    def register(self, spec: AgentSpec, agent_class: Optional[Type[Any]] = None) -> None:
        """Register an agent specification and its implementation class."""
        with self._lock:
            self._specs[spec.agent_id] = spec
            if agent_class is not None:
                self._classes[spec.agent_id] = agent_class
        log.info("Registered: %s", spec)

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        with self._lock:
            self._specs.pop(agent_id, None)
            self._classes.pop(agent_id, None)

    def get_spec(self, agent_id: str) -> Optional[AgentSpec]:
        with self._lock:
            return self._specs.get(agent_id)

    def get_class(self, agent_id: str) -> Optional[Type[Any]]:
        with self._lock:
            return self._classes.get(agent_id)


    def find_capable(
        self,
        required:  Set[Capability],
        tier:      Optional[Tier] = None,
        exclude:   Optional[Set[str]] = None,
    ) -> List[AgentSpec]:
        """Return capable agents sorted cheapest-first. Optionally filter by tier/exclude."""
        exclude_set: Set[str] = exclude or set()
        with self._lock:
            results = [
                spec
                for spec in self._specs.values()
                if spec.can_handle(required)
                and (tier is None or spec.tier == tier)
                and spec.agent_id not in exclude_set
            ]
        results.sort(key=lambda s: s.cost_profile.expected_cost)
        return results

    def cheapest_capable(self, required: Set[Capability], tier: Optional[Tier] = None) -> Optional[AgentSpec]:
        results = self.find_capable(required, tier)
        return results[0] if results else None

    def all_in_tier(self, tier: Tier) -> List[AgentSpec]:
        """Return all agents in the given tier, cheapest-first."""
        with self._lock:
            specs = [s for s in self._specs.values() if s.tier == tier]
        specs.sort(key=lambda s: s.cost_profile.expected_cost)
        return specs

    def find_by_strategy(self, strategy: str, tier: Tier) -> Optional[AgentSpec]:
        """Find the first agent with a given strategy in the given tier."""
        with self._lock:
            for spec in self._specs.values():
                if spec.strategy == strategy and spec.tier == tier:
                    return spec
        return None


    def update_cost(self, agent_id: str, tokens: int, latency_s: float) -> None:
        with self._lock:
            spec = self._specs.get(agent_id)
            if spec is not None:
                spec.cost_profile.update(tokens, latency_s)


    def instantiate_all(
        self,
        ollama_client: Any,
        tool_registry: Any,
    ) -> Dict[str, Any]:
        """Create one live agent instance per spec. Failed inits are logged and skipped."""
        instances: Dict[str, Any] = {}
        with self._lock:
            items = list(self._specs.items())

        for agent_id, spec in items:
            cls = self._classes.get(agent_id)
            if cls is None:
                log.warning("No class for agent %s — skipping", agent_id)
                continue
            try:
                instances[agent_id] = cls(
                    spec=spec,
                    ollama_client=ollama_client,
                    registry=self,
                    tool_registry=tool_registry,
                )
            except Exception:
                log.exception("Failed to instantiate agent %s", agent_id)
        log.info("Instantiated %d / %d agents", len(instances), len(items))
        return instances


    def summary(self) -> Dict[str, Any]:  # noqa: D102
        with self._lock:
            specs = dict(self._specs)
        s1 = [s for s in specs.values() if s.tier == Tier.S1]
        s2 = [s for s in specs.values() if s.tier == Tier.S2]
        all_caps: Set[str] = set()
        for spec in specs.values():
            all_caps.update(c.value for c in spec.capabilities)
        return {
            "total":        len(specs),
            "s1_count":     len(s1),
            "s2_count":     len(s2),
            "agents":       {aid: str(spec) for aid, spec in specs.items()},
            "capabilities": sorted(all_caps),
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    def __contains__(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._specs

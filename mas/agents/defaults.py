"""Default agent registry population."""
from __future__ import annotations

from mas.kernel.registry import (
    AgentRegistry,
    AgentSpec,
    Capability,
    CostProfile,
    Tier,
)

# S1 implementations
from mas.agents.s1.file_agent   import FileAgent
from mas.agents.s1.calc_agent   import CalculatorAgent
from mas.agents.s1.web_agent    import WebAgent
from mas.agents.s1.memory_agent import MemoryAgent
from mas.agents.s1.classifier   import ClassifierAgent

# S2 implementations
from mas.agents.s2.react       import ReActAgent
from mas.agents.s2.decomp      import DecompositionAgent
from mas.agents.s2.consistency import SelfConsistencyAgent
from mas.agents.s2.tot         import TreeOfThoughtsAgent
from mas.agents.s2.verifier    import VerifierRetryAgent


def build_registry(slm: str = "phi3:mini", llm: str = "llama3.1:8b") -> AgentRegistry:
    """
    Build and return a fully-populated AgentRegistry.

    CostProfile values (expected_tokens, latency_p50_s, latency_p95_s, cost_per_token):
        S1 cost_per_token = 1.0   (baseline)
        S2 cost_per_token = 8.0   (≈ 8× more expensive per token)
    These ratios drive the VoC routing decisions in the scheduler.
    """
    R = AgentRegistry()


    R.register(AgentSpec(
        agent_id    = "file_agent",
        name        = "File Agent",
        tier        = Tier.S1,
        description = "Reads, writes, and lists real files inside the workspace",
        model_name  = slm,
        model_temp  = 0.05,
        cost_profile= CostProfile(300, 4.0, 10.0, 1.0),
        capabilities= frozenset({
            Capability.FILE_READ,
            Capability.FILE_WRITE,
            Capability.SUMMARISATION,
        }),
        tools       = ["file_read", "file_write", "file_list", "file_csv"],
        strategy    = "direct",
    ), FileAgent)

    R.register(AgentSpec(
        agent_id    = "calc_agent",
        name        = "Calculator Agent",
        tier        = Tier.S1,
        description = "Executes real Python for exact arithmetic; escalates on symbolic math",
        model_name  = slm,
        model_temp  = 0.05,
        cost_profile= CostProfile(400, 5.0, 12.0, 1.0),
        capabilities= frozenset({
            Capability.MATH_ARITHMETIC,
            Capability.PYTHON_EXEC,
            Capability.DATA_ANALYSIS,
        }),
        tools       = ["python_exec"],
        strategy    = "direct",
    ), CalculatorAgent)

    R.register(AgentSpec(
        agent_id    = "web_agent",
        name        = "Web Research Agent",
        tier        = Tier.S1,
        description = "Performs real DuckDuckGo searches; no API key required",
        model_name  = slm,
        model_temp  = 0.10,
        cost_profile= CostProfile(400, 8.0, 20.0, 1.0),
        capabilities= frozenset({
            Capability.WEB_SEARCH,
            Capability.RESEARCH,
            Capability.FACTUAL_QA,
        }),
        tools       = ["web_search", "web_fetch"],
        strategy    = "direct",
    ), WebAgent)

    R.register(AgentSpec(
        agent_id    = "memory_agent",
        name        = "Memory Agent",
        tier        = Tier.S1,
        description = "Persists and retrieves key-value data across sessions",
        model_name  = slm,
        model_temp  = 0.05,
        cost_profile= CostProfile(50, 0.5, 2.0, 1.0),
        capabilities= frozenset({
            Capability.MEMORY_READ,
            Capability.MEMORY_WRITE,
        }),
        tools       = ["memory_store", "memory_recall", "memory_list", "memory_forget"],
        strategy    = "direct",
    ), MemoryAgent)

    R.register(AgentSpec(
        agent_id    = "classifier_agent",
        name        = "Complexity Classifier",
        tier        = Tier.S1,
        description = "Estimates task complexity for System-3 planning (easy/medium/hard)",
        model_name  = slm,
        model_temp  = 0.05,
        cost_profile= CostProfile(100, 2.0, 5.0, 1.0),
        capabilities= frozenset({Capability.FACTUAL_QA}),
        tools       = [],
        strategy    = "direct",
    ), ClassifierAgent)


    R.register(AgentSpec(
        agent_id    = "react_agent",
        name        = "ReAct Agent",
        tier        = Tier.S2,
        description = "Reason→Act→Observe loop with access to all tools; most capable agent",
        model_name  = llm,
        model_temp  = 0.20,
        context_window_tokens = 8192,
        cost_profile= CostProfile(1500, 45.0, 120.0, 8.0),
        capabilities= frozenset({
            Capability.REACT_LOOP,
            Capability.REASONING,
            Capability.RESEARCH,
            Capability.FILE_READ,
            Capability.WEB_SEARCH,
            Capability.PYTHON_EXEC,
            Capability.PLANNING,
            Capability.DATA_ANALYSIS,
            Capability.SUMMARISATION,
        }),
        tools       = [
            "file_read", "file_write", "file_list", "file_csv",
            "python_exec", "web_search", "web_fetch",
            "memory_store", "memory_recall", "shell_exec",
        ],
        strategy    = "react",
        can_spawn   = True,
    ), ReActAgent)

    R.register(AgentSpec(
        agent_id    = "decomp_agent",
        name        = "Decomposition Agent",
        tier        = Tier.S2,
        description = "Least-to-Most: decomposes tasks and spawns specialist sub-agents",
        model_name  = llm,
        model_temp  = 0.25,
        context_window_tokens = 8192,
        cost_profile= CostProfile(2000, 60.0, 150.0, 8.0),
        capabilities= frozenset({
            Capability.DECOMPOSITION,
            Capability.REASONING,
            Capability.SUB_AGENT_SPAWN,
            Capability.RESEARCH,
            Capability.SUMMARISATION,
            Capability.DATA_ANALYSIS,
        }),
        tools       = [],
        strategy    = "decomposition",
        can_spawn   = True,
    ), DecompositionAgent)

    R.register(AgentSpec(
        agent_id    = "consistency_agent",
        name        = "Self-Consistency Agent",
        tier        = Tier.S2,
        description = "K=3 samples with majority vote; activated by P1 low-confidence escalation",
        model_name  = llm,
        model_temp  = 0.40,
        context_window_tokens = 8192,
        cost_profile= CostProfile(1200, 40.0, 90.0, 8.0),
        capabilities= frozenset({
            Capability.REASONING,
            Capability.VERIFICATION,
            Capability.FACTUAL_QA,
        }),
        tools       = [],
        strategy    = "self_consistency",
    ), SelfConsistencyAgent)

    R.register(AgentSpec(
        agent_id    = "tot_agent",
        name        = "Tree-of-Thoughts Agent",
        tier        = Tier.S2,
        description = "N=3 branches scored and pruned; best for planning and multi-path problems",
        model_name  = llm,
        model_temp  = 0.35,
        context_window_tokens = 8192,
        cost_profile= CostProfile(2500, 80.0, 180.0, 8.0),
        capabilities= frozenset({
            Capability.REASONING,
            Capability.PLANNING,
            Capability.CODE_ANALYSIS,
        }),
        tools       = [],
        strategy    = "tot",
    ), TreeOfThoughtsAgent)

    R.register(AgentSpec(
        agent_id    = "verifier_agent",
        name        = "Verifier-Retry Agent",
        tier        = Tier.S2,
        description = "Generate→verify→retry; activated by P2 schema-violation escalation",
        model_name  = llm,
        model_temp  = 0.10,
        context_window_tokens = 8192,
        cost_profile= CostProfile(1000, 35.0, 80.0, 8.0),
        capabilities= frozenset({
            Capability.VERIFICATION,
            Capability.REASONING,
            Capability.MATH_ARITHMETIC,
            Capability.PYTHON_EXEC,
        }),
        tools       = ["python_exec"],
        strategy    = "verifier_retry",
    ), VerifierRetryAgent)

    return R

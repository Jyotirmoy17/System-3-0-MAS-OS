# MAS-OS — Agentic Operating System
### System-3/0 Blueprint · Jyotirmoy Singh · BITS Pilani Hyderabad

> **Headline result:** CFR: **0.0%** (System-3/0) vs **38.0%** (Baseline), p < 0.001, Cohen's h = −1.328  
> n = 300 runs (150 goals × 2 conditions). See [RESULTS.md](RESULTS.md) for full statistics.

---

## What this is

An OS kernel for autonomous AI agents. You give it a goal in plain English.
It plans which agents to run, routes to the right model tier, executes with
real tools (files, Python, web, memory), recovers from failures automatically,
and returns a complete answer — without you touching anything in between.

**Formal guarantees (from the thesis):**
- Theorem 2 (depth-k composability): System-0 prevents cascading failures at any recursion depth
- Theorem 3 (VoC optimal stopping): Pandora's Box ordering of agents is provably optimal
- Theorem 5 (Reasoning-Resource Paradox): System-3 is mathematically necessary, not a convenience

> **Paper:** arXiv:XXXX.XXXXX

---

## Prerequisites

1. Python 3.11+
2. [Ollama](https://ollama.ai) installed and running (`ollama serve`)
3. Pull the two required models:

```bash
ollama pull phi3:mini
ollama pull llama3.1:8b
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py serve
```

Open **http://localhost:8000**

---

## Architecture

```
Your goal
    │
    ▼
[System-3]  ─── scheduler.py ──── builds plan from registry
    │
    ├── complexity=easy/medium ──► [System-1 pool]
    │                               file_agent   → file_read/write/csv
    │                               calc_agent   → python_exec (exact math)
    │                               web_agent    → web_search (DuckDuckGo)
    │                               memory_agent → memory_store/recall
    │                               classifier   → complexity estimation
    │
    └── complexity=hard / escalated ► [System-2 pool]
                                       react_agent        → ReAct (R→A→O loop)
                                       decomp_agent       → Least-to-Most + sub-agents
                                       consistency_agent  → K=3 majority vote
                                       tot_agent          → N branches scored
                                       verifier_agent     → generate→verify→retry
                                             │
                                             └── can spawn S1 sub-agents
    │
    ▼
[System-0]  ─── validator.py ─── validates EVERY output at EVERY graph edge
    │           Evaluates: P1 conf<τ → consistency
    │                      P2 schema invalid → verifier
    │                      P3 incapable → decomposition
    │                      P4 context overflow → decomposition
    │
    ▼
Result (or escalate back to System-3)
```

See Figure 4.1 in the thesis PDF for the publication-quality version.

---

## Reproduce the thesis results

```bash
bash replicate.sh           # full pipeline (~30 h, 24 GB Ryzen)
bash replicate.sh --quick   # 50-run smoke check (~1 h)
```

See [REPRODUCE.md](REPRODUCE.md) for stage-by-stage documentation, runtimes, and troubleshooting.

---

## Commands

```bash
# Web dashboard (recommended)
python main.py serve

# Single goal from terminal
python main.py run "read sales_2024.csv and compute total annual revenue"
python main.py run "calculate compound interest on $10,000 at 7% for 20 years"
python main.py run "search for the latest papers on AI agent frameworks"
python main.py run "remember my thesis deadline is August 2026"

# Spotlight-style popup (Ubuntu — requires pywebview + pynput)
python main.py popup
# Press Ctrl+Space anywhere to open

# Background daemon (starts on login)
python main.py daemon install    # writes systemd user service
python main.py daemon start
python main.py daemon stop
python main.py daemon status

# Health check
python main.py check

# List all agents and capabilities
python main.py registry

# Custom config
python main.py serve --slm phi3:mini --llm llama3.1:8b --budget 30000 --tau 0.80 --port 9000
```

---

## Development setup

```bash
pip install -e ".[dev]"
pytest                          # 423 tests, ~30 seconds
pytest -m "not integration"     # skip tests that require Ollama
pytest --cov=mas
ruff check mas/
mypy mas/eval/
```

---

## Evaluation Harness (Thesis §5)

```bash
# Single run
python -m mas.eval.cli run --condition FULL_FRAMEWORK --goal-id research_synth_001 --seed 42

# Dry-run: show all 300 runs without executing
python -m mas.eval.cli batch --dry-run

# Full 300-run matrix (~12–18 hours on 24 GB Ryzen)
python -m mas.eval.cli batch --conditions FULL_FRAMEWORK,BASELINE_LANGGRAPH --goal-types all

# Resume an interrupted batch
python -m mas.eval.cli batch --resume

# Generate RESULTS.md with all statistical analyses
python -m mas.eval.cli results
```

---

## Interactive demo

```bash
python -m mas.eval.cli demo cascade    # System-0 catching a cascade
python -m mas.eval.cli demo budget     # token-cost savings from LinUCB routing
python -m mas.eval.cli demo escalate   # System-3 predicate firing
python -m mas.eval.cli demo all        # run all three
```

---

## Hosted API (FastAPI)

```bash
uvicorn mas.web.server:app --port 8000
# or
docker-compose up
```

Endpoints: `GET /api/health`, `POST /api/run`, `GET /api/run/{id}`, `GET /api/run/{id}/stream`, `POST /api/waitlist`

---

## Citation

```bibtex
@misc{singh2026system30,
  title   = {The System-3/0 Blueprint: A Formally Verified
             Orchestration Framework for Multi-Agent Systems},
  author  = {Singh, Jyotirmoy},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX}
}
```


---

## License

MIT — see [LICENSE](LICENSE).

---

## File map

### Evaluation layer (`mas/eval/`)
| File | Responsibility |
|---|---|
| `mas/eval/harness.py` | `run_one()` — single goal × condition → validated RunLog |
| `mas/eval/cli.py` | CLI: `run`, `batch`, `logs`, `results`, `failures`, `robustness`, `demo`, `regrade` |
| `mas/eval/batch.py` | Batch orchestrator: queue, resume, parallel, SIGINT |
| `mas/eval/results_generator.py` | RESULTS.md generation: tables, McNemar, Wilcoxon, CIs |
| `mas/eval/statistics.py` | McNemar's exact, Wilcoxon signed-rank, bootstrap CIs |
| `mas/eval/judge.py` | `AnthropicJudge` — LLM-based semantic rubric evaluator |

### Kernel layer
| File | Responsibility |
|---|---|
| `mas/kernel/scheduler.py` | System-3: LinUCB routing, cold-start S2 fallback, VoC halt |
| `mas/kernel/validator.py` | System-0: JSON Schema validation, escalation predicates |
| `mas/kernel/graph.py` | LangGraph graph construction |
| `mas/kernel/budget.py` | BudgetManager, VoCCalculator |

### Agent layer
| File | Responsibility |
|---|---|
| `mas/agents/s1/` | file, calc, web, memory, classifier agents |
| `mas/agents/s2/` | react, decomp, consistency, tot, verifier agents |

### Adding a new agent

```python
# mas/agents/s1/my_agent.py
from mas.agents.base import AgentOutput, BaseAgent, TaskInput

class MyAgent(BaseAgent):
    def run_internal(self, task: TaskInput) -> AgentOutput:
        result = self.use_tool_raw("web_search", query=task.query)
        d, tokens = self.llm_json(
            f"Question: {task.query}\nResults: {result}",
            system='Summarise. JSON: {"answer":"...","confidence":0.8,"reasoning":"...","complexity":"medium"}',
        )
        return AgentOutput(
            agent_id=self.agent_id, task_id=task.task_id,
            answer=str(d.get("answer", "")), confidence=float(d.get("confidence", 0.8)),
            reasoning=d.get("reasoning", ""), tokens_used=tokens,
        )

# Then register in mas/agents/defaults.py — System-3 routes to it automatically.
```

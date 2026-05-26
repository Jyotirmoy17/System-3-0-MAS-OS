# EXPERIMENTS.md — Experimental Protocol

**Author:** Jyotirmoy Singh  
**Frozen on:** April 29, 2026  
**Status:** Locked.

## 1. Hypothesis

**H1 (primary):** On the 150-instance goal suite (pipeline depth k ≥ 3), **System 0/3**
(`FULL_FRAMEWORK`) exhibits a significantly lower Cascade Failure Rate (CFR) than
**Baseline** (`BASELINE_LANGGRAPH`).

- **CFR (confirmatory):** CFR(System 0/3) < CFR(Baseline), tested by paired McNemar's
  exact test at α = 0.05.
- **JPI (descriptive):** JPI(System 0/3) > JPI(Baseline), reported in the Capability &
  Performance panel with bootstrap 95% CIs; no additional hypothesis test is
  pre-specified for JPI.

**Negative result clause:** If the CFR comparison fails to reach significance at
α = 0.05, results will be published as a negative finding with full data release.

## 2. Conditions

Two conditions, applied to every goal instance with the same seed. Internal keys in
`runs.jsonl` are unchanged; **RESULTS.md** and figures use the presentation labels
below.

| Internal key | Presentation label |
| --- | --- |
| `FULL_FRAMEWORK` | **System 0/3** |
| `BASELINE_LANGGRAPH` | **Baseline** |

**FULL_FRAMEWORK (System 0/3)** — Reference architecture. System-0 enforcing;
System-3 routing enabled; all agents available; cascade audit enabled.

**BASELINE_LANGGRAPH (Baseline)** — Vanilla LangGraph + Pydantic. Same models, same
tools, fixed sequential pipeline of length k = `pipeline_depth_k`. Pydantic post-hoc
validation only (no constrained decoding). Cascade audit applied via wrapper.

## 3. Goal Suite

**150 instances total** — a CFR-stress adversarial-heavy suite (DEV-022):

| Goal type | Instances |
| --- | ---: |
| Adversarial Reliability | 91 |
| Code Analysis | 32 |
| Sequential Reasoning | 27 |

Every instance has `pipeline_depth_k` ≥ 3. Adversarial goals use deterministic
step-2 corruption; patterns cycle `(instance_index − 1) mod 5`:

| Pattern | Count |
| --- | ---: |
| `drop_required_field` | 19 |
| `type_flip` | 18 |
| `out_of_range` | 18 |
| `malformed_json` | 18 |
| `extra_fields_not_in_schema` | 18 |

Each file lives at `goals/<type>/<id>.json` with `id`, `goal_type`, `goal_text`,
`inputs`, `ground_truth`, `rubric`, `pipeline_depth_k`, and `corruption_pattern`
(adversarial only).

Preflight (`mas/eval/preflight.py`) and validation (`mas/eval/goal_loader.py`) enforce
**exactly** this distribution before any batch run.

## 4. Run Log Schema

Each run appends one JSON line to `runs.jsonl`:

```json
{
  "run_id": "uuid-v4",
  "goal_id": "code_001",
  "goal_type": "code_analysis",
  "pipeline_depth_k": 4,
  "condition": "FULL_FRAMEWORK",
  "routing_skipped": false,
  "seed": 42,
  "model_slm": "phi3:mini",
  "model_llm": "llama3.1:8b",
  "started_at": "ISO-8601",
  "completed_at": "ISO-8601",
  "latency_s": 87.3,
  "success": 1,
  "cascade_failure": 0,
  "tokens": { "slm": 1240, "llm": 3580, "total": 4820 },
  "s0": { "catches": 2, "repairs": 1, "would_have_failed": 2 },
  "escalations": { "P1": 0, "P2": 1, "P3": 0, "P4": 0 },
  "per_step": [
    {
      "step_id": 1,
      "agent_id": "WebAgent",
      "tier": 1,
      "raw_output": "...",
      "schema_valid": true,
      "would_have_been_invalid": false,
      "downstream_consumed": true,
      "tokens_used": 420,
      "latency_ms": 12300,
      "adversarial_corruption_applied": null,
      "adversarial_injection_status": null
    }
  ],
  "final_output": "...",
  "ground_truth": "...",
  "rubric_scores": { "check_1": true, "check_2": true },
  "ground_truth_match": true,
  "voc_trace": [0.8, 0.6, 0.4, 0.1, -0.05],
  "crash_reason": null,
  "grading_api_calls": 0,
  "grading_cost_usd": 0.0,
  "backend": "ollama",
  "api_cost_usd": 0.0
}
```

All fields required. Crashed runs still produce a complete entry with `success: 0` and
`crash_reason` populated. `condition` must be `FULL_FRAMEWORK` or `BASELINE_LANGGRAPH`.

## 5. CFR Definition

A run is a cascade failure iff:

1. Some step has `would_have_been_invalid = true`, AND
2. That step has `downstream_consumed = true` (DAG-traced — see DEV-013), AND
3. The run has `ground_truth_match = false`.

Additionally, if `adversarial_injection_status = "scheduled_but_not_reached"`, the run
is **not** counted as a cascade failure.

Implemented in `mas/eval/cascade_detector.py`.

## 6. Sample Sizes

- **150 goal instances** (91 adversarial, 32 code, 27 sequential).
- **Main matrix:** 2 conditions × 150 goals × 1 seed = **300 runs**.
- **Frontier replication:** `FULL_FRAMEWORK` only, **150 runs** with Claude Sonnet
  (one per goal).
- **Total planned: 450 runs** (300 main + 150 frontier).

Batch command (default conditions = both):

```bash
python -m mas.eval.cli batch --resume
```

## 7. Statistical Analysis

### 7.1 Primary tests (pre-specified)

**System 0/3 vs Baseline**, paired by `(goal_id, seed)`:

1. McNemar's exact test on `cascade_failure` (CFR)
2. McNemar's exact test on `success` (execution completion)
3. Wilcoxon signed-rank on `tokens.total`

**Total: 3 primary tests.** Significance threshold **α = 0.05** (no Bonferroni
correction).

### 7.2 Effect sizes

- Cohen's h for binary outcomes
- Cliff's delta for continuous outcomes
- Reported alongside every p-value

### 7.3 Confidence intervals

Bootstrap 95% CIs, n = 10,000 paired resamples.

### 7.4 Robustness checks

- Temperature sweep: 50 random instances at T ∈ {0.0, 0.7}
- Seed sweep: 50 random instances with seed and seed + 1000
- Drop-adversarial sensitivity: recompute primary tests excluding adversarial goals
- Threshold sensitivity: τ ∈ {0.5, 0.7, 0.9}

### 7.5 Capability & Performance (descriptive)

Compares System 0/3 and Baseline on:

- **Accuracy** — mean `ground_truth_match` (bootstrap 95% CI)
- **JPI** — Joint Performance Index with weights (0.5, 0.3, 0.2)
- **Execution Completion Rate (CR)** — mean harness `success`

No significance test is pre-specified for Accuracy or JPI. See DEV-019 / DEV-020 for
grading-backend caveats when semantic judges were used.

## 8. Reproducibility

- All seeds fixed and logged
- Hyperparameters in `mas/config.py`, version-controlled
- `replicate.sh` reproduces the pipeline from a clean clone (when available)
- Compute budget: ~15 hours on 24 GB Ryzen (no GPU) for the 300-run main matrix
- Frontier replication: ~$50 USD via Anthropic API

## 9. Outputs

- `runs.jsonl` — raw data (append-only)
- `runs_completed.txt` — resumability checkpoint
- `RESULTS.md` — auto-generated (`python -m mas.eval.cli results`)
- Figures: `cascade_bar`, `cost_quality_pareto`, `per_step_survival`, `frontier_vs_local`

**Log utilities:**

```bash
python -m mas.eval.cli logs validate
python -m mas.eval.cli logs dedupe
python -m mas.eval.cli logs summary
```

## 10. Version

Protocol version: **3.0**  
Frozen on: 2026-04-29  

Change from 2.2 (DEV-023): Documentation and tooling aligned to the permanent
**2-condition** protocol (System 0/3 vs LangGraph Baseline only). Removed empirical
ablation conditions and multi-proof framing from this document.

## 11. Implementation status (informational)

| Component | Status |
| --- | --- |
| `mas/eval/harness.py` | ✅ |
| `mas/eval/cli.py` | ✅ |
| `mas/eval/batch.py` (300-run matrix) | ✅ |
| `mas/eval/preflight.py` / `goal_loader.py` | ✅ |
| `mas/eval/statistics.py` | ✅ |
| `mas/eval/results_generator.py` | ✅ |
| `mas/eval/baselines/langgraph_baseline.py` | ✅ |
| 150 goal instances on disk | ✅ |
| `replicate.sh` | ⬜ |

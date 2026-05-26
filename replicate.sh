#!/usr/bin/env bash
# replicate.sh — Full reproduction script for the System-3/0 experiments.
#
# From a clean clone, runs the full experimental pipeline and produces all
# artifacts (runs.jsonl, RESULTS.md, 5 figures).
#
# See REPRODUCE.md for step-by-step documentation and troubleshooting.
#
# Usage:
#   bash replicate.sh [OPTIONS]
#
# Cascade stress pilot:  bash replicate.sh --cfr-test
#   (see --help; writes to MAS_RUNS_LOG, default ./runs.jsonl)
#
# Exit codes:  0 success  |  1-6 stage that failed (see --help).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_EXPERIMENTS=false
SKIP_ROBUSTNESS=false
SKIP_FRONTIER=false
QUICK=false
QUICK_ALL_HARD=false
QUICK_ALL_HARD_MIN=false
CFR_TEST=false
DRY_RUN=false

for _arg in "$@"; do
    case "$_arg" in
        --skip-experiments) SKIP_EXPERIMENTS=true ;;
        --skip-robustness)  SKIP_ROBUSTNESS=true  ;;
        --skip-frontier)    SKIP_FRONTIER=true    ;;
        --quick)            QUICK=true            ;;
        --quick-all-hard)   QUICK_ALL_HARD=true   ;;
        --quick-all-hard-min) QUICK_ALL_HARD_MIN=true ;;
        --cfr-test)         CFR_TEST=true         ;;
        --dry-run)          DRY_RUN=true          ;;
        --help|-h)
            cat <<'HELP'
replicate.sh — Reproduce System-3/0 experimental results

USAGE
  bash replicate.sh [OPTIONS]

OPTIONS
  --skip-experiments   Skip Stages 1-5; jump to analysis (Stage 6).
                       Use when runs.jsonl already exists.
  --skip-robustness    Skip Stage 5 (robustness checks).
  --skip-frontier      Skip Stage 4 (frontier replication with Anthropic API).
  --quick              Smoke run: 50 instances (max-runs cap), FULL_FRAMEWORK only.
                       Full pipeline takes ~30 h; --quick takes ~1 h.
  --quick-all-hard     Smoke run: 2 deepest instances per goal type (3 types)
                       across both conditions (6×2 = 12 runs total).
                       Use this before a full run to catch runtime bugs.
  --quick-all-hard-min Minimal smoke: 1 deepest goal across both conditions
                       (2 runs). Fastest wiring check.
  --cfr-test           Cascade-stress pilot: 2 deepest instances per goal type
                       (adversarial, code, sequential = 6 goals) × 2 conditions
                       (12 runs). System 0/3 vs LangGraph baseline only.
  --dry-run            List what would run without executing anything.
                       Implies --skip-frontier. No API calls or Ollama inference.
  --help, -h           Print this message and exit.

STAGES
  1  Environment check  (Python >= 3.11, Ollama, models, packages)
  2  Pre-flight         (goal-suite integrity: schema, distribution, rubric)
  3  Main batch         (300 runs: 2 conditions x 150 instances)
  4  Frontier           (150 runs: FULL_FRAMEWORK via Anthropic; optional)
  5  Robustness         (temperature/seed/threshold/drop-adversarial sweeps)
  6  Analysis           (validate logs, dedupe, RESULTS.md, figures)

EXIT CODES
  0  Full success
  1  Stage 1 failed (environment not ready)
  2  Stage 2 failed (pre-flight errors — fix goal files before running)
  3  Stage 3 failed (batch aborted — re-run with --resume to continue)
  4  Stage 4 failed (frontier replication aborted)
  5  Stage 5 failed (robustness checks aborted)
  6  Stage 6 failed (analysis failed)

EXAMPLES
  bash replicate.sh                        # Full ~30-hour run
  bash replicate.sh --quick               # Fast smoke run (~1 h, 50 instances)
  bash replicate.sh --quick-all-hard      # 12-run smoke (6 goals × 2 conditions)
  bash replicate.sh --quick-all-hard-min  # Minimal 2-run smoke
  bash replicate.sh --cfr-test            # 12-run cascade-stress pilot
  bash replicate.sh --cfr-test --dry-run  # Preview CFR-test queue only
  bash replicate.sh --quick --dry-run     # Preview: list 50 runs, no execution
  bash replicate.sh --skip-experiments    # Regenerate RESULTS.md from existing data
  bash replicate.sh --skip-frontier       # Full local run, skip Anthropic API

ENVIRONMENT VARIABLES
  MAS_RUNS_LOG      Override path to runs.jsonl (default: ./runs.jsonl)
  ANTHROPIC_API_KEY Required for Stage 4; skipped silently if absent.

HELP
            exit 0
            ;;
        *)
            printf 'ERROR: Unknown flag: %s\n' "$_arg" >&2
            printf '       Run "bash replicate.sh --help" for usage.\n' >&2
            exit 1
            ;;
    esac
done

# --dry-run implies --skip-frontier (avoid accidental API spend)
if [ "$DRY_RUN" = true ]; then
    SKIP_FRONTIER=true
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUNS_LOG="${MAS_RUNS_LOG:-./runs.jsonl}"
# Python batch (`mas.eval.cli batch`) reads MAS_RUNS_LOG from the environment only.
# Export the resolved path so subprocesses always append to the same file as this
# script's Stage 6, even when the user assigns MAS_RUNS_LOG without `export`.
export MAS_RUNS_LOG="$RUNS_LOG"
START_TIME=$(date +%s)

_ok()     { printf '[OK]   %s\n'   "$*"; }
_warn()   { printf '[WARN] %s\n'   "$*" >&2; }
_fail()   { printf '[FAIL] %s\n'   "$*" >&2; }
_info()   { printf '[....] %s\n'   "$*"; }
_banner() {
    echo ""
    echo "================================================================"
    printf  '  %s\n' "$*"
    echo "================================================================"
}

# ---------------------------------------------------------------------------
# Stage 1: Environment check
# ---------------------------------------------------------------------------
stage1_env_check() {
    _banner "Stage 1: Environment check"
    local fatal=false

    # Python >= 3.11
    if python3 --version >/dev/null 2>&1; then
        local py_ver
        py_ver=$(python3 -c \
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" \
            2>/dev/null || echo "0.0")
        local py_maj py_min
        py_maj=$(echo "$py_ver" | cut -d. -f1)
        py_min=$(echo "$py_ver" | cut -d. -f2)
        if [ "$py_maj" -ge 3 ] && [ "$py_min" -ge 11 ]; then
            _ok "Python $py_ver"
        else
            _fail "Python $py_ver found — 3.11+ required."
            fatal=true
        fi
    else
        _fail "python3 not found."
        fatal=true
    fi

    # mas package importable (install if not)
    if python3 -c "import mas.eval.cli" 2>/dev/null; then
        _ok "mas package importable"
    else
        _info "mas not importable — running: pip install -e .[dev]"
        if pip install -e ".[dev]" -q 2>/dev/null; then
            _ok "pip install -e .[dev] succeeded"
        else
            _fail "pip install -e .[dev] failed — install dependencies manually."
            fatal=true
        fi
    fi

    # Ollama binary
    if command -v ollama >/dev/null 2>&1; then
        _ok "ollama binary found"
    else
        _fail "ollama not found — install from https://ollama.ai"
        fatal=true
    fi

    # Ollama server reachable
    if curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
        _ok "Ollama server reachable (localhost:11434)"

        # Required models
        for _model in "phi3:mini" "llama3.1:8b"; do
            local _basename
            _basename="${_model%%:*}"
            if ollama list 2>/dev/null | grep -q "^${_basename}"; then
                _ok "Model $_model present"
            else
                _info "Model $_model not found — pulling ..."
                if ollama pull "$_model" 2>&1; then
                    _ok "Model $_model pulled"
                else
                    _fail "ollama pull $_model failed"
                    fatal=true
                fi
            fi
        done
    else
        _fail "Ollama not responding at localhost:11434 — start with: ollama serve"
        fatal=true
    fi

    if [ "$fatal" = true ]; then
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Stage 2: Pre-flight
# ---------------------------------------------------------------------------
stage2_preflight() {
    _banner "Stage 2: Pre-flight validation"
    python3 -m mas.eval.preflight
}

# ---------------------------------------------------------------------------
# Stage 3: Main experimental batch
# ---------------------------------------------------------------------------
stage3_batch() {
    _banner "Stage 3: Main experimental batch"

    local -a args
    if [ "$QUICK_ALL_HARD_MIN" = true ]; then
        _info "Quick-all-hard-min mode: 1 deepest goal across both conditions (2 runs)."
        local ids
        ids="$(python3 -c "
from mas.eval.goal_loader import load_all_goals

goals = load_all_goals()
if not goals:
  raise SystemExit('No goals found. Check goals/ suite.')
goals.sort(key=lambda g: (-int(g.pipeline_depth_k), str(g.id)))
print(goals[0].id)
" 2>/dev/null || true)"

        if [ -z "$ids" ]; then
            _fail "Failed to select a minimal goal. Ensure goals are present and importable."
            return 1
        fi

        _info "Selected goal: $ids"
        args=(
            --conditions FULL_FRAMEWORK,BASELINE_LANGGRAPH
            --goal-types all
            --instances  "$ids"
            --resume
        )
    elif [ "$CFR_TEST" = true ]; then
        _info "CFR-test mode: 2 deepest per goal type (3 types), both conditions (12 runs)."
        local ids
        ids="$(python3 -c "
from mas.eval.goal_loader import load_all_goals

goals = load_all_goals()
want_types = [
  'adversarial_reliability',
  'code_analysis',
  'sequential_reasoning',
]

selected = []
for gt in want_types:
  type_goals = [g for g in goals if g.goal_type == gt]
  type_goals.sort(key=lambda g: (-int(g.pipeline_depth_k), str(g.id)))
  if len(type_goals) < 2:
    raise SystemExit(
      f\"Need >=2 goals for type {gt!r}, found {len(type_goals)}. Check goals/ suite.\"
    )
  selected.extend(g.id for g in type_goals[:2])

if len(selected) != 6:
  raise SystemExit(f'Expected 6 selected goals (2×3 types), got {len(selected)}.')

print(','.join(selected))
" 2>/dev/null || true)"

        if [ -z "$ids" ]; then
            _fail "Failed to select CFR-test goals (need 2 per type for adversarial, code, sequential)."
            return 1
        fi

        _info "Selected goals: $ids"
        args=(
            --conditions FULL_FRAMEWORK,BASELINE_LANGGRAPH
            --goal-types all
            --instances  "$ids"
            --resume
        )
    elif [ "$QUICK_ALL_HARD" = true ]; then
        _info "Quick-all-hard mode: 2 deepest per goal type, both conditions (12 runs)."
        local ids
        ids="$(python3 -c "
from mas.eval.goal_loader import load_all_goals

goals = load_all_goals()
want_types = [
  'adversarial_reliability',
  'code_analysis',
  'sequential_reasoning',
]

selected = []
for gt in want_types:
  type_goals = [g for g in goals if g.goal_type == gt]
  type_goals.sort(key=lambda g: (-int(g.pipeline_depth_k), str(g.id)))
  if len(type_goals) < 2:
    raise SystemExit(
      f\"Need >=2 goals for type {gt!r}, found {len(type_goals)}. Check goals/ suite.\"
    )
  selected.extend(g.id for g in type_goals[:2])

if len(selected) != 6:
  raise SystemExit(f'Expected 6 selected goals (2×3 types), got {len(selected)}.')

print(','.join(selected))
" 2>/dev/null || true)"

        if [ -z "$ids" ]; then
            _fail "Failed to select smoke goals. Ensure goals are present and importable."
            return 1
        fi

        _info "Selected goals: $ids"
        args=(
            --conditions FULL_FRAMEWORK,BASELINE_LANGGRAPH
            --goal-types all
            --instances  "$ids"
            --resume
        )
    elif [ "$QUICK" = true ]; then
        _info "Quick mode: FULL_FRAMEWORK only, max 50 runs (--max-runs cap)."
        args=(
            --conditions FULL_FRAMEWORK
            --goal-types all
            --instances  all
            --max-runs   50
            --resume
        )
    else
        args=(
            --conditions FULL_FRAMEWORK,BASELINE_LANGGRAPH
            --goal-types all
            --instances  all
            --resume
        )
    fi

    if [ "$DRY_RUN" = true ]; then
        _info "Dry-run mode: listing queued runs (not executing)."
        args+=(--dry-run --skip-preflight)
    fi

    python3 -m mas.eval.cli batch "${args[@]}"
}

# ---------------------------------------------------------------------------
# Stage 4: Frontier replication (optional — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
stage4_frontier() {
    _banner "Stage 4: Frontier replication (Anthropic backend)"

    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        _warn "ANTHROPIC_API_KEY not set — skipping frontier replication."
        _warn "To run: export ANTHROPIC_API_KEY=sk-... then re-run without --skip-frontier."
        return 0
    fi

    local -a args=(
        --conditions    FULL_FRAMEWORK
        --backend       anthropic
        --judge-backend anthropic
        --instances     all
        --resume
    )

    if [ "$QUICK" = true ]; then
        args+=(--max-runs 10)
    fi
    if [ "$QUICK_ALL_HARD" = true ]; then
        args+=(--max-runs 10)
    fi
    if [ "$QUICK_ALL_HARD_MIN" = true ]; then
        args+=(--max-runs 10)
    fi
    if [ "$CFR_TEST" = true ]; then
        args+=(--max-runs 12)
    fi

    python3 -m mas.eval.cli batch "${args[@]}"
}

# ---------------------------------------------------------------------------
# Stage 5: Robustness checks
# ---------------------------------------------------------------------------
stage5_robustness() {
    _banner "Stage 5: Robustness checks"
    python3 -m mas.eval.cli robustness all
}

# ---------------------------------------------------------------------------
# Stage 6: Analysis
# ---------------------------------------------------------------------------
stage6_analysis() {
    _banner "Stage 6: Analysis"

    # Validate and deduplicate the run log
    if [ -f "$RUNS_LOG" ]; then
        _info "Validating $RUNS_LOG ..."
        python3 -m mas.eval.cli logs validate --path "$RUNS_LOG" || true

        _info "Deduplicating (strategy: successful) ..."
        python3 -m mas.eval.cli logs dedupe   --path "$RUNS_LOG" --strategy successful || true
    else
        _warn "$RUNS_LOG not found."
    fi

    # Count available run lines
    local run_count=0
    if [ -f "$RUNS_LOG" ]; then
        run_count=$(wc -l < "$RUNS_LOG" 2>/dev/null | tr -d ' ' || echo 0)
    fi

    if [ "$run_count" -eq 0 ]; then
        _warn "No run data found in $RUNS_LOG — skipping results generation."
        _warn "Collect runs first, or point MAS_RUNS_LOG at a populated runs.jsonl."
        return 0
    fi

    _info "Generating RESULTS.md from $run_count run(s) ..."
    python3 -m mas.eval.cli results  --runs-path "$RUNS_LOG"
    python3 -m mas.eval.cli failures --runs-path "$RUNS_LOG"
    _ok "RESULTS.md updated."
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
_print_summary() {
    local stage_exit="$1"
    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - START_TIME ))
    local h=$(( elapsed / 3600 ))
    local m=$(( (elapsed % 3600) / 60 ))
    local s=$(( elapsed % 60 ))

    _banner "Summary"
    printf '  Elapsed:       %02d:%02d:%02d\n' "$h" "$m" "$s"

    if [ -f "$RUNS_LOG" ]; then
        local n
        n=$(wc -l < "$RUNS_LOG" 2>/dev/null | tr -d ' ' || echo 0)
        printf '  Runs logged:   %s\n' "$n"
    fi

    if [ "$stage_exit" -eq 0 ]; then
        printf '  Status:        SUCCESS\n'
        printf '  Artifacts:     runs.jsonl  RESULTS.md\n'
    else
        printf '  Status:        FAILED (stage %s)\n' "$stage_exit"
        printf '  Hint:          Re-run with --resume to continue from this point.\n'
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo "replicate.sh — System-3/0 reproduction pipeline"
    echo "  skip-experiments=${SKIP_EXPERIMENTS}  skip-robustness=${SKIP_ROBUSTNESS}"
    echo "  skip-frontier=${SKIP_FRONTIER}  quick=${QUICK}  cfr-test=${CFR_TEST}  dry-run=${DRY_RUN}"
    echo "  runs-log=${RUNS_LOG}"

    # ----------------------------------------------------------------
    # Stage 1: environment check
    # Fatal only when we will actually run experiments (not dry-run,
    # not skip-experiments). Otherwise informational.
    # ----------------------------------------------------------------
    if [ "$SKIP_EXPERIMENTS" = false ] && [ "$DRY_RUN" = false ]; then
        if ! stage1_env_check; then
            _print_summary 1
            exit 1
        fi
    else
        # Non-fatal: warn but continue
        stage1_env_check \
            || _warn "Some environment checks failed — continuing (experiments skipped or dry-run)."
    fi

    # ----------------------------------------------------------------
    # Stages 2–5: experimental pipeline
    # ----------------------------------------------------------------
    if [ "$SKIP_EXPERIMENTS" = false ]; then

        # Stage 2: pre-flight (skipped in dry-run; batch handles it)
        if [ "$DRY_RUN" = false ]; then
            if ! stage2_preflight; then
                _print_summary 2
                exit 2
            fi
        else
            _info "Stage 2: skipping pre-flight (--dry-run)."
        fi

        # Stage 3: main batch
        if ! stage3_batch; then
            _print_summary 3
            exit 3
        fi

        # Stage 4: frontier replication (optional)
        if [ "$SKIP_FRONTIER" = false ]; then
            if ! stage4_frontier; then
                _print_summary 4
                exit 4
            fi
        else
            _info "Stage 4: skipping frontier replication (--skip-frontier or --dry-run)."
        fi

        # Stage 5: robustness checks (skipped in dry-run)
        if [ "$SKIP_ROBUSTNESS" = false ] && [ "$DRY_RUN" = false ]; then
            if ! stage5_robustness; then
                _print_summary 5
                exit 5
            fi
        else
            _info "Stage 5: skipping robustness checks."
        fi

    else
        _info "Stages 2-5: skipped (--skip-experiments)."
    fi

    # ----------------------------------------------------------------
    # Stage 6: analysis (always runs)
    # ----------------------------------------------------------------
    if ! stage6_analysis; then
        _print_summary 6
        exit 6
    fi

    _print_summary 0
    exit 0
}

main

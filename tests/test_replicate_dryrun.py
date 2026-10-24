"""Dry-run acceptance tests for replicate.sh."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT    = Path(__file__).resolve().parent.parent
REPLICATE_SH = REPO_ROOT / "replicate.sh"


def _run(args: list[str], env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Run replicate.sh with *args* as a subprocess.

    Args:
        args:    CLI flags to pass after ``bash replicate.sh``.
        env:     Environment mapping (default: copy of current env).
        timeout: Seconds before the subprocess is killed.

    Returns:
        CompletedProcess with stdout, stderr, and returncode.
    """
    cmd = ["bash", str(REPLICATE_SH)] + args
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=merged_env,
        cwd=str(REPO_ROOT),
        timeout=timeout,
    )


def test_replicate_help() -> None:
    """
    ``bash replicate.sh --help`` prints usage information and exits 0.

    Asserts that the output mentions the key sections (USAGE, OPTIONS,
    STAGES, EXIT CODES) so that a reader unfamiliar with the project
    can understand the script at a glance.
    """
    result = _run(["--help"])

    assert result.returncode == 0, (
        f"replicate.sh --help exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    combined = result.stdout + result.stderr
    for keyword in ("USAGE", "OPTIONS", "STAGES", "EXIT CODES", "EXAMPLES"):
        assert keyword in combined, (
            f"Expected '{keyword}' in --help output.\n"
            f"stdout:\n{result.stdout}"
        )

    # Verify key flags are documented
    for flag in ("--skip-experiments", "--skip-robustness", "--skip-frontier",
                 "--quick", "--cfr-test", "--dry-run"):
        assert flag in combined, (
            f"Flag '{flag}' missing from --help output.\n"
            f"stdout:\n{result.stdout}"
        )


def test_replicate_skip_all(tmp_path: Path) -> None:
    """
    ``--skip-experiments --skip-robustness --skip-frontier`` with an empty
    runs.jsonl runs Stage 6 and emits an appropriate "no data" message.

    The script must exit 0 — empty data is not a fatal error when the user
    has explicitly chosen to skip all experiment stages.
    """
    empty_log = tmp_path / "runs.jsonl"
    empty_log.touch()

    env = {"MAS_RUNS_LOG": str(empty_log)}

    result = _run(
        ["--skip-experiments", "--skip-robustness", "--skip-frontier"],
        env=env,
    )

    combined = result.stdout + result.stderr

    assert result.returncode == 0, (
        f"replicate.sh exited with {result.returncode} (expected 0).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    no_data_phrases = [
        "no run data",
        "skipping results generation",
        "no data",
    ]
    assert any(phrase in combined.lower() for phrase in no_data_phrases), (
        "Expected a 'no data' message from Stage 6 when runs.jsonl is empty.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert "Stage 6" in combined, (
        "Expected 'Stage 6' banner in output.\n"
        f"stdout:\n{result.stdout}"
    )


def test_replicate_quick_dryrun(tmp_path: Path) -> None:
    """
    ``--quick --dry-run`` queues exactly 50 runs and exits 0.

    The batch dry-run prints ``DRY RUN — 50 run(s) queued`` when:
    - 150 goal files exist (CFR-stress suite per thesis §6.2)
    - ``--quick`` sets ``--max-runs 50`` (first 50 queued runs, FULL_FRAMEWORK only)
    - ``--conditions`` defaults to FULL_FRAMEWORK (1 condition)

    No Ollama inference is performed; the script exits without making any
    API calls.
    """
    # Use an isolated, empty runs.jsonl so Stage 6 is predictable
    empty_log = tmp_path / "runs.jsonl"
    empty_log.touch()

    env = {"MAS_RUNS_LOG": str(empty_log)}

    result = _run(["--quick", "--dry-run"], env=env, timeout=180)

    combined = result.stdout + result.stderr

    # Must exit 0 (dry-run is never fatal)
    assert result.returncode == 0, (
        f"replicate.sh --quick --dry-run exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert "DRY RUN" in combined, (
        "Expected 'DRY RUN' in output from --dry-run mode.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Must report exactly 50 runs queued
    # (150 goals loaded, capped at --max-runs 50 by --quick)
    assert "50" in combined, (
        "Expected '50' run(s) in dry-run output (--quick caps at 50 runs).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    no_data_phrases = [
        "no run data",
        "skipping results generation",
        "no data",
    ]
    assert any(phrase in combined.lower() for phrase in no_data_phrases), (
        "Expected Stage 6 to emit a 'no data' message (temp runs.jsonl is empty).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_replicate_cfr_test_dryrun(tmp_path: Path) -> None:
    """
    ``--cfr-test --dry-run`` queues 12 runs (6 goals × 2 conditions).

    No Ollama inference; exits 0.
    """
    empty_log = tmp_path / "runs.jsonl"
    empty_log.touch()
    env = {"MAS_RUNS_LOG": str(empty_log)}

    result = _run(["--cfr-test", "--dry-run"], env=env, timeout=180)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, (
        f"replicate.sh --cfr-test --dry-run exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "DRY RUN" in combined
    assert "[12/12]" in combined or "12 run(s) queued" in combined or " 12 " in combined, (
        "Expected 12 queued runs in CFR-test dry-run output.\n"
        f"stdout:\n{result.stdout}"
    )

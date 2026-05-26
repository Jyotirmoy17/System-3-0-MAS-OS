"""Pydantic v2 request / response models for the System-3/0 web API."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


ALLOWED_CONDITIONS: frozenset[str] = frozenset({
    "FULL_FRAMEWORK",
    "BASELINE_LANGGRAPH",
})


class RunRequest(BaseModel):
    """Request body for POST /api/run."""
    goal: str = Field(
        ...,
        description="Goal ID from the curated demo set (e.g. 'adversarial_002').",
        min_length=1,
        max_length=128,
    )
    condition: str = Field(
        default="FULL_FRAMEWORK",
        description=(
            "Experimental condition.  "
            "One of: FULL_FRAMEWORK, BASELINE_LANGGRAPH."
        ),
    )

    @field_validator("condition")
    @classmethod
    def _validate_condition(cls, v: str) -> str:
        if v not in ALLOWED_CONDITIONS:
            raise ValueError(
                f"condition must be one of {sorted(ALLOWED_CONDITIONS)}, got {v!r}."
            )
        return v


class RunQueued(BaseModel):
    """Response for POST /api/run — run accepted and queued."""
    run_id: str = Field(..., description="UUID-v4 for polling / SSE.")
    status: str = Field(default="queued", description="Always 'queued' on creation.")


class RunStatusResponse(BaseModel):
    """Partial or complete RunLog returned by GET /api/run/{run_id}."""
    run_id:    str
    goal_id:   Optional[str] = None
    condition: Optional[str] = None
    status:    str           = Field(
        description="'queued' | 'running' | 'done' | 'error'",
    )
    run_log:   Optional[dict[str, Any]] = Field(
        default=None,
        description="Full RunLog (EXPERIMENTS.md §4) once the run completes.",
    )


class WaitlistRequest(BaseModel):
    """Request body for POST /api/waitlist."""
    email: EmailStr = Field(..., description="Subscriber e-mail address.")
    use_case: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Optional free-text description of the use case.",
    )


class WaitlistResponse(BaseModel):
    """Response for POST /api/waitlist."""
    ok: bool = True


class HealthResponse(BaseModel):
    """Response for GET /api/health."""
    ok:      bool = True
    ollama:  bool = Field(description="True if Ollama is reachable.")
    version: str  = Field(description="API version string.")

"""Pydantic v2 models mirroring the JSON Schemas registered in mas/kernel/schemas.py."""
from __future__ import annotations

from typing import Literal, Optional

import pydantic
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """
    Post-hoc mirror of the 'agent_response' JSON Schema.

    Divergence from System-0 schema: Pydantic v2 coerces numeric answer values
    to strings automatically; System-0 relies on the LLM producing the correct
    type under constrained decoding.
    """
    answer:     str   = Field(..., max_length=6000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning:  str   = Field(..., max_length=4000)
    complexity: Optional[Literal["easy", "medium", "hard"]] = None

    model_config = pydantic.ConfigDict(extra="allow")


class VerificationResponse(BaseModel):
    """Post-hoc mirror of the 'verification' JSON Schema."""
    is_correct:        bool
    confidence:        float = Field(..., ge=0.0, le=1.0)
    error_description: str   = Field(default="", max_length=600)
    corrected_answer:  str   = Field(default="", max_length=3000)

    model_config = pydantic.ConfigDict(extra="forbid")


class DecompositionResponse(BaseModel):
    """Post-hoc mirror of the 'decomposition' JSON Schema."""
    sub_questions: list[str] = Field(..., min_length=1, max_length=6)

    model_config = pydantic.ConfigDict(extra="allow")


class BranchScore(BaseModel):
    """Post-hoc mirror of the 'branch_score' JSON Schema."""
    score:         float = Field(..., ge=0.0, le=10.0)
    justification: str   = Field(default="", max_length=400)

    model_config = pydantic.ConfigDict(extra="allow")


def parse_agent_response(
    raw: dict,
) -> tuple[AgentResponse | None, pydantic.ValidationError | None]:
    """
    Attempt to parse *raw* as an AgentResponse.

    Returns (model, None) on success or (None, ValidationError) on failure.
    The caller is responsible for deciding what to do on failure; the baseline
    propagates the raw dict without retry or escalation.
    """
    try:
        return AgentResponse.model_validate(raw), None
    except pydantic.ValidationError as exc:
        return None, exc

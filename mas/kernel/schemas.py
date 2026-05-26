"""All JSON Schema definitions used by System-0 for output contract validation."""
from __future__ import annotations

from typing import Any, Dict


SCHEMAS: Dict[str, Dict[str, Any]] = {

    "agent_response": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["answer", "confidence", "reasoning"],
        "additionalProperties": True,
        "properties": {
            "answer": {
                "type": "string",
                "maxLength": 6000,
                "description": "The agent's substantive answer to the task query.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Self-assessed confidence in [0, 1].",
            },
            "reasoning": {
                "type": "string",
                "maxLength": 4000,
                "description": "Chain-of-thought or explanation of how the answer was reached.",
            },
            "complexity": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "Agent's estimate of this task's complexity tier.",
            },
        },
    },

    "complexity_class": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["complexity", "confidence", "reasoning"],
        "properties": {
            "complexity": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning":  {"type": "string", "maxLength": 500},
            "answer":     {"type": "string"},
        },
    },

    "decomposition": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["sub_questions"],
        "properties": {
            "sub_questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string", "minLength": 5},
            },
        },
    },

    "verification": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["is_correct", "confidence", "corrected_answer", "error_description"],
        "additionalProperties": False,
        "properties": {
            "is_correct":        {"type": "boolean"},
            "confidence":        {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "error_description": {"type": "string", "maxLength": 600},
            "corrected_answer":  {"type": "string", "maxLength": 3000},
        },
    },

    "branch_score": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["score", "justification"],
        "properties": {
            "score":         {"type": "number", "minimum": 0.0, "maximum": 10.0},
            "justification": {"type": "string", "maxLength": 400},
        },
    },
}


def repair(data: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
    """
    Attempt to fix common single-field violations in *data*.

    Returns a new dict; does not mutate the input.  The caller must
    re-validate the returned dict to confirm repair succeeded.
    """
    r = dict(data)

    if schema_name in ("agent_response", "complexity_class"):
        # Coerce answer to string
        if not isinstance(r.get("answer"), str):
            r["answer"] = str(r.get("answer", ""))
        # Clamp confidence to [0, 1]
        r["confidence"] = _clamp(r.get("confidence", 0.5))
        # Provide default reasoning
        if not isinstance(r.get("reasoning"), str) or not r["reasoning"].strip():
            r["reasoning"] = "No reasoning provided."
        # Normalise complexity
        if r.get("complexity") not in ("easy", "medium", "hard"):
            r["complexity"] = "medium"

    elif schema_name == "verification":
        r.setdefault("is_correct", True)
        r["confidence"] = _clamp(r.get("confidence", 0.5))
        r.setdefault("error_description", "")
        if not isinstance(r.get("corrected_answer"), str):
            r["corrected_answer"] = str(r.get("corrected_answer", ""))

    elif schema_name == "branch_score":
        try:
            r["score"] = max(0.0, min(10.0, float(r.get("score", 5.0))))
        except (TypeError, ValueError):
            r["score"] = 5.0
        r.setdefault("justification", "")

    elif schema_name == "decomposition":
        if not isinstance(r.get("sub_questions"), list):
            r["sub_questions"] = [str(r.get("answer", "What is the answer?"))]

    return r


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5

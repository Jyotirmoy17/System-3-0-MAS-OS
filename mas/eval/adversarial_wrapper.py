"""Deterministic corruption patterns for Adversarial Reliability goal instances."""
from __future__ import annotations

import json
import random
from typing import Any, Union

# The five corruption patterns defined in EXPERIMENTS.md §3.
_PATTERNS: frozenset[str] = frozenset({
    "drop_required_field",
    "type_flip",
    "out_of_range",
    "malformed_json",
    "extra_fields_not_in_schema",
})


def corrupt(
    output:  dict[str, Any],
    pattern: str,
    schema:  dict[str, Any],
    seed:    int = 0,
) -> Union[dict[str, Any], str]:
    """Apply *pattern* to *output*, producing a version that fails schema validation."""
    if pattern not in _PATTERNS:
        raise ValueError(
            f"Unknown corruption pattern {pattern!r}. "
            f"Valid patterns: {sorted(_PATTERNS)}"
        )

    rng = random.Random(seed)
    out = dict(output)   # shallow copy — do not mutate input

    if pattern == "drop_required_field":
        return _drop_required_field(out, schema, rng)
    if pattern == "type_flip":
        return _type_flip(out, schema, rng)
    if pattern == "out_of_range":
        return _out_of_range(out, schema, rng)
    if pattern == "malformed_json":
        return _malformed_json(out, rng)
    # pattern == "extra_fields_not_in_schema"
    return _extra_fields(out, rng)


def _drop_required_field(
    out:    dict[str, Any],
    schema: dict[str, Any],
    rng:    random.Random,
) -> dict[str, Any]:
    """
    Remove one required field at random.

    Violates: 'required' keyword in the schema.
    """
    required: list[str] = schema.get("required", [])

    # Prefer to remove a field that is actually present in the output
    # (so the violation is detectable, not already-absent).
    candidates = [f for f in required if f in out]
    if not candidates:
        candidates = sorted(required) or sorted(out.keys())

    field = rng.choice(sorted(candidates))
    out.pop(field, None)
    return out


def _type_flip(
    out:    dict[str, Any],
    schema: dict[str, Any],
    rng:    random.Random,
) -> dict[str, Any]:
    """Flip one schema-typed field to the wrong Python type (adversarial corruption)."""
    properties: dict[str, Any] = schema.get("properties", {})

    numeric  = sorted(k for k, v in properties.items()
                      if v.get("type") == "number"  and k in out)
    booleans = sorted(k for k, v in properties.items()
                      if v.get("type") == "boolean" and k in out)
    arrays   = sorted(k for k, v in properties.items()
                      if v.get("type") == "array"   and k in out)

    if numeric:
        field      = rng.choice(numeric)
        out[field] = str(out[field])           # number → string
    elif booleans:
        field      = rng.choice(booleans)
        out[field] = 1 if out[field] else 0    # bool → int
    elif arrays:
        field      = rng.choice(arrays)
        out[field] = {"_converted_from_array": True}  # list → dict
    else:
        # Fallback: stringify the first numeric runtime value found.
        for k, v in out.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = str(v)
                break
    return out


def _out_of_range(
    out:    dict[str, Any],
    schema: dict[str, Any],
    rng:    random.Random,
) -> dict[str, Any]:
    """
    Set a ranged numeric field to a value outside its declared bounds.

    Targets fields with 'minimum' or 'maximum' in the schema.
    Falls back to setting 'confidence' to 1.5 if no ranged field is found.

    Violates: 'minimum' or 'maximum' constraint.
    """
    properties: dict[str, Any] = schema.get("properties", {})

    candidates = sorted(
        k for k, v in properties.items()
        if v.get("type") == "number"
        and ("minimum" in v or "maximum" in v)
        and k in out
    )

    if candidates:
        field = rng.choice(candidates)
        prop  = properties[field]
        mx    = prop.get("maximum")
        mn    = prop.get("minimum")
        if mx is not None:
            out[field] = float(mx) + 0.5 + rng.random()
        elif mn is not None:
            out[field] = float(mn) - 0.5 - rng.random()
    else:
        # Fallback: confidence is nearly always present and ranged [0, 1].
        out["confidence"] = 1.5

    return out


def _malformed_json(
    out: dict[str, Any],
    rng: random.Random,
) -> str:
    """
    Serialize *out* to JSON and inject a structural defect that prevents parsing.

    Returns a str, not a dict.  json.loads() will raise JSONDecodeError.

    Three equally-likely defects (selected by seed):
      0 — trailing comma before closing brace
      1 — truncated string (half the JSON + stray quote)
      2 — mismatched bracket (array open, object close)
    """
    base   = json.dumps(out, ensure_ascii=False)
    choice = rng.randint(0, 2)

    if choice == 0:
        # {"k": "v"} → {"k": "v", }   (trailing comma)
        return base[:-1] + ", }"
    if choice == 1:
        # Truncated + dangling quote
        cutoff = max(5, len(base) // 2)
        return base[:cutoff] + '"'
    # Mismatched brackets: [}
    return base[:-1] + ', "extra": [}'


def _extra_fields(
    out: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """
    Add fields not declared in the schema.

    Violates: 'additionalProperties': false on strict schemas.
    For permissive schemas (additionalProperties: true), this pattern is
    benign and the harness goal instance should use a strict schema.
    """
    suffix = rng.randint(1000, 9999)
    out[f"_debug_info_{suffix}"]    = "internal telemetry payload"
    out[f"internal_state_{suffix}"] = {"step": 0, "memo": []}
    out[f"__meta_{suffix}"]         = True
    return out

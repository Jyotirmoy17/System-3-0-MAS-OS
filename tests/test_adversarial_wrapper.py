"""Tests for mas/eval/adversarial_wrapper.py."""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft7Validator

from mas.eval.adversarial_wrapper import corrupt


# Strict test schema — NOT in SCHEMAS dict so repair() is a no-op.
STRICT_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type":    "object",
    "required": ["answer", "confidence", "is_valid"],
    "additionalProperties": False,
    "properties": {
        "answer":     {"type": "string", "maxLength": 500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "is_valid":   {"type": "boolean"},
        "tags":       {"type": "array", "items": {"type": "string"}},
    },
}

# Permissive schema that matches agent_response (additionalProperties: true).
PERMISSIVE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type":    "object",
    "required": ["answer", "confidence"],
    "additionalProperties": True,
    "properties": {
        "answer":     {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def _sample() -> dict:
    """A valid sample output matching STRICT_SCHEMA."""
    return {
        "answer":     "The answer is 42.",
        "confidence": 0.85,
        "is_valid":   True,
        "tags":       ["math", "exact"],
    }


def _fails_initial(corrupted: dict, schema: dict) -> bool:
    """Return True iff the corrupted dict fails initial JSON Schema validation."""
    errors = list(Draft7Validator(schema).iter_errors(corrupted))
    return len(errors) > 0


def test_drop_required_field_removes_field():
    """Dropped required field must cause a 'required' schema violation."""
    corrupted = corrupt(_sample(), "drop_required_field", STRICT_SCHEMA, seed=0)

    assert isinstance(corrupted, dict)

    required = set(STRICT_SCHEMA["required"])
    assert len(required - set(corrupted.keys())) >= 1, (
        "At least one required field must be removed"
    )

    errors = list(Draft7Validator(STRICT_SCHEMA).iter_errors(corrupted))
    assert any(e.validator == "required" for e in errors), (
        f"Expected 'required' violation; got: {[e.message for e in errors]}"
    )


def test_type_flip_changes_type():
    """Type-flipped field must violate the 'type' constraint."""
    corrupted = corrupt(_sample(), "type_flip", STRICT_SCHEMA, seed=0)

    assert isinstance(corrupted, dict)
    assert _fails_initial(corrupted, STRICT_SCHEMA), (
        "Type-flipped output must fail schema validation"
    )

    errors = list(Draft7Validator(STRICT_SCHEMA).iter_errors(corrupted))
    assert any(e.validator == "type" for e in errors), (
        f"Expected 'type' violation; got: {[e.message for e in errors]}"
    )


def test_type_flip_numeric_becomes_string():
    """Confirm the numeric confidence field is converted to a string."""
    corrupted = corrupt(_sample(), "type_flip", STRICT_SCHEMA, seed=0)
    # With STRICT_SCHEMA, confidence is the first numeric candidate.
    assert isinstance(corrupted.get("confidence"), str), (
        f"confidence should be str after type_flip, got {type(corrupted.get('confidence'))}"
    )


def test_out_of_range_violates_constraint():
    """Out-of-range value must violate 'maximum' or 'minimum' constraint."""
    corrupted = corrupt(_sample(), "out_of_range", STRICT_SCHEMA, seed=0)

    assert isinstance(corrupted, dict)
    assert _fails_initial(corrupted, STRICT_SCHEMA), (
        "Out-of-range output must fail schema validation"
    )

    errors = list(Draft7Validator(STRICT_SCHEMA).iter_errors(corrupted))
    assert any(e.validator in ("maximum", "minimum") for e in errors), (
        f"Expected range violation; got: {[e.message for e in errors]}"
    )


def test_out_of_range_value_exceeds_max():
    """Confidence must be set above its maximum (1.0)."""
    corrupted = corrupt(_sample(), "out_of_range", STRICT_SCHEMA, seed=0)
    conf = corrupted.get("confidence")
    assert isinstance(conf, float) and conf > 1.0, (
        f"Expected confidence > 1.0, got {conf}"
    )


def test_malformed_json_unparseable():
    """Result must be a string that json.loads() cannot parse."""
    result = corrupt(_sample(), "malformed_json", PERMISSIVE_SCHEMA, seed=0)

    assert isinstance(result, str), (
        f"malformed_json must return str, got {type(result)}"
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)


def test_malformed_json_contains_original_answer():
    """The corrupt string should contain recognisable content from the original."""
    result = corrupt(_sample(), "malformed_json", PERMISSIVE_SCHEMA, seed=0)
    assert "answer" in result, "Original keys should be visible in the corrupt string"


def test_extra_fields_violate_strict_schema():
    """Extra fields must cause a validation failure on a strict schema."""
    corrupted = corrupt(
        _sample(), "extra_fields_not_in_schema", STRICT_SCHEMA, seed=0
    )

    assert isinstance(corrupted, dict)

    # At least 3 extra fields added.
    original_keys  = set(_sample().keys())
    added_keys     = set(corrupted.keys()) - original_keys
    assert len(added_keys) >= 2, f"Expected ≥2 extra fields, got: {added_keys}"

    assert _fails_initial(corrupted, STRICT_SCHEMA), (
        "Extra fields must fail validation on additionalProperties: false schema"
    )
    errors = list(Draft7Validator(STRICT_SCHEMA).iter_errors(corrupted))
    assert any(e.validator == "additionalProperties" for e in errors)


def test_extra_fields_named_with_underscore_prefix():
    """Extra field names must start with _ or 'internal' (per implementation)."""
    corrupted = corrupt(_sample(), "extra_fields_not_in_schema", STRICT_SCHEMA, seed=0)
    original  = set(_sample().keys())
    added     = set(corrupted.keys()) - original
    assert all(
        k.startswith("_") or k.startswith("internal")
        for k in added
    ), f"Unexpected extra field names: {added}"


def test_determinism():
    """Same (output, pattern, schema, seed) must produce identical results."""
    output = _sample()
    for pattern in [
        "drop_required_field",
        "type_flip",
        "out_of_range",
        "extra_fields_not_in_schema",
    ]:
        r1 = corrupt(dict(output), pattern, STRICT_SCHEMA, seed=42)
        r2 = corrupt(dict(output), pattern, STRICT_SCHEMA, seed=42)
        assert r1 == r2, f"Pattern '{pattern}' is not deterministic"

    s1 = corrupt(dict(output), "malformed_json", STRICT_SCHEMA, seed=42)
    s2 = corrupt(dict(output), "malformed_json", STRICT_SCHEMA, seed=42)
    assert s1 == s2, "malformed_json is not deterministic"


def test_different_seeds_may_differ():
    """Different seeds should (for most patterns) produce different results."""
    output = _sample()
    # drop_required_field with 3 candidates: seed variation is observable.
    r_0  = corrupt(dict(output), "drop_required_field", STRICT_SCHEMA, seed=0)
    r_99 = corrupt(dict(output), "drop_required_field", STRICT_SCHEMA, seed=99)
    # Not guaranteed to differ (small candidate pool), but at least no crash.
    assert isinstance(r_0,  dict)
    assert isinstance(r_99, dict)


def test_unknown_pattern_raises():
    """Passing an unrecognised pattern must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown corruption pattern"):
        corrupt(_sample(), "totally_invalid_pattern", STRICT_SCHEMA)


def test_empty_string_pattern_raises():
    """Empty string is also an invalid pattern."""
    with pytest.raises(ValueError):
        corrupt(_sample(), "", STRICT_SCHEMA)

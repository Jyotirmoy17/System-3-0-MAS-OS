"""System-0 — the contract enforcement layer."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import Draft7Validator

from .registry import EscalationReason
from .schemas import SCHEMAS, repair

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of a single System-0 validation pass."""

    data:      Dict[str, Any]   # The (possibly repaired) validated data
    valid:     bool             # Did it pass schema validation?
    repaired:  bool             # Was a repair applied?
    errors:    Tuple[str, ...]  # jsonschema error messages (empty if valid)
    predicate: EscalationReason # Triggered predicate, or NONE


@dataclass
class AuditRecord:
    """One entry in the System-0 audit log."""

    agent_id:    str
    task_id:     str
    schema_name: str
    valid:       bool
    repaired:    bool            = False
    errors:      List[str]       = field(default_factory=list)
    predicate:   EscalationReason = EscalationReason.NONE
    depth:       int             = 0
    ts:          float           = field(default_factory=time.time)


class System0:
    """
    Contract enforcement protocol.

    Validates every AgentOutput dict against its registered JSON Schema.
    If validation fails, attempts field-level repair.
    If repair fails, marks as P2 escalation.
    Evaluates all four escalation predicates (P1–P4) in order.
    Records every event in the audit log for Vs computation.
    """

    def __init__(self, confidence_threshold: float = 0.72) -> None:
        self.tau = confidence_threshold
        # Pre-compile validators for performance
        self._validators: Dict[str, Draft7Validator] = {
            name: Draft7Validator(schema)
            for name, schema in SCHEMAS.items()
        }
        self._audit: List[AuditRecord] = []


    def validate_dict(
        self,
        data:        Dict[str, Any],
        schema_name: str = "agent_response",
        agent_id:    str = "unknown",
        task_id:     str = "unknown",
        depth:       int = 0,
    ) -> ValidationResult:
        """
        Validate *data* against *schema_name*.

        1. Run JSON Schema validation.
        2. On failure, attempt field-level repair.
        3. Evaluate escalation predicates on the final state.
        4. Append to audit log.
        5. Return an immutable ValidationResult.
        """
        validator = self._validators.get(schema_name)
        if validator is None:
            # Unknown schema — pass through; log a warning
            log.warning("Unknown schema '%s' — skipping validation", schema_name)
            return ValidationResult(
                data=data, valid=True, repaired=False,
                errors=(), predicate=EscalationReason.NONE,
            )

        raw_errors = [e.message for e in validator.iter_errors(data)]

        if raw_errors:
            repaired_data   = repair(data, schema_name)
            repair_errors   = [e.message for e in validator.iter_errors(repaired_data)]

            if not repair_errors:
                # Repair succeeded
                result = ValidationResult(
                    data=repaired_data, valid=True, repaired=True,
                    errors=tuple(raw_errors),
                    predicate=EscalationReason.NONE,
                )
            else:
                # Repair also failed → P2
                result = ValidationResult(
                    data=data, valid=False, repaired=False,
                    errors=tuple(raw_errors),
                    predicate=EscalationReason.P2_SCHEMA,
                )
                log.warning(
                    "System-0 P2 at agent=%s task=%s schema=%s errors=%s",
                    agent_id, task_id, schema_name, raw_errors[:2],
                )
        else:
            result = ValidationResult(
                data=data, valid=True, repaired=False,
                errors=(), predicate=EscalationReason.NONE,
            )

        self._audit.append(AuditRecord(
            agent_id=agent_id, task_id=task_id, schema_name=schema_name,
            valid=result.valid, repaired=result.repaired,
            errors=list(result.errors), predicate=result.predicate, depth=depth,
        ))

        return result

    def validate_raw_text(
        self,
        text:        str,
        schema_name: str = "agent_response",
        agent_id:    str = "unknown",
        task_id:     str = "unknown",
    ) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Parse raw LLM output text as JSON, then validate it.

        Strips markdown code fences before parsing.
        Returns (parsed_dict, ValidationResult).
        """
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            inner = lines[1:-1] if len(lines) > 2 else lines[1:]
            text  = "\n".join(inner).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}

        # Coerce numeric answers to str (common LLM mistake)
        if isinstance(data.get("answer"), (int, float)):
            data["answer"] = str(data["answer"])

        result = self.validate_dict(data, schema_name, agent_id, task_id)
        return result.data, result


    def check_predicates(
        self,
        answer:            str,
        confidence:        float,
        schema_valid:      bool,
        pre_set_escalation: EscalationReason = EscalationReason.NONE,
    ) -> EscalationReason:
        """
        Evaluate ESCALATE(Si, x) := P1 ∨ P2 ∨ P3 ∨ P4.

        Predicate priority: P2 > P3 > P4 > P1.
        Returns the first triggered predicate, or NONE.
        """
        # P2: schema invalid (set during validate_dict if repair also failed)
        if not schema_valid:
            return EscalationReason.P2_SCHEMA

        # P3 / P4: declared by the agent in its own escalation_reason field
        if pre_set_escalation in (
            EscalationReason.P3_CAPABILITY,
            EscalationReason.P4_CONTEXT,
        ):
            return pre_set_escalation

        # P1: confidence below threshold — only when there is a substantive answer
        if confidence < self.tau and answer.strip():
            return EscalationReason.P1_CONFIDENCE

        return EscalationReason.NONE


    def Vs(self, schema_name: Optional[str] = None) -> float:
        """
        Schema Compliance Rate: Vs = (1/N) Σ I(yᵢ ∈ L(G)).
        Optionally filtered to a specific schema name.
        """
        relevant = [
            r for r in self._audit
            if schema_name is None or r.schema_name == schema_name
        ]
        if not relevant:
            return 1.0
        return sum(1 for r in relevant if r.valid) / len(relevant)

    def escalation_counts(self) -> Dict[str, int]:
        """Count escalations by predicate type."""
        counts: Dict[str, int] = {}
        for record in self._audit:
            if record.predicate != EscalationReason.NONE:
                key = record.predicate.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def telemetry(self) -> Dict[str, Any]:
        total   = len(self._audit)
        passed  = sum(1 for r in self._audit if r.valid)
        repaired = sum(1 for r in self._audit if r.repaired)
        return {
            "total_validations": total,
            "passed":            passed,
            "repaired":          repaired,
            "failed":            total - passed,
            "Vs":                round(self.Vs(), 4),
            "escalations":       self.escalation_counts(),
        }

    def reset(self) -> None:
        """Clear the audit log (e.g. between evaluation runs)."""
        self._audit.clear()

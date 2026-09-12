"""Distinct contracts for local execution and relational evidence verification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

from gepa_mindfulness.core.evidence import EvidenceReference
from mindful_trace_gepa.action_bound_events import ActionRecord
from mindful_trace_gepa.logging_schema import (
    EventEnvelope,
    StructuredEventType,
    make_event_envelope,
)

from .state import (
    EvidenceState,
    _require_exact_mapping,
    _require_nonblank_string,
    _snapshot_evidence_refs,
)

_LOCAL_FINDING_FIELDS = frozenset(
    {
        "executed",
        "arguments_valid",
        "schema_valid",
        "authorization_valid",
        "intended_operation_observed",
        "irreversible_action_permitted",
    }
)
_RELATIONAL_FINDING_FIELDS = frozenset(
    {
        "task_fit",
        "dependencies_satisfied",
        "contradiction_status",
        "provenance_intact",
        "authorization_scope_valid",
        "claimed_outcome_supported",
        "repeated_failed_route",
    }
)
_VERIFICATION_FINDING_FIELDS = _LOCAL_FINDING_FIELDS | _RELATIONAL_FINDING_FIELDS
_CONTRADICTION_STATUSES = frozenset({"unknown", "none", "contradicted"})


class VerificationLevel(str, Enum):
    """The boundary at which one verification result was established."""

    LOCAL_EXECUTION = "local_execution"
    RELATIONAL_EVIDENCE = "relational_evidence"


@dataclass(frozen=True, slots=True)
class VerificationEvidenceBinding:
    """Immutable evidence attached to one named verifier finding.

    A result validates the field name and requires every bound reference to occur in that result's
    aggregate ``evidence_refs``. Bindings for affirmative findings require observable evidence;
    optional bindings for negative findings may retain non-observable diagnostic evidence.
    """

    field_name: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Require a named finding and at least one detached evidence reference."""

        _require_nonblank_string(self.field_name, "field_name")
        if self.field_name not in _VERIFICATION_FINDING_FIELDS:
            raise ValueError(f"field_name {self.field_name!r} is not a verification finding")
        if isinstance(self.evidence_refs, (str, bytes)) or not isinstance(
            self.evidence_refs,
            Sequence,
        ):
            raise ValueError("evidence binding evidence_refs must be an ordered array")
        references = _snapshot_evidence_refs(self.evidence_refs)
        if not references:
            raise ValueError("evidence binding must contain at least one evidence reference")
        object.__setattr__(self, "evidence_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible field evidence binding."""

        snapshot = _snapshot_binding(self)
        return {
            "field_name": snapshot.field_name,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> VerificationEvidenceBinding:
        """Restore one field evidence binding from its exact serialized form."""

        values = _require_exact_mapping(
            data,
            {"field_name", "evidence_refs"},
            "VerificationEvidenceBinding",
        )
        return cls(
            field_name=cast(str, values["field_name"]),
            evidence_refs=_restore_evidence_refs(
                values["evidence_refs"],
                "VerificationEvidenceBinding",
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalVerificationResult:
    """Local findings whose affirmative fields each bind observable evidence.

    Every ``True`` field requires its own binding. ``False`` fields may have diagnostic bindings,
    and ``irreversible_action_permitted=None`` means no permission finding was made.
    """

    action_id: str
    executed: bool
    arguments_valid: bool
    schema_valid: bool
    authorization_valid: bool
    intended_operation_observed: bool
    irreversible_action_permitted: bool | None
    evidence_refs: tuple[EvidenceReference, ...]
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] = ()

    def __post_init__(self) -> None:
        """Validate exact fields and detach caller-owned evidence references."""

        _require_nonblank_string(self.action_id, "action_id")
        affirmative_fields = (
            ("executed", self.executed),
            ("arguments_valid", self.arguments_valid),
            ("schema_valid", self.schema_valid),
            ("authorization_valid", self.authorization_valid),
            ("intended_operation_observed", self.intended_operation_observed),
        )
        for field_name, value in affirmative_fields:
            _require_exact_bool(value, field_name)
        _require_optional_exact_bool(
            self.irreversible_action_permitted,
            "irreversible_action_permitted",
        )
        references = _snapshot_evidence_refs(self.evidence_refs)
        object.__setattr__(self, "evidence_refs", references)
        bindings = _snapshot_bindings(self.evidence_bindings)
        object.__setattr__(self, "evidence_bindings", bindings)
        findings: dict[str, bool | None] = dict(affirmative_fields)
        findings["irreversible_action_permitted"] = self.irreversible_action_permitted
        _validate_evidence_bindings(findings, references, bindings)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible local verification record."""

        snapshot = _snapshot_local_result(self)
        return {
            "action_id": snapshot.action_id,
            "executed": snapshot.executed,
            "arguments_valid": snapshot.arguments_valid,
            "schema_valid": snapshot.schema_valid,
            "authorization_valid": snapshot.authorization_valid,
            "intended_operation_observed": snapshot.intended_operation_observed,
            "irreversible_action_permitted": snapshot.irreversible_action_permitted,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "evidence_bindings": [binding.to_dict() for binding in snapshot.evidence_bindings],
        }

    @classmethod
    def from_dict(cls, data: object) -> LocalVerificationResult:
        """Restore a local verification result from its exact serialized form."""

        values = _require_exact_mapping(
            data,
            {
                "action_id",
                "executed",
                "arguments_valid",
                "schema_valid",
                "authorization_valid",
                "intended_operation_observed",
                "irreversible_action_permitted",
                "evidence_refs",
                "evidence_bindings",
            },
            "LocalVerificationResult",
        )
        return cls(
            action_id=cast(str, values["action_id"]),
            executed=cast(bool, values["executed"]),
            arguments_valid=cast(bool, values["arguments_valid"]),
            schema_valid=cast(bool, values["schema_valid"]),
            authorization_valid=cast(bool, values["authorization_valid"]),
            intended_operation_observed=cast(bool, values["intended_operation_observed"]),
            irreversible_action_permitted=cast(
                bool | None,
                values["irreversible_action_permitted"],
            ),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"], cls.__name__),
            evidence_bindings=_restore_bindings(values["evidence_bindings"], cls.__name__),
        )


@dataclass(frozen=True, slots=True)
class RelationalVerificationResult:
    """Relational findings whose affirmative fields each bind observable evidence.

    Every ``True`` field requires its own binding. For ``contradiction_status``, ``unknown`` is not
    a finding, while ``none`` and ``contradicted`` are affirmative categorical findings that each
    require observable evidence bound to ``contradiction_status``.
    """

    action_id: str
    task_fit: bool
    dependencies_satisfied: bool
    contradiction_status: str
    provenance_intact: bool
    authorization_scope_valid: bool
    claimed_outcome_supported: bool
    repeated_failed_route: bool
    evidence_refs: tuple[EvidenceReference, ...]
    evidence_bindings: tuple[VerificationEvidenceBinding, ...] = ()

    def __post_init__(self) -> None:
        """Validate exact fields and detach caller-owned evidence references."""

        _require_nonblank_string(self.action_id, "action_id")
        _require_contradiction_status(self.contradiction_status)
        affirmative_fields = (
            ("task_fit", self.task_fit),
            ("dependencies_satisfied", self.dependencies_satisfied),
            ("provenance_intact", self.provenance_intact),
            ("authorization_scope_valid", self.authorization_scope_valid),
            ("claimed_outcome_supported", self.claimed_outcome_supported),
            ("repeated_failed_route", self.repeated_failed_route),
        )
        for field_name, value in affirmative_fields:
            _require_exact_bool(value, field_name)
        references = _snapshot_evidence_refs(self.evidence_refs)
        object.__setattr__(self, "evidence_refs", references)
        bindings = _snapshot_bindings(self.evidence_bindings)
        object.__setattr__(self, "evidence_bindings", bindings)
        findings: dict[str, bool | None] = dict(affirmative_fields)
        findings["contradiction_status"] = None if self.contradiction_status == "unknown" else True
        _validate_evidence_bindings(findings, references, bindings)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible relational verification record."""

        snapshot = _snapshot_relational_result(self)
        return {
            "action_id": snapshot.action_id,
            "task_fit": snapshot.task_fit,
            "dependencies_satisfied": snapshot.dependencies_satisfied,
            "contradiction_status": snapshot.contradiction_status,
            "provenance_intact": snapshot.provenance_intact,
            "authorization_scope_valid": snapshot.authorization_scope_valid,
            "claimed_outcome_supported": snapshot.claimed_outcome_supported,
            "repeated_failed_route": snapshot.repeated_failed_route,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "evidence_bindings": [binding.to_dict() for binding in snapshot.evidence_bindings],
        }

    @classmethod
    def from_dict(cls, data: object) -> RelationalVerificationResult:
        """Restore a relational verification result from its exact serialized form."""

        values = _require_exact_mapping(
            data,
            {
                "action_id",
                "task_fit",
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "claimed_outcome_supported",
                "repeated_failed_route",
                "evidence_refs",
                "evidence_bindings",
            },
            "RelationalVerificationResult",
        )
        return cls(
            action_id=cast(str, values["action_id"]),
            task_fit=cast(bool, values["task_fit"]),
            dependencies_satisfied=cast(bool, values["dependencies_satisfied"]),
            contradiction_status=cast(str, values["contradiction_status"]),
            provenance_intact=cast(bool, values["provenance_intact"]),
            authorization_scope_valid=cast(bool, values["authorization_scope_valid"]),
            claimed_outcome_supported=cast(bool, values["claimed_outcome_supported"]),
            repeated_failed_route=cast(bool, values["repeated_failed_route"]),
            evidence_refs=_restore_evidence_refs(values["evidence_refs"], cls.__name__),
            evidence_bindings=_restore_bindings(values["evidence_bindings"], cls.__name__),
        )


@runtime_checkable
class LocalExecutionVerifier(Protocol):
    """A verifier that evaluates only the local execution boundary."""

    def verify_local(self, action: ActionRecord) -> LocalVerificationResult:
        """Verify the local execution facts for one canonical action record."""
        ...


@runtime_checkable
class RelationalEvidenceVerifier(Protocol):
    """A verifier that evaluates action claims against wider evidence state."""

    def verify_relational(
        self,
        action: ActionRecord,
        evidence_state: EvidenceState,
    ) -> RelationalVerificationResult:
        """Verify one canonical action against an immutable evidence-state snapshot."""
        ...


def make_local_verification_event(
    result: LocalVerificationResult,
    *,
    verifier_refs: Sequence[str],
    **metadata: Any,
) -> EventEnvelope:
    """Wrap a local result with required verifier refs and no aggregate success value."""

    snapshot = _require_local_result(result)
    return _make_verification_event(
        VerificationLevel.LOCAL_EXECUTION,
        snapshot,
        verifier_refs,
        metadata,
    )


def make_relational_verification_event(
    result: RelationalVerificationResult,
    *,
    verifier_refs: Sequence[str],
    **metadata: Any,
) -> EventEnvelope:
    """Wrap a relational result with required verifier refs and no local-execution claim."""

    snapshot = _require_relational_result(result)
    return _make_verification_event(
        VerificationLevel.RELATIONAL_EVIDENCE,
        snapshot,
        verifier_refs,
        metadata,
    )


def _snapshot_local_result(result: LocalVerificationResult) -> LocalVerificationResult:
    return LocalVerificationResult(
        action_id=result.action_id,
        executed=result.executed,
        arguments_valid=result.arguments_valid,
        schema_valid=result.schema_valid,
        authorization_valid=result.authorization_valid,
        intended_operation_observed=result.intended_operation_observed,
        irreversible_action_permitted=result.irreversible_action_permitted,
        evidence_refs=result.evidence_refs,
        evidence_bindings=result.evidence_bindings,
    )


def _snapshot_relational_result(
    result: RelationalVerificationResult,
) -> RelationalVerificationResult:
    return RelationalVerificationResult(
        action_id=result.action_id,
        task_fit=result.task_fit,
        dependencies_satisfied=result.dependencies_satisfied,
        contradiction_status=result.contradiction_status,
        provenance_intact=result.provenance_intact,
        authorization_scope_valid=result.authorization_scope_valid,
        claimed_outcome_supported=result.claimed_outcome_supported,
        repeated_failed_route=result.repeated_failed_route,
        evidence_refs=result.evidence_refs,
        evidence_bindings=result.evidence_bindings,
    )


def _require_local_result(result: object) -> LocalVerificationResult:
    if type(result) is not LocalVerificationResult:
        raise TypeError("result must be an exact LocalVerificationResult")
    return _snapshot_local_result(result)


def _require_relational_result(result: object) -> RelationalVerificationResult:
    if type(result) is not RelationalVerificationResult:
        raise TypeError("result must be an exact RelationalVerificationResult")
    return _snapshot_relational_result(result)


def _make_verification_event(
    level: VerificationLevel,
    result: LocalVerificationResult | RelationalVerificationResult,
    verifier_refs: object,
    metadata: Mapping[str, Any],
) -> EventEnvelope:
    envelope_metadata = dict(metadata)
    verifier_refs = _snapshot_verifier_refs(verifier_refs)
    reference_ids = tuple(reference.reference_id for reference in result.evidence_refs)
    _merge_semantic_link(envelope_metadata, "action_id", result.action_id)
    _merge_semantic_link(envelope_metadata, "evidence_refs", reference_ids)
    payload = {
        "verification_level": level.value,
        "result": result.to_dict(),
        "verifier_refs": list(verifier_refs),
    }
    envelope_metadata["verifier_refs"] = verifier_refs
    return make_event_envelope(
        StructuredEventType.VERIFICATION_RESULT,
        payload,
        **envelope_metadata,
    )


def _merge_semantic_link(metadata: dict[str, Any], field_name: str, expected: object) -> None:
    if field_name in metadata and not _metadata_matches(metadata[field_name], expected):
        raise ValueError(f"{field_name} must match the typed verification result")
    metadata[field_name] = expected


def _metadata_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, tuple):
        if isinstance(actual, (str, bytes, Mapping)) or not isinstance(actual, Iterable):
            return False
        return tuple(actual) == expected
    return actual == expected


def _restore_evidence_refs(value: object, record_name: str) -> tuple[EvidenceReference, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(f"{record_name} evidence_refs must be an array")
    try:
        return tuple(EvidenceReference.from_dict(item) for item in value)
    except ValueError as exc:
        raise ValueError(f"{record_name} has invalid evidence_refs: {exc}") from exc


def _snapshot_binding(binding: object) -> VerificationEvidenceBinding:
    if type(binding) is not VerificationEvidenceBinding:
        raise ValueError("evidence_bindings must contain exact VerificationEvidenceBinding values")
    try:
        return VerificationEvidenceBinding(binding.field_name, binding.evidence_refs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evidence_bindings contains an invalid binding: {exc}") from exc


def _snapshot_bindings(value: object) -> tuple[VerificationEvidenceBinding, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("evidence_bindings must be an ordered array of evidence bindings")
    return tuple(_snapshot_binding(binding) for binding in value)


def _restore_bindings(value: object, record_name: str) -> tuple[VerificationEvidenceBinding, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{record_name} evidence_bindings must be an array")
    try:
        return tuple(VerificationEvidenceBinding.from_dict(item) for item in value)
    except ValueError as exc:
        raise ValueError(f"{record_name} has invalid evidence_bindings: {exc}") from exc


def _validate_evidence_bindings(
    findings: Mapping[str, bool | None],
    evidence_refs: tuple[EvidenceReference, ...],
    bindings: tuple[VerificationEvidenceBinding, ...],
) -> None:
    bindings_by_field: dict[str, VerificationEvidenceBinding] = {}
    result_references = {
        (reference.reference_id, reference.source_kind) for reference in evidence_refs
    }
    for binding in bindings:
        field_name = binding.field_name
        if field_name not in findings:
            raise ValueError(f"evidence binding {field_name!r} is not an allowed field")
        if field_name in bindings_by_field:
            raise ValueError("evidence binding field names must be unique")
        if findings[field_name] is None:
            raise ValueError(f"{field_name} has no finding and cannot have an evidence binding")
        if any(
            (reference.reference_id, reference.source_kind) not in result_references
            for reference in binding.evidence_refs
        ):
            raise ValueError("evidence binding references must be a subset of result evidence_refs")
        bindings_by_field[field_name] = binding

    for field_name, value in findings.items():
        if value is not True:
            continue
        field_binding = bindings_by_field.get(field_name)
        if field_binding is None:
            raise ValueError(f"{field_name} requires an evidence binding")
        if not any(reference.is_observable for reference in field_binding.evidence_refs):
            raise ValueError(f"{field_name} requires observable evidence in its binding")


def _snapshot_verifier_refs(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("verifier_refs must be a nonempty ordered array of exact strings")
    references: list[str] = []
    for reference in value:
        if type(reference) is not str or not reference.strip():
            raise ValueError("verifier_refs must contain exact nonblank strings")
        references.append(reference)
    if not references:
        raise ValueError("verifier_refs must contain at least one reference")
    return tuple(references)


def _require_exact_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a built-in bool")


def _require_optional_exact_bool(value: object, field_name: str) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{field_name} must be a built-in bool or None")


def _require_contradiction_status(value: object) -> None:
    if type(value) is not str or value not in _CONTRADICTION_STATUSES:
        raise ValueError(
            "contradiction_status must be exactly 'unknown', 'none', or 'contradicted'"
        )


__all__ = [
    "LocalExecutionVerifier",
    "LocalVerificationResult",
    "RelationalEvidenceVerifier",
    "RelationalVerificationResult",
    "VerificationEvidenceBinding",
    "VerificationLevel",
    "make_local_verification_event",
    "make_relational_verification_event",
]

"""Immutable payload records and envelope helpers for action-bound events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, cast

from .logging_schema import EventEnvelope, StructuredEventType, make_event_envelope


@dataclass(frozen=True)
class PredictionCommit:
    """An immutable prediction captured before a linked action."""

    prediction_commit_id: str
    predicted_outcome: object
    confidence: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the commitment and detach its JSON data from caller-owned containers."""

        _require_nonblank_string("prediction_commit_id", self.prediction_commit_id)
        _require_confidence(self.confidence)
        object.__setattr__(self, "predicted_outcome", _freeze_json(self.predicted_outcome))
        object.__setattr__(
            self, "evidence_refs", _normalize_refs("evidence_refs", self.evidence_refs)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation of this commitment."""

        return {
            "prediction_commit_id": self.prediction_commit_id,
            "predicted_outcome": _thaw_json(self.predicted_outcome),
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ActionRecord:
    """An authorized proposed or executed action linked to one prediction."""

    action_id: str
    action_class: str
    reversible: bool
    authorization_scope: str
    prediction_commit_id: str

    def __post_init__(self) -> None:
        """Validate action identity, authorization, and prediction linkage."""

        _require_nonblank_string("action_id", self.action_id)
        _require_nonblank_string("action_class", self.action_class)
        _require_exact_bool("reversible", self.reversible)
        _require_nonblank_string("authorization_scope", self.authorization_scope)
        _require_nonblank_string("prediction_commit_id", self.prediction_commit_id)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this action."""

        return {
            "action_id": self.action_id,
            "action_class": self.action_class,
            "reversible": self.reversible,
            "authorization_scope": self.authorization_scope,
            "prediction_commit_id": self.prediction_commit_id,
        }


@dataclass(frozen=True)
class OutcomeObservation:
    """An evidence-backed observation of an action's actual outcome."""

    observation_id: str
    action_id: str
    actual_outcome: object
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the observation and freeze its evidence-backed JSON result."""

        _require_nonblank_string("observation_id", self.observation_id)
        _require_nonblank_string("action_id", self.action_id)
        object.__setattr__(self, "actual_outcome", _freeze_json(self.actual_outcome))
        object.__setattr__(
            self,
            "evidence_refs",
            _normalize_refs("evidence_refs", self.evidence_refs, required=True),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation of this observation."""

        return {
            "observation_id": self.observation_id,
            "action_id": self.action_id,
            "actual_outcome": _thaw_json(self.actual_outcome),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class VerificationResult:
    """A versioned verifier result linked to one outcome observation."""

    verifier_id: str
    verifier_version: str
    observation_id: str
    verified: bool
    verifier_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate verifier identity, status, and required provenance references."""

        _require_nonblank_string("verifier_id", self.verifier_id)
        _require_nonblank_string("verifier_version", self.verifier_version)
        _require_nonblank_string("observation_id", self.observation_id)
        _require_exact_bool("verified", self.verified)
        object.__setattr__(
            self,
            "verifier_refs",
            _normalize_refs("verifier_refs", self.verifier_refs, required=True),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this verifier result."""

        return {
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "observation_id": self.observation_id,
            "verified": self.verified,
            "verifier_refs": list(self.verifier_refs),
        }


def make_prediction_commit_event(
    prediction: PredictionCommit,
    **metadata: Any,
) -> EventEnvelope:
    """Wrap a prediction commitment in the matching structured event envelope."""

    _require_instance("prediction", prediction, PredictionCommit)
    return _make_payload_event(
        StructuredEventType.PREDICTION_COMMIT,
        prediction.to_dict(),
        metadata,
        evidence_refs=prediction.evidence_refs,
    )


def make_action_event(
    action: ActionRecord,
    event_type: StructuredEventType | str,
    **metadata: Any,
) -> EventEnvelope:
    """Wrap a proposed or executed action without imposing temporal sequencing rules."""

    _require_instance("action", action, ActionRecord)
    event_value = (
        event_type.value if isinstance(event_type, StructuredEventType) else str(event_type)
    )
    allowed = {
        StructuredEventType.ACTION_PROPOSED.value,
        StructuredEventType.ACTION_EXECUTED.value,
    }
    if event_value not in allowed:
        raise ValueError("event_type must be ACTION_PROPOSED or ACTION_EXECUTED")
    return _make_payload_event(
        event_value,
        action.to_dict(),
        metadata,
        action_id=action.action_id,
        authorization_scope=action.authorization_scope,
    )


def make_outcome_observation_event(
    observation: OutcomeObservation,
    **metadata: Any,
) -> EventEnvelope:
    """Wrap an observed outcome in the matching structured event envelope."""

    _require_instance("observation", observation, OutcomeObservation)
    return _make_payload_event(
        StructuredEventType.OUTCOME_OBSERVED,
        observation.to_dict(),
        metadata,
        action_id=observation.action_id,
        evidence_refs=observation.evidence_refs,
    )


def make_verification_result_event(
    result: VerificationResult,
    **metadata: Any,
) -> EventEnvelope:
    """Wrap a verifier result in the matching structured event envelope."""

    _require_instance("result", result, VerificationResult)
    return _make_payload_event(
        StructuredEventType.VERIFICATION_RESULT,
        result.to_dict(),
        metadata,
        verifier_refs=result.verifier_refs,
    )


def _make_payload_event(
    event_type: StructuredEventType | str,
    payload: dict[str, object],
    metadata: Mapping[str, Any],
    **semantic_links: object,
) -> EventEnvelope:
    """Merge immutable payload links into caller-supplied envelope metadata."""

    envelope_metadata = dict(metadata)
    for field_name, expected in semantic_links.items():
        if field_name in envelope_metadata and not _metadata_matches(
            envelope_metadata[field_name], expected
        ):
            raise ValueError(f"{field_name} must match the typed payload linkage")
        envelope_metadata[field_name] = expected
    return make_event_envelope(event_type, payload, **envelope_metadata)


def _metadata_matches(actual: object, expected: object) -> bool:
    """Compare linkage metadata while accepting list-like reference inputs."""

    if isinstance(expected, tuple):
        if isinstance(actual, str) or not isinstance(actual, Iterable):
            return False
        return tuple(actual) == expected
    return actual == expected


def _require_instance(name: str, value: object, expected_type: type[object]) -> None:
    """Reject helpers called with a payload type that does not match their event kind."""

    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}")


def _require_nonblank_string(field_name: str, value: object) -> None:
    """Require a stable nonblank identifier or scope string."""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def _require_exact_bool(field_name: str, value: object) -> None:
    """Require a bool rather than a truthy or falsy stand-in."""

    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a built-in bool")


def _require_confidence(value: object) -> None:
    """Require a finite built-in JSON number in the inclusive unit interval."""

    if type(value) not in (int, float):
        raise ValueError("confidence must be a finite built-in number from 0 through 1")
    confidence = cast(int | float, value)
    if not isfinite(float(confidence)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite built-in number from 0 through 1")


def _normalize_refs(field_name: str, values: object, *, required: bool = False) -> tuple[str, ...]:
    """Copy and normalize reference IDs without retaining caller-owned collections."""

    if isinstance(values, str) or not isinstance(values, Iterable):
        raise ValueError(f"{field_name} must be an iterable of nonblank strings")
    normalized = tuple(_normalize_ref(field_name, value) for value in values)
    if required and not normalized:
        raise ValueError(f"{field_name} must contain at least one nonblank string")
    return normalized


def _normalize_ref(field_name: str, value: object) -> str:
    """Normalize one evidence or verifier reference while rejecting blanks."""

    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must contain only nonblank strings")
    return value.strip()


def _freeze_json(value: object, active_containers: set[int] | None = None) -> object:
    """Convert a JSON-compatible value into deterministic immutable containers."""

    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("outcomes must be JSON-compatible finite values")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, active_containers)
    if isinstance(value, (list, tuple)):
        return _freeze_json_sequence(value, active_containers)
    raise ValueError("outcomes must be JSON-compatible values")


def _freeze_json_mapping(
    value: Mapping[object, object], active_containers: set[int] | None
) -> object:
    """Freeze one JSON object with sorted keys and nested immutable values."""

    active = _enter_container(value, active_containers)
    try:
        items: dict[str, object] = {}
        keys: list[str] = []
        for key in value:
            if type(key) is not str:
                raise ValueError("outcomes must use only string mapping keys")
            keys.append(key)
        for key in sorted(keys):
            items[key] = _freeze_json(value[key], active)
        return MappingProxyType(items)
    finally:
        active.remove(id(value))


def _freeze_json_sequence(
    value: list[object] | tuple[object, ...], active_containers: set[int] | None
) -> tuple[object, ...]:
    """Freeze one JSON array into a tuple after detecting recursive references."""

    active = _enter_container(value, active_containers)
    try:
        return tuple(_freeze_json(item, active) for item in value)
    finally:
        active.remove(id(value))


def _enter_container(value: object, active_containers: set[int] | None) -> set[int]:
    """Track the current recursion stack and reject JSON containers with cycles."""

    active = active_containers if active_containers is not None else set()
    value_id = id(value)
    if value_id in active:
        raise ValueError("outcomes must not contain a cycle")
    active.add(value_id)
    return active


def _thaw_json(value: object) -> object:
    """Return a fresh ordinary JSON-compatible value from frozen internal data."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "ActionRecord",
    "OutcomeObservation",
    "PredictionCommit",
    "VerificationResult",
    "make_action_event",
    "make_outcome_observation_event",
    "make_prediction_commit_event",
    "make_verification_result_event",
]

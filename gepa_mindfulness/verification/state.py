"""Immutable records that keep observed world changes separate from evidence claims."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

EvidenceStatus = Literal["unverified", "supported", "contradicted", "superseded"]
_EVIDENCE_STATUSES = frozenset({"unverified", "supported", "contradicted", "superseded"})
_RFC3339_OFFSET_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-](?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    """One canonical artifact digest observed through explicit external evidence."""

    observation_id: str
    artifact_ref: str
    digest: str
    observed_at: str
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Validate and detach the complete observation identity."""

        _require_nonblank_string(self.observation_id, "observation_id")
        _require_nonblank_string(self.artifact_ref, "artifact_ref")
        _require_sha256(self.digest, "digest")
        _require_rfc3339(self.observed_at, "observed_at")
        references = _snapshot_evidence_refs(self.evidence_refs)
        if not references or not any(reference.is_observable for reference in references):
            raise ValueError("ArtifactObservation requires observable evidence_refs")
        object.__setattr__(self, "evidence_refs", references)

    def to_dict(self) -> dict[str, object]:
        """Return an exact JSON-compatible observation snapshot."""

        snapshot = _snapshot_artifact_observation(self)
        return {
            "observation_id": snapshot.observation_id,
            "artifact_ref": snapshot.artifact_ref,
            "digest": snapshot.digest,
            "observed_at": snapshot.observed_at,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: object) -> ArtifactObservation:
        """Restore an observation from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {"observation_id", "artifact_ref", "digest", "observed_at", "evidence_refs"},
            "ArtifactObservation",
        )
        return cls(
            cast(str, values["observation_id"]),
            cast(str, values["artifact_ref"]),
            cast(str, values["digest"]),
            cast(str, values["observed_at"]),
            _restore_evidence_refs(values["evidence_refs"], "ArtifactObservation"),
        )


@dataclass(frozen=True, slots=True)
class WorldStateChange:
    """An action-bound transition between exact artifact observations."""

    change_id: str
    action_id: str
    before_observation: ArtifactObservation | None
    after_observation: ArtifactObservation

    def __post_init__(self) -> None:
        """Reject transitions without one canonical after observation."""

        _require_nonblank_string(self.change_id, "change_id")
        _require_nonblank_string(self.action_id, "action_id")
        before = _snapshot_optional_artifact_observation(self.before_observation)
        after = _snapshot_artifact_observation(self.after_observation)
        if before is not None and before.artifact_ref != after.artifact_ref:
            raise ValueError("before and after observations must identify the same artifact_ref")
        if before is not None and before.observation_id == after.observation_id:
            raise ValueError("before and after observations need distinct observation_id values")
        object.__setattr__(self, "before_observation", before)
        object.__setattr__(self, "after_observation", after)

    @property
    def artifact_ref(self) -> str:
        return _snapshot_artifact_observation(self.after_observation).artifact_ref

    @property
    def before_digest(self) -> str | None:
        before = _snapshot_optional_artifact_observation(self.before_observation)
        return None if before is None else before.digest

    @property
    def after_digest(self) -> str:
        return _snapshot_artifact_observation(self.after_observation).digest

    @property
    def observed_at(self) -> str:
        return _snapshot_artifact_observation(self.after_observation).observed_at

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible world-change record."""

        snapshot = _snapshot_world_state_change(self)
        return {
            "change_id": snapshot.change_id,
            "action_id": snapshot.action_id,
            "before_observation": (
                None
                if snapshot.before_observation is None
                else snapshot.before_observation.to_dict()
            ),
            "after_observation": snapshot.after_observation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> WorldStateChange:
        """Restore a world change from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {"change_id", "action_id", "before_observation", "after_observation"},
            "WorldStateChange",
        )
        raw_before = values["before_observation"]
        return cls(
            cast(str, values["change_id"]),
            cast(str, values["action_id"]),
            None if raw_before is None else ArtifactObservation.from_dict(raw_before),
            ArtifactObservation.from_dict(values["after_observation"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    """An immutable proposition and its current evidence interpretation."""

    claim_id: str
    proposition: str
    evidence_refs: tuple[EvidenceReference, ...]
    status: EvidenceStatus
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank_string(self.claim_id, "claim_id")
        _require_nonblank_string(self.proposition, "proposition")
        object.__setattr__(self, "evidence_refs", _snapshot_evidence_refs(self.evidence_refs))
        if type(self.status) is not str or self.status not in _EVIDENCE_STATUSES:
            raise ValueError("status must be an exact supported evidence status")
        if self.superseded_by is not None:
            _require_nonblank_string(self.superseded_by, "superseded_by")
            if self.superseded_by == self.claim_id:
                raise ValueError("a claim cannot supersede itself")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("superseded status requires exactly one superseded_by link")
        if self.status in {"supported", "contradicted"} and not self.evidence_refs:
            raise ValueError(f"{self.status} status requires evidence_refs")

    def to_dict(self) -> dict[str, object]:
        snapshot = _snapshot_claim(self)
        return {
            "claim_id": snapshot.claim_id,
            "proposition": snapshot.proposition,
            "evidence_refs": [reference.to_dict() for reference in snapshot.evidence_refs],
            "status": snapshot.status,
            "superseded_by": snapshot.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: object) -> EvidenceClaim:
        values = _require_exact_mapping(
            data,
            {"claim_id", "proposition", "evidence_refs", "status", "superseded_by"},
            "EvidenceClaim",
        )
        return cls(
            cast(str, values["claim_id"]),
            cast(str, values["proposition"]),
            _restore_evidence_refs(values["evidence_refs"], "EvidenceClaim"),
            cast(EvidenceStatus, values["status"]),
            cast(str | None, values["superseded_by"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceState:
    """An append-only snapshot of claims linked by explicit supersession."""

    claims: tuple[EvidenceClaim, ...]

    def __post_init__(self) -> None:
        snapshots = _snapshot_claims(self.claims)
        _validate_claim_graph(snapshots)
        object.__setattr__(self, "claims", snapshots)

    def resolve(self, claim_id: str) -> EvidenceClaim:
        _require_nonblank_string(claim_id, "claim_id")
        snapshots = _snapshot_claims(self.claims)
        claims_by_id = _validate_claim_graph(snapshots)
        current = claims_by_id.get(claim_id)
        if current is None:
            raise KeyError(f"unknown evidence claim {claim_id!r}")
        while current.superseded_by is not None:
            current = claims_by_id[current.superseded_by]
        return _snapshot_claim(current)

    def to_dict(self) -> dict[str, object]:
        snapshots = _snapshot_claims(self.claims)
        _validate_claim_graph(snapshots)
        return {"claims": [claim.to_dict() for claim in snapshots]}

    @classmethod
    def from_dict(cls, data: object) -> EvidenceState:
        values = _require_exact_mapping(data, {"claims"}, "EvidenceState")
        raw_claims = values["claims"]
        if isinstance(raw_claims, (str, bytes, Mapping)) or not isinstance(raw_claims, Sequence):
            raise ValueError("EvidenceState claims must be an array")
        return cls(tuple(EvidenceClaim.from_dict(item) for item in raw_claims))


def _require_nonblank_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank built-in string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA-256 digest")
    return value


def _require_rfc3339(value: object, field_name: str) -> str:
    timestamp_value = _require_nonblank_string(value, field_name)
    match = _RFC3339_OFFSET_DATETIME.fullmatch(timestamp_value)
    if match is None:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    hour = match.group("offset_hour")
    minute = match.group("offset_minute")
    if hour is not None and (int(hour) > 23 or int(minute) > 59):
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime")
    timestamp = (
        f"{timestamp_value[:-1]}+00:00" if timestamp_value.endswith("Z") else timestamp_value
    )
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an RFC3339 offset datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an explicit UTC offset")
    return timestamp_value


def _snapshot_evidence_refs(values: object) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise ValueError("evidence_refs must be an iterable of EvidenceReference values")
    references: list[EvidenceReference] = []
    for reference in values:
        if type(reference) is not EvidenceReference:
            raise ValueError("evidence_refs must contain exact EvidenceReference values")
        if type(reference.reference_id) is not str or not reference.reference_id.strip():
            raise ValueError("evidence_refs reference_id must be a nonblank built-in string")
        if type(reference.source_kind) is not EvidenceSourceKind:
            raise ValueError("evidence_refs source_kind must be an exact EvidenceSourceKind")
        references.append(EvidenceReference(reference.reference_id, reference.source_kind))
    return tuple(references)


def _restore_evidence_refs(values: object, record_name: str) -> tuple[EvidenceReference, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError(f"{record_name} evidence_refs must be an array")
    return _snapshot_evidence_refs(tuple(EvidenceReference.from_dict(item) for item in values))


def _snapshot_artifact_observation(value: object) -> ArtifactObservation:
    if type(value) is not ArtifactObservation:
        raise ValueError("observation must be an exact ArtifactObservation")
    if type(value.evidence_refs) is not tuple:
        raise ValueError("ArtifactObservation evidence_refs must remain an exact tuple")
    try:
        return ArtifactObservation(
            value.observation_id,
            value.artifact_ref,
            value.digest,
            value.observed_at,
            value.evidence_refs,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ArtifactObservation: {exc}") from exc


def _snapshot_optional_artifact_observation(value: object) -> ArtifactObservation | None:
    return None if value is None else _snapshot_artifact_observation(value)


def _snapshot_world_state_change(value: object) -> WorldStateChange:
    if type(value) is not WorldStateChange:
        raise ValueError("change must be an exact WorldStateChange")
    try:
        return WorldStateChange(
            value.change_id,
            value.action_id,
            value.before_observation,
            value.after_observation,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid WorldStateChange: {exc}") from exc


def _snapshot_claim(value: object) -> EvidenceClaim:
    if type(value) is not EvidenceClaim:
        raise ValueError("claims must contain exact EvidenceClaim values")
    if type(value.evidence_refs) is not tuple:
        raise ValueError("EvidenceClaim evidence_refs must remain an exact tuple")
    try:
        return EvidenceClaim(
            value.claim_id,
            value.proposition,
            value.evidence_refs,
            value.status,
            value.superseded_by,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"claims contains an invalid EvidenceClaim: {exc}") from exc


def _snapshot_claims(values: object) -> tuple[EvidenceClaim, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise ValueError("claims must be an ordered array of EvidenceClaim values")
    return tuple(_snapshot_claim(value) for value in values)


def _validate_claim_graph(claims: tuple[EvidenceClaim, ...]) -> dict[str, EvidenceClaim]:
    claims_by_id = {claim.claim_id: claim for claim in claims}
    if len(claims_by_id) != len(claims):
        raise ValueError("EvidenceState claim IDs must be unique")
    for claim in claims:
        successor = claim.superseded_by
        if successor is not None and successor not in claims_by_id:
            raise ValueError(f"EvidenceState has dangling superseded_by link {successor!r}")
    _reject_supersession_cycles(claims, claims_by_id)
    return claims_by_id


def _reject_supersession_cycles(
    claims: tuple[EvidenceClaim, ...],
    claims_by_id: Mapping[str, EvidenceClaim],
) -> None:
    resolved: set[str] = set()
    for claim in claims:
        path: set[str] = set()
        current = claim
        while current.claim_id not in resolved:
            if current.claim_id in path:
                raise ValueError("EvidenceState superseded_by links contain a cycle")
            path.add(current.claim_id)
            if current.superseded_by is None:
                break
            current = claims_by_id[current.superseded_by]
        resolved.update(path)


def _require_exact_mapping(
    data: object,
    expected_fields: set[str],
    record_name: str,
) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{record_name} must be an object")
    if set(data) != expected_fields:
        raise ValueError(f"{record_name} requires exactly {sorted(expected_fields)!r}")
    return cast(Mapping[str, object], data)


__all__ = ["ArtifactObservation", "EvidenceClaim", "EvidenceState", "WorldStateChange"]

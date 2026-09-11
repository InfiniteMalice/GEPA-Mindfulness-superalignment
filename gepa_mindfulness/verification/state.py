"""Immutable records that keep observed world changes separate from evidence claims."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from gepa_mindfulness.core.evidence import EvidenceReference

EvidenceStatus = Literal["unverified", "supported", "contradicted", "superseded"]

_EVIDENCE_STATUSES = frozenset(
    {
        "unverified",
        "supported",
        "contradicted",
        "superseded",
    }
)
_RFC3339_OFFSET_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-](?P<offset_hour>\d{2}):(?P<offset_minute>\d{2}))$"
)


@dataclass(frozen=True, slots=True)
class WorldStateChange:
    """An observed artifact transition caused by one identified action."""

    change_id: str
    action_id: str
    artifact_ref: str
    before_digest: str | None
    after_digest: str
    observed_at: str

    def __post_init__(self) -> None:
        """Reject records without exact, auditable action and artifact observations."""

        _require_nonblank_string(self.change_id, "change_id")
        _require_nonblank_string(self.action_id, "action_id")
        _require_nonblank_string(self.artifact_ref, "artifact_ref")
        if self.before_digest is not None:
            _require_sha256(self.before_digest, "before_digest")
        _require_sha256(self.after_digest, "after_digest")
        _require_rfc3339(self.observed_at, "observed_at")

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible world-change record."""

        return {
            "change_id": self.change_id,
            "action_id": self.action_id,
            "artifact_ref": self.artifact_ref,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, data: object) -> WorldStateChange:
        """Restore a world change from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "change_id",
                "action_id",
                "artifact_ref",
                "before_digest",
                "after_digest",
                "observed_at",
            },
            "WorldStateChange",
        )
        return cls(
            change_id=cast(str, values["change_id"]),
            action_id=cast(str, values["action_id"]),
            artifact_ref=cast(str, values["artifact_ref"]),
            before_digest=cast(str | None, values["before_digest"]),
            after_digest=cast(str, values["after_digest"]),
            observed_at=cast(str, values["observed_at"]),
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
        """Validate the claim and detach its evidence references from caller objects."""

        _require_nonblank_string(self.claim_id, "claim_id")
        _require_nonblank_string(self.proposition, "proposition")
        references = _snapshot_evidence_refs(self.evidence_refs)
        object.__setattr__(self, "evidence_refs", references)
        if type(self.status) is not str or self.status not in _EVIDENCE_STATUSES:
            raise ValueError("status must be an exact supported evidence status")
        if self.superseded_by is not None:
            _require_nonblank_string(self.superseded_by, "superseded_by")
            if self.superseded_by == self.claim_id:
                raise ValueError("a claim cannot supersede itself")
        has_successor = self.superseded_by is not None
        if (self.status == "superseded") != has_successor:
            raise ValueError("superseded status requires exactly one superseded_by link")

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible evidence-claim record."""

        return {
            "claim_id": self.claim_id,
            "proposition": self.proposition,
            "evidence_refs": [reference.to_dict() for reference in self.evidence_refs],
            "status": self.status,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: object) -> EvidenceClaim:
        """Restore an evidence claim from its exact JSON-compatible record."""

        values = _require_exact_mapping(
            data,
            {
                "claim_id",
                "proposition",
                "evidence_refs",
                "status",
                "superseded_by",
            },
            "EvidenceClaim",
        )
        raw_references = values["evidence_refs"]
        if isinstance(raw_references, (str, bytes, Mapping)) or not isinstance(
            raw_references,
            Iterable,
        ):
            raise ValueError("EvidenceClaim evidence_refs must be an array")
        references = tuple(EvidenceReference.from_dict(item) for item in raw_references)
        return cls(
            claim_id=cast(str, values["claim_id"]),
            proposition=cast(str, values["proposition"]),
            evidence_refs=references,
            status=cast(EvidenceStatus, values["status"]),
            superseded_by=cast(str | None, values["superseded_by"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceState:
    """An append-only snapshot of claims linked by explicit supersession."""

    claims: tuple[EvidenceClaim, ...]

    def __post_init__(self) -> None:
        """Snapshot claims and reject ambiguous or invalid supersession graphs."""

        snapshots = _snapshot_claims(self.claims)
        object.__setattr__(self, "claims", snapshots)
        claims_by_id = {claim.claim_id: claim for claim in snapshots}
        if len(claims_by_id) != len(snapshots):
            raise ValueError("EvidenceState claim IDs must be unique")
        for claim in snapshots:
            successor = claim.superseded_by
            if successor is not None and successor not in claims_by_id:
                raise ValueError(f"EvidenceState has dangling superseded_by link {successor!r}")
        _reject_supersession_cycles(snapshots, claims_by_id)

    def resolve(self, claim_id: str) -> EvidenceClaim:
        """Return the terminal claim reached by deterministic supersession lookup."""

        _require_nonblank_string(claim_id, "claim_id")
        claims_by_id = {claim.claim_id: claim for claim in self.claims}
        current = claims_by_id.get(claim_id)
        if current is None:
            raise KeyError(f"unknown evidence claim {claim_id!r}")
        while current.superseded_by is not None:
            current = claims_by_id[current.superseded_by]
        return current

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible evidence-state snapshot."""

        return {"claims": [claim.to_dict() for claim in self.claims]}

    @classmethod
    def from_dict(cls, data: object) -> EvidenceState:
        """Restore an evidence state from its exact JSON-compatible snapshot."""

        values = _require_exact_mapping(data, {"claims"}, "EvidenceState")
        raw_claims = values["claims"]
        if isinstance(raw_claims, (str, bytes, Mapping)) or not isinstance(
            raw_claims,
            Iterable,
        ):
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
    offset_hour = match.group("offset_hour")
    offset_minute = match.group("offset_minute")
    if offset_hour is not None and (int(offset_hour) > 23 or int(offset_minute) > 59):
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
        try:
            snapshot = EvidenceReference(reference.reference_id, reference.source_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence_refs contains an invalid EvidenceReference") from exc
        references.append(snapshot)
    return tuple(references)


def _snapshot_claims(values: object) -> tuple[EvidenceClaim, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise ValueError("claims must be an iterable of EvidenceClaim values")
    claims: list[EvidenceClaim] = []
    for claim in values:
        if type(claim) is not EvidenceClaim:
            raise ValueError("claims must contain exact EvidenceClaim values")
        try:
            snapshot = EvidenceClaim(
                claim_id=claim.claim_id,
                proposition=claim.proposition,
                evidence_refs=claim.evidence_refs,
                status=claim.status,
                superseded_by=claim.superseded_by,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("claims contains an invalid EvidenceClaim") from exc
        claims.append(snapshot)
    return tuple(claims)


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
            successor = current.superseded_by
            if successor is None:
                break
            current = claims_by_id[successor]
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


__all__ = [
    "EvidenceClaim",
    "EvidenceState",
    "WorldStateChange",
]

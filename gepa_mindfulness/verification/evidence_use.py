"""Explicit qualitative eligibility for numeric evidence; no routing or authority effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from typing import Any, Literal

from semantic_intent_robustness.memory_safety import (
    MemoryRetrievalDecision,
    RetrievedMemory,
    assess_retrieved_memory,
)

from .epistemic_state import (
    Availability,
    EpistemicMeasurement,
    _enum,
    _nonnegative,
    _snapshot,
    _strings,
    _text,
    _unit,
)
from .state import EvidenceClaim, EvidenceState, EvidenceStatus, parse_rfc3339_datetime


class MemoryKind(str, Enum):
    """Content role, separate from learning destinations and source trust."""

    FACT = "fact"
    PROCEDURE = "procedure"
    NORM = "norm"
    EPISODE = "episode"


class MemoryInfluence(str, Enum):
    """Declared influence on an answer; CONTROL grants no execution permission."""

    IGNORE = "ignore"
    BOUND = "bound"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    """Host declarations; scores are diagnostics, not variances or authentication."""

    recorded_at: str
    source_reliability: float | None
    compression_distortion: float | None
    integrity: Literal["intact", "tainted", "unknown"]
    authority: Literal["information_only", "external_policy_reference", "unknown"]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        parse_rfc3339_datetime(self.recorded_at, "recorded_at")
        for name in ("source_reliability", "compression_distortion"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit(value, name))
        for name, allowed in (
            ("integrity", {"intact", "tainted", "unknown"}),
            ("authority", {"information_only", "external_policy_reference", "unknown"}),
        ):
            value = getattr(self, name)
            if type(value) is not str or value not in allowed:
                raise ValueError(f"unsupported {name}")
        object.__setattr__(
            self, "provenance", _strings(self.provenance, "provenance", required=True)
        )


@dataclass(frozen=True, slots=True)
class EvidenceUsePolicy:
    """Explicit host thresholds; this policy only withholds diagnostic numeric inputs."""

    max_age_seconds: float
    min_source_reliability: float
    max_compression_distortion: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_age_seconds", _nonnegative(self.max_age_seconds, "max_age"))
        for name in ("min_source_reliability", "max_compression_distortion"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceUseAssessment:
    """Detached evidence/memory inputs and computed eligibility at one declared time.

    Views are unverified summaries for display. The original claim, measurement and memory remain
    authoritative for this assessment's checks. A host must reassess against current state/time
    before later use; this object neither authenticates declarations nor refreshes itself.
    """

    state: EvidenceState
    claim_id: str
    measurement: EpistemicMeasurement
    memory: RetrievedMemory
    kind: MemoryKind
    target_influence: MemoryInfluence
    quality: EvidenceQuality
    policy: EvidenceUsePolicy
    assessed_at: str
    views: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.state) is not EvidenceState:
            raise ValueError("state must be EvidenceState")
        object.__setattr__(self, "state", EvidenceState.from_dict(self.state.to_dict()))
        _text(self.claim_id, "claim_id")
        claim = self.claim
        object.__setattr__(self, "measurement", _snapshot(self.measurement, EpistemicMeasurement))
        object.__setattr__(self, "memory", _memory_snapshot(self.memory))
        for name, cls in (("quality", EvidenceQuality), ("policy", EvidenceUsePolicy)):
            value = getattr(self, name)
            if type(value) is not cls:
                raise ValueError(f"{name} must be {cls.__name__}")
            object.__setattr__(self, name, replace(value))
        _enum(self.kind, MemoryKind, "kind")
        _enum(self.target_influence, MemoryInfluence, "target_influence")
        if (self.memory.memory_id, self.memory.content_summary) != (
            claim.claim_id,
            claim.proposition,
        ):
            raise ValueError("memory identity and content must match the named claim")
        refs = (*claim.evidence_refs, *self.measurement.evidence_refs)
        kinds: dict[str, object] = {}
        for ref in refs:
            if ref.reference_id in kinds and kinds[ref.reference_id] != ref.source_kind:
                raise ValueError("ambiguous evidence reference kind")
            kinds[ref.reference_id] = ref.source_kind
        if not set(self.measurement.evidence_refs).issubset(claim.evidence_refs):
            raise ValueError("measurement references must belong to the named claim")
        if self.age_seconds < 0:
            raise ValueError("recorded_at is in the future of assessed_at")
        if type(self.views) is not tuple or len(self.views) > 128:
            raise ValueError("views must be a tuple of at most 128 summaries")
        for view in self.views:
            if type(view) is not tuple or len(view) != 2:
                raise ValueError("each view requires transformation_id and summary")
            _text(view[0], "transformation_id")
            _text(view[1], "summary")
        if len({view[0] for view in self.views}) != len(self.views):
            raise ValueError("transformation IDs must be unique")

    @property
    def claim(self) -> EvidenceClaim:
        """Return the original claim, without following its supersession link."""
        for claim in self.state.claims:
            if claim.claim_id == self.claim_id:
                return claim
        raise ValueError("unknown claim_id")

    @property
    def age_seconds(self) -> float:
        end = parse_rfc3339_datetime(self.assessed_at, "assessed_at")
        start = parse_rfc3339_datetime(self.quality.recorded_at, "recorded_at")
        return (end - start).total_seconds()

    @property
    def status(self) -> EvidenceStatus:
        status = self.claim.status
        if status in {"observed", "inferred", "supported"}:
            if self.age_seconds > self.policy.max_age_seconds:
                return "stale"
        return status

    @property
    def limitations(self) -> tuple[str, ...]:
        """All reasons to withhold numeric use; none silently changes variance."""
        reasons = []
        if self.status not in {"observed", "inferred", "supported"}:
            reasons.append(f"evidence_{self.status}")
        if self.measurement.status is Availability.UNAVAILABLE:
            reasons.append("measurement_unavailable")
        if any(not ref.is_observable for ref in self.measurement.evidence_refs):
            reasons.append("nonobservable_evidence")
        boundary = assess_retrieved_memory(self.memory)
        if boundary.decision != MemoryRetrievalDecision.USE_WITH_PROVENANCE:
            reasons.extend((f"memory_{boundary.decision.value}", *boundary.reasons))
        if self.quality.integrity != "intact":
            reasons.append(f"integrity_{self.quality.integrity}")
        if self.quality.authority == "unknown":
            reasons.append("authority_unknown")
        if self.quality.source_reliability is None:
            reasons.append("source_reliability_unknown")
        elif self.quality.source_reliability < self.policy.min_source_reliability:
            reasons.append("source_reliability_below_policy")
        if self.quality.compression_distortion is None:
            reasons.append("compression_distortion_unknown")
        elif self.quality.compression_distortion > self.policy.max_compression_distortion:
            reasons.append("compression_distortion_above_policy")
        if self.kind in {MemoryKind.PROCEDURE, MemoryKind.NORM}:
            reasons.append("nonmeasurement_memory_kind")
        if self.target_influence is MemoryInfluence.IGNORE:
            reasons.append("target_influence_ignore")
        return tuple(dict.fromkeys(reasons))

    @property
    def influence(self) -> MemoryInfluence:
        """Effective numeric influence only; target influence remains available for other uses."""
        return MemoryInfluence.IGNORE if self.limitations else self.target_influence

    def measurement_for_update(self) -> EpistemicMeasurement:
        """Return the unchanged eligible input or raise; this does not certify causal history."""
        current = replace(self)
        if current.limitations:
            raise ValueError("ineligible measurement: " + ", ".join(current.limitations))
        return current.measurement

    def summarize(self, summary: str, *, transformation_id: str) -> EvidenceUseAssessment:
        """Append an unverified display view without replacing any source or boundary metadata."""
        return replace(self, views=(*self.views, (transformation_id, summary)))

    def to_dict(self) -> dict[str, Any]:
        """Export a detached audit report, not an authenticated import or authorization token."""
        current = replace(self)
        quality = asdict(current.quality)
        quality["provenance"] = list(current.quality.provenance)
        return {
            "schema_version": "evidence-use-v1",
            "STATUS": current.status,
            "EVIDENCE": [ref.to_dict() for ref in current.measurement.evidence_refs],
            "LIMITATION": list(current.limitations),
            "NEXT_ACTION": "review_evidence" if current.limitations else "consider_measurement",
            "state": current.state.to_dict(),
            "claim_id": current.claim_id,
            "measurement": current.measurement.to_dict(),
            "memory": current.memory.to_dict(),
            "kind": current.kind.value,
            "target_influence": current.target_influence.value,
            "influence": current.influence.value,
            "quality": quality,
            "policy": asdict(current.policy),
            "assessed_at": current.assessed_at,
            "age_seconds": current.age_seconds,
            "views": [{"transformation_id": i, "summary": s} for i, s in current.views],
        }


def _memory_snapshot(memory: RetrievedMemory) -> RetrievedMemory:
    if type(memory) is not RetrievedMemory:
        raise ValueError("memory must be RetrievedMemory")
    for name in ("memory_id", "content_summary", "source_identity"):
        _text(getattr(memory, name), name)
    # Legacy memory construction permits truthy values; this numeric boundary requires booleans.
    for item in fields(memory):
        if item.type == "bool" and type(getattr(memory, item.name)) is not bool:
            raise ValueError(f"memory.{item.name} must be bool")
    provenance = memory.representation_provenance
    return replace(
        memory,
        representation_provenance=None if provenance is None else replace(provenance),
    )

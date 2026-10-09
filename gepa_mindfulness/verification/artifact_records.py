"""Bound artifact provenance and transformation ancestry, without memory or authority effects."""

# Standard library
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, TypeVar

# Third-party
# Local
from semantic_intent_robustness.memory_safety import (
    RepresentationMemoryProvenance,
    RetrievedMemory,
)
from semantic_intent_robustness.representation import (
    CandidateOutcome,
    RepresentationCandidate,
    RepresentationChannel,
    SourceSpan,
)

from ..core.evidence import EvidenceReference
from ..training.eligibility import TrainingEligibility
from .debate_records import _digest, _record
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    choice,
    public_refs,
    restore_records,
    restore_refs,
    strings,
)
from .evidence_use import EvidenceQuality, _memory_snapshot
from .state import ArtifactObservation, EvidenceClaim, EvidenceState, parse_rfc3339_datetime

ArtifactKey = tuple[str, str]
ADMISSIONS = (
    TrainingEligibility.DEVELOPMENT,
    TrainingEligibility.REGRESSION,
    TrainingEligibility.HIDDEN_EVAL,
)
R = TypeVar("R", bound=DiagnosticRecord)


def json_value(value: Any) -> Any:
    """Encode composed legacy dataclasses while respecting their existing wire formats."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    return value


def payload_digest(value: Any) -> str:
    """Hash exact JSON contents; a digest does not authenticate them."""
    encoded = json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def artifact_digest(record: DiagnosticRecord) -> str:
    """Bind complete revalidated record contents."""
    if not isinstance(record, DiagnosticRecord):
        raise ValueError("expected diagnostic record")
    return payload_digest(record.to_dict())


class ArtifactDiagnosticRecord(DiagnosticRecord):
    """Exact diagnostic serialization for composed legacy evidence and memory dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Revalidate and detach nested values, including legacy quality and memory records."""
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "training_eligibility": "DEVELOPMENT",
            **{f.name: json_value(getattr(self, f.name)) for f in fields(self)},
        }


def _mapping(value: object, cls: Any, optional: tuple[str, ...] = ()) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} requires an object")
    names = {f.name for f in fields(cls)}
    if not names - set(optional) <= set(value) <= names:
        raise ValueError(f"incorrect {cls.__name__} fields")
    return dict(value)


def _tuple(value: object) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("expected array")
    return tuple(value)


def artifact_key(value: object) -> ArtifactKey:
    """Validate one exact artifact/version key, never a name-based alias."""
    values = _tuple(value)
    if len(values) != 2:
        raise ValueError("artifact key needs ID and version")
    for v in values:
        _text(v, "artifact key")
    return values[0], values[1]


def record_tuple(value: object, cls: type[R]) -> tuple[R, ...]:
    """Detach an exact tuple of known record types."""
    if type(value) is not tuple:
        raise ValueError("records require a tuple")
    return tuple(_record(v, cls) for v in value)


def unique(values: list[Any], name: str) -> None:
    """Reject duplicate identities rather than silently replacing records."""
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")


def _restore_quality(data: object) -> EvidenceQuality:
    values = _mapping(data, EvidenceQuality)
    values["provenance"] = _tuple(values["provenance"])
    return EvidenceQuality(**values)


def restore_memory(data: object) -> RetrievedMemory:
    """Restore the original memory schema, including nested representation provenance."""
    values = _mapping(
        data, RetrievedMemory, ("representation_derived", "representation_provenance")
    )
    provenance = values.get("representation_provenance")
    if provenance is not None:
        p = _mapping(provenance, RepresentationMemoryProvenance)
        c = _mapping(p["candidate"], RepresentationCandidate)
        c["source_span"] = SourceSpan(**_mapping(c["source_span"], SourceSpan))
        c["transform_channel"] = RepresentationChannel(c["transform_channel"])
        c["outcome"] = CandidateOutcome(c["outcome"])
        c["provenance"] = _tuple(c["provenance"])
        p["candidate"] = RepresentationCandidate(**c)
        p["transform_provenance"] = _tuple(p["transform_provenance"])
        values["representation_provenance"] = RepresentationMemoryProvenance(**p)
    return _memory_snapshot(RetrievedMemory(**values))


def _claim_memory(owner: Any) -> None:
    if type(owner.claim) is not EvidenceClaim or type(owner.memory) is not RetrievedMemory:
        raise ValueError("exact claim and memory required")
    claim = EvidenceClaim.from_dict(owner.claim.to_dict())
    memory = restore_memory(_memory_snapshot(owner.memory).to_dict())
    if (memory.memory_id, memory.content_summary) != (claim.claim_id, claim.proposition):
        raise ValueError("memory identity/content must match claim")
    object.__setattr__(owner, "claim", claim)
    object.__setattr__(owner, "memory", memory)
    object.__setattr__(owner, "entity_ids", strings(owner.entity_ids, "entity_ids", required=True))
    public_refs(claim.evidence_refs)
    _text(owner.item_id, "item_id")


@dataclass(frozen=True)
class ArtifactLocation(ArtifactDiagnosticRecord):
    """Host locator of a quotation or observation; bytes are not independently inspected."""

    kind: str
    value: str
    schema_version = "artifact-location-v1"

    def __post_init__(self) -> None:
        choice(self.kind, "kind", ("page", "section", "span", "observation"))
        _text(self.value, "value")


@dataclass(frozen=True)
class ArtifactContribution(ArtifactDiagnosticRecord):
    """Previously declared contribution with retained evaluator provenance, not certification."""

    task_id: str
    description: str
    evaluator_refs: tuple[EvidenceReference, ...]
    schema_version = "artifact-contribution-v1"
    restorers = {"evaluator_refs": restore_refs}

    def __post_init__(self) -> None:
        _text(self.task_id, "task_id")
        _text(self.description, "description")
        object.__setattr__(self, "evaluator_refs", public_refs(self.evaluator_refs, required=True))


@dataclass(frozen=True)
class ArtifactRecord(ArtifactDiagnosticRecord):
    """One exact artifact version and its source restrictions."""

    artifact_id: str
    version: str
    observation: ArtifactObservation
    entity_ids: tuple[str, ...]
    availability: str
    source_training_eligibility: TrainingEligibility
    schema_version = "artifact-record-v1"
    restorers = {
        "observation": ArtifactObservation.from_dict,
        "source_training_eligibility": TrainingEligibility,
    }

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.version, "version")
        if type(self.observation) is not ArtifactObservation:
            raise ValueError("exact ArtifactObservation required")
        object.__setattr__(
            self, "observation", ArtifactObservation.from_dict(self.observation.to_dict())
        )
        public_refs(self.observation.evidence_refs, required=True)
        object.__setattr__(
            self, "entity_ids", strings(self.entity_ids, "entity_ids", required=True)
        )
        choice(self.availability, "availability", ("available", "restricted", "deleted", "unknown"))
        if type(self.source_training_eligibility) is not TrainingEligibility or (
            self.source_training_eligibility not in ADMISSIONS
        ):
            raise ValueError("artifact source requires non-training admission")


@dataclass(frozen=True)
class SourceFragment(ArtifactDiagnosticRecord):
    """A source quotation, claim and memory remain separately inspectable."""

    item_id: str
    artifact_key: ArtifactKey
    artifact_digest: str
    entity_ids: tuple[str, ...]
    location: ArtifactLocation
    quotation: str
    claim: EvidenceClaim
    memory: RetrievedMemory
    quality: EvidenceQuality
    contributions: tuple[ArtifactContribution, ...]
    schema_version = "source-fragment-v1"
    restorers = {
        "artifact_key": lambda v: artifact_key(v),
        "location": ArtifactLocation.from_dict,
        "claim": EvidenceClaim.from_dict,
        "memory": restore_memory,
        "quality": _restore_quality,
        "contributions": lambda v: restore_records(v, ArtifactContribution),
    }

    def __post_init__(self) -> None:
        _claim_memory(self)
        object.__setattr__(self, "artifact_key", artifact_key(self.artifact_key))
        _digest(self.artifact_digest)
        object.__setattr__(self, "location", _record(self.location, ArtifactLocation))
        _text(self.quotation, "quotation")
        if type(self.quality) is not EvidenceQuality:
            raise ValueError("exact EvidenceQuality required")
        object.__setattr__(self, "quality", _restore_quality(json_value(self.quality)))
        object.__setattr__(
            self, "contributions", record_tuple(self.contributions, ArtifactContribution)
        )


@dataclass(frozen=True)
class DerivedInterpretation(ArtifactDiagnosticRecord):
    """Generated interpretation bound to all original inputs; its claim starts unverified."""

    item_id: str
    entity_ids: tuple[str, ...]
    claim: EvidenceClaim
    memory: RetrievedMemory
    inputs: tuple[tuple[str, str], ...]
    transform_id: str
    transform_version: str
    created_at: str
    output_digest: str
    schema_version = "derived-interpretation-v1"
    restorers = {
        "claim": EvidenceClaim.from_dict,
        "memory": restore_memory,
        "inputs": lambda v: tuple(_tuple(x) for x in _tuple(v)),
    }

    def __post_init__(self) -> None:
        _claim_memory(self)
        if self.claim.status != "unverified":
            raise ValueError("derived claim must remain unverified")
        if type(self.inputs) is not tuple or not self.inputs or len(self.inputs) > 256:
            raise ValueError("interpretation requires 1..256 input bindings")
        for item in self.inputs:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("input needs item ID and digest")
            _text(item[0], "input ID")
            _digest(item[1])
            if item[0] == self.item_id:
                raise ValueError("self ancestry")
        unique([item[0] for item in self.inputs], "input IDs")
        for name in ("transform_id", "transform_version"):
            _text(getattr(self, name), name)
        parse_rfc3339_datetime(self.created_at, "created_at")
        _digest(self.output_digest)
        if self.output_digest != payload_digest(self.claim.to_dict()):
            raise ValueError("output digest differs from claim")


@dataclass(frozen=True)
class ArtifactSnapshot(ArtifactDiagnosticRecord):
    """Bounded host-only source roster with validated claim and transformation lineage."""

    entity_ids: tuple[str, ...]
    artifacts: tuple[ArtifactRecord, ...]
    sources: tuple[SourceFragment, ...]
    interpretations: tuple[DerivedInterpretation, ...]
    state: EvidenceState
    schema_version = "artifact-snapshot-v1"
    restorers = {
        "artifacts": lambda v: restore_records(v, ArtifactRecord),
        "sources": lambda v: restore_records(v, SourceFragment),
        "interpretations": lambda v: restore_records(v, DerivedInterpretation),
        "state": EvidenceState.from_dict,
    }

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_ids", strings(self.entity_ids, "entity_ids", required=True)
        )
        for name, cls in (
            ("artifacts", ArtifactRecord),
            ("sources", SourceFragment),
            ("interpretations", DerivedInterpretation),
        ):
            object.__setattr__(self, name, record_tuple(getattr(self, name), cls))
        if type(self.state) is not EvidenceState:
            raise ValueError("exact EvidenceState required")
        object.__setattr__(self, "state", EvidenceState.from_dict(self.state.to_dict()))
        if len(self.artifacts) > 64 or len(self.sources) + len(self.interpretations) > 256:
            raise ValueError("artifact/item roster exceeds bounds")
        if len(self.state.claims) > 256:
            raise ValueError("claim roster exceeds bounds")
        unique([(a.artifact_id, a.version) for a in self.artifacts], "artifact versions")
        unique([s.item_id for s in self.sources + self.interpretations], "item IDs")
        unique([(d.transform_id, d.transform_version) for d in self.interpretations], "transforms")
        artifacts = {(a.artifact_id, a.version): a for a in self.artifacts}
        claims = {c.claim_id: c for c in self.state.claims}
        for a in self.artifacts:
            if not set(a.entity_ids) <= set(self.entity_ids):
                raise ValueError("unknown artifact entity")
        for item in self.sources + self.interpretations:
            if claims.get(item.claim.claim_id) != item.claim:
                raise ValueError("item claim differs from state")
            if not set(item.entity_ids) <= set(self.entity_ids):
                raise ValueError("unknown item entity")
        for source in self.sources:
            artifact = artifacts.get(source.artifact_key)
            if artifact is None or source.artifact_digest != artifact.observation.digest:
                raise ValueError("source artifact/digest mismatch")
            if not set(source.entity_ids) <= set(artifact.entity_ids):
                raise ValueError("source entity differs from artifact")
            if not set(source.claim.evidence_refs) <= set(artifact.observation.evidence_refs):
                raise ValueError("source claim references differ from artifact")
            if source.quality.recorded_at != artifact.observation.observed_at:
                raise ValueError("source time differs from original observation")
        kinds: dict[str, Any] = {}
        refs = [r for a in self.artifacts for r in a.observation.evidence_refs]
        refs += [r for c in self.state.claims for r in c.evidence_refs]
        for ref in refs:
            if ref.reference_id in kinds and kinds[ref.reference_id] != ref.source_kind:
                raise ValueError("ambiguous source kind")
            kinds[ref.reference_id] = ref.source_kind
        _lineage(self)


def _lineage(snapshot: ArtifactSnapshot) -> dict[str, tuple[str, ...]]:
    items = {s.item_id: s for s in snapshot.sources + snapshot.interpretations}
    ancestry: dict[str, tuple[str, ...]] = {s.item_id: (s.item_id,) for s in snapshot.sources}
    active: set[str] = set()
    digests = {key: artifact_digest(item) for key, item in items.items()}

    def visit(key: str) -> tuple[str, ...]:
        if key in ancestry:
            return ancestry[key]
        if key in active:
            raise ValueError("cyclic transformation lineage")
        if key not in items:
            raise ValueError("unknown lineage input")
        item = items[key]
        if not isinstance(item, DerivedInterpretation):
            raise ValueError("unknown source")
        active.add(key)
        refs: set[EvidenceReference] = set()
        entities: set[str] = set()
        sources: set[str] = set()
        created = parse_rfc3339_datetime(item.created_at, "created_at")
        for parent_id, digest in item.inputs:
            sources.update(visit(parent_id))
            if digests[parent_id] != digest:
                raise ValueError("input digest mismatch")
            parent = items[parent_id]
            refs.update(parent.claim.evidence_refs)
            entities.update(parent.entity_ids)
            time = (
                parent.quality.recorded_at
                if isinstance(parent, SourceFragment)
                else parent.created_at
            )
            if created < parse_rfc3339_datetime(time, "input time"):
                raise ValueError("transformation predates input")
        if set(item.claim.evidence_refs) != refs or set(item.entity_ids) != entities:
            raise ValueError("interpretation must preserve input evidence/entities")
        active.remove(key)
        ancestry[key] = tuple(sorted(sources))
        return ancestry[key]

    for key in items:
        visit(key)
    return ancestry


def source_ancestors(snapshot: ArtifactSnapshot, item_id: str) -> tuple[str, ...]:
    """Return original source IDs after revalidating the complete host snapshot."""
    snapshot = _record(snapshot, ArtifactSnapshot)
    ancestors = _lineage(snapshot)
    if item_id not in ancestors:
        raise ValueError("unknown item ID")
    return ancestors[item_id]

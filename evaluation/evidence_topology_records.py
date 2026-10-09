"""Exact host protocols and independent receipts for opt-in artifact diagnostics."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.artifact_evidence import ArtifactQuery
from gepa_mindfulness.verification.artifact_records import (
    ArtifactDiagnosticRecord,
    ArtifactSnapshot,
    artifact_digest,
    record_tuple,
    unique,
)
from gepa_mindfulness.verification.artifact_topology import EvidenceTopology, validate_topology
from gepa_mindfulness.verification.debate_records import _digest, _record
from gepa_mindfulness.verification.diagnostic_records import (
    _text,
    choice,
    public_refs,
    restore_records,
    restore_refs,
    strings,
)

from .causal_records import CausalVariant, MetricVerdict, canonical_json
from .debate_records import _identity, _restore_identity
from .ladder import Severity
from .pluralistic_records import _verdicts

TOPOLOGY_METRICS = (
    "correctness",
    "missing_evidence_failure",
    "stale_exposure",
    "source_attribution",
    "entity_error",
    "denied_content_exposure",
)
CONDITIONS = ("existing_retrieval", "artifact_index", "artifact_topology")


@dataclass(frozen=True)
class TopologyOpportunity(ArtifactDiagnosticRecord):
    """One predeclared semantic metric and its host-supplied severity/cohort."""

    opportunity_id: str
    metric: str
    severity: Severity
    cohort: str
    schema_version = "topology-opportunity-v1"
    restorers = {"severity": Severity}

    def __post_init__(self) -> None:
        _text(self.opportunity_id, "opportunity_id")
        _text(self.cohort, "cohort")
        choice(self.metric, "metric", TOPOLOGY_METRICS)
        if type(self.severity) is not Severity:
            raise ValueError("severity requires Severity")


@dataclass(frozen=True)
class TopologyProtocol(ArtifactDiagnosticRecord):
    """Complete public subject, evidence and query binding; expected actions stay private."""

    protocol_id: str
    rubric_id: str
    subject: CausalVariant
    snapshot: ArtifactSnapshot
    topology: EvidenceTopology
    query: ArtifactQuery
    opportunities: tuple[TopologyOpportunity, ...]
    evaluator: TrustedEvaluatorContract
    schema_version = "topology-protocol-v1"
    restorers = {
        "subject": CausalVariant.from_dict,
        "snapshot": ArtifactSnapshot.from_dict,
        "topology": EvidenceTopology.from_dict,
        "query": ArtifactQuery.from_dict,
        "opportunities": lambda v: restore_records(v, TopologyOpportunity),
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
    }

    def __post_init__(self) -> None:
        _text(self.protocol_id, "protocol_id")
        _text(self.rubric_id, "rubric_id")
        if type(self.subject) is not CausalVariant:
            raise ValueError("subject requires exact CausalVariant")
        object.__setattr__(self, "subject", CausalVariant.from_dict(self.subject.to_dict()))
        for name, cls in (
            ("snapshot", ArtifactSnapshot),
            ("topology", EvidenceTopology),
            ("query", ArtifactQuery),
        ):
            object.__setattr__(self, name, _record(getattr(self, name), cls))
        object.__setattr__(
            self, "opportunities", record_tuple(self.opportunities, TopologyOpportunity)
        )
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        unique([o.opportunity_id for o in self.opportunities], "opportunity IDs")
        unique([o.metric for o in self.opportunities], "opportunity metrics")
        validate_topology(self.snapshot, self.topology)
        if self.query.public_query != canonical_json([t.to_dict() for t in self.subject.turns]):
            raise ValueError("query must bind exact public turns")
        keys = {(a.artifact_id, a.version) for a in self.snapshot.artifacts}
        if (
            not set(self.query.entity_ids) <= set(self.snapshot.entity_ids)
            or not (set(self.query.artifact_keys) | set(self.query.excluded_artifacts)) <= keys
        ):
            raise ValueError("query references unknown entities or artifact versions")


def _measurement(value: float | None, name: str) -> None:
    if value is not None and (type(value) not in (float, int) or not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite nonnegative measurement or None")


@dataclass(frozen=True)
class TopologyCapture(ArtifactDiagnosticRecord):
    """Host-observed IDs and output, never a saved permission to replay source content."""

    protocol_digest: str
    condition: str
    status: str
    retrieved_item_ids: tuple[str, ...]
    attributed_item_ids: tuple[str, ...]
    response: str | None
    actions: tuple[str, ...]
    latency_seconds: float | None
    retrieval_cost: float | None
    cost_unit: str | None
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str
    schema_version = "topology-capture-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        _digest(self.protocol_digest)
        choice(self.condition, "condition", CONDITIONS)
        choice(self.status, "status", ("observed", "censored"))
        for name in ("retrieved_item_ids", "attributed_item_ids", "actions"):
            object.__setattr__(self, name, strings(getattr(self, name), name))
        if not set(self.attributed_item_ids) <= set(self.retrieved_item_ids):
            raise ValueError("attribution requires retrieved items")
        if len(self.retrieved_item_ids) > 256:
            raise ValueError("too many retrieved items")
        if self.status == "observed":
            _text(self.response, "response")
        elif self.response is not None or self.actions or self.retrieved_item_ids:
            raise ValueError("censored capture cannot contain observed output")
        _measurement(self.latency_seconds, "latency_seconds")
        _measurement(self.retrieval_cost, "retrieval_cost")
        if (self.retrieval_cost is None) != (self.cost_unit is None):
            raise ValueError("cost and unit must both be present or absent")
        if self.cost_unit is not None:
            _text(self.cost_unit, "cost_unit")
        object.__setattr__(
            self,
            "evidence_refs",
            public_refs(self.evidence_refs, required=self.status == "observed"),
        )
        _text(self.reason, "reason")


@dataclass(frozen=True)
class ClaimSupportVerdict(ArtifactDiagnosticRecord):
    """Independent sufficiency of named routes, bound to the full protocol and claim."""

    claim_id: str
    supported: bool | None
    contradictions_resolved: bool | None
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str
    sufficient_route_ids: tuple[str, ...] = ()
    schema_version = "claim-support-verdict-v1"
    restorers = {"evidence_refs": restore_refs}

    def __post_init__(self) -> None:
        _text(self.claim_id, "claim_id")
        object.__setattr__(
            self, "sufficient_route_ids", strings(self.sufficient_route_ids, "sufficient routes")
        )
        if len(self.sufficient_route_ids) > 64:
            raise ValueError("too many sufficient routes")
        for v in (self.supported, self.contradictions_resolved):
            if v is not None and type(v) is not bool:
                raise ValueError("claim judgments require boolean or None")
        object.__setattr__(self, "evidence_refs", public_refs(self.evidence_refs, required=True))
        _text(self.reason, "reason")


@dataclass(frozen=True)
class TopologyAssessment(ArtifactDiagnosticRecord):
    """A fully bound receipt requiring fresh independent authentication on each use."""

    protocol_digest: str
    capture_digest: str
    evaluator: TrustedEvaluatorContract
    status: str
    human_required: bool
    verdicts: tuple[MetricVerdict, ...]
    claim_verdicts: tuple[ClaimSupportVerdict, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str
    schema_version = "topology-assessment-v1"
    restorers = {
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
        "verdicts": lambda v: tuple(MetricVerdict.from_dict(x) for x in v),
        "claim_verdicts": lambda v: restore_records(v, ClaimSupportVerdict),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        _digest(self.protocol_digest)
        _digest(self.capture_digest)
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        choice(self.status, "status", ("verified", "unresolved", "disputed"))
        if type(self.human_required) is not bool:
            raise ValueError("human_required requires boolean")
        object.__setattr__(self, "verdicts", _verdicts(self.verdicts))
        object.__setattr__(
            self, "claim_verdicts", record_tuple(self.claim_verdicts, ClaimSupportVerdict)
        )
        unique([v.claim_id for v in self.claim_verdicts], "claim verdict IDs")
        if len(self.verdicts) > 6 or len(self.claim_verdicts) > 256:
            raise ValueError("assessment exceeds metric or claim bounds")
        object.__setattr__(
            self,
            "evidence_refs",
            public_refs(self.evidence_refs, required=self.status == "verified"),
        )
        _text(self.reason, "reason")


def topology_protocol_digest(protocol: TopologyProtocol) -> str:
    """Bind exact subject, snapshot, topology, scope, policy, rubric and evaluator."""
    return artifact_digest(_record(protocol, TopologyProtocol))

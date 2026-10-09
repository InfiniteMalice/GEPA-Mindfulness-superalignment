"""Provenance-bound independent judgments for public stakeholder diagnostics."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass

# Third-party
# Local
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.verification.debate_records import _digest, _record
from gepa_mindfulness.verification.diagnostic_records import (
    _text,
    choice,
    public_refs,
    restore_records,
    restore_refs,
)
from gepa_mindfulness.verification.state import EvidenceClaim
from semantic_intent_robustness.perspective_protocol import (
    PerspectivePlan,
    PerspectiveSource,
    RoleAssignment,
    _items,
    _unique,
)

from .causal_records import CausalPair, CausalVariant, MetricVerdict, canonical_json, content_digest
from .debate_records import DebateRecord, _identity, _restore_identity
from .ladder import Severity

PLURALISTIC_METRICS = (
    "social_sycophancy",
    "overcritical_response",
    "perspective_robustness",
    "legitimate_third_party_interest",
)


@dataclass(frozen=True)
class PluralisticOpportunity(DebateRecord):
    """One predeclared pair-level semantic measurement, not an optimization signal."""

    opportunity_id: str
    metric: str
    severity: Severity
    cohort: str
    schema_version = "pluralistic-opportunity-v1"
    restorers = {"severity": Severity}

    def __post_init__(self) -> None:
        _text(self.opportunity_id, "opportunity_id")
        _text(self.cohort, "cohort")
        choice(self.metric, "metric", PLURALISTIC_METRICS)
        if type(self.severity) is not Severity:
            raise ValueError("severity requires Severity")


@dataclass(frozen=True)
class PluralisticProtocol(DebateRecord):
    """Exact pair, public stakeholder plan and declared semantic evaluator."""

    protocol_id: str
    rubric_id: str
    pair: CausalPair
    plan: PerspectivePlan
    opportunities: tuple[PluralisticOpportunity, ...]
    evaluator: TrustedEvaluatorContract
    schema_version = "pluralistic-protocol-v1"
    restorers = {
        "pair": CausalPair.from_dict,
        "plan": PerspectivePlan.from_dict,
        "opportunities": lambda v: restore_records(v, PluralisticOpportunity),
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
    }

    def __post_init__(self) -> None:
        _text(self.protocol_id, "protocol_id")
        _text(self.rubric_id, "rubric_id")
        if type(self.pair) is not CausalPair:
            raise ValueError("pair requires CausalPair")
        object.__setattr__(self, "pair", CausalPair.from_dict(self.pair.to_dict()))
        object.__setattr__(self, "plan", _record(self.plan, PerspectivePlan))
        object.__setattr__(
            self, "opportunities", _items(self.opportunities, PluralisticOpportunity)
        )
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        _unique([o.opportunity_id for o in self.opportunities], "opportunity IDs")
        _unique([o.metric for o in self.opportunities], "metrics per pair")
        source = self.plan.source
        if source.variant_id != self.pair.after.variant_id or source.source_text != canonical_json(
            [t.to_dict() for t in self.pair.after.turns]
        ):
            raise ValueError("plan source does not bind the after variant's public turns")
        admissions = (
            TrainingEligibility.DEVELOPMENT,
            TrainingEligibility.REGRESSION,
            TrainingEligibility.HIDDEN_EVAL,
        )
        if admissions.index(source.source_training_eligibility) < admissions.index(
            self.pair.training_eligibility
        ):
            raise ValueError("plan source cannot weaken pair admission")


def _verdicts(values: tuple[MetricVerdict, ...]) -> tuple[MetricVerdict, ...]:
    if type(values) is not tuple or any(type(v) is not MetricVerdict for v in values):
        raise ValueError("requires exact MetricVerdict tuple")
    result = tuple(MetricVerdict.from_dict(v.to_dict()) for v in values)
    _unique([v.opportunity_id for v in result], "verdict IDs")
    return result


@dataclass(frozen=True)
class PluralisticAssessment(DebateRecord):
    """A complete receipt requiring fresh host authentication on every analysis."""

    protocol_digest: str
    capture_digest: str
    perspective_capture_digest: str
    evaluator: TrustedEvaluatorContract
    status: str
    human_required: bool
    verdicts: tuple[MetricVerdict, ...]
    claim_verdicts: tuple[MetricVerdict, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    reason: str
    schema_version = "pluralistic-assessment-v1"
    restorers = {
        "evaluator": lambda v: _restore_identity(v, TrustedEvaluatorContract),
        "verdicts": lambda v: tuple(MetricVerdict.from_dict(x) for x in v),
        "claim_verdicts": lambda v: tuple(MetricVerdict.from_dict(x) for x in v),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        for digest in (self.protocol_digest, self.capture_digest, self.perspective_capture_digest):
            _digest(digest)
        object.__setattr__(self, "evaluator", _identity(self.evaluator, TrustedEvaluatorContract))
        choice(self.status, "status", ("verified", "unresolved", "disputed"))
        if type(self.human_required) is not bool:
            raise ValueError("human_required requires boolean")
        for name in ("verdicts", "claim_verdicts"):
            object.__setattr__(self, name, _verdicts(getattr(self, name)))
        object.__setattr__(
            self,
            "evidence_refs",
            public_refs(self.evidence_refs, required=self.status == "verified"),
        )
        _text(self.reason, "reason")


def pluralistic_protocol_digest(protocol: PluralisticProtocol) -> str:
    """Bind the full pair, roster, assertions, rubric and semantic evaluator."""
    return content_digest(_record(protocol, PluralisticProtocol).to_dict())


def source_from_variant(
    variant: CausalVariant,
    *,
    semantic_core_id: str,
    facts: tuple[EvidenceClaim, ...],
    constraints: tuple[EvidenceClaim, ...],
    roles: tuple[RoleAssignment, ...],
    source_refs: tuple[EvidenceReference, ...],
    source_training_eligibility: TrainingEligibility,
) -> PerspectiveSource:
    """Copy only public turns; expected outcomes remain evaluator-only."""
    if type(variant) is not CausalVariant:
        raise ValueError("variant requires CausalVariant")
    variant = CausalVariant.from_dict(variant.to_dict())
    return PerspectiveSource(
        variant.variant_id,
        semantic_core_id,
        canonical_json([t.to_dict() for t in variant.turns]),
        facts,
        constraints,
        roles,
        source_refs,
        source_training_eligibility,
    )

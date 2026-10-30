"""Default-off, evidence-reviewed advice over existing controlled-improvement intake."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace

# Third-party
# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_records import ImprovementRecord, _encode, record_digest
from gepa_mindfulness.coevolution import CoevolutionStore, CorrectionProposal
from gepa_mindfulness.controlled_improvement import TriageDiagnostics, assess_improvement
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.learning_surfaces import (
    LearningSurface,
    LessonCharacteristics,
    LessonKind,
    classify_learning_surface,
)
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from gepa_mindfulness.verification.epistemic_state import EpistemicStateEstimate

PROCEDURES = frozenset(
    (
        "context_compaction",
        "retrieval_sequencing",
        "tool_selection",
        "review_routing",
        "task_decomposition",
        "reasoning_budget",
        "recovery",
        "non_authoritative_cache",
        "diagnostic_instrumentation",
    )
)
_DESTINATIONS = {
    LearningSurface.TRACE_ONLY: "no_justified_change",
    LearningSurface.MEMORY: "memory_repair",
    LearningSurface.HARNESS: "harness_procedure",
    LearningSurface.SKILL_GRAPH: "skill_refinement",
    LearningSurface.MODEL: "model_correction",
    LearningSurface.HUMAN: "human_review",
}


def _refs(values: tuple[EvidenceReference, ...]) -> None:
    if not 1 <= len(values) <= 32 or len(set(values)) != len(values):
        raise ValueError("requires bounded unique public evidence references")
    if any(len(v.reference_id) > 256 for v in values):
        raise ValueError("evidence identity exceeds limit")


@dataclass(frozen=True, kw_only=True)
class _WorkflowRecord(ImprovementRecord):
    training_eligibility: str = "DEVELOPMENT"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.training_eligibility != "DEVELOPMENT"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("workflow records confer no optimizer or runtime authority")
        validate_input_envelope(_encode(self))


@dataclass(frozen=True)
class WorkflowDiagnosis(_WorkflowRecord):
    """A host or model hypothesis, not a self-certified diagnosis or change permission."""

    proposal_digest: str
    characteristics: LessonCharacteristics
    evidence_refs: tuple[EvidenceReference, ...]
    selector_version: str
    expected_effect: str
    preservation_requirements: tuple[str, ...]
    dataset_defect: bool = False
    no_justified_change: bool = False
    procedure: str | None = None
    schema_version: str = "workflow-diagnosis-v1"

    def _validate(self) -> None:
        if self.schema_version != "workflow-diagnosis-v1":
            raise ValueError("unsupported diagnosis schema")
        _refs(self.evidence_refs)
        if len(self.selector_version) > 256 or len(self.expected_effect) > 4096:
            raise ValueError("diagnosis text exceeds limit")
        if not 1 <= len(self.preservation_requirements) <= 16 or any(
            len(v) > 512 for v in self.preservation_requirements
        ):
            raise ValueError("bounded preservation requirements required")
        if self.dataset_defect and self.no_justified_change:
            raise ValueError("conflicting diagnosis alternatives")
        harness = self.characteristics.kind is LessonKind.STABLE_PROCEDURAL_CONVENTION
        if (harness and self.procedure not in PROCEDURES) or (
            not harness and self.procedure is not None
        ):
            raise ValueError("harness diagnosis requires an eligible procedure only")


@dataclass(frozen=True)
class WorkflowReview(_WorkflowRecord):
    """Host-configured independent review; its identity declaration is not authentication."""

    diagnosis_digest: str
    evaluator: TrustedEvaluatorContract
    review_id: str
    status: str
    evidence_refs: tuple[EvidenceReference, ...]
    schema_version: str = "workflow-review-v1"

    def _validate(self) -> None:
        if self.schema_version != "workflow-review-v1":
            raise ValueError("unsupported workflow review schema")
        if self.status not in ("supported", "rejected", "uncertain"):
            raise ValueError("invalid diagnosis review status")
        if len(self.review_id) > 256 or any(
            len(getattr(self.evaluator, f.name)) > 256 for f in fields(self.evaluator)
        ):
            raise ValueError("review identity exceeds limit")
        _refs(self.evidence_refs)


def select_improvement_workflow(
    store: CoevolutionStore,
    correction: CorrectionProposal,
    estimate: EpistemicStateEstimate,
    triage: TriageDiagnostics,
    diagnosis: WorkflowDiagnosis,
    *,
    evaluator: TrustedEvaluatorContract | None = None,
    review: Callable[[WorkflowDiagnosis], WorkflowReview] | None = None,
    enabled: bool = False,
) -> dict:
    """Recommend review work, never execute or register an intervention.

    The host authenticates the independent reviewer, protects the catalog and isolates trials.
    Typed diagnosis facts require a supported review and the existing intake's sandbox route.
    Human or unresolved intake routes cannot be overruled by review agreement. All outputs are
    DEVELOPMENT, not optimizer feedback or an acceptance decision. Callbacks are trusted host
    integrations, not a Python sandbox; exceptions return only a fixed failure category.
    """
    if enabled is not True:
        raise ValueError("workflow selection requires enabled=True")
    if type(diagnosis) is not WorkflowDiagnosis or type(correction) is not CorrectionProposal:
        raise ValueError("exact diagnosis and correction required")
    request = WorkflowDiagnosis.from_dict(WorkflowDiagnosis.to_dict(diagnosis))
    proposal = CorrectionProposal.from_dict(CorrectionProposal.to_dict(correction))
    # Check identity before touching registered source evidence.
    if request.proposal_digest != content_digest(CorrectionProposal.to_dict(proposal)):
        raise ValueError("diagnosis must bind the exact correction proposal")
    if type(evaluator) is not TrustedEvaluatorContract or not callable(review):
        raise ValueError("independent host evaluator and review callback required")
    evaluator = replace(evaluator)
    if any(
        type(getattr(evaluator, f.name)) is not str or len(getattr(evaluator, f.name)) > 256
        for f in fields(evaluator)
    ):
        raise ValueError("invalid evaluator identity")
    node = next(
        n for n in proposal.failure_graph.nodes if n.failure_id == proposal.localized_failure_id
    )
    if not set(request.evidence_refs).issubset(node.evidence_refs):
        raise ValueError("diagnosis evidence must name the localized failure")
    assessment = assess_improvement(store, proposal, estimate, triage, enabled=True)
    validate_input_envelope(assessment)
    surface = classify_learning_surface(request.characteristics)
    result = {
        "schema_version": "improvement-workflow-v1",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "execute_candidate": False,
        "proposal_id": proposal.proposal_id,
        "source_trajectory_id": proposal.source_trajectory_id,
        "source_epoch_id": proposal.source_epoch_id,
        "diagnosis": request.to_dict(),
        "learning_surface": surface.value,
        "intake_route": assessment["route"],
        "intake_reasons": assessment["reasons"],
        "independent_review": None,
        "review_failure": None,
        "workflow": "further_verification",
    }
    # Reserve the complete review record, including worst-case JSON escaping, before callback.
    validate_input_envelope(dict(result, review_capacity="x" * 131072))
    if surface is LearningSurface.HUMAN or assessment["route"] == "human_review":
        result["workflow"] = "human_review"
        return result
    if assessment["route"] != "sandbox_review":
        return result
    digest = record_digest(request)
    try:
        detached = WorkflowDiagnosis.from_dict(request.to_dict())
        candidate = review(detached)
        if record_digest(detached) != digest or type(candidate) is not WorkflowReview:
            raise ValueError("invalid or mutated review")
        checked = WorkflowReview.from_dict(WorkflowReview.to_dict(candidate))
        if checked.diagnosis_digest != digest or checked.evaluator != evaluator:
            raise ValueError("review binding mismatch")
    except Exception:
        result["review_failure"] = "invalid_review"
        return result
    result["independent_review"] = checked.to_dict()
    if checked.status == "supported":
        result["workflow"] = (
            "no_justified_change"
            if request.no_justified_change
            else "data_repair" if request.dataset_defect else _DESTINATIONS[surface]
        )
    validate_input_envelope(result)
    return result

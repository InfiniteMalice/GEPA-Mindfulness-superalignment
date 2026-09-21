"""Disabled-by-default diagnostic integration for the semantic intent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mindful_trace_gepa.logging_schema import EventEnvelope

from ._continuity_validation import boolean
from .epistemic_continuity import (
    EpistemicContinuityAssessment,
    HistoricalSupport,
    assess_epistemic_continuity,
    recall_historical_support,
)
from .epistemic_records import CommitmentUpdate, EpistemicCommitment
from .internal_state_trajectory import SoTStateSnapshot
from .motivated_forgetting import (
    DirectionalPressure,
    MotivatedForgettingAssessment,
    assess_motivated_forgetting,
)
from .schemas import SemanticSafetyRecord
from .semantic_state_continuity import (
    SemanticStateContinuityAssessment,
    assess_semantic_state_continuity,
)


@dataclass(frozen=True, slots=True)
class DiagnosticFeature:
    """Explicit experimental switch; enforcement modes are not supported."""

    enabled: bool = False
    maturity: str = "research"

    def __post_init__(self) -> None:
        """Reject invalid diagnostic switches, maturity labels and dependencies."""
        boolean(self.enabled, "enabled")
        if self.maturity not in {"research", "shadow"}:
            raise ValueError("continuity features support research/shadow diagnostics only")


@dataclass(frozen=True, slots=True)
class ContinuityConfig:
    """Four independent opt-ins with explicit dependencies and fixed maturity boundaries."""

    state_of_thought: DiagnosticFeature = field(default_factory=DiagnosticFeature)
    semantic_state_continuity: DiagnosticFeature = field(
        default_factory=lambda: DiagnosticFeature(maturity="shadow")
    )
    epistemic_continuity: DiagnosticFeature = field(
        default_factory=lambda: DiagnosticFeature(maturity="shadow")
    )
    motivated_forgetting: DiagnosticFeature = field(default_factory=DiagnosticFeature)

    def __post_init__(self) -> None:
        """Reject invalid diagnostic switches, maturity labels and dependencies."""
        for feature, maturity in (
            (self.state_of_thought, "research"),
            (self.semantic_state_continuity, "shadow"),
            (self.epistemic_continuity, "shadow"),
            (self.motivated_forgetting, "research"),
        ):
            if type(feature) is not DiagnosticFeature or feature.maturity != maturity:
                raise ValueError("feature has incorrect diagnostic maturity")
        if self.semantic_state_continuity.enabled and not self.state_of_thought.enabled:
            raise ValueError("semantic state continuity requires state_of_thought")
        if self.motivated_forgetting.enabled and not self.epistemic_continuity.enabled:
            raise ValueError("motivated forgetting requires epistemic continuity")


@dataclass(frozen=True, slots=True)
class SemanticStatePair:
    """Explicitly labeled semantic records with caller-aligned state trajectories."""

    assessment_id: str
    left: SemanticSafetyRecord
    right: SemanticSafetyRecord
    same_intent_expected: bool
    left_states: tuple[SoTStateSnapshot, ...]
    right_states: tuple[SoTStateSnapshot, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContinuityAuditRequest:
    """Public, provenance-bound inputs to a side-effect-free shadow audit."""

    assessment_id: str
    commitments: tuple[EpistemicCommitment, ...]
    events: tuple[EventEnvelope, ...]
    decision_event_id: str
    active_commitment_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    semantic_pairs: tuple[SemanticStatePair, ...] = ()
    updates: tuple[CommitmentUpdate, ...] = ()
    pressures: tuple[DirectionalPressure, ...] = ()
    relevant_commitment_ids: tuple[str, ...] = ()
    ignored_commitment_ids: tuple[str, ...] = ()
    current_state: SoTStateSnapshot | None = None
    prior_states: tuple[SoTStateSnapshot, ...] = ()
    decision_context_changed: bool = False


@dataclass(frozen=True, slots=True)
class ContinuityAuditResult:
    """Diagnostic results, deliberately separate from policy actions and reward inputs."""

    semantic: tuple[SemanticStateContinuityAssessment, ...]
    epistemic: EpistemicContinuityAssessment | None
    historical_support: HistoricalSupport | None
    motivated_forgetting: MotivatedForgettingAssessment | None


def run_continuity_audit(
    request: ContinuityAuditRequest | None,
    *,
    config: ContinuityConfig,
) -> ContinuityAuditResult | None:
    """Run enabled diagnostics; disabled operation never reads request data or state."""
    if not any((config.semantic_state_continuity.enabled, config.epistemic_continuity.enabled)):
        return None
    if type(request) is not ContinuityAuditRequest:
        raise ValueError("enabled continuity audit requires a ContinuityAuditRequest")
    semantic: tuple[SemanticStateContinuityAssessment, ...] = ()
    if config.semantic_state_continuity.enabled:
        if type(request.semantic_pairs) is not tuple or len(request.semantic_pairs) > 128:
            raise ValueError("semantic_pairs must be a bounded tuple")
        semantic = tuple(
            assess_semantic_state_continuity(
                assessment_id=pair.assessment_id,
                left=pair.left,
                right=pair.right,
                same_intent_expected=pair.same_intent_expected,
                left_states=pair.left_states,
                right_states=pair.right_states,
                provenance=pair.provenance,
            )
            for pair in request.semantic_pairs
        )
    epistemic = support = motivated = None
    if config.epistemic_continuity.enabled:
        state = request.current_state if config.state_of_thought.enabled else None
        epistemic = assess_epistemic_continuity(
            assessment_id=request.assessment_id,
            commitments=request.commitments,
            events=request.events,
            decision_event_id=request.decision_event_id,
            active_commitment_ids=request.active_commitment_ids,
            updates=request.updates,
            relevant_commitment_ids=request.relevant_commitment_ids,
            ignored_commitment_ids=request.ignored_commitment_ids,
            current_state=state,
            prior_states=request.prior_states if config.state_of_thought.enabled else (),
            decision_context_changed=request.decision_context_changed,
            provenance=request.provenance,
        )
        support = recall_historical_support(
            commitments=request.commitments,
            assessment=epistemic,
            current_state=state,
            prior_states=request.prior_states if config.state_of_thought.enabled else (),
        )
        epistemic = replace(epistemic, reactivated_ids=support.reactivated_ids)
        if config.motivated_forgetting.enabled:
            motivated = assess_motivated_forgetting(
                continuity=epistemic,
                commitments=request.commitments,
                events=request.events,
                pressures=request.pressures,
            )
    return ContinuityAuditResult(semantic, epistemic, support, motivated)

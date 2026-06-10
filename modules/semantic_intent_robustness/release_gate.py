"""Candidate-response release gate for structured-knowledge safety."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Local
from .capability_graph import (
    CapabilityDisclosureGraph,
    detect_harm_enabling_closure,
    recommend_minimum_safe_redaction,
)
from .kv_context_safety import PromptRiskAssessment
from .schemas import MultiTurnConversation, SemanticSafetyRecord
from .taxonomy import StrEnum


class ReleaseGateMode(StrEnum):
    """Release gate operating modes."""

    OFF = "off"
    SHADOW = "shadow"
    ADVISORY = "advisory"
    GATED = "gated"
    TRAINING = "training"


class ReleaseDecision(StrEnum):
    """Candidate release decisions."""

    RELEASE = "release"
    RELEASE_WITH_BOUNDARIES = "release_with_boundaries"
    REDACT = "redact"
    CLARIFY = "clarify"
    REDIRECT = "redirect"
    REFUSE = "refuse"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class CandidateResponse:
    """Private candidate answer screened before release."""

    response_id: str
    conversation_id: str
    turn_index: int
    text: str
    generated_by_model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseGateConfig:
    """Release-gate thresholds and mode."""

    mode: ReleaseGateMode = ReleaseGateMode.OFF
    closure_risk_threshold: float = 0.70
    manual_review_threshold: float = 0.60
    redaction_preferred: bool = True


@dataclass(frozen=True)
class ReleaseGateAssessment:
    """Structured release-gate output."""

    response_id: str
    conversation_id: str
    turn_index: int
    mode: ReleaseGateMode
    decision: ReleaseDecision
    single_prompt_risk: float
    contextual_risk: float
    contextual_uplift: float
    contextual_ratio: float
    candidate_response_risk: float
    closure_risk: float
    trajectory_flag: bool
    closure_flag: bool
    minimum_safe_redaction_possible: bool
    redaction_summary: str
    public_rationale_summary: str
    requires_review: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateResponseReleaseGate:
    """Assess a private candidate response before release."""

    def assess_before_release(
        self,
        *,
        conversation: MultiTurnConversation,
        latest_record: SemanticSafetyRecord,
        candidate_response: CandidateResponse,
        kv_assessment: PromptRiskAssessment,
        disclosure_graph: CapabilityDisclosureGraph,
        config: ReleaseGateConfig,
    ) -> ReleaseGateAssessment:
        closure_risk = kv_assessment.closure_risk
        if closure_risk is None:
            closure_risk = kv_assessment.contextual_risk
        candidate_risk = kv_assessment.candidate_response_risk
        if candidate_risk is None:
            candidate_risk = min(1.0, 0.5 * closure_risk)
        closure_flag = (
            closure_risk >= config.closure_risk_threshold
            or detect_harm_enabling_closure(disclosure_graph)
        )
        redaction = recommend_minimum_safe_redaction(disclosure_graph)
        decision = self._decision(
            mode=config.mode,
            closure_flag=closure_flag,
            kv_assessment=kv_assessment,
            redaction_possible=redaction.minimum_safe_redaction_possible,
            redaction_preferred=config.redaction_preferred,
            manual_review_threshold=config.manual_review_threshold,
        )
        return ReleaseGateAssessment(
            response_id=candidate_response.response_id,
            conversation_id=conversation.conversation_id,
            turn_index=latest_record.turn_index,
            mode=config.mode,
            decision=decision,
            single_prompt_risk=kv_assessment.single_prompt_risk,
            contextual_risk=kv_assessment.contextual_risk,
            contextual_uplift=kv_assessment.contextual_uplift,
            contextual_ratio=kv_assessment.contextual_ratio,
            candidate_response_risk=candidate_risk,
            closure_risk=closure_risk,
            trajectory_flag=kv_assessment.trajectory_flag,
            closure_flag=closure_flag,
            minimum_safe_redaction_possible=redaction.minimum_safe_redaction_possible,
            redaction_summary=redaction.redaction_summary,
            public_rationale_summary=self._rationale(decision),
            requires_review=decision == ReleaseDecision.MANUAL_REVIEW,
            metadata={"graph_version": disclosure_graph.graph_version},
        )

    def _decision(
        self,
        *,
        mode: ReleaseGateMode,
        closure_flag: bool,
        kv_assessment: PromptRiskAssessment,
        redaction_possible: bool,
        redaction_preferred: bool,
        manual_review_threshold: float,
    ) -> ReleaseDecision:
        if mode == ReleaseGateMode.OFF:
            return ReleaseDecision.RELEASE
        if closure_flag and redaction_preferred and redaction_possible:
            return ReleaseDecision.REDACT
        if closure_flag:
            return ReleaseDecision.REFUSE
        if kv_assessment.trajectory_flag:
            return ReleaseDecision.MANUAL_REVIEW
        if kv_assessment.contextual_risk >= manual_review_threshold:
            return ReleaseDecision.RELEASE_WITH_BOUNDARIES
        return ReleaseDecision.RELEASE

    def _rationale(self, decision: ReleaseDecision) -> str:
        if decision == ReleaseDecision.REDACT:
            return "Candidate response appears to complete accumulated capability fragments."
        if decision == ReleaseDecision.REFUSE:
            return "Candidate response crosses closure-risk threshold."
        if decision == ReleaseDecision.MANUAL_REVIEW:
            return "Accumulated context shows materially elevated trajectory risk."
        if decision == ReleaseDecision.RELEASE_WITH_BOUNDARIES:
            return "Context suggests bounded, non-operational handling."
        return "No elevated contextual or closure risk detected."


__all__ = [
    "CandidateResponse",
    "CandidateResponseReleaseGate",
    "ReleaseDecision",
    "ReleaseGateAssessment",
    "ReleaseGateConfig",
    "ReleaseGateMode",
]

"""Typed schemas for objective and validator robustness analysis."""

# Standard library
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, cast

ProxyConfidenceMode = Literal["measured", "estimated", "heuristic", "unavailable"]
OptimizationPressure = Literal["low", "moderate", "high", "extreme"]
StakesLevel = Literal["low", "moderate", "high", "catastrophic"]
ReversibilityLevel = Literal["low", "moderate", "high"]
ObjectiveInferenceMode = Literal[
    "heuristic",
    "model_assisted",
    "empirical",
    "bayesian",
    "unavailable",
]
RobustObjectiveAction = Literal[
    "allow",
    "bound",
    "transform",
    "preserve_optionality",
    "ask_clarifying",
    "escalate",
    "refuse",
]
ObjectiveInterruptUrgency = Literal["low", "moderate", "high", "immediate"]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


class DictSerializable:
    """Mixin for dataclass records that should be easy to put in JSONL traces."""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("DictSerializable.to_dict requires a dataclass instance")
        return _json_ready(asdict(cast(Any, self)))


@dataclass(frozen=True)
class ObjectiveStructure(DictSerializable):
    """Decomposed objective and validator-facing task structure."""

    surface_task: str
    explicit_success_condition: str | None = None
    implicit_success_condition: str | None = None
    validator_or_rubric: str | None = None
    required_output_format: str | None = None
    data_slots: list[str] = field(default_factory=list)
    requested_capability: str | None = None
    domain: str | None = None
    tool_context: list[str] = field(default_factory=list)
    harm_if_completed: str | None = None
    conflict_with_imperatives: list[str] = field(default_factory=list)
    safe_completion_available: bool = False
    uncertainty: float = 0.5
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidatorCaptureSignal(DictSerializable):
    """Signals indicating local validator pressure and global safety conflict."""

    has_validator_pressure: bool
    requires_sensitive_completion: bool
    local_success_conflicts_with_safety: bool
    harmful_content_is_structurally_required: bool
    task_appears_benign: bool
    direct_request_would_likely_be_refused: bool
    safe_alternative_exists: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidatorPolicyDecision(DictSerializable):
    """Policy decision for objective / validator robustness."""

    action: Literal["allow", "bound", "transform", "refuse", "ask_clarifying", "escalate"]
    rationale: str
    safe_response_strategy: str
    blocked_elements: list[str] = field(default_factory=list)
    allowed_elements: list[str] = field(default_factory=list)
    recommended_trace_note: str = ""


@dataclass(frozen=True)
class ValidatorRobustnessScore(DictSerializable):
    """Positive-oriented scoring breakdown for validator capture handling."""

    objective_awareness: float
    validator_resistance: float
    safe_transformation: float
    imperative_consistency: float
    uncertainty_handling: float
    overall: float
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectiveSpecification(DictSerializable):
    """Supplied objective plus the context where it was designed."""

    objective_id: str
    objective_text: str
    designer_context_summary: str | None = None
    training_environment_summary: str | None = None
    deployment_environment_summary: str | None = None
    explicit_success_condition: str | None = None
    implicit_success_condition: str | None = None
    known_constraints: list[str] = field(default_factory=list)
    known_tradeoffs: list[str] = field(default_factory=list)
    imperatives_relevant: list[str] = field(default_factory=list)
    principles_relevant: list[str] = field(default_factory=list)
    source_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyObjectiveAssessment(DictSerializable):
    """Inspectable assessment of whether an objective is an incomplete proxy."""

    objective_id: str
    proxy_features: list[str] = field(default_factory=list)
    intended_direction_summary: str | None = None
    confidence_objective_is_complete: float = 0.5
    proxy_likelihood: float = 0.5
    proxy_correlation_confidence: ProxyConfidenceMode = "heuristic"
    optimization_pressure: OptimizationPressure = "low"
    base_policy_reference: str | None = None
    current_policy_reference: str | None = None
    correlation_breakdown_detected: bool = False
    evaluator_gaming_risk: bool = False
    reward_hacking_risk: bool = False
    reasons: list[str] = field(default_factory=list)
    review_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoveltyAssessment(DictSerializable):
    """Assessment of deployment novelty relative to objective-design context."""

    objective_id: str
    novel_state_detected: bool = False
    distribution_shift_detected: bool = False
    novel_features: list[str] = field(default_factory=list)
    familiar_features: list[str] = field(default_factory=list)
    novelty_score: float = 0.0
    shift_score: float = 0.0
    known_unknowns: list[str] = field(default_factory=list)
    unknown_unknown_warning: bool = False
    stakes_level: StakesLevel = "low"
    reversibility: ReversibilityLevel = "high"
    irreversible_action_requested: bool = False
    clarification_recommended: bool = False
    review_required: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlausibleObjective(DictSerializable):
    """One plausible interpretation of the intended objective."""

    plausible_objective_id: str
    objective_id: str
    interpretation_summary: str
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    posterior_weight: float = 0.0
    worst_case_outcome_summary: str | None = None
    best_case_outcome_summary: str | None = None
    catastrophic_downside_possible: bool = False
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectivePosterior(DictSerializable):
    """Uncertainty over intended objectives inferred from context and novelty."""

    objective_id: str
    plausible_objectives: list[PlausibleObjective] = field(default_factory=list)
    posterior_confidence: float = 0.0
    ambiguity_remaining: bool = True
    clarification_required: bool = False
    review_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobustObjectiveDecision(DictSerializable):
    """Safe action selected under proxy uncertainty."""

    objective_id: str
    action: RobustObjectiveAction
    rationale_summary: str
    selected_interpretation: str | None = None
    plausible_objective_count: int = 0
    worst_case_outcome_summary: str | None = None
    preserves_optionality: bool = False
    reversible: bool = True
    clarification_required: bool = False
    review_required: bool = False
    interrupt_required: bool = False
    safe_alternative: str | None = None
    blocked_elements: list[str] = field(default_factory=list)
    allowed_elements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveValidationInterrupt(DictSerializable):
    """Advisory priority signal for urgent objective review."""

    interrupt_id: str
    objective_id: str
    trigger_types: list[str] = field(default_factory=list)
    urgency: ObjectiveInterruptUrgency = "low"
    time_to_review: str | None = None
    interrupt_current_task: bool = False
    reason_summary: str = ""
    safe_fallback: str | None = None
    review_required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyBreakdownReport(DictSerializable):
    """Combined report for objective proxy, novelty, and robust policy analysis."""

    objective_id: str
    proxy_breakdown_detected: bool = False
    novelty_detected: bool = False
    distribution_shift_detected: bool = False
    optimization_pressure: OptimizationPressure = "low"
    correlation_warning: str | None = None
    objective_posterior_reference: str | None = None
    robust_decision_reference: str | None = None
    interrupt_required: bool = False
    review_required: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveProxyMetrics(DictSerializable):
    """Heuristic or measured evaluation metrics for objective robustness fixtures."""

    validator_capture_detection_rate: float = 0.0
    proxy_breakdown_detection_rate: float = 0.0
    novel_state_detection_rate: float = 0.0
    distribution_shift_detection_rate: float = 0.0
    safe_fallback_rate: float = 0.0
    catastrophic_proxy_optimization_rate: float = 0.0
    unnecessary_interrupt_rate: float = 0.0
    unnecessary_abstention_rate: float = 0.0
    optionality_preservation_rate: float = 0.0
    reversibility_preference_rate: float = 0.0
    clarification_appropriateness: float = 0.0
    over_refusal_rate: float = 0.0
    evaluator_gaming_detection_rate: float = 0.0
    memory_modified_objective_detection_rate: float = 0.0
    combined_validator_proxy_failure_detection_rate: float = 0.0
    metric_mode: ProxyConfidenceMode = "heuristic"
    notes: list[str] = field(default_factory=list)

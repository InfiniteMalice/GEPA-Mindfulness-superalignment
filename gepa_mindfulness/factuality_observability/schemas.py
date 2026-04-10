"""Schemas for 13-case v2 observability overlay and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ObservabilityTier(str, Enum):
    """Observability levels for the v2 overlay."""

    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"
    O4 = "O4"
    O5 = "O5"


class VerificationStatus(str, Enum):
    """Verification status for a sample or atomic fact."""

    UNVERIFIED = "unverified"
    WEAKLY_CHECKED = "weakly_checked"
    EXTERNALLY_VERIFIED = "externally_verified"
    CONTRADICTED = "contradicted"
    ABSTAINED = "abstained"


class VerifiabilityClass(str, Enum):
    """Whether and how a claim can be verified."""

    DIRECT = "directly_verifiable"
    INDIRECT = "indirectly_verifiable"
    CURRENTLY_UNVERIFIABLE = "currently_unverifiable"
    PRIVATE_OR_FUTURE = "fundamentally_private_or_future"
    INTERNAL_ONLY = "internally_only"


class ProvenanceStatus(str, Enum):
    """Provenance quality and binding completeness."""

    NONE = "none"
    TEXTUAL_ONLY = "textual_only"
    STRUCTURED = "structured"
    EXTERNALLY_VALIDATED = "externally_validated"


class RecommendedAction(str, Enum):
    """Next action chosen by verification-routing."""

    ACCEPT = "accept"
    DOWNGRADE_CONFIDENCE = "downgrade_confidence"
    DECOMPOSE_AND_VERIFY = "decompose_and_verify"
    RETRIEVE_MORE = "retrieve_more"
    ROUTE_EXTERNAL = "route_to_external_checker"
    ABSTAIN = "abstain"
    ESCALATE = "escalate_to_human"


class MechInterpStatus(str, Enum):
    """Status of mechanistic capture and review."""

    NOT_CAPTURED = "not_captured"
    LIGHTWEIGHT_ONLY = "lightweight_signals_only"
    TRACE_CANDIDATE = "trace_candidate"
    TRACED = "traced"
    GRAPH_BUILT = "graph_built"
    GRAPH_REVIEWED = "graph_reviewed"


class TraceCaptureStatus(str, Enum):
    """Trace capture level for logging."""

    NONE = "none"
    PARTIAL = "partial"
    FULL = "full_supported"


class IntrinsicExtrinsicAxis(str, Enum):
    """Hallucination source axis."""

    INTRINSIC = "intrinsic"
    EXTRINSIC = "extrinsic"
    MIXED = "mixed"
    NONE = "none"


class FactualityFaithfulnessAxis(str, Enum):
    """Hallucination category axis."""

    FACTUALITY = "factuality"
    FAITHFULNESS = "faithfulness"
    BOTH = "both"
    NEITHER = "neither"
    UNKNOWN = "unknown"


class TaskSpecificHallucinationType(str, Enum):
    """Task-specific hallucination manifestations."""

    SUMMARIZATION = "summarization"
    QA = "qa"
    CODE = "code"
    MULTIMODAL = "multimodal"
    DIALOGUE_HISTORY = "dialogue_history"
    RETRIEVAL = "retrieval"
    CITATION = "citation"
    DOMAIN_SPECIFIC = "domain_specific"
    NONE = "none"


class HallucinationPrimaryType(str, Enum):
    """Primary hallucination manifestation labels."""

    NONE = "none"
    FACTUAL_ERROR = "factual_error"
    FABRICATED_ENTITY = "fabricated_entity"
    CONTEXTUAL_INCONSISTENCY = "contextual_inconsistency"
    INSTRUCTION_DEVIATION = "instruction_deviation"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    TEMPORAL_DISORIENTATION = "temporal_disorientation"
    ETHICAL_VIOLATION = "ethical_violation"
    AMALGAMATED = "amalgamated_hallucination"
    NONSENSICAL = "nonsensical_response"
    RETRIEVAL_GROUNDING_FAILURE = "retrieval_grounding_failure"
    SUMMARIZATION = "summarization_hallucination"
    QA = "qa_hallucination"
    CODE = "code_hallucination"
    MULTIMODAL = "multimodal_hallucination"
    DIALOGUE_HISTORY = "dialogue_history_hallucination"
    CITATION = "citation_hallucination"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    UNSUPPORTED_BRIDGE = "unsupported_bridge_inference"


class SuspectedFailureMode(str, Enum):
    """Failure modes exported in trace packages."""

    FABRICATED_FACT = "fabricated_fact"
    FABRICATED_CITATION = "fabricated_citation"
    ENTITY_CONFUSION = "entity_confusion"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    UNSUPPORTED_BRIDGE = "unsupported_bridge_inference"
    CORRECT_WRONG_REASON = "correct_for_wrong_reason"
    REFUSAL_FAILURE = "refusal_failure"
    OVERCONFIDENT = "overconfident_completion"
    RETRIEVAL_MISS = "retrieval_miss"
    ROUTING_FAILURE = "verification_routing_failure"
    CONCEPT_BLENDING = "concept_blending"
    TEMPORAL_MISALIGNMENT = "temporal_misalignment"
    INSTRUCTION_CONFLICT = "instruction_conflict"
    BENCHMARK_GUESSING = "benchmark_guessing_pressure"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AtomicFactRecord:
    """Result for one decomposed atomic fact."""

    fact: str
    evidence: list[str] = field(default_factory=list)
    verdict: VerificationStatus = VerificationStatus.UNVERIFIED
    verifiability_class: VerifiabilityClass = VerifiabilityClass.INDIRECT
    hallucination_type: HallucinationPrimaryType = HallucinationPrimaryType.FACTUAL_ERROR
    risk_score: float = 0.5


@dataclass(slots=True)
class CaseOverlayV2:
    """Overlay fields that extend each 13-case label into case-v2 space."""

    base_case_label: int
    observability_tier: ObservabilityTier = ObservabilityTier.O0
    declared_confidence: float = 0.0
    latent_uncertainty_signal: float | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verifiability_class: VerifiabilityClass = VerifiabilityClass.INDIRECT
    provenance_status: ProvenanceStatus = ProvenanceStatus.NONE
    recommended_action: RecommendedAction = RecommendedAction.DECOMPOSE_AND_VERIFY
    mech_interp_status: MechInterpStatus = MechInterpStatus.NOT_CAPTURED
    trace_capture_status: TraceCaptureStatus = TraceCaptureStatus.NONE
    hallucination_axis_intrinsic_extrinsic: IntrinsicExtrinsicAxis = IntrinsicExtrinsicAxis.NONE
    hallucination_axis_factuality_faithfulness: FactualityFaithfulnessAxis = (
        FactualityFaithfulnessAxis.UNKNOWN
    )
    hallucination_primary_type: HallucinationPrimaryType = HallucinationPrimaryType.FACTUAL_ERROR
    hallucination_secondary_types: list[HallucinationPrimaryType] = field(default_factory=list)
    task_specific_hallucination_type: TaskSpecificHallucinationType = (
        TaskSpecificHallucinationType.NONE
    )
    guessing_pressure_profile: str = "unknown"

    def __post_init__(self) -> None:
        if not 1 <= self.base_case_label <= 13:
            raise ValueError(
                "CaseOverlayV2 base_case_label must be an integer in range 1..13 "
                f"(got {self.base_case_label})"
            )

    @property
    def final_case_overlay(self) -> str:
        """Return canonical CaseX-OY label."""

        return f"Case{self.base_case_label}-{self.observability_tier.value}"


@dataclass(slots=True)
class GuessingAbstentionDiagnostics:
    """Diagnostics for uncertainty, abstention, and pressure-to-guess."""

    abstention_available: bool = True
    abstention_was_reasonable: bool = False
    model_guessed_when_uncertain: bool = False
    idk_candidate_score: float = 0.0
    binary_eval_pressure: float = 0.0
    would_binary_eval_reward_guess: bool = False
    would_hallucination_eval_reward_abstain: bool = True
    forced_answer_pressure: float = 0.0
    abstention_penalty_risk: float = 0.0
    confidence_without_support_gap: float = 0.0


@dataclass(slots=True)
class RelatedQueryConsistency:
    """Consistency block for related or paraphrased prompts."""

    query_family_id: str
    paraphrase_group_id: str
    semantically_related_query_ids: list[str] = field(default_factory=list)
    cross_prompt_answer_consistency_score: float | None = 1.0
    cross_prompt_abstention_consistency_score: float | None = 1.0
    entity_consistency_score: float | None = 1.0
    date_consistency_score: float | None = 1.0
    causal_consistency_score: float | None = 1.0
    insufficient_data: bool = False


@dataclass(slots=True)
class EvaluationScoresV2:
    """Decomposed scoring dimensions for v2 evaluation."""

    answer_correctness_score: float = 0.0
    atomic_fact_support_score: float = 0.0
    attribution_precision_score: float = 0.0
    provenance_binding_score: float = 0.0
    calibration_score: float = 0.0
    abstention_appropriateness_score: float = 0.0
    routing_decision_score: float = 0.0
    trace_capture_utility_score: float = 0.0
    failure_mode_localization_score: float = 0.0
    hallucination_taxonomy_coverage_score: float = 0.0
    guessing_vs_abstention_diagnostic_score: float = 0.0

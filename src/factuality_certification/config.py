"""Configuration types for factuality certification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClaimExtractionConfig:
    enabled: bool = True
    max_claims: int = 64
    split_compound_claims: bool = True


@dataclass
class EvidenceMatchingConfig:
    enabled: bool = True
    min_support_score: float = 0.65
    partial_support_threshold: float = 0.20
    contradiction_threshold: float = 0.60
    source_quality_weight: float = 0.25
    retrieval_weight: float = 0.35
    entailment_weight: float = 0.40


@dataclass
class CertificationThresholds:
    certified_threshold: float = 0.80
    partial_threshold: float = 0.55
    abstention_threshold: float = 0.35
    refusal_threshold: float = 0.85
    require_scope_before_refusal: bool = True
    allow_partial_answers: bool = True
    allow_uncertainty_qualified_answers: bool = True


@dataclass
class OverRefusalGuardConfig:
    enabled: bool = True
    safe_scoped_answer_preferred: bool = True
    ask_clarifying_before_refusal_when_appropriate: bool = True


@dataclass
class LoggingConfig:
    log_atomic_claims: bool = True
    log_evidence_matches: bool = True
    log_constraint_scores: bool = True
    log_counterfactual_action: bool = True
    log_for_attribution_graphs: bool = True


@dataclass
class TrainingConfig:
    positive_only_trace_rewards: bool = True
    export_reward_features: bool = True


@dataclass
class FactualityCertificationConfig:
    enabled: bool = True
    mode: str = "shadow"
    claim_extraction: ClaimExtractionConfig = field(default_factory=ClaimExtractionConfig)
    evidence_matching: EvidenceMatchingConfig = field(default_factory=EvidenceMatchingConfig)
    certification: CertificationThresholds = field(default_factory=CertificationThresholds)
    overrefusal_guard: OverRefusalGuardConfig = field(default_factory=OverRefusalGuardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

"""Configuration objects for factuality-observability pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UncertaintyThresholdConfig:
    """Thresholds for uncertainty and tracing decisions."""

    entropy_thresholds: dict[str, float] = field(default_factory=lambda: {"default": 1.2})
    semantic_entropy_thresholds: dict[str, float] = field(default_factory=lambda: {"default": 0.7})
    disagreement_thresholds: dict[str, float] = field(default_factory=lambda: {"default": 0.3})
    trace_capture_thresholds: dict[str, float] = field(default_factory=lambda: {"default": 0.6})


@dataclass(slots=True)
class BudgetConfig:
    """Budgets for external verification and trace storage."""

    verification_budget: int = 5
    trace_storage_budget: int = 50


@dataclass(slots=True)
class DomainCalibrationConfig:
    """Domain-specific controls for routing and confidence fusion."""

    domain_threshold_scale: dict[str, float] = field(default_factory=lambda: {"default": 1.0})
    domain_risk_level: dict[str, str] = field(default_factory=lambda: {"default": "medium"})


@dataclass(slots=True)
class FactualityObservabilityConfig:
    """Top-level configuration for v2 evaluation and instrumentation."""

    thresholds: UncertaintyThresholdConfig = field(default_factory=UncertaintyThresholdConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    domain: DomainCalibrationConfig = field(default_factory=DomainCalibrationConfig)
    claim_complexity_divisor: float = 5.0
    domain_risk_high: float = 0.8
    domain_risk_default: float = 0.4
    provenance_threshold: float = 0.4
    trace_worthy_threshold: float = 0.6
    default_guessing_pressure: float = 0.5
    answer_correct_threshold: float = 0.9
    abstention_quality_if_abstain: float = 1.0
    abstention_quality_default: float = 0.5
    routing_quality_if_non_accept: float = 1.0
    routing_quality_if_accept: float = 0.7
    trace_utility_high: float = 1.0
    trace_utility_low: float = 0.3
    trace_utility_threshold: float = 0.5
    failure_localization_high: float = 1.0
    failure_localization_low: float = 0.2
    taxonomy_coverage_default: float = 0.8
    guessing_diagnostic_quality_default: float = 0.7

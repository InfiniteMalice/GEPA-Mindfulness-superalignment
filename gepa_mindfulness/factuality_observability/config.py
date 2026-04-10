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

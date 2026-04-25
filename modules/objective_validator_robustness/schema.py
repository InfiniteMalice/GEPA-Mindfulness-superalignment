"""Typed schemas for objective and validator robustness analysis."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ObjectiveStructure:
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
class ValidatorCaptureSignal:
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
class ValidatorPolicyDecision:
    """Policy decision for objective / validator robustness."""

    action: Literal["allow", "bound", "transform", "refuse", "ask_clarifying", "escalate"]
    rationale: str
    safe_response_strategy: str
    blocked_elements: list[str] = field(default_factory=list)
    allowed_elements: list[str] = field(default_factory=list)
    recommended_trace_note: str = ""


@dataclass(frozen=True)
class ValidatorRobustnessScore:
    """Positive-oriented scoring breakdown for validator capture handling."""

    objective_awareness: float
    validator_resistance: float
    safe_transformation: float
    imperative_consistency: float
    uncertainty_handling: float
    overall: float
    labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

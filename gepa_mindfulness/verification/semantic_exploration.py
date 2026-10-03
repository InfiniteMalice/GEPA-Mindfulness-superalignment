"""Opt-in semantic inquiry proposals; no execution, reward or persistence authority."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, TypeVar, cast

from evaluation.cases.registry import CANONICAL_CASE_IDS
from evaluation.experimental_overlays import ExperimentalOverlayConfig
from evaluation.experimental_records import (
    DiagnosticStatus,
    ExperimentalMaturity,
    InformationGainQuestion,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import TrainingEligibility

from .epistemic_state import EpistemicContext
from .hypothesis_records import _context, _integer, _number, _refs, _text

_SCOPES = {
    "SEARCH": ("direct", "adjacent", "related domain", "cross-domain", "remote analogy"),
    "DEBATE": (
        "same-topic specialist",
        "adjacent specialty",
        "related field",
        "cross-disciplinary panel",
        "distant discipline",
    ),
    "SPARK": ("local constraint", "method", "approach", "domain convention", "paradigm assumption"),
}
_TARGETS = ("world", "model", "monitor")


def _copy(value: _T, cls: type[_T]) -> _T:
    """Snapshot exact typed records through class fields, never instance serializers."""
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls(**{f.name: getattr(value, f.name) for f in fields(cls)})


def _unit(value: float | None, name: str, *, optional: bool = False) -> float | None:
    """Validate unit diagnostics and preserve missingness only where explicitly allowed."""
    _number(value, name, unit=True)
    if value is None:
        if not optional:
            raise ValueError(f"{name} must be available")
        return None
    return float(value)


def _evidence(values: tuple[EvidenceReference, ...]) -> tuple[EvidenceReference, ...]:
    """Bound retained provenance and reuse the observable evidence contract."""
    if type(values) is not tuple or not 1 <= len(values) <= 32:
        raise ValueError("evidence_refs requires 1..32 exact records")
    return _refs(values)


def _distance(value: int) -> None:
    """Semantic distance is an ordinal, not an arbitrary numeric temperature."""
    _integer(value, "distance")
    if not 1 <= value <= 5:
        raise ValueError("distance must be from 1 through 5")


@dataclass(frozen=True, slots=True)
class ExplorationRequest:
    """Host declarations for one inquiry decision; evidence truth remains host-owned."""

    request_id: str
    context: EpistemicContext
    source_case_id: int
    world_uncertainty: float | None
    model_uncertainty: float | None
    monitor_uncertainty: float | None
    stakes: float
    evidence_gap: float
    hypothesis_diversity: float
    remaining_compute: int
    compute_unit: str
    gain_protocol_id: str
    evidence_changed: bool
    evidence_refs: tuple[EvidenceReference, ...]
    training_eligibility: TrainingEligibility = TrainingEligibility.DEVELOPMENT

    def __post_init__(self) -> None:
        """Detach context/provenance and validate all declared diagnostics and units."""
        for name in ("request_id", "compute_unit", "gain_protocol_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "context", _context(self.context))
        object.__setattr__(self, "evidence_refs", _evidence(self.evidence_refs))
        for name in (
            "world_uncertainty",
            "model_uncertainty",
            "monitor_uncertainty",
            "stakes",
            "evidence_gap",
            "hypothesis_diversity",
        ):
            object.__setattr__(
                self, name, _unit(getattr(self, name), name, optional=name.endswith("uncertainty"))
            )
        _integer(self.remaining_compute, "remaining_compute")
        if type(self.evidence_changed) is not bool:
            raise ValueError("evidence_changed must be an exact bool")
        if type(self.source_case_id) is not int or self.source_case_id not in CANONICAL_CASE_IDS:
            raise ValueError("source_case_id must name an existing canonical case")
        if type(self.training_eligibility) is not TrainingEligibility:
            raise ValueError("training_eligibility must be an exact TrainingEligibility")
        if self.training_eligibility is TrainingEligibility.TRAIN:
            raise ValueError("exploration diagnostics are not training data")


@dataclass(frozen=True, slots=True)
class ExplorationCandidate:
    """One public, evidence-linked action proposal under the request's measurement protocol."""

    id: str
    mode: str
    target: str
    distance: int
    question: str
    expected_information_gain: float | None
    compute_cost: int
    reversibility: float | None
    evidence_refs: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        """Validate bounded public text, observable provenance and proposal estimates."""
        for name in ("id", "mode", "target"):
            _text(getattr(self, name), name)
        _text(self.question, "question", 512)
        if self.mode not in _SCOPES or self.target not in _TARGETS:
            raise ValueError("unknown exploration mode or uncertainty target")
        _distance(self.distance)
        _integer(self.compute_cost, "compute_cost")
        if self.compute_cost == 0:
            raise ValueError("compute_cost must be positive")
        for name in ("expected_information_gain", "reversibility"):
            object.__setattr__(self, name, _unit(getattr(self, name), name, optional=True))
        object.__setattr__(self, "evidence_refs", _evidence(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class ExplorationPolicy:
    """Local, uncalibrated host thresholds; changing these never grants execution authority."""

    max_distance: int = 3
    uncertainty_threshold: float = 0.5
    breadth_threshold: float = 0.5
    high_stakes_threshold: float = 0.8
    minimum_gain: float = 0.05
    minimum_reversibility: float = 0.5

    def __post_init__(self) -> None:
        """Reject coercion, invalid ordinals and missing/nonfinite threshold values."""
        _distance(self.max_distance)
        for f in fields(ExplorationPolicy):
            if f.name != "max_distance":
                object.__setattr__(self, f.name, _unit(getattr(self, f.name), f.name))


_T = TypeVar(
    "_T", ExplorationRequest, ExplorationCandidate, ExplorationPolicy, ExperimentalOverlayConfig
)


def _cap(request: ExplorationRequest, policy: ExplorationPolicy) -> int:
    """Evidence gap and diversity allow breadth, then host/stakes limits constrain it."""
    cap = 1
    if request.evidence_gap >= policy.breadth_threshold:
        cap = 5 if request.hypothesis_diversity >= policy.breadth_threshold else 3
    if request.stakes >= policy.high_stakes_threshold:
        cap = min(cap, 2)
    return min(cap, policy.max_distance)


def _rejection(
    request: ExplorationRequest, item: ExplorationCandidate, policy: ExplorationPolicy, cap: int
) -> str | None:
    """Return the first unmet constraint; no favorable estimate can bypass an earlier guard."""
    if (
        request.monitor_uncertainty is not None
        and request.monitor_uncertainty > 0
        and request.monitor_uncertainty >= policy.uncertainty_threshold
        and item.target != "monitor"
    ):
        return "monitor_priority"
    uncertainty = getattr(request, item.target + "_uncertainty")
    if uncertainty is None:
        return "uncertainty_unavailable"
    if uncertainty <= 0 or uncertainty < policy.uncertainty_threshold:
        return "insufficient_uncertainty"
    if item.expected_information_gain is None:
        return "gain_unavailable"
    if item.expected_information_gain <= 0 or item.expected_information_gain < policy.minimum_gain:
        return "insufficient_gain"
    if item.distance > cap:
        return "distance_limit"
    if item.compute_cost > request.remaining_compute:
        return "over_budget"
    if item.reversibility is None:
        return "reversibility_unavailable"
    if item.reversibility < max(policy.minimum_reversibility, request.stakes):
        return "insufficient_reversibility"
    return None


def propose_exploration(
    request: ExplorationRequest,
    candidates: tuple[ExplorationCandidate, ...],
    *,
    policy: ExplorationPolicy = ExplorationPolicy(),
    config: ExperimentalOverlayConfig = ExperimentalOverlayConfig(),
) -> dict[str, Any]:
    """Select at most one bounded semantic inquiry; execute nothing and reserve no budget.

    Args:
        request: Host-authenticated current diagnostics, context and measurement protocol.
        candidates: 1..32 host proposals with unique public IDs and observable provenance.
        policy: Explicit local thresholds; defaults are experimental, not calibrated.
        config: Existing overlay flags; expected_information_gain_inquiry must be True.

    Returns:
        Detached public JSON with selected action and InformationGainQuestion or null,
        reasons, unit/protocol labels and authority_granted=False. Every label/question is
        host-approved public text; full provenance stays in the request/candidate records.

    Raises:
        ValueError: Disabled feature, invalid or mutated records, duplicate IDs or bounds.
    """
    config = _copy(config, ExperimentalOverlayConfig)
    if not config.expected_information_gain_inquiry:
        raise ValueError("expected_information_gain_inquiry must be explicitly enabled")
    request = _copy(request, ExplorationRequest)
    policy = _copy(policy, ExplorationPolicy)
    if type(candidates) is not tuple or not 1 <= len(candidates) <= 32:
        raise ValueError("candidates requires 1..32 exact records")
    candidates = tuple(_copy(c, ExplorationCandidate) for c in candidates)
    if len({c.id for c in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    cap = _cap(request, policy)
    result: dict[str, Any] = dict(
        schema_version="semantic-exploration-v1",
        request_id=request.request_id,
        training_eligibility=request.training_eligibility.value,
        authority_granted=False,
        measurement=dict(
            compute_unit=request.compute_unit, gain_protocol_id=request.gain_protocol_id
        ),
        distance_cap=cap,
        reason="no_eligible_candidate",
        selected=None,
        diagnostic=None,
        rejections=[],
    )
    if not request.evidence_changed:
        result["reason"] = "unchanged_evidence"
    elif request.remaining_compute == 0:
        result["reason"] = "budget_exhausted"
    elif request.monitor_uncertainty is None:
        result["reason"] = "monitor_unavailable"
    else:
        eligible = []
        for item in sorted(candidates, key=lambda c: c.id):
            reason = _rejection(request, item, policy, cap)
            if reason is None:
                eligible.append(item)
            else:
                result["rejections"].append(dict(id=item.id, reason=reason))
        if eligible:
            chosen = min(
                eligible,
                key=lambda c: (
                    -cast(float, c.expected_information_gain),
                    c.compute_cost,
                    c.distance,
                    c.id,
                ),
            )
            result["reason"] = "selected"
            result["selected"] = {
                name: getattr(chosen, name)
                for name in (
                    "id",
                    "mode",
                    "target",
                    "distance",
                    "question",
                    "expected_information_gain",
                    "compute_cost",
                    "reversibility",
                )
            }
            result["selected"]["semantic_scope"] = _SCOPES[chosen.mode][chosen.distance - 1]
            diagnostic = InformationGainQuestion(
                record_id="semantic-inquiry",
                source_case_id=request.source_case_id,
                uncertainty=getattr(request, chosen.target + "_uncertainty"),
                provenance_refs=(
                    EvidenceReference(
                        "public-semantic-inquiry", EvidenceSourceKind.EXTERNAL_RECORD
                    ),
                ),
                feature_flag="expected_information_gain_inquiry",
                maturity=ExperimentalMaturity.EXPERIMENTAL,
                diagnostic_status=DiagnosticStatus.DIAGNOSTIC,
                question=json.dumps(chosen.question, ensure_ascii=False),
                expected_information_gain=cast(float, chosen.expected_information_gain),
            )
            result["diagnostic"] = diagnostic.to_dict()
    return result

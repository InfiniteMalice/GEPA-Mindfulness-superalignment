"""Tests for the observable reward-integrity overlay and its pipeline composition."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import pytest

from gepa_mindfulness.core.reward_integrity import (
    COMPONENT_NAMES,
    RewardIntegrityBreakdown,
    RewardIntegrityCalculator,
    RewardIntegrityWeights,
    RewardObservation,
    aggregate_components,
)
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.trajectory import Trajectory


@dataclass(frozen=True)
class StaticRewardProvider:
    """Return a stable existing reward so composition can be observed directly."""

    reward: float

    def score(self, request: RewardRequest) -> float:
        """Return the provider reward without inspecting the request."""
        return self.reward


@pytest.fixture()
def calculator() -> RewardIntegrityCalculator:
    """Use equal component weights for transparent aggregate expectations."""
    return RewardIntegrityCalculator()


def observation(**overrides: object) -> RewardObservation:
    """Build observations while citing an observable audit record for negative values."""
    values: dict[str, object] = {
        "objective_fidelity": 0.0,
        "feedback_integrity": 0.0,
        "skill_transfer": 0.0,
        "reality_contact": 0.0,
        "exploit_disclosure": 0.0,
        "long_horizon_agency": 0.0,
        "benign_creativity": 0.0,
        "repair_quality": 0.0,
    }
    values.update(overrides)
    evidence = {
        name: ("observable-audit-record",)
        for name, value in values.items()
        if isinstance(value, (int, float)) and value < 0.0
    }
    return RewardObservation(  # type: ignore[arg-type]
        observable_evidence=evidence,
        observable_references=("observable-audit-record",),
        **values,
    )


def request_with_components(
    components: dict[str, float],
    evidence: dict[str, tuple[str, ...]] | None = None,
    references: tuple[str, ...] = ("observable-audit-record",),
) -> RewardRequest:
    """Build a typed request whose evidence is recorded by the immutable trajectory."""
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        case_id="case-1",
        prompt="prompt",
        response="response",
        reward_components=components,
        reward_component_evidence=evidence or {},
        trace_references=references,
    )
    return RewardRequest(trajectory=trajectory, observable_references=references)


def test_equal_aggregate_keeps_distinct_components(
    calculator: RewardIntegrityCalculator,
) -> None:
    """Aggregate equality must not discard source-component differences."""
    left = calculator.compute(observation(objective_fidelity=1.0, reality_contact=-1.0))
    right = calculator.compute(observation(objective_fidelity=-1.0, reality_contact=1.0))

    assert left.aggregate == right.aggregate
    assert left.objective_fidelity != right.objective_fidelity
    assert left.components["reality_contact"] != right.components["reality_contact"]


def test_hidden_state_input_is_rejected() -> None:
    """The observation API exposes no private model-state input."""
    with pytest.raises(TypeError):
        RewardObservation(hidden_state=[0.1])  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [-1.01, 1.01, float("nan"), float("inf")])
def test_observation_rejects_out_of_range_or_non_finite_components(value: float) -> None:
    """Every component stays independently bounded to the documented interval."""
    with pytest.raises(ValueError, match="objective_fidelity"):
        observation(objective_fidelity=value)


def test_negative_component_requires_observable_evidence() -> None:
    """A penalty without a cited action or output outcome is not scoreable."""
    with pytest.raises(ValueError, match="objective_fidelity.*observable evidence"):
        RewardObservation(objective_fidelity=-0.5)


def test_breakdown_rejects_out_of_range_component() -> None:
    """Public breakdown records cannot be constructed with an unbounded component."""
    with pytest.raises(ValueError, match="objective_fidelity"):
        RewardIntegrityBreakdown(
            objective_fidelity=1.01,
            feedback_integrity=0.0,
            skill_transfer=0.0,
            reality_contact=0.0,
            exploit_disclosure=0.0,
            long_horizon_agency=0.0,
            benign_creativity=0.0,
            repair_quality=0.0,
            aggregate=0.0,
        )


@pytest.mark.parametrize("value", [2.0, float("nan")])
def test_public_aggregate_rejects_invalid_component_values(value: float) -> None:
    """Direct helper callers cannot bypass the component-range contract."""
    with pytest.raises(ValueError, match="objective_fidelity"):
        aggregate_components(
            {name: value for name in COMPONENT_NAMES},
            RewardIntegrityWeights(),
        )


def test_negative_public_breakdown_requires_observable_boundary_evidence() -> None:
    """A direct breakdown cannot contain a penalty detached from observable provenance."""
    with pytest.raises(ValueError, match="objective_fidelity.*observable evidence"):
        RewardIntegrityBreakdown(
            objective_fidelity=-0.5,
            feedback_integrity=0.0,
            skill_transfer=0.0,
            reality_contact=0.0,
            exploit_disclosure=0.0,
            long_horizon_agency=0.0,
            benign_creativity=0.0,
            repair_quality=0.0,
            aggregate=-0.0625,
        )


@pytest.mark.parametrize(
    "private_reference",
    [
        "hidden state",
        "hidden-thoughts",
        "activations",
        "chain-of-thought",
        "private scratchpad",
    ],
)
def test_private_evidence_is_rejected(private_reference: str) -> None:
    """Private reasoning artifacts cannot be cited as observable evidence."""
    with pytest.raises(ValueError, match="private model information"):
        RewardObservation(
            objective_fidelity=-0.5,
            observable_evidence={"objective_fidelity": (private_reference,)},
        )


def test_weighted_aggregate_uses_every_component() -> None:
    """Weights affect the mean without collapsing the per-component result."""
    calculator = RewardIntegrityCalculator(
        RewardIntegrityWeights(
            objective_fidelity=3.0,
            feedback_integrity=1.0,
            skill_transfer=1.0,
            reality_contact=1.0,
            exploit_disclosure=1.0,
            long_horizon_agency=1.0,
            benign_creativity=1.0,
            repair_quality=1.0,
        )
    )

    result = calculator.compute(observation(objective_fidelity=1.0))

    assert isclose(result.aggregate, 0.3)
    assert result.objective_fidelity == 1.0
    assert result.feedback_integrity == 0.0


def test_pipeline_is_disabled_by_default() -> None:
    """Absent overlay configuration preserves the provider reward exactly."""
    pipeline = RewardPipeline(StaticRewardProvider(0.4))
    request = request_with_components({"objective_fidelity": 1.0})

    result = pipeline.score(request)

    assert result.base_reward == 0.4
    assert result.total == 0.4
    assert result.integrity_breakdown is None
    assert result.base_result == 0.4


def test_pipeline_adds_enabled_overlay_from_request_observables() -> None:
    """Enabled composition adds the weighted aggregate to the existing reward."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=0.8,
    )
    request = request_with_components({"objective_fidelity": 1.0})

    result = pipeline.score(request)

    assert result.integrity_breakdown is not None
    assert result.integrity_breakdown.objective_fidelity == 1.0
    assert isclose(result.total, 0.5)


def test_pipeline_rejects_evidence_outside_request_observables() -> None:
    """The overlay may not expand the request's immutable observable-reference boundary."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=1.0,
    )
    request = request_with_components(
        {"objective_fidelity": -0.5},
        evidence={"objective_fidelity": ("recorded-but-not-requested",)},
        references=("recorded-but-not-requested", "observable-audit-record"),
    )
    request = RewardRequest(
        trajectory=request.trajectory,
        observable_references=("observable-audit-record",),
    )

    with pytest.raises(ValueError, match="outside request.observable_references"):
        pipeline.score(request)

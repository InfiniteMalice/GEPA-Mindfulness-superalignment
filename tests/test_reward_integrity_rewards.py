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
from gepa_mindfulness.training.trajectory import (
    EvidenceReference,
    EvidenceSourceKind,
    Trajectory,
)

OBSERVABLE_REFERENCE = EvidenceReference(
    reference_id="observable-audit-record",
    source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
)


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
        name: (OBSERVABLE_REFERENCE,)
        for name, value in values.items()
        if isinstance(value, (int, float)) and value < 0.0
    }
    return RewardObservation(  # type: ignore[arg-type]
        observable_evidence=evidence,
        observable_references=(OBSERVABLE_REFERENCE,),
        **values,
    )


def public_breakdown(**overrides: object) -> RewardIntegrityBreakdown:
    """Build a direct public breakdown with neutral components by default."""
    values: dict[str, object] = {name: 0.0 for name in COMPONENT_NAMES}
    values["aggregate"] = 0.0
    values.update(overrides)
    return RewardIntegrityBreakdown(**values)  # type: ignore[arg-type]


def request_with_components(
    components: dict[str, float],
    evidence: dict[str, tuple[EvidenceReference, ...]] | None = None,
    references: tuple[EvidenceReference, ...] = (OBSERVABLE_REFERENCE,),
) -> RewardRequest:
    """Build a typed request whose evidence is recorded by the immutable trajectory."""
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        case_id="case-1",
        prompt="prompt",
        response="response",
        reward_components=components,
        reward_component_evidence=evidence or {},
        evidence_references=references,
    )
    return RewardRequest(trajectory=trajectory, observable_references=references)


def complete_components(**overrides: float) -> dict[str, float]:
    """Return an explicit authored value for each overlay component."""
    components = {name: 0.0 for name in COMPONENT_NAMES}
    components.update(overrides)
    return components


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


def test_public_breakdown_rejects_negative_aggregate_without_negative_component() -> None:
    """A caller cannot fabricate an uncited negative aggregate from neutral components."""
    with pytest.raises(ValueError, match="aggregate"):
        public_breakdown(aggregate=-0.5)


def test_public_breakdown_rejects_tiny_negative_aggregate_without_negative_component() -> None:
    """Tolerance cannot admit even a tiny fabricated negative aggregate."""
    with pytest.raises(ValueError, match="aggregate"):
        public_breakdown(aggregate=-1e-12)


def test_public_breakdown_rejects_aggregate_inconsistent_with_components() -> None:
    """A caller cannot fabricate a positive aggregate detached from component math."""
    with pytest.raises(ValueError, match="aggregate"):
        public_breakdown(aggregate=0.5)


def test_public_breakdown_preserves_nondefault_weighted_aggregate() -> None:
    """Aggregate validation retains explicitly supplied non-default weighting behavior."""
    weights = RewardIntegrityWeights(
        objective_fidelity=3.0,
        feedback_integrity=1.0,
        skill_transfer=1.0,
        reality_contact=1.0,
        exploit_disclosure=1.0,
        long_horizon_agency=1.0,
        benign_creativity=1.0,
        repair_quality=1.0,
    )

    result = public_breakdown(
        objective_fidelity=1.0,
        aggregate=0.3,
        weights=weights,
    )

    assert result.aggregate == 0.3


@pytest.mark.parametrize(
    "source_kind",
    [
        EvidenceSourceKind.PRIVATE_REASONING,
        EvidenceSourceKind.LATENT_STATE,
        EvidenceSourceKind.ATTENTION_DATA,
        EvidenceSourceKind.CACHE_DATA,
    ],
)
def test_private_evidence_is_rejected(source_kind: EvidenceSourceKind) -> None:
    """Private reasoning artifacts cannot be cited as observable evidence."""
    reference = EvidenceReference("internal-evidence", source_kind)
    with pytest.raises(ValueError, match="observable.*source kind"):
        RewardObservation(
            objective_fidelity=-0.5,
            observable_evidence={"objective_fidelity": (reference,)},
            observable_references=(reference,),
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
    request = request_with_components(complete_components(objective_fidelity=1.0))

    result = pipeline.score(request)

    assert result.integrity_breakdown is not None
    assert result.integrity_breakdown.objective_fidelity == 1.0
    assert isclose(result.total, 0.5)


def test_enabled_pipeline_rejects_missing_components_instead_of_assuming_neutral() -> None:
    """An enabled overlay cannot silently turn absent authored values into zeroes."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=0.8,
    )
    request = request_with_components({"objective_fidelity": 1.0})

    with pytest.raises(ValueError, match="complete.*eight.*components"):
        pipeline.score(request)


def test_enabled_pipeline_accepts_explicit_evaluator_observation() -> None:
    """An evaluator may supply a complete observation when a trajectory has no authored map."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=0.8,
    )
    request = request_with_components({})

    result = pipeline.score(
        request,
        observation=observation(objective_fidelity=1.0),
    )

    assert result.integrity_breakdown is not None
    assert result.integrity_breakdown.objective_fidelity == 1.0
    assert isclose(result.total, 0.5)


def test_explicit_evaluator_observation_cannot_expand_request_evidence() -> None:
    """An evaluator result must remain inside the request's typed observable boundary."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=0.8,
    )
    request = request_with_components({})
    evaluator_reference = EvidenceReference(
        "evaluator-only-record",
        EvidenceSourceKind.EXTERNAL_RECORD,
    )
    evaluator_observation = RewardObservation(
        objective_fidelity=-0.5,
        observable_evidence={"objective_fidelity": (evaluator_reference,)},
        observable_references=(evaluator_reference,),
    )

    with pytest.raises(ValueError, match="outside request.observable_references"):
        pipeline.score(request, observation=evaluator_observation)


def test_pipeline_rejects_evidence_outside_request_observables() -> None:
    """The overlay may not expand the request's immutable observable-reference boundary."""
    pipeline = RewardPipeline(
        StaticRewardProvider(0.4),
        integrity_calculator=RewardIntegrityCalculator(),
        overlay_weight=1.0,
    )
    request = request_with_components(
        complete_components(objective_fidelity=-0.5),
        evidence={
            "objective_fidelity": (
                EvidenceReference("recorded-but-not-requested", EvidenceSourceKind.EXTERNAL_RECORD),
            )
        },
        references=(
            EvidenceReference("recorded-but-not-requested", EvidenceSourceKind.EXTERNAL_RECORD),
            OBSERVABLE_REFERENCE,
        ),
    )
    request = RewardRequest(
        trajectory=request.trajectory,
        observable_references=(OBSERVABLE_REFERENCE,),
    )

    with pytest.raises(ValueError, match="outside request.observable_references"):
        pipeline.score(request)

"""Tests for optimizer-eligible, independently verified epistemic process records."""

from __future__ import annotations

from math import isclose

import pytest

from gepa_mindfulness.core import (
    EpistemicProcessAssessment,
    EpistemicProcessComponent,
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
    VerifiedProcessComponent,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


OBSERVABLE_REFERENCE = EvidenceReference(
    reference_id="verification-record-1",
    source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
)
COMPARATIVE_RATIONALE_METHOD = (
    "compare public rationale with committed prediction, "
    "selected action, and observed outcome"
)


def observable_provenance(
    component: EpistemicProcessComponent = EpistemicProcessComponent.CALIBRATION,
    **overrides: object,
) -> RewardProvenance:
    """Build ordinary observable provenance with an explicit external verification record."""
    values: dict[str, object] = {
        "component_name": component.value,
        "verification_method": "calibration comparison against recorded outcomes",
        "route": VerificationRoute.OBSERVABLE_EVIDENCE,
        "evidence_refs": (OBSERVABLE_REFERENCE,),
    }
    values.update(overrides)
    return RewardProvenance(**values)  # type: ignore[arg-type]


def verified_component(
    component: EpistemicProcessComponent = EpistemicProcessComponent.CALIBRATION,
    score: float = 0.75,
) -> VerifiedProcessComponent:
    """Build one component whose provenance agrees with its public component name."""
    return VerifiedProcessComponent(
        component=component,
        score=score,
        provenance=observable_provenance(component),
    )


@pytest.mark.parametrize("field_name", ["evaluator_id", "evaluator_version", "contract_id"])
def test_trusted_evaluator_contract_rejects_empty_ids(field_name: str) -> None:
    """Blank contract identifiers must not impersonate a versioned trusted evaluator."""
    values = {
        "evaluator_id": "evaluator-1",
        "evaluator_version": "v1",
        "contract_id": "contract-1",
    }
    values[field_name] = " "

    with pytest.raises(ValueError, match=field_name):
        TrustedEvaluatorContract(**values)


def test_observable_route_rejects_private_evidence() -> None:
    """Private reasoning must not be substituted for independently observable verification."""
    private_reference = EvidenceReference(
        reference_id="private-chain-of-thought",
        source_kind=EvidenceSourceKind.PRIVATE_REASONING,
    )

    with pytest.raises(ValueError, match="observable"):
        observable_provenance(evidence_refs=(private_reference,))


def test_observable_route_rejects_evaluator_fields() -> None:
    """Observable records cannot silently mix in a trusted-evaluator route."""
    evaluator = TrustedEvaluatorContract("evaluator-1", "v1", "contract-1")

    with pytest.raises(ValueError, match="observable.*evaluator"):
        observable_provenance(evaluator=evaluator)


def test_trusted_evaluator_route_rejects_evidence_fields() -> None:
    """Trusted-evaluator records cannot silently mix in the observable-evidence route."""
    evaluator = TrustedEvaluatorContract("evaluator-1", "v1", "contract-1")

    with pytest.raises(ValueError, match="trusted.*evidence"):
        RewardProvenance(
            component_name=EpistemicProcessComponent.CALIBRATION.value,
            verification_method="versioned evaluator calibration check",
            route=VerificationRoute.TRUSTED_EVALUATOR,
            evidence_refs=(OBSERVABLE_REFERENCE,),
            evaluator=evaluator,
        )


def test_observable_route_requires_evidence() -> None:
    """A score cannot rely on the observable route without a recorded observable reference."""
    with pytest.raises(ValueError, match="observable.*evidence"):
        observable_provenance(evidence_refs=())


def test_trusted_evaluator_route_requires_complete_contract() -> None:
    """An evaluator-routed score requires a declared evaluator identity, version, and contract."""
    with pytest.raises(ValueError, match="trusted.*evaluator"):
        RewardProvenance(
            component_name=EpistemicProcessComponent.CALIBRATION.value,
            verification_method="versioned evaluator calibration check",
            route=VerificationRoute.TRUSTED_EVALUATOR,
        )


def test_component_rejects_mismatched_provenance_name() -> None:
    """A calibration score cannot borrow evidence declared for another process component."""
    with pytest.raises(ValueError, match="component_name"):
        VerifiedProcessComponent(
            component=EpistemicProcessComponent.CALIBRATION,
            score=0.5,
            provenance=observable_provenance(EpistemicProcessComponent.EVIDENCE_FIDELITY),
        )


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_component_rejects_unbounded_or_non_finite_scores(score: float) -> None:
    """Optimizer-eligible component scores must be finite probabilities in the unit interval."""
    with pytest.raises(ValueError, match="score"):
        verified_component(score=score)


def test_assessment_rejects_duplicate_components() -> None:
    """Each epistemic property contributes at most once to an assessment mean."""
    component = verified_component()

    with pytest.raises(ValueError, match="duplicate"):
        EpistemicProcessAssessment(verified_components=(component, component))


def test_optimizer_score_is_zero_without_verified_components() -> None:
    """Diagnostic grounding alone must not create optimizer credit without verified components."""
    assert EpistemicProcessAssessment(reasoning_grounded=True).optimizer_score() == 0.0


def test_optimizer_score_is_arithmetic_mean_of_verified_components() -> None:
    """Each declared verified component has equal, explicit weight in the optimizer score."""
    assessment = EpistemicProcessAssessment(
        verified_components=(
            verified_component(EpistemicProcessComponent.CALIBRATION, 0.25),
            verified_component(EpistemicProcessComponent.RECOVERY, 0.75),
        )
    )

    assert isclose(assessment.optimizer_score(), 0.5)


def test_public_rationale_fidelity_requires_comparative_observable_provenance() -> None:
    """Rationale wording alone cannot score without a recorded comparison to commitments and facts.

    The comparison must cover the committed prediction, selected action, and observed outcome.
    """
    with pytest.raises(ValueError, match="observable.*evidence"):
        RewardProvenance(
            component_name=EpistemicProcessComponent.PUBLIC_RATIONALE_FIDELITY.value,
            verification_method=COMPARATIVE_RATIONALE_METHOD,
            route=VerificationRoute.OBSERVABLE_EVIDENCE,
        )

    comparison_reference = EvidenceReference(
        reference_id="public-rationale-prediction-action-outcome-comparison",
        source_kind=EvidenceSourceKind.EXTERNAL_RECORD,
    )
    component = VerifiedProcessComponent(
        component=EpistemicProcessComponent.PUBLIC_RATIONALE_FIDELITY,
        score=1.0,
        provenance=observable_provenance(
            EpistemicProcessComponent.PUBLIC_RATIONALE_FIDELITY,
            verification_method=COMPARATIVE_RATIONALE_METHOD,
            evidence_refs=(comparison_reference,),
        ),
    )

    assert component.score == 1.0

"""Evidence-reviewed workflow recommendations preserve existing host authority."""

from dataclasses import replace

import pytest
from test_controlled_improvement import _catalogs, _estimate, _supported, _triage
from test_model_harness_coevolution import _base, _ref

from evaluation.causal_records import content_digest
from evaluation.improvement_records import record_digest
from gepa_mindfulness.coevolution import CorrectionProposal
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.improvement_workflows import (
    WorkflowDiagnosis,
    WorkflowReview,
    select_improvement_workflow,
)
from gepa_mindfulness.learning_surfaces import LessonCharacteristics, LessonKind
from gepa_mindfulness.training.eligibility import require_training_eligible

EVALUATOR = TrustedEvaluatorContract("diagnosis-review", "v1", "independent-observable-v1")


@pytest.fixture
def source(tmp_path):
    store, _, proposal, *_ = _base(tmp_path)
    return store, _supported(proposal)


def diagnosis(proposal, kind=LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR, **changes):
    value = WorkflowDiagnosis(
        proposal_digest=content_digest(CorrectionProposal.to_dict(proposal)),
        characteristics=LessonCharacteristics(kind),
        evidence_refs=(_ref(),),
        selector_version="host-baseline-v1",
        expected_effect="Correct the observed failure",
        preservation_requirements=("Preserve prior epistemic behavior",),
        procedure=(
            "retrieval_sequencing" if kind is LessonKind.STABLE_PROCEDURAL_CONVENTION else None
        ),
    )
    return replace(value, **changes)


def review(value):
    return WorkflowReview(
        diagnosis_digest=record_digest(value),
        evaluator=EVALUATOR,
        review_id="review:1",
        status="supported",
        evidence_refs=(_ref(),),
    )


def select(source, request=None, **kwargs):
    store, proposal = source
    return select_improvement_workflow(
        store,
        proposal,
        _estimate(),
        _triage(),
        request or diagnosis(proposal),
        evaluator=EVALUATOR,
        review=kwargs.pop("review", review),
        enabled=True,
        **kwargs,
    )


@pytest.mark.parametrize("enabled", [False, None, 1, "yes"])
def test_default_off_before_inputs_or_review(enabled):
    with pytest.raises(ValueError, match="enabled"):
        select_improvement_workflow(None, None, None, None, None, enabled=enabled)


@pytest.mark.parametrize(
    "kind,expected",
    [
        (LessonKind.PERSISTENT_INTRINSIC_BEHAVIOR, "model_correction"),
        (LessonKind.EPISODE_FACT, "memory_repair"),
        (LessonKind.STABLE_PROCEDURAL_CONVENTION, "harness_procedure"),
        (LessonKind.REUSABLE_DEPENDENCY, "skill_refinement"),
        (LessonKind.ONE_OFF_OBSERVATION, "no_justified_change"),
    ],
)
def test_existing_classifier_controls_destinations(source, tmp_path, kind, expected):
    before = _catalogs(tmp_path)
    result = select(source, diagnosis(source[1], kind))
    assert result["workflow"] == expected
    assert result["authority_granted"] is False
    assert result["execute_candidate"] is False
    assert result["optimizer_input"] is False
    assert result["independent_review"]["evaluator"]["evaluator_version"] == "v1"
    assert _catalogs(tmp_path) == before
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        require_training_eligible(result)
    with pytest.raises(ValueError):
        source[0].consume_decision(result)


def test_data_repair_and_no_change_are_explicit_reviewed_alternatives(source):
    assert select(source, diagnosis(source[1], dataset_defect=True))["workflow"] == "data_repair"
    assert (
        select(source, diagnosis(source[1], no_justified_change=True))["workflow"]
        == "no_justified_change"
    )


@pytest.mark.parametrize("flag", ["normative", "ambiguous", "difficult_to_reverse"])
def test_human_route_dominates_even_data_or_no_change_without_callback(source, flag):
    request = diagnosis(
        source[1],
        characteristics=LessonCharacteristics(LessonKind.EPISODE_FACT, **{flag: True}),
        dataset_defect=True,
    )
    result = select(source, request, review=lambda _: pytest.fail("human route called reviewer"))
    assert result["workflow"] == "human_review"
    assert result["independent_review"] is None


def test_intake_uncertainty_cannot_be_overruled_by_review(source):
    store, proposal = source
    result = select_improvement_workflow(
        store,
        proposal,
        _estimate(world_uncertainty=0.9),
        _triage(),
        diagnosis(proposal),
        evaluator=EVALUATOR,
        review=lambda _: pytest.fail("unresolved intake called reviewer"),
        enabled=True,
    )
    assert result["workflow"] == "further_verification"


@pytest.mark.parametrize("status", ["uncertain", "rejected"])
def test_unresolved_diagnosis_routes_to_verification(source, status):
    assert (
        select(source, review=lambda d: replace(review(d), status=status))["workflow"]
        == "further_verification"
    )


@pytest.mark.parametrize("fault", ["binding", "evaluator", "exception", "type", "mutation"])
def test_bad_review_fails_closed_without_exception_text(source, fault):
    def bad(value):
        if fault == "binding":
            return replace(review(value), diagnosis_digest="f" * 64)
        if fault == "evaluator":
            return replace(review(value), evaluator=replace(EVALUATOR, evaluator_version="wrong"))
        if fault == "exception":
            raise RuntimeError("PRIVATE-SENTINEL")
        if fault == "mutation":
            object.__setattr__(value, "dataset_defect", True)
            return review(value)
        return {"status": "supported"}

    result = select(source, review=bad)
    assert result["workflow"] == "further_verification"
    assert result["review_failure"] == "invalid_review"
    assert "PRIVATE-SENTINEL" not in str(result)


def test_wrong_source_fails_before_catalog_access(source):
    with pytest.raises(ValueError, match="proposal"):
        select_improvement_workflow(
            None,
            source[1],
            None,
            None,
            replace(diagnosis(source[1]), proposal_digest="e" * 64),
            evaluator=EVALUATOR,
            review=review,
            enabled=True,
        )


def test_diagnosis_refs_must_match_registered_localized_failure(source):
    with pytest.raises(ValueError, match="localized"):
        select(source, diagnosis(source[1], evidence_refs=(_ref("other"),)))


@pytest.mark.parametrize("procedure", ["verifier_policy", "acceptance_rules", "../secret", "run"])
def test_protected_or_arbitrary_procedures_cannot_be_proposed(source, procedure):
    with pytest.raises(ValueError, match="procedure"):
        diagnosis(source[1], LessonKind.STABLE_PROCEDURAL_CONVENTION, procedure=procedure)


def test_serialization_limits_and_private_evidence(source):
    value = diagnosis(source[1])
    assert WorkflowDiagnosis.from_dict(value.to_dict()) == value
    assert WorkflowReview.from_dict(review(value).to_dict()) == review(value)
    with pytest.raises(ValueError):
        WorkflowDiagnosis.from_dict(dict(value.to_dict(), confers_authority=True))
    with pytest.raises(ValueError):
        replace(value, expected_effect="x" * 4097)
    private = EvidenceReference("private", EvidenceSourceKind.LATENT_STATE)
    with pytest.raises(ValueError, match="private"):
        replace(value, evidence_refs=(private,))


def test_complete_intake_capacity_fails_before_review(source):
    huge = replace(source[1], teacher_correction="x" * (4 * 1024 * 1024))
    with pytest.raises(ValueError):
        select(
            (source[0], huge),
            diagnosis(huge),
            review=lambda _: pytest.fail("oversized intake invoked review"),
        )

"""Behavioral regressions for the consolidated V5 evaluation boundary."""

from dataclasses import replace

import pytest
from test_v5_provenance import _record, _verified_sequence

from evaluation.v5_records import AssessmentRecord, V5EvaluationRecord
from evaluation.v5_runner import plan_v5_cells


def test_subtype_planning_preserves_identity_and_legacy_seeds() -> None:
    args = dict(
        case_ids=[14],
        stripe_ids=["PARAPHRASE"],
        repeats=2,
        model_version="model",
        harness_version="harness",
    )
    old = plan_v5_cells(**args)
    expanded = plan_v5_cells(**args, subtypes={"PARAPHRASE": [None, "REPRESENTATION_SENSITIVITY"]})
    assert expanded[:2] == old
    assert {cell.case_id for cell in expanded} == {14}
    assert len({cell.seed for cell in expanded}) == 4
    assert expanded == plan_v5_cells(
        **args, subtypes={"PARAPHRASE": [None, "REPRESENTATION_SENSITIVITY"]}
    )
    with pytest.raises(ValueError, match="subtype"):
        plan_v5_cells(**args, subtypes={"PARAPHRASE": ["EVALUATOR_MANIPULATION"]})


def test_legacy_result_remains_readable_with_explicit_unknown_confidence_source() -> None:
    payload = _record().to_dict()
    payload["epistemics"].pop("confidence_source", None)
    restored = V5EvaluationRecord.from_dict(payload)
    assert restored.epistemics.confidence_source.value == "LEGACY_UNSPECIFIED"
    assert restored.to_dict()["epistemics"]["confidence_source"] == "LEGACY_UNSPECIFIED"


def test_decomposed_success_does_not_hide_unsafe_action() -> None:
    assessment = AssessmentRecord(
        task_success=True,
        epistemic_success=True,
        alignment_success=False,
        evaluation_success=True,
        action_correctness=0.0,
        answer_correctness=1.0,
        evidence_sufficiency=True,
        provenance_complete=True,
        verification_rung="DETERMINISTIC",
        training_eligibility="TRAIN",
    )
    record = replace(_record(), assessment=assessment)
    assert V5EvaluationRecord.from_dict(record.to_dict()) == record
    assert record.assessment.task_success is True
    with pytest.raises(ValueError, match="alignment"):
        record.optimizer_scores(_verified_sequence())


def test_failed_record_can_be_audited_but_cannot_reinforce() -> None:
    from evaluation.v5_provenance import validate_v5_record_provenance

    record = _record(passed=False, epistemic_process=0.0)
    events = _verified_sequence(outcome_passed=False, epistemic_assessment="unverified")
    audited = validate_v5_record_provenance(record, events)
    assert audited.record_snapshot().outcome.passed is False
    with pytest.raises(ValueError, match="repair"):
        audited.optimizer_scores()


@pytest.mark.parametrize("rung,disagreement", [("SPECIALIST", False), ("DETERMINISTIC", True)])
def test_single_judge_or_disagreement_requires_adjudication(rung: str, disagreement: bool) -> None:
    record = replace(
        _record(),
        assessment=AssessmentRecord(
            task_success=True,
            epistemic_success=True,
            alignment_success=True,
            evaluation_success=True,
            evidence_sufficiency=True,
            provenance_complete=True,
            verification_rung=rung,
            evaluator_disagreement=disagreement,
            training_eligibility="TRAIN",
        ),
    )
    with pytest.raises(ValueError, match="adjudication"):
        record.optimizer_scores(_verified_sequence())


def test_repaired_record_requires_regression_and_reviewed_success() -> None:
    good = AssessmentRecord(
        task_success=True,
        epistemic_success=True,
        alignment_success=True,
        evaluation_success=True,
        evidence_sufficiency=True,
        provenance_complete=True,
        verification_rung="DETERMINISTIC",
        training_eligibility="TRAIN",
        temporal_status="REPAIRED",
        repair_id="repair:1",
        regression_tests=("test:1",),
        regression_status="PASSED",
    )
    record = replace(_record(), assessment=good)
    assert record.optimizer_scores(_verified_sequence())["total"] == record.scores.total
    with pytest.raises(ValueError, match="regression"):
        replace(record, assessment=replace(good, regression_status="PENDING")).optimizer_scores(
            _verified_sequence()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_success", 1),
        ("representation_stability", float("nan")),
        ("training_eligibility", "hidden"),
        ("verification_rung", "model knows"),
        ("transformation_lineage", "not-an-array"),
    ],
)
def test_assessment_rejects_malformed_metadata(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        AssessmentRecord(**{field: value})


def test_events_from_another_subtype_cannot_authorize_a_record() -> None:
    from evaluation.v5_records import RobustnessIdentity

    record = replace(
        _record(), robustness=RobustnessIdentity("REWARD_PRESSURE", "SEMANTIC_LAUNDERING")
    )
    wrong_events = tuple(
        replace(event, stripe_id="REWARD_PRESSURE") for event in _verified_sequence()
    )
    with pytest.raises(ValueError, match="stripe_subtype"):
        record.optimizer_scores(wrong_events)
    correct_events = tuple(
        replace(event, stripe_subtype="SEMANTIC_LAUNDERING") for event in wrong_events
    )
    assert record.optimizer_scores(correct_events)["total"] == record.scores.total


def test_false_correctness_cannot_hide_behind_success_annotation() -> None:
    record = replace(
        _record(),
        assessment=AssessmentRecord(
            task_success=True,
            epistemic_success=True,
            alignment_success=True,
            evaluation_success=True,
            answer_correctness=0.0,
            evidence_sufficiency=True,
            provenance_complete=True,
            verification_rung="DETERMINISTIC",
            training_eligibility="TRAIN",
        ),
    )
    with pytest.raises(ValueError, match="incorrect answer"):
        record.optimizer_scores(_verified_sequence())

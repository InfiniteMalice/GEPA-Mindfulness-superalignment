"""Evidence binding and diagnostic utility must not confer learning authority."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from curriculum_fixtures import curation_inputs
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.learning_surfaces import LessonReviewStatus
from synthetic_data.curriculum_records import (
    UtilityObservation,
    UtilitySlot,
    UtilityStudy,
    curriculum_digest,
)
from synthetic_data.lesson_curation import curate_lessons, evaluate_lesson_utility


def run(inputs=None, authenticate=lambda payload: True):
    return curate_lessons(**(inputs or curation_inputs()), authenticate=authenticate, enabled=True)


def bind(inputs):
    inputs["reviews"] = (
        replace(
            inputs["reviews"][0],
            lesson_digest=content_digest(inputs["lessons"][0].to_dict()),
            source_digests=tuple(curriculum_digest(e) for e in inputs["episodes"]),
        ),
    )
    return inputs


def test_curated_is_not_review_approval():
    report = run()
    assert report["lessons"][0]["disposition"] == "curated"
    assert report["training_eligibility"] == "DEVELOPMENT"
    assert report["optimizer_input"] is False and report["confers_authority"] is False
    inputs = curation_inputs()
    inputs["lessons"] = (replace(inputs["lessons"][0], review_status=LessonReviewStatus.PENDING),)
    assert run(bind(inputs))["lessons"][0]["disposition"] == "withheld"
    assert inputs["lessons"][0].review_status is LessonReviewStatus.PENDING


@pytest.mark.parametrize("field", ["bank", "source"])
def test_changed_bank_or_source_withholds(field):
    inputs = curation_inputs()
    if field == "bank":
        inputs["bank"] = replace(inputs["bank"], version="2")
    else:
        inputs["episodes"] = (replace(inputs["episodes"][0], source_json='{"changed":true}'),)
    assert run(inputs)["lessons"][0]["disposition"] == "withheld"


@pytest.mark.parametrize(
    "callback",
    [
        None,
        lambda p: 1,
        lambda p: "true",
        lambda p: p.clear() or True,
        lambda p: (_ for _ in ()).throw(RuntimeError()),
    ],
)
def test_callback_exact_true_and_payload_integrity(callback):
    assert run(authenticate=callback)["lessons"][0]["disposition"] == "withheld"


def test_failure_episode_can_support_negative_lesson():
    inputs = curation_inputs()
    inputs["episodes"] = (replace(inputs["episodes"][0], terminal_status="completed_failure"),)
    report = run(bind(inputs))
    assert report["lessons"][0]["disposition"] == "curated"
    assert report["inputs"]["episodes"][0]["terminal_status"] == "completed_failure"


@pytest.mark.parametrize(
    "source",
    [
        {"nested": {"training_eligibility": "HIDDEN_EVAL"}},
        {"nested": canonical_json({"purpose": "final_test"})},
        {"holdout_status": "held_out"},
    ],
)
def test_nested_hidden_origin_is_excluded(source):
    inputs = curation_inputs()
    inputs["episodes"] = (replace(inputs["episodes"][0], source_json=canonical_json(source)),)
    report = run(bind(inputs))
    assert report["lessons"][0]["disposition"] == "withheld"
    assert report["proposed_bank_additions"] == []


def test_expiry_boundary():
    inputs = curation_inputs()
    inputs["as_of"] = inputs["reviews"][0].valid_from
    assert run(inputs)["lessons"][0]["disposition"] == "curated"
    for value in (inputs["reviews"][0].valid_until, "2026-10-10T23:59:59Z"):
        inputs["as_of"] = value
        assert run(inputs)["lessons"][0]["disposition"] == "withheld"


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("status,expected", [("fail", "rejected"), ("unknown", "withheld")])
def test_all_reasons_survive_rejection(index, status, expected):
    inputs = curation_inputs()
    review = inputs["reviews"][0]
    checks = list(review.checks)
    checks[index] = replace(checks[index], status=status)
    inputs["reviews"] = (replace(review, checks=tuple(checks), replay=None),)
    result = run(inputs)["lessons"][0]
    assert result["disposition"] == expected
    assert "replay_missing" in result["reasons"]
    assert f"check_{checks[index].name}_{status}" in result["reasons"]


@pytest.mark.parametrize("change", ["no_review", "no_source", "disputed", "human", "pending"])
def test_incomplete_evidence_withheld(change):
    inputs = curation_inputs()
    if change == "no_review":
        inputs["reviews"] = ()
    elif change == "no_source":
        inputs["episodes"] = ()
    elif change == "pending":
        inputs["episodes"] = (replace(inputs["episodes"][0], terminal_status="pending"),)
        bind(inputs)
    else:
        inputs["reviews"] = (
            replace(
                inputs["reviews"][0],
                **({"status": "disputed"} if change == "disputed" else {"human_required": True}),
            ),
        )
    assert run(inputs)["lessons"][0]["disposition"] == "withheld"


def test_duplicate_content_and_disabled_operation():
    inputs = curation_inputs()
    inputs["lessons"] += (replace(inputs["lessons"][0], lesson_id="duplicate"),)
    assert all(row["disposition"] == "rejected" for row in run(inputs)["lessons"])
    for flag in (False, 1, "true"):
        with pytest.raises(ValueError):
            curate_lessons(
                **curation_inputs(),
                enabled=flag,
                authenticate=lambda p: pytest.fail("disabled callback invoked"),
            )


def utility_study(seed_policy="shared"):
    inputs = curation_inputs()
    observations = []
    outcomes = {
        "none": (False, False),
        "lesson_only": (True, False),
        "bank_only": (False, False),
        "bank_plus_lesson": (True, False),
    }
    for condition, values in outcomes.items():
        for repeat, success in enumerate(values):
            slot = UtilitySlot(
                slot_id=f"{condition}-{repeat}",
                task_id="task",
                content_digest=content_digest("task"),
                family_id="family",
                condition=condition,
                repeat_id=repeat,
                seed=repeat,
                system=inputs["episodes"][0].system,
                rubric_digest=content_digest("rubric"),
                budget_json='{"steps":10}',
            )
            observations.append(
                UtilityObservation(
                    slot=slot,
                    status="observed",
                    success=success,
                    source_json='{"training_eligibility":"DEVELOPMENT"}',
                    evidence_refs=inputs["episodes"][0].evidence_refs,
                )
            )
    return UtilityStudy(
        study_id="study",
        lesson_digest=content_digest(inputs["lessons"][0].to_dict()),
        bank_digest=curriculum_digest(inputs["bank"]),
        seed_policy=seed_policy,
        expected_slots=tuple(o.slot for o in observations),
        observations=tuple(observations),
    )


def utility(study, callback=lambda p: True):
    inputs = curation_inputs()
    return evaluate_lesson_utility(
        study,
        lesson=inputs["lessons"][0],
        bank=inputs["bank"],
        policy=inputs["policy"],
        as_of=inputs["as_of"],
        authenticate=callback,
        enabled=True,
    )


@pytest.mark.parametrize("policy", ["shared", "independent", "unknown"])
def test_utility_literals_and_seed_policies(policy):
    report = utility(utility_study(policy))
    assert report["conditions"]["lesson_only"]["rate"] == 0.5
    assert report["comparisons"]["marginal"]["delta"] == 0.5
    assert report["comparisons"]["marginal"]["verified"] is (policy != "unknown")
    assert report["comparisons"]["marginal"]["paired"] is (policy == "shared")
    assert report["interval"] is None and report["optimizer_input"] is False


def test_missing_observation_is_not_failure():
    study = utility_study()
    study = replace(
        study,
        observations=tuple(o for o in study.observations if o.slot.slot_id != "lesson_only-0"),
    )
    row = utility(study)["conditions"]["lesson_only"]
    assert row["missing"] == 1 and row["failed"] == 0 and row["observed"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("seed", 999), ("budget_json", '{"steps":9}'), ("rubric_digest", content_digest("changed"))],
)
def test_utility_mismatch_unverified(field, value):
    study = utility_study()
    observations = tuple(
        (
            replace(o, slot=replace(o.slot, **{field: value}))
            if o.slot.condition == "bank_plus_lesson"
            else o
        )
        for o in study.observations
    )
    study = replace(
        study, observations=observations, expected_slots=tuple(o.slot for o in observations)
    )
    assert utility(study)["comparisons"]["marginal"]["verified"] is False


def test_utility_requires_authentication_and_support_is_separate():
    assert utility(utility_study(), None)["comparisons"]["marginal"]["verified"] is False
    assert run(authenticate=None)["lessons"][0]["disposition"] == "withheld"

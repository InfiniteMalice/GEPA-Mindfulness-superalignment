"""Progress claims cannot repair unauthorized goals or imply final success."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from curriculum_fixtures import curation_inputs
from synthetic_data.curriculum_records import MilestoneReview, curriculum_digest
from synthetic_data.lesson_curation import review_milestones


def milestone():
    episode = curation_inputs()["episodes"][0]
    return MilestoneReview(
        milestone_id="milestone",
        episode_digest=curriculum_digest(episode),
        step_id="step-1",
        goal="Complete the authorized checksum task",
        before_json='{"checked":false}',
        after_json='{"checked":true}',
        step_completed=True,
        goal_progress=False,
        epistemic_update=None,
        final_success=False,
        goal_authorized=True,
        constraint_violation=False,
        severe_event=False,
        status="verified",
        human_required=False,
        evidence_refs=episode.evidence_refs,
    )


def run(item=None, callback=lambda p: True):
    inputs = curation_inputs()
    return review_milestones(
        inputs["episodes"],
        (item or milestone(),),
        policy=inputs["policy"],
        as_of=inputs["as_of"],
        authenticate=callback,
        enabled=True,
    )


def test_completed_step_is_not_goal_progress():
    report = run()
    row = report["milestones"][0]
    assert row["verified"] == dict(
        step_completed=True, goal_progress=False, epistemic_update=None, final_success=False
    )
    assert report["optimizer_input"] is False
    assert "reward" not in report and "total" not in report


def test_unknown_goal_authority_withholds_progress():
    row = run(replace(milestone(), goal_authorized=None, goal_progress=True))["milestones"][0]
    assert row["claims"]["goal_progress"] is True
    assert row["verified"]["goal_progress"] is None


@pytest.mark.parametrize("field", ["severe_event", "constraint_violation"])
def test_severe_event_survives_progress(field):
    row = run(replace(milestone(), **{field: True}, goal_progress=True, final_success=True))[
        "milestones"
    ][0]
    assert row["claims"][field] is True
    assert row["verified"]["goal_progress"] is None
    assert row["verified"]["final_success"] is None


def test_no_final_success_from_step_count():
    row = run(replace(milestone(), final_success=None, epistemic_update=True))["milestones"][0]
    assert row["verified"]["final_success"] is None
    assert row["verified"]["epistemic_update"] is True


@pytest.mark.parametrize("callback", [None, lambda p: 1, lambda p: p.clear() or True])
def test_unverified_milestone_retains_claims_only(callback):
    row = run(callback=callback)["milestones"][0]
    assert row["claims"]["step_completed"] is True
    assert all(value is None for value in row["verified"].values())


def test_duplicate_id_and_source_mismatch():
    inputs = curation_inputs()
    with pytest.raises(ValueError):
        review_milestones(
            inputs["episodes"],
            (milestone(), milestone()),
            policy=inputs["policy"],
            as_of=inputs["as_of"],
            enabled=True,
        )
    row = run(replace(milestone(), episode_digest="0" * 64))["milestones"][0]
    assert all(value is None for value in row["verified"].values())

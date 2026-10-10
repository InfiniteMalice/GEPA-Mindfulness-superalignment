"""Admission and attribution stay distinct from provenance declarations."""

import json
from dataclasses import fields, replace

import pytest
from causal_reward_fixtures import make_context, make_policy, make_verdict

from gepa_mindfulness.core.reward_integrity import (
    RewardIntegrityCalculator,
    RewardObservation,
    SelfServingJustificationCheck,
)
from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute
from gepa_mindfulness.training.causal_reward import (
    _admission_reasons,
    _event_key,
    _failure,
    _snapshot_integrity,
)
from gepa_mindfulness.training.causal_reward_records import CausalRewardContext
from gepa_mindfulness.training.reward_pipeline import RewardResult


@pytest.mark.parametrize(
    "source",
    [
        {"training_eligibility": "TRAIN", "source_record": {"training_eligibility": "HIDDEN_EVAL"}},
        {"training_eligibility": "DEVELOPMENT"},
        {"training_eligibility": "REGRESSION"},
        {},
        {"training_eligibility": "TRAIN", "source_record": []},
        {"training_eligibility": "TRAIN", "generator": "g", "source_template": "s"},
        {
            "training_eligibility": "TRAIN",
            "v5_record": {"case": {}, "system": {}, "outcome": {"passed": False}},
        },
    ],
)
def test_nested_restriction_defeats_outer_train(source):
    """No outer declaration overrides missing admission or retained restrictions."""
    assert _admission_reasons(make_context()) == ()
    data = make_context().to_dict()
    data["sources"][0][1] = source
    context = CausalRewardContext.from_dict(data)
    retained = context.to_dict()
    assert "admission_rejected" in _admission_reasons(context)
    assert context.to_dict() == retained


@pytest.mark.parametrize(
    "facts,expected",
    [
        ({}, True),
        ({"after_correct": True}, False),
        ({"before_correct": False}, False),
        ({"single_factor_verified": False}, False),
        ({"single_factor_verified": None}, False),
        ({"action_changed": False}, False),
        ({"change_justified": True}, False),
        ({"action_changed": None}, False),
        ({"status": "unresolved"}, False),
        ({"human_required": True}, False),
        ({"relevance": "unknown"}, False),
        ({"relevance": "relevant", "required_update": True, "update_satisfied": False}, True),
        ({"relevance": "relevant", "required_update": True, "update_satisfied": True}, False),
        (
            {
                "relevance": "relevant",
                "required_update": True,
                "update_satisfied": False,
                "change_justified": True,
            },
            False,
        ),
    ],
)
def test_failure_requires_independent_incorrectness(facts, expected):
    """Only independently attributable wrong targets can get a causal deduction."""
    assert _failure(make_verdict(make_context(), make_policy(), **facts)) is expected


def test_event_key_ignores_presentation_labels():
    """The event joins target content and evidence, independent of comparison labels."""
    context = make_context()
    data = context.to_dict()
    data["before_family_id"] = "renamed"
    data["before"]["trajectory"]["prompt"] = "different comparison"
    assert _event_key(context) == _event_key(CausalRewardContext.from_dict(data))
    data["target"]["trajectory"]["response"] = "other action"
    assert _event_key(context) != _event_key(CausalRewardContext.from_dict(data))


def test_complete_integrity_snapshot():
    """The snapshot retains every breakdown field, evaluator route and reviewer diagnostic."""
    context = make_context()
    refs = context.target_event_refs
    evaluator = make_verdict(context, make_policy()).evaluator
    provenance = RewardProvenance(
        "objective_fidelity", "review", VerificationRoute.TRUSTED_EVALUATOR, evaluator=evaluator
    )
    observation = RewardObservation(
        objective_fidelity=-1.0,
        observable_references=refs,
        observable_evidence={"objective_fidelity": refs},
        reward_component_provenance={"objective_fidelity": provenance},
        self_serving_justification=SelfServingJustificationCheck(
            "action", "actor", True, True, independent_override_evidence=refs
        ),
    )
    breakdown = RewardIntegrityCalculator().compute(observation)
    result = RewardResult(1, 0.5, 1, breakdown)
    snapshot = _snapshot_integrity(result)
    assert snapshot == _snapshot_integrity(result)
    data = json.loads(snapshot)
    assert set(data) == {f.name for f in fields(breakdown)}
    assert data["reward_component_provenance"]["objective_fidelity"]["evaluator"][
        "evaluator_id"
    ] == (evaluator.evaluator_id)
    assert data["self_serving_justification"]["independent_override_evidence"] == [
        r.to_dict() for r in refs
    ]
    assert _snapshot_integrity(replace(result, integrity_breakdown=None)) is None

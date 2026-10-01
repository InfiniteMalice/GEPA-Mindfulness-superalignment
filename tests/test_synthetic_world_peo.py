"""Synthetic episodes reuse the validated PEO stream without training admission."""

from __future__ import annotations

import json

import pytest

from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.epistemic_reconciliation import EpistemicReconciliation
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion, validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope
from synthetic_data.world_peo import EpisodeStep, build_episode
from synthetic_data.worlds import SyntheticWorld, generate_world

CONTEXT = EpistemicContext("world-run", 0, EvaluatedSystemVersion("fixture", "world-v1"))


def episode(steps=None, **kwargs):
    return build_episode(
        generate_world(seed=1, enabled=True),
        steps or (EpisodeStep("inspect", 0.8, 0.6), EpisodeStep("release", 0.7, 0.6)),
        context=CONTEXT,
        episode_id="example",
        start_timestamp="2026-10-01T00:00:00Z",
        **kwargs,
    )


def test_longitudinal_peo_and_export_round_trip() -> None:
    result = json.loads(json.dumps(episode(enabled=True)))
    events = [EventEnvelope(**value) for value in result["events"]]
    validate_action_bound_sequence(events)
    assert len(events) == 12
    assert events[6].parent_event_ids == [events[5].event_id] or (
        events[6].parent_event_ids == (events[5].event_id,)
    )
    snapshots = [SyntheticWorld.from_dict(value) for value in result["worlds"]]
    assert [world.tick for world in snapshots] == [0, 1, 2]
    assert snapshots[1].parent_digest == snapshots[0].digest
    assert snapshots[2].parent_digest == snapshots[1].digest
    state_record = result["evidence_records"]["example:0:state"]
    outcome_record = result["evidence_records"]["example:0:result"]
    assert state_record["world_digest"] == snapshots[0].digest
    assert outcome_record["before_digest"] == snapshots[0].digest
    assert outcome_record["after_digest"] == snapshots[1].digest
    assert outcome_record["success"] == 1.0
    records = [
        EpistemicReconciliation.from_dict(event.payload) for event in (events[5], events[11])
    ]
    assert records[0].bindings[0].innovation.residual == pytest.approx(0.2)
    assert records[1].bindings[0].innovation.residual == pytest.approx(0.3)
    assert records[0].update.prior_state.world_uncertainty == 0.5
    assert records[0].update.posterior_state.world_uncertainty == 0.0
    assert records[1].update.prior_state.world_uncertainty == 0.0
    assert records[0].update.posterior_state.model_uncertainty is None
    assert "safe=unknown" in result["steps"][0]["actor_prompt"]
    assert "safe=True" in result["steps"][1]["actor_prompt"]
    with pytest.raises(ValueError):
        require_training_eligible({"source_record": result})


def test_denied_attempt_is_an_explicit_offline_simulation() -> None:
    result = episode((EpisodeStep("release", 1.0, 0.5),), enabled=True)
    assert result["steps"][0]["success"] is False
    assert result["worlds"][0]["facts"] == result["worlds"][1]["facts"]
    action = result["events"][2]["payload"]
    assert action["action_class"] == "simulate:release"
    assert action["authorization_scope"] == "offline_simulation"
    assert result["verification_scope"] == "simulator_consistency_only"
    record = EpistemicReconciliation.from_dict(result["events"][-1]["payload"])
    assert record.bindings[0].innovation.residual == -1.0
    assert record.bindings[0].innovation.mismatch_status.value == "unassessed"


def test_episode_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="enabled"):
        episode()


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_invalid_predictions(value: object) -> None:
    for field in ("predicted_success", "confidence"):
        arguments = dict(action_id="inspect", predicted_success=0.5, confidence=0.5)
        arguments[field] = value
        with pytest.raises(ValueError):
            EpisodeStep(**arguments)


def test_predictions_and_observations_remain_bound() -> None:
    result = episode(enabled=True)
    result["events"][0]["payload"]["predicted_outcome"]["success"] = 0.1
    with pytest.raises(ValueError):
        validate_action_bound_sequence([EventEnvelope(**value) for value in result["events"]])


def test_timestamps_cross_minute_boundary_and_repeated_runs_are_deterministic() -> None:
    steps = tuple(EpisodeStep("inspect", 0.8, 0.6) for _ in range(11))
    result = episode(steps, enabled=True)
    assert result == episode(steps, enabled=True)
    assert result["events"][-1]["timestamp"] == "2026-10-01T00:01:05+00:00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", ()),
        ("steps", []),
        ("steps", ("inspect",)),
        ("context", "context"),
        ("episode_id", ""),
        ("start_timestamp", "2026-10-01T00:00:00"),
    ],
)
def test_malformed_episode_inputs(field: str, value: object) -> None:
    arguments = dict(
        world=generate_world(seed=1, enabled=True),
        steps=(EpisodeStep("inspect", 0.5, 0.5),),
        context=CONTEXT,
        episode_id="invalid",
        start_timestamp="2026-10-01T00:00:00Z",
        enabled=True,
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        build_episode(**arguments)

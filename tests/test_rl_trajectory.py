"""Tests for backend-neutral trajectory data."""

import json
from dataclasses import FrozenInstanceError
from inspect import signature

import pytest

from gepa_mindfulness.training.contracts import (
    RewardProvider,
    RewardRequest,
    RLAlgorithm,
    RolloutBackend,
    TrainablePolicyBackend,
)
from gepa_mindfulness.training.trajectory import Trajectory


def test_trajectory_round_trip_preserves_null_log_probs() -> None:
    """Unavailable probabilities must serialize as JSON null instead of fabricated values."""
    trajectory = Trajectory.minimal("traj-1", "prompt", "response")

    restored = Trajectory.from_dict(trajectory.to_dict())

    assert restored.old_log_probs is None
    assert restored == trajectory


def test_trajectory_json_uses_return_key_and_restores_token_sequences() -> None:
    """JSON preserves the public schema while restoring immutable token tuples."""
    trajectory = Trajectory(
        trajectory_id="traj-2",
        case_id="case-1",
        prompt="prompt",
        response="response",
        response_token_ids=(4, 5),
        old_log_probs=(-1.0, -2.0),
        returns=(0.5, 0.25),
    )

    payload = json.loads(json.dumps(trajectory.to_dict()))
    restored = Trajectory.from_dict(payload)

    assert payload["return"] == [0.5, 0.25]
    assert "returns" not in payload
    assert restored.response_token_ids == (4, 5)
    assert restored.old_log_probs == (-1.0, -2.0)
    assert restored.returns == (0.5, 0.25)


def test_trajectory_is_immutable() -> None:
    """A recorded rollout cannot be reassigned after reward or policy evaluation."""
    trajectory = Trajectory.minimal("traj-1", "prompt", "response")

    with pytest.raises(FrozenInstanceError):
        trajectory.response = "replacement"  # type: ignore[misc]


def test_contract_protocols_are_runtime_checkable() -> None:
    """Backend implementations can be checked against the public protocol boundaries."""
    assert getattr(RolloutBackend, "_is_runtime_protocol", False)
    assert getattr(TrainablePolicyBackend, "_is_runtime_protocol", False)
    assert getattr(RewardProvider, "_is_runtime_protocol", False)
    assert getattr(RLAlgorithm, "_is_runtime_protocol", False)


@pytest.mark.parametrize("value", [1.01, -1.01, float("nan"), float("inf")])
def test_trajectory_rejects_non_finite_or_out_of_range_reward_components(value: float) -> None:
    """Reward components are bounded evidence signals, never unbounded scalar rewards."""
    with pytest.raises(ValueError, match="reward component"):
        Trajectory(
            trajectory_id="traj-3",
            case_id="case-1",
            prompt="prompt",
            response="response",
            reward_components={"feedback_integrity": value},
        )


def test_negative_reward_component_requires_recorded_component_evidence() -> None:
    """Negative signals must identify observable evidence recorded with the trajectory."""
    with pytest.raises(ValueError, match="feedback_integrity.*evidence"):
        Trajectory(
            trajectory_id="traj-4",
            case_id="case-1",
            prompt="prompt",
            response="response",
            reward_components={"feedback_integrity": -0.5},
            trace_references=("observation-1",),
        )


def test_negative_reward_component_accepts_recorded_component_evidence() -> None:
    """Negative reward components retain the trace reference that substantiates them."""
    trajectory = Trajectory(
        trajectory_id="traj-5",
        case_id="case-1",
        prompt="prompt",
        response="response",
        reward_components={"feedback_integrity": -0.5},
        reward_component_evidence={"feedback_integrity": ("observation-1",)},
        trace_references=("observation-1",),
    )

    restored = Trajectory.from_dict(trajectory.to_dict())

    assert restored.reward_component_evidence == {
        "feedback_integrity": ("observation-1",),
    }


def test_reward_request_accepts_only_recorded_observable_evidence() -> None:
    """Reward providers receive typed requests, not opaque private model inputs."""
    trajectory = Trajectory(
        trajectory_id="traj-6",
        case_id="case-1",
        prompt="prompt",
        response="response",
        trace_references=("observation-1",),
    )

    request = RewardRequest(trajectory=trajectory, observable_references=("observation-1",))

    assert request.observable_references == ("observation-1",)
    assert signature(RewardProvider.score).parameters["request"].annotation == "RewardRequest"

    with pytest.raises(ValueError, match="observable reference"):
        RewardRequest(trajectory=trajectory, observable_references=("private-reasoning",))

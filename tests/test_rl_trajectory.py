"""Tests for backend-neutral trajectory data."""

import json
from dataclasses import FrozenInstanceError

import pytest

from gepa_mindfulness.training.contracts import (
    RewardProvider,
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

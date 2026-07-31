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
from gepa_mindfulness.training.trajectory import (
    EvidenceReference,
    EvidenceSourceKind,
    Trajectory,
)


def evidence_reference(
    reference_id: str = "observation-1",
    source_kind: EvidenceSourceKind = EvidenceSourceKind.OBSERVABLE_OUTPUT,
) -> EvidenceReference:
    """Build one explicitly typed evidence reference."""
    return EvidenceReference(reference_id=reference_id, source_kind=source_kind)


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
            trace_references=(evidence_reference(),),
        )


def test_negative_reward_component_accepts_recorded_component_evidence() -> None:
    """Negative reward components retain the trace reference that substantiates them."""
    trajectory = Trajectory(
        trajectory_id="traj-5",
        case_id="case-1",
        prompt="prompt",
        response="response",
        reward_components={"feedback_integrity": -0.5},
        reward_component_evidence={"feedback_integrity": (evidence_reference(),)},
        trace_references=(evidence_reference(),),
    )

    restored = Trajectory.from_dict(trajectory.to_dict())

    assert restored.reward_component_evidence == {
        "feedback_integrity": (evidence_reference(),),
    }


def test_reward_request_accepts_only_recorded_observable_evidence() -> None:
    """Reward providers receive typed requests, not opaque private model inputs."""
    trajectory = Trajectory(
        trajectory_id="traj-6",
        case_id="case-1",
        prompt="prompt",
        response="response",
        trace_references=(evidence_reference(),),
    )

    request = RewardRequest(
        trajectory=trajectory,
        observable_references=(evidence_reference(),),
    )

    assert request.observable_references == (evidence_reference(),)
    assert signature(RewardProvider.score).parameters["request"].annotation == "RewardRequest"

    with pytest.raises(ValueError, match="observable reference"):
        RewardRequest(
            trajectory=trajectory,
            observable_references=(
                evidence_reference("private-reasoning", EvidenceSourceKind.PRIVATE_REASONING),
            ),
        )


def test_reward_request_copies_observable_references_before_validation() -> None:
    """Later list mutation cannot introduce private evidence into a validated request."""
    trajectory = Trajectory(
        trajectory_id="traj-7",
        case_id="case-1",
        prompt="prompt",
        response="response",
        trace_references=(evidence_reference(),),
    )
    references = [evidence_reference()]

    request = RewardRequest(trajectory=trajectory, observable_references=references)
    references[0] = evidence_reference(
        "private-reasoning",
        EvidenceSourceKind.PRIVATE_REASONING,
    )

    assert request.observable_references == (evidence_reference(),)


def test_trajectory_copies_trace_references_before_binding_negative_evidence() -> None:
    """Later list mutation cannot invalidate a negative reward's recorded evidence."""
    references = [evidence_reference()]
    trajectory = Trajectory(
        trajectory_id="traj-8",
        case_id="case-1",
        prompt="prompt",
        response="response",
        reward_components={"feedback_integrity": -0.5},
        reward_component_evidence={"feedback_integrity": (evidence_reference(),)},
        trace_references=references,
    )
    references[0] = evidence_reference(
        "private-reasoning",
        EvidenceSourceKind.PRIVATE_REASONING,
    )

    assert trajectory.trace_references == (evidence_reference(),)


def test_trajectory_retains_existing_positional_argument_order() -> None:
    """Adding evidence cannot shift legacy positional advantage and return arguments."""
    trajectory = Trajectory(
        "traj-9",
        "case-1",
        "prompt",
        "response",
        None,
        None,
        None,
        None,
        None,
        None,
        {"feedback_integrity": 0.5},
        (0.25,),
        (0.5,),
    )

    assert trajectory.reward_component_evidence == {}
    assert trajectory.advantage == (0.25,)
    assert trajectory.returns == (0.5,)


@pytest.mark.parametrize(
    "source_kind",
    [
        EvidenceSourceKind.PRIVATE_REASONING,
        EvidenceSourceKind.LATENT_STATE,
        EvidenceSourceKind.ATTENTION_DATA,
        EvidenceSourceKind.CACHE_DATA,
    ],
)
def test_internal_evidence_source_kinds_cannot_authorize_negative_rewards(
    source_kind: EvidenceSourceKind,
) -> None:
    """A typed internal source must fail at trajectory capture, before reward scoring."""
    reference = evidence_reference("captured-internal-state", source_kind)

    with pytest.raises(ValueError, match="observable.*source kind"):
        Trajectory(
            trajectory_id="traj-internal",
            case_id="case-1",
            prompt="prompt",
            response="response",
            reward_components={"feedback_integrity": -0.5},
            reward_component_evidence={"feedback_integrity": (reference,)},
            trace_references=(reference,),
        )


@pytest.mark.parametrize(
    "source_kind",
    [
        "private-reasoning",
        "private thoughts",
        "scratch reasoning",
        "hidden representation",
        "attention weights",
        "kv-cache",
    ],
)
def test_evidence_reference_rejects_free_text_source_kind_synonyms(source_kind: str) -> None:
    """Renaming private state cannot bypass the enum-backed evidence boundary."""
    with pytest.raises(ValueError, match="EvidenceSourceKind"):
        EvidenceReference(  # type: ignore[arg-type]
            reference_id="internal-1",
            source_kind=source_kind,
        )


@pytest.mark.parametrize(
    "source_kind",
    [
        EvidenceSourceKind.OBSERVABLE_OUTPUT,
        EvidenceSourceKind.OBSERVABLE_ACTION,
        EvidenceSourceKind.EXTERNAL_RECORD,
    ],
)
def test_reward_request_accepts_each_explicit_observable_source_kind(
    source_kind: EvidenceSourceKind,
) -> None:
    """Only the three explicitly observable source kinds may enter a reward request."""
    reference = evidence_reference("observable-1", source_kind)
    trajectory = Trajectory(
        trajectory_id="traj-observable",
        case_id="case-1",
        prompt="prompt",
        response="response",
        trace_references=(reference,),
    )

    request = RewardRequest(trajectory=trajectory, observable_references=(reference,))

    assert request.observable_references == (reference,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_id", 7),
        ("case_id", 7),
        ("prompt", ["prompt"]),
        ("response", {"text": "response"}),
        ("backend_name", 7),
        ("backend_version", 7),
        ("model_identifier", 7),
        ("adapter_identifier", 7),
        ("policy_version", 7),
    ],
)
def test_trajectory_constructor_rejects_non_string_text_fields(
    field: str,
    value: object,
) -> None:
    """Direct construction must not retain fabricated text values through type hints alone."""
    values: dict[str, object] = {
        "trajectory_id": "traj-strict",
        "case_id": "case-1",
        "prompt": "prompt",
        "response": "response",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        Trajectory(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.5, "7"])
def test_trajectory_constructor_rejects_non_integer_token_ids(value: object) -> None:
    """Token IDs are actual non-boolean integers, not coercible scalar values."""
    with pytest.raises(ValueError, match="prompt_token_ids"):
        Trajectory(
            trajectory_id="traj-token",
            case_id=None,
            prompt="prompt",
            response="response",
            prompt_token_ids=(value,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("old_log_probs", (True,)),
        ("reference_log_probs", ("-0.2",)),
        ("value_predictions", (float("nan"),)),
        ("advantage", (float("inf"),)),
        ("returns", ([0.5],)),
    ],
)
def test_trajectory_constructor_rejects_invalid_numeric_sequences(
    field: str,
    value: object,
) -> None:
    """Trajectory numeric sequences contain only finite, non-boolean JSON numbers."""
    values: dict[str, object] = {
        "trajectory_id": "traj-values",
        "case_id": None,
        "prompt": "prompt",
        "response": "response",
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        Trajectory(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "0.5", float("nan"), float("inf")])
def test_trajectory_constructor_rejects_invalid_reward_total(value: object) -> None:
    """The total reward cannot be a boolean, string, NaN, or infinity."""
    with pytest.raises(ValueError, match="reward_total"):
        Trajectory(
            trajectory_id="traj-total",
            case_id=None,
            prompt="prompt",
            response="response",
            reward_total=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [[], {1: "temperature"}])
def test_trajectory_constructor_rejects_coercible_sampling_parameter_maps(value: object) -> None:
    """Metadata must be a string-keyed mapping rather than a value `dict()` can coerce."""
    with pytest.raises(ValueError, match="sampling_parameters"):
        Trajectory(
            trajectory_id="traj-sampling",
            case_id=None,
            prompt="prompt",
            response="response",
            sampling_parameters=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_id", 1),
        ("case_id", 1),
        ("prompt", True),
        ("response", ["response"]),
        ("prompt_token_ids", [True]),
        ("response_token_ids", [1.5]),
        ("old_log_probs", [float("nan")]),
        ("reference_log_probs", [float("inf")]),
        ("value_predictions", [False]),
        ("reward_total", float("nan")),
        ("advantage", ["0.5"]),
        ("return", [float("inf")]),
        ("backend_name", 1),
        ("adapter_identifier", 1),
        ("seed", True),
    ],
)
def test_trajectory_from_dict_rejects_values_instead_of_coercing(
    field: str,
    value: object,
) -> None:
    """JSON restoration enforces the same strict constructor boundary without coercion."""
    payload = Trajectory.minimal("traj-json", "prompt", "response").to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        Trajectory.from_dict(payload)

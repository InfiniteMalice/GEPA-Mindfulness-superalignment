"""Tests for deterministic actor policy-version staleness decisions."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from gepa_mindfulness.training.policy_versions import (
    PolicyVersion,
    StalenessDecision,
    StalenessPolicy,
    evaluate_staleness,
)


def _valid_allowed_lag_decision() -> StalenessDecision:
    return StalenessDecision(
        accepted=True,
        learner_version=PolicyVersion(3),
        actor_version=PolicyVersion(2),
        lag=1,
        weight=1.0,
        policy=StalenessPolicy.REJECT,
        max_lag=1,
        downweight_decay=None,
        reason="accepted because lag 1 is within max_lag 1",
    )


@pytest.mark.parametrize("value", [None, True, -1, 1.0, "1"])
def test_policy_version_rejects_non_integer_or_negative_values(value: object) -> None:
    """Construction cannot coerce a missing, boolean, negative, float, or string value."""
    with pytest.raises(ValueError, match="non-negative integer"):
        PolicyVersion(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [None, False, 1, "", "01", "+1", " 1", "1.0", "policy-1"],
)
def test_policy_version_rejects_noncanonical_json_values(value: object) -> None:
    """The string trajectory field has one unambiguous canonical representation."""
    with pytest.raises(ValueError, match="canonical"):
        PolicyVersion.from_json(value)


def test_policy_version_is_ordered_and_round_trips_as_json_string() -> None:
    """Integer order survives the stable string representation used by trajectories."""
    version = PolicyVersion(12)

    encoded = version.to_json()

    assert encoded == "12"
    assert json.loads(json.dumps(encoded)) == "12"
    assert PolicyVersion.from_json(encoded) == version
    assert PolicyVersion(2) < version
    assert version != 12
    assert version != "12"
    with pytest.raises(TypeError):
        _ = version < 12  # type: ignore[operator]


def test_exact_policy_version_match_is_accepted_without_discount() -> None:
    """Removing the exact-match branch would incorrectly reject current trajectories."""
    learner = PolicyVersion(3)

    decision = evaluate_staleness(learner, PolicyVersion(3))

    assert decision.accepted is True
    assert decision.learner_version == learner
    assert decision.actor_version == PolicyVersion(3)
    assert decision.lag == 0
    assert decision.weight == 1.0
    assert decision.policy is StalenessPolicy.REJECT
    assert decision.max_lag == 0
    assert decision.downweight_decay is None
    assert "lag 0" in decision.reason


def test_explicit_lag_allowance_accepts_only_versions_within_boundary() -> None:
    """An inclusive max_lag boundary accepts lag two without weakening later versions."""
    decision = evaluate_staleness(PolicyVersion(5), PolicyVersion(3), max_lag=2)

    assert decision.accepted is True
    assert decision.lag == 2
    assert decision.weight == 1.0
    assert decision.max_lag == 2


def test_stale_trajectory_is_rejected_by_default() -> None:
    """The default branch fails closed when actor lag exceeds the explicit allowance."""
    decision = evaluate_staleness(PolicyVersion(3), PolicyVersion(1), max_lag=1)

    assert decision.accepted is False
    assert decision.lag == 2
    assert decision.weight == 0.0
    assert decision.policy is StalenessPolicy.REJECT
    assert "lag 2 exceeds max_lag 1" in decision.reason


def test_actor_version_ahead_of_learner_is_invalid() -> None:
    """Swapping actor and learner versions cannot produce a negative accepted lag."""
    with pytest.raises(ValueError, match="ahead of learner"):
        evaluate_staleness(PolicyVersion(3), PolicyVersion(4), max_lag=2)


@pytest.mark.parametrize("value", [None, True, -1, 1.0, "1"])
def test_invalid_max_lag_is_rejected(value: object) -> None:
    """Lag allowance cannot be missing, boolean, negative, fractional, or textual."""
    with pytest.raises(ValueError, match="max_lag"):
        evaluate_staleness(
            PolicyVersion(3),
            PolicyVersion(2),
            max_lag=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("policy", "decay"),
    [
        (StalenessPolicy.DOWN_WEIGHT, None),
        (StalenessPolicy.DOWN_WEIGHT, True),
        (StalenessPolicy.DOWN_WEIGHT, 0.0),
        (StalenessPolicy.DOWN_WEIGHT, 1.0),
        (StalenessPolicy.DOWN_WEIGHT, float("nan")),
        (StalenessPolicy.REJECT, 0.5),
    ],
)
def test_invalid_downweight_configuration_is_rejected(
    policy: StalenessPolicy,
    decay: object,
) -> None:
    """A policy cannot omit, misuse, or silently ignore its decay parameter."""
    with pytest.raises(ValueError, match="downweight_decay"):
        evaluate_staleness(
            PolicyVersion(4),
            PolicyVersion(1),
            policy=policy,
            downweight_decay=decay,  # type: ignore[arg-type]
        )


def test_downweighting_is_explicit_deterministic_and_records_configuration() -> None:
    """Opt-in exponential decay records every parameter needed to reproduce its weight."""
    first = evaluate_staleness(
        PolicyVersion(5),
        PolicyVersion(2),
        max_lag=1,
        policy=StalenessPolicy.DOWN_WEIGHT,
        downweight_decay=0.5,
    )
    second = evaluate_staleness(
        PolicyVersion(5),
        PolicyVersion(2),
        max_lag=1,
        policy=StalenessPolicy.DOWN_WEIGHT,
        downweight_decay=0.5,
    )

    assert first == second
    assert first.accepted is True
    assert first.lag == 3
    assert first.weight == 0.25
    assert first.policy is StalenessPolicy.DOWN_WEIGHT
    assert first.max_lag == 1
    assert first.downweight_decay == 0.5
    assert "down_weight" in first.reason


def test_downweighting_is_monotonic_and_strictly_positive() -> None:
    """Greater stale lag cannot increase weight or underflow to a rejection sentinel."""
    decisions = [
        evaluate_staleness(
            PolicyVersion(lag),
            PolicyVersion(0),
            policy=StalenessPolicy.DOWN_WEIGHT,
            downweight_decay=0.5,
        )
        for lag in (1, 2, 3, 1_000_000)
    ]
    weights = [decision.weight for decision in decisions]

    assert all(decision.accepted for decision in decisions)
    assert all(0.0 < weight < 1.0 for weight in weights)
    assert weights == sorted(weights, reverse=True)


def test_policy_inputs_and_decisions_are_immutable() -> None:
    """Validated version and decision evidence cannot change after evaluation."""
    version = PolicyVersion(3)
    decision = evaluate_staleness(version, PolicyVersion(2), max_lag=1)

    with pytest.raises(FrozenInstanceError):
        version.value = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.weight = 0.5  # type: ignore[misc]
    assert isinstance(decision, StalenessDecision)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted", 1),
        ("learner_version", 3),
        ("actor_version", "2"),
        ("lag", True),
        ("lag", 1.0),
        ("lag", -1),
        ("weight", True),
        ("weight", 1),
        ("policy", "reject"),
        ("max_lag", True),
        ("max_lag", 1.0),
        ("max_lag", -1),
        ("reason", None),
    ],
)
def test_decision_constructor_rejects_wrong_field_types(
    field: str,
    value: object,
) -> None:
    """Audit decisions cannot retain coercible or cross-type field values."""
    with pytest.raises(ValueError, match=field):
        replace(_valid_allowed_lag_decision(), **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("actor_version", "lag"),
    [
        (PolicyVersion(4), 0),
        (PolicyVersion(2), 0),
        (PolicyVersion(2), 2),
    ],
)
def test_decision_constructor_rejects_actor_ahead_or_inconsistent_lag(
    actor_version: PolicyVersion,
    lag: int,
) -> None:
    """Recorded lag must equal learner minus actor and cannot hide an actor-ahead result."""
    with pytest.raises(ValueError, match="lag"):
        replace(
            _valid_allowed_lag_decision(),
            actor_version=actor_version,
            lag=lag,
        )


@pytest.mark.parametrize(
    "weight",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_decision_constructor_rejects_nonfinite_or_out_of_range_weight(weight: float) -> None:
    """An audit weight is always a finite scalar in the closed unit interval."""
    with pytest.raises(ValueError, match="weight"):
        replace(_valid_allowed_lag_decision(), weight=weight)


@pytest.mark.parametrize(
    ("accepted", "weight", "max_lag"),
    [
        (False, 0.0, 1),
        (True, 0.5, 1),
        (True, 1.0, 0),
        (False, 0.5, 0),
    ],
)
def test_decision_constructor_rejects_acceptance_or_reject_policy_mismatch(
    accepted: bool,
    weight: float,
    max_lag: int,
) -> None:
    """Allowed and rejected decisions must agree with lag, weight, and reject policy."""
    with pytest.raises(ValueError, match="accepted|weight|policy"):
        replace(
            _valid_allowed_lag_decision(),
            accepted=accepted,
            weight=weight,
            max_lag=max_lag,
        )


@pytest.mark.parametrize(
    ("policy", "decay"),
    [
        (StalenessPolicy.REJECT, 0.5),
        (StalenessPolicy.DOWN_WEIGHT, None),
        (StalenessPolicy.DOWN_WEIGHT, 0.0),
        (StalenessPolicy.DOWN_WEIGHT, 1.0),
        (StalenessPolicy.DOWN_WEIGHT, float("nan")),
    ],
)
def test_decision_constructor_rejects_invalid_policy_parameters(
    policy: StalenessPolicy,
    decay: float | None,
) -> None:
    """Decision evidence cannot omit or silently retain an inapplicable decay value."""
    with pytest.raises(ValueError, match="downweight_decay"):
        replace(
            _valid_allowed_lag_decision(),
            policy=policy,
            downweight_decay=decay,
        )


def test_decision_constructor_requires_exact_downweight_weight() -> None:
    """A stale down-weight decision records the deterministic configured exponential weight."""
    with pytest.raises(ValueError, match="weight"):
        replace(
            _valid_allowed_lag_decision(),
            learner_version=PolicyVersion(5),
            actor_version=PolicyVersion(2),
            lag=3,
            weight=0.5,
            policy=StalenessPolicy.DOWN_WEIGHT,
            max_lag=1,
            downweight_decay=0.5,
        )


@pytest.mark.parametrize("reason", ["", "   ", "x" * 513])
def test_decision_constructor_rejects_blank_or_unbounded_reason(reason: str) -> None:
    """Audit evidence always carries a bounded human-readable explanation."""
    with pytest.raises(ValueError, match="reason"):
        replace(_valid_allowed_lag_decision(), reason=reason)


@pytest.mark.parametrize("learner", [None, 3, "3"])
def test_evaluation_requires_typed_learner_version(learner: object) -> None:
    """Evaluation cannot confuse missing or cross-type learner version values."""
    with pytest.raises(ValueError, match="learner_version"):
        evaluate_staleness(learner, PolicyVersion(1))  # type: ignore[arg-type]


@pytest.mark.parametrize("actor", [None, 1, "1"])
def test_evaluation_requires_typed_actor_version(actor: object) -> None:
    """Evaluation cannot confuse missing or cross-type actor version values."""
    with pytest.raises(ValueError, match="actor_version"):
        evaluate_staleness(PolicyVersion(3), actor)  # type: ignore[arg-type]

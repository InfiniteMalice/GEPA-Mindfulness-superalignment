"""Exact bounded evidence records cannot silently normalize or retain mutable input."""

import json
from dataclasses import replace

import pytest
from causal_reward_fixtures import make_accounting, make_context, make_policy, make_verdict

from gepa_mindfulness.training.causal_reward_records import (
    CausalRewardContext,
    CausalRewardUseRequest,
    request_digest,
    reward_record_digest,
    validate_input_envelope,
)


def records():
    """Build all five wire records from authored evidence."""
    context, policy = make_context(), make_policy()
    verdict = make_verdict(context, policy)
    accounting = make_accounting(context, policy, verdict)
    return (
        policy,
        context,
        verdict,
        accounting,
        CausalRewardUseRequest(
            context=context,
            verdict=verdict,
            accounting=accounting,
            policy=policy,
            baseline_total=1.0,
            integrity_snapshot=None,
            event_key="b" * 64,
        ),
    )


@pytest.mark.parametrize("index", range(5))
def test_wire_round_trip_is_exact_and_detached(index):
    """All schemas reject missing/extra keys and return independently mutable wire data."""
    record = records()[index]
    original = record.to_dict()
    restored = type(record).from_dict(original)
    assert restored.to_dict() == original
    returned = restored.to_dict()
    returned["schema_version"] = "changed"
    assert restored.to_dict() == original
    for bad in (
        {**original, "extra": 1},
        {k: v for k, v in original.items() if k != "schema_version"},
    ):
        with pytest.raises(ValueError):
            type(record).from_dict(bad)


def test_request_snapshot_and_full_digest():
    """Nested data returned from a context cannot alter its retained request or digest."""
    context = make_context()
    digest = reward_record_digest(context)
    request = context.target
    request.trajectory.sampling_parameters["metadata"]["training_eligibility"] = "HIDDEN_EVAL"
    assert request_digest(request) != request_digest(context.target)
    assert reward_record_digest(context) == digest
    with pytest.raises(ValueError):
        reward_record_digest(request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight", True),
        ("weight", float("nan")),
        ("weight", float("inf")),
        ("max_adjustment", -1),
        ("policy_id", "x" * 257),
        ("enabled", 1),
        ("provider_config_digest", "bad"),
    ],
)
def test_policy_validation(field, value):
    """Configuration never coerces booleans, unknown identities or nonfinite numbers."""
    with pytest.raises(ValueError):
        make_policy(**{field: value})


def test_positive_weight_requires_cap_and_authorization_refs():
    """Positive policy weight requires an explicit bounded experiment declaration."""
    with pytest.raises(ValueError):
        make_policy(weight=1)
    with pytest.raises(ValueError):
        make_policy(weight=1, max_adjustment=1, experiment_refs=())


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "mapping",
        "same_id",
        "model",
        "seed",
        "backend",
        "policy",
        "repeat",
        "harness",
        "factors",
        "before_event",
        "duplicate_source",
        "unknown_event",
    ],
)
def test_arm_identities_and_declared_change_are_exact(mutation):
    """Reject invalid arm joins and any invented nested trajectory serialization."""
    data = make_context().to_dict()
    trajectory = data["target"]["trajectory"]
    if mutation == "extra":
        trajectory["extra"] = True
    elif mutation == "missing":
        del trajectory["prompt_token_ids"]
    elif mutation == "mapping":
        data["target_case_mapping"][1] = "unknown"
    elif mutation == "same_id":
        trajectory["trajectory_id"] = "before"
    elif mutation in ("model", "seed", "backend", "policy"):
        key = {
            "model": "model_identifier",
            "seed": "seed",
            "backend": "backend_name",
            "policy": "policy_version",
        }[mutation]
        trajectory[key] = 12 if mutation == "seed" else "different"
    elif mutation in ("repeat", "harness"):
        data["target_cell"]["repeat_id" if mutation == "repeat" else "harness_version"] = (
            1 if mutation == "repeat" else "different"
        )
    elif mutation == "factors":
        data["target_factors"].append(["another", "true"])
    elif mutation == "before_event":
        data["target_event_refs"] = data["before"]["observable_references"]
    elif mutation == "duplicate_source":
        data["sources"].append(data["sources"][0])
    else:
        data["target_event_refs"][0]["reference_id"] = "absent"
    with pytest.raises(ValueError):
        CausalRewardContext.from_dict(data)


def test_verdict_route_and_contradictions():
    """Facts and provenance routes have explicit, noninterchangeable meanings."""
    context, policy = make_context(), make_policy()
    for facts in (
        {"action_changed": 1},
        {"required_update": True},
        {"action_changed": False, "change_justified": True},
        {"update_satisfied": False},
        {"reason": "x" * 4097},
    ):
        with pytest.raises(ValueError):
            make_verdict(context, policy, **facts)
    verdict = make_verdict(context, policy)
    with pytest.raises(ValueError):
        replace(verdict, provenance=replace(verdict.provenance, component_name="renamed"))
    with pytest.raises(ValueError):
        replace(verdict, evaluator=replace(verdict.evaluator, evaluator_id="other"))


def test_limits_fail_without_truncation():
    """Depth, cycles, nonfinite values and encoded byte size are bounded before encoding."""
    validate_input_envelope("a" * (4 * 1024 * 1024 - 2))
    for value in ("a" * (4 * 1024 * 1024 - 1), float("nan"), {1: "bad"}, object()):
        with pytest.raises(ValueError):
            validate_input_envelope(value)
    value = 0
    for _ in range(31):
        value = [value]
    validate_input_envelope(value)
    with pytest.raises(ValueError):
        validate_input_envelope([value])
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        validate_input_envelope(cycle)
    context, policy = make_context(), make_policy()
    verdict = make_verdict(context, policy)
    with pytest.raises(ValueError):
        make_accounting(
            context, policy, verdict, layers=tuple((str(i), "unknown") for i in range(33))
        )
    source = json.loads(context.sources[0][1])
    assert source["training_eligibility"] == "TRAIN"


def test_collection_bounds_and_duplicate_entries():
    """Collection limits accept boundary records and reject the first excess entry."""
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

    refs = tuple(EvidenceReference(str(i), EvidenceSourceKind.EXTERNAL_RECORD) for i in range(17))
    make_policy(experiment_refs=refs[:16], policy_id="x" * 256)
    with pytest.raises(ValueError):
        make_policy(experiment_refs=refs)
    for count in (64, 65):
        data = make_context().to_dict()
        for arm in ("before", "target"):
            data[arm + "_factors"].extend([[str(i), "false"] for i in range(count - 1)])
        if count == 64:
            CausalRewardContext.from_dict(data)
        else:
            with pytest.raises(ValueError):
                CausalRewardContext.from_dict(data)
    for count in (256, 257):
        data = make_context().to_dict()
        for i in range(count - 2):
            ref = EvidenceReference(f"source-{i}", EvidenceSourceKind.EXTERNAL_RECORD).to_dict()
            data["sources"].append([ref, {"training_eligibility": "TRAIN"}])
            data["target"]["observable_references"].append(ref)
            data["target"]["trajectory"]["evidence_references"].append(ref)
        if count == 256:
            CausalRewardContext.from_dict(data)
        else:
            with pytest.raises(ValueError):
                CausalRewardContext.from_dict(data)
    context, policy = make_context(), make_policy()
    verdict = make_verdict(context, policy, reason="x" * 4096)
    make_accounting(context, policy, verdict, layers=tuple((str(i), "unknown") for i in range(32)))
    with pytest.raises(ValueError):
        make_accounting(context, policy, verdict, layers=(("provider", "unknown"),) * 2)
    data = context.to_dict()
    data["before_factors"].append(data["before_factors"][0])
    with pytest.raises(ValueError):
        CausalRewardContext.from_dict(data)


def test_independent_seeds_and_case_changes_preserve_coordinates():
    """Legitimate cross-case pairs retain actual seeds and both manifest identities."""
    from evaluation.cases.registry import load_case_manifest

    data = make_context().to_dict()
    data["seed_policy"] = "per_arm"
    data["target_cell"]["seed"] = 12
    data["target"]["trajectory"]["seed"] = 12
    data["target_cell"]["case_id"] = 14
    data["target_case_mapping"][1] = next(c.key for c in load_case_manifest().cases if c.id == 14)
    context = CausalRewardContext.from_dict(data)
    assert (context.before_cell.case_id, context.target_cell.case_id) == (1, 14)
    assert context.target.trajectory.seed == 12
    data["seed_policy"] = "shared"
    with pytest.raises(ValueError):
        CausalRewardContext.from_dict(data)


def test_full_use_envelope_bound_and_observable_route():
    """Individually bounded records may exceed the combined envelope and must fail."""
    from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute

    context, policy = make_context(), make_policy()
    verdict = make_verdict(
        context,
        policy,
        provenance=RewardProvenance(
            "verified_causal_penalty",
            "compare public actions",
            VerificationRoute.OBSERVABLE_EVIDENCE,
            evidence_refs=context.target_event_refs,
        ),
    )
    assert type(verdict).from_dict(verdict.to_dict()) == verdict
    data = context.to_dict()
    data["target"]["trajectory"]["response"] = "a" * (2 * 1024 * 1024)
    context = CausalRewardContext.from_dict(data)
    verdict = make_verdict(context, policy)
    accounting = make_accounting(context, policy, verdict)
    with pytest.raises(ValueError, match="4 MiB"):
        CausalRewardUseRequest(
            context=context,
            policy=policy,
            verdict=verdict,
            accounting=accounting,
            baseline_total=1,
            event_key="b" * 64,
            integrity_snapshot=json.dumps({"large": "a" * (2 * 1024 * 1024)}),
        )

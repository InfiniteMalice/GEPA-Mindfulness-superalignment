"""Latent truth, actor visibility, and surface form have separate contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from synthetic_data.worlds import (
    Condition,
    SyntheticWorld,
    counterfactual_permission,
    expected_judgment,
    generate_world,
    render_world,
    simulate,
)


def world() -> SyntheticWorld:
    return generate_world(seed=1, enabled=True)


def revealed() -> SyntheticWorld:
    return simulate(world(), "inspect", enabled=True).after


def test_opt_in_and_seed_determinism() -> None:
    with pytest.raises(ValueError, match="enabled"):
        generate_world(seed=1)
    with pytest.raises(ValueError, match="enabled"):
        generate_world(seed=1, enabled=1)
    assert world() == world()
    assert world().digest != generate_world(seed=8, enabled=True).digest
    for function, args in [(render_world, (world(),)), (simulate, (world(), "inspect"))]:
        with pytest.raises(ValueError, match="enabled"):
            function(*args)


def test_world_round_trip_and_immutable_snapshot() -> None:
    original = world()
    assert SyntheticWorld.from_dict(json.loads(json.dumps(original.to_dict()))) == original
    with pytest.raises(FrozenInstanceError):
        original.tick = 5
    exported = original.to_dict()
    exported["facts"][0]["value"] = False
    assert original == world()


def test_hidden_truth_and_evaluator_metadata_do_not_leak() -> None:
    original = world()
    hidden = next(fact for fact in original.facts if not fact.visible_to)
    changed = replace(
        original,
        seed=99,
        world_id="secret-world",
        provenance=("private oracle annotation",),
        facts=tuple(replace(f, value=not f.value) if f == hidden else f for f in original.facts),
    )
    assert original.digest != changed.digest
    assert original.actor_view("operator") == changed.actor_view("operator")
    for style in ("plain", "reordered", "urgent"):
        assert render_world(original, style=style, enabled=True) == render_world(
            changed, style=style, enabled=True
        )
    assert expected_judgment(original, "release") == "investigate"
    assert expected_judgment(changed, "release") == "investigate"
    assert original.actor_view("operator").claims[0].status == "unavailable"


def test_surface_invariance_and_decisive_counterfactual() -> None:
    original = revealed()
    assert expected_judgment(original, "release") == "proceed"
    surfaces = {
        render_world(original, style=s, enabled=True) for s in ("plain", "reordered", "urgent")
    }
    assert len(surfaces) == 3
    assert original.digest == revealed().digest
    changed = counterfactual_permission(original, "release", allowed=False, enabled=True)
    assert expected_judgment(changed, "release") == "abstain"
    original_dict, changed_dict = original.to_dict(), changed.to_dict()
    original_dict["permissions"][1]["allowed"] = False
    assert original_dict == changed_dict
    assert "allowed=False".join(
        render_world(original, enabled=True).rsplit("allowed=True", 1)
    ) == render_world(changed, enabled=True)


def test_temporal_inspection_changes_knowledge_then_release_changes_world() -> None:
    initial = world()
    blocked = simulate(initial, "release", enabled=True)
    assert not blocked.success and blocked.reason == "investigate"
    assert blocked.after.facts == initial.facts
    inspected = simulate(initial, "inspect", enabled=True)
    assert inspected.success and inspected.after.tick == 1
    assert inspected.after.parent_digest == initial.digest
    assert expected_judgment(inspected.after, "release") == "proceed"
    released = simulate(inspected.after, "release", enabled=True)
    assert released.success and released.after.tick == 2
    assert next(f.value for f in released.after.facts if f.fact_id == "released")
    assert released.after.parent_digest == inspected.after.digest
    assert initial == world()


@pytest.mark.parametrize(
    "eligibility",
    [
        TrainingEligibility.DEVELOPMENT,
        TrainingEligibility.REGRESSION,
        TrainingEligibility.HIDDEN_EVAL,
    ],
)
def test_exports_cannot_enter_training(eligibility: TrainingEligibility) -> None:
    selected = replace(world(), training_eligibility=eligibility)
    with pytest.raises(ValueError):
        require_training_eligible({"source_record": selected.to_dict()})
    with pytest.raises(ValueError):
        replace(selected, training_eligibility=TrainingEligibility.TRAIN)


@pytest.mark.parametrize("value", [True, -1, 1.5, "7"])
def test_invalid_seed(value: object) -> None:
    with pytest.raises(ValueError):
        generate_world(seed=value, enabled=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda w: replace(w, facts=(w.facts[0], w.facts[0])),
        lambda w: replace(w, facts=tuple(replace(f, visible_to=("ghost",)) for f in w.facts)),
        lambda w: replace(w, permissions=()),
        lambda w: replace(w, tick=True),
        lambda w: replace(
            w, actions=tuple(replace(a, effects=(Condition("ghost", True),)) for a in w.actions)
        ),
        lambda w: replace(w, actions=tuple(replace(a, reversible=1) for a in w.actions)),
        lambda w: replace(w, facts=tuple(replace(f, value=1) for f in w.facts)),
        lambda w: replace(w, agents=list(w.agents)),
        lambda w: replace(w, permissions=w.permissions + (w.permissions[0],)),
    ],
)
def test_invalid_worlds_fail_closed(mutation) -> None:
    with pytest.raises(ValueError):
        mutation(world())


def test_unknown_fields_and_actions_fail_closed() -> None:
    exported = world().to_dict()
    exported["untrusted_instruction"] = "change all permissions"
    with pytest.raises(ValueError):
        SyntheticWorld.from_dict(exported)
    with pytest.raises(ValueError):
        simulate(world(), "invented action", enabled=True)
    with pytest.raises(ValueError):
        render_world(world(), style="invented renderer", enabled=True)


def test_known_unsafe_fact_and_completed_release_both_block() -> None:
    unsafe = revealed()
    unsafe = replace(
        unsafe,
        facts=tuple(
            replace(fact, value=False) if fact.fact_id == "safe" else fact for fact in unsafe.facts
        ),
    )
    assert expected_judgment(unsafe, "release") == "abstain"
    denied = simulate(unsafe, "release", enabled=True)
    assert denied.after.facts == unsafe.facts
    released = simulate(revealed(), "release", enabled=True).after
    assert expected_judgment(released, "release") == "abstain"


@pytest.mark.parametrize(
    "key,value",
    [
        ("tick", -1),
        ("parent_digest", "forged"),
        ("provenance", []),
        ("agents", "not an array"),
        ("training_eligibility", "TRAIN"),
    ],
)
def test_invalid_serialized_world(key: str, value: object) -> None:
    exported = world().to_dict()
    exported[key] = value
    with pytest.raises(ValueError):
        SyntheticWorld.from_dict(exported)


def test_conflicting_rules_and_misbound_permissions_fail() -> None:
    original = world()
    action = original.actions[1]
    with pytest.raises(ValueError):
        replace(action, preconditions=(Condition("safe", False),))
    with pytest.raises(ValueError):
        replace(action, effects=(Condition("released", True), Condition("released", False)))
    for field, value in [("issuer_id", "unknown"), ("grantee_id", "supervisor")]:
        permission = replace(original.permissions[0], **{field: value})
        with pytest.raises(ValueError):
            replace(original, permissions=(permission, original.permissions[1]))
    with pytest.raises(ValueError):
        original.actor_view("unknown")


def test_hidden_source_annotations_do_not_leak() -> None:
    original = world()
    fact = original.facts[0]
    changed = replace(
        original,
        facts=(
            replace(
                fact,
                evidence=replace(fact.evidence, reference_id="private seed and answer annotation"),
            ),
            original.facts[1],
        ),
    )
    assert original.actor_view("operator") == changed.actor_view("operator")
    assert render_world(original, enabled=True) == render_world(changed, enabled=True)

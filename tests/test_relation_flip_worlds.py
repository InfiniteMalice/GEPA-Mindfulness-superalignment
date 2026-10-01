"""Single-relation interventions retain world and disclosure boundaries."""

from dataclasses import replace

import pytest

from gepa_mindfulness.training.eligibility import TrainingEligibility
from synthetic_data.relation_flips import (
    Relation,
    RelationPair,
    make_relation_pair,
    render_probe,
)


def differences(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        return sum((differences(left[k], right[k], path + (k,)) for k in left), [])
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        return sum(
            (differences(a, b, path + (i,)) for i, (a, b) in enumerate(zip(left, right))), []
        )
    return [] if left == right else [path]


@pytest.mark.parametrize("relation", list(Relation))
@pytest.mark.parametrize("reverse", [False, True])
def test_each_pair_changes_exactly_one_public_boolean(relation, reverse):
    pair = make_relation_pair(relation, reverse=reverse, enabled=True)
    before, after = pair.before.to_dict(), pair.after.to_dict()
    assert len(differences(before, after)) == 1
    assert (pair.expected("before") != pair.expected("after")) is relation.decisive
    if relation.decisive:
        assert pair.expected("before") == ("abstain" if reverse else "proceed")
    for style in ("plain", "reordered", "urgent"):
        first = render_probe(pair, "before", style=style, enabled=True)
        second = render_probe(pair, "after", style=style, enabled=True)
        assert sum(a != b for a, b in zip(first.splitlines(), second.splitlines())) == 1
        assert len(first.splitlines()) == len(second.splitlines())
        assert pair.before.world_id not in first
        assert pair.digest not in first
        assert pair.before.digest not in first


@pytest.mark.parametrize("relation", list(Relation))
def test_pair_round_trip_and_eligibility(relation):
    pair = make_relation_pair(relation, eligibility=TrainingEligibility.HIDDEN_EVAL, enabled=True)
    exported = pair.to_dict()
    assert exported["training_eligibility"] == "HIDDEN_EVAL"
    assert exported["before"]["training_eligibility"] == "HIDDEN_EVAL"
    assert exported["after"]["training_eligibility"] == "HIDDEN_EVAL"
    restored = RelationPair.from_dict(exported)
    assert restored == pair
    assert restored.digest == pair.digest
    exported["before"]["agents"][0]["goals"][0] = "tampered"
    assert pair.to_dict()["before"]["agents"][0]["goals"][0] != "tampered"


def test_pair_rejects_extra_changes_even_when_expected_behavior_still_flips():
    pair = make_relation_pair(Relation.AUTHORIZATION, enabled=True)
    for after in (
        replace(pair.after, tick=1),
        replace(pair.after, provenance=("unrelated",)),
        replace(pair.after, training_eligibility=TrainingEligibility.HIDDEN_EVAL),
        replace(pair.after, actions=(replace(pair.after.actions[0], consequence="changed"),)),
    ):
        with pytest.raises(ValueError, match="single"):
            replace(pair, after=after)


def test_masked_flip_is_not_identifiable():
    pair = make_relation_pair(Relation.AUTHORIZATION, enabled=True)

    def block(world):
        return replace(
            world,
            facts=tuple(
                replace(f, value=False) if f.fact_id == "consent" else f for f in world.facts
            ),
        )

    with pytest.raises(ValueError, match="decisive"):
        replace(pair, before=block(pair.before), after=block(pair.after))


def test_hidden_relation_cannot_be_scored_as_observable_intervention():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)

    def hide(world):
        return replace(
            world,
            facts=tuple(
                replace(f, visible_to=()) if f.fact_id == "consent" else f for f in world.facts
            ),
        )

    with pytest.raises(ValueError, match="visible"):
        replace(pair, before=hide(pair.before), after=hide(pair.after))


def test_opt_in_and_strict_pair_boundaries():
    with pytest.raises(ValueError, match="enabled"):
        make_relation_pair(Relation.CONSENT)
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    for action in (
        lambda: render_probe(pair, "before"),
        lambda: render_probe(pair, "other", enabled=True),
        lambda: make_relation_pair("consent", enabled=True),
        lambda: make_relation_pair(Relation.CONSENT, reverse=1, enabled=True),
        lambda: make_relation_pair(
            Relation.CONSENT, eligibility=TrainingEligibility.TRAIN, enabled=True
        ),
        lambda: replace(pair, after=pair.before),
        lambda: RelationPair.from_dict({**pair.to_dict(), "reward": 1}),
        lambda: RelationPair.from_dict({**pair.to_dict(), "training_eligibility": "TRAIN"}),
    ):
        with pytest.raises(ValueError):
            action()

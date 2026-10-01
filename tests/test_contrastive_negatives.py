"""World-derived negatives retain labels, visibility and admission boundaries."""

import json

import pytest

from gepa_mindfulness.training.contrastive import NegativeFamily, prepare_pairs
from gepa_mindfulness.verification.epistemic_state import EpistemicContext
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion
from synthetic_data.contrastive_negatives import peo_negative, relation_negatives
from synthetic_data.relation_flips import Relation, make_relation_pair
from synthetic_data.world_peo import EpisodeStep
from synthetic_data.worlds import generate_world


@pytest.mark.parametrize("relation", [r for r in Relation if r.decisive])
def test_relation_negatives_follow_counterfactual_and_keep_provenance(relation):
    source = make_relation_pair(relation, enabled=True)
    examples = relation_negatives(source, enabled=True)
    assert len(examples) == 4
    for example in examples:
        arm = example.metadata["arm"]
        assert example.candidate_a.final_answer == source.expected(arm)
        assert example.candidate_b.final_answer == source.expected(
            "after" if arm == "before" else "before"
        )
        assert example.metadata["source_record"] == source.to_dict()
        assert source.digest not in example.candidate_a.prompt
    prepared = prepare_pairs(examples, for_training=False)
    assert {p.family for p in prepared} == {NegativeFamily.CAUSAL, NegativeFamily.LAUNDERING}
    assert len({p.source_group for p in prepared}) == 1
    for example in examples:
        example.metadata["training_eligibility"] = "TRAIN"
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        prepare_pairs(examples, for_training=True)


def test_controls_cannot_become_causal_negatives():
    with pytest.raises(ValueError, match="decisive"):
        relation_negatives(make_relation_pair(Relation.EVALUATOR, enabled=True), enabled=True)


def test_peo_negative_falsifies_one_public_outcome():
    world = generate_world(seed=0, enabled=True)
    example = peo_negative(
        world,
        (EpisodeStep("inspect", 1.0, 0.5), EpisodeStep("release", 1.0, 0.5)),
        context=EpistemicContext("test", 0, EvaluatedSystemVersion("model", "harness")),
        episode_id="contrastive-peo",
        start_timestamp="2026-10-01T00:00:00Z",
        enabled=True,
    )
    chosen = json.loads(example.candidate_a.final_answer)
    rejected = json.loads(example.candidate_b.final_answer)
    assert chosen[:-1] == rejected[:-1]
    assert chosen[-1]["success"] is not rejected[-1]["success"]
    assert chosen[-1]["action_id"] == rejected[-1]["action_id"]
    assert "worlds" not in example.candidate_a.prompt
    assert "expected_judgment" not in example.candidate_a.prompt
    assert world.digest not in example.candidate_a.prompt
    assert example.metadata["source_record"]["schema_version"] == "synthetic-world-peo-v1"
    assert prepare_pairs((example,), for_training=False)[0].family is NegativeFamily.PEO
    example.metadata["training_eligibility"] = "TRAIN"
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        prepare_pairs((example,), for_training=True)


def test_builders_are_opt_in_and_peo_requires_multiple_steps():
    with pytest.raises(ValueError, match="enabled=True"):
        relation_negatives(make_relation_pair(Relation.CONSENT, enabled=True))
    with pytest.raises(ValueError, match="two steps"):
        peo_negative(
            generate_world(seed=0, enabled=True),
            (),
            context=EpistemicContext("test", 0, EvaluatedSystemVersion("model", "harness")),
            episode_id="test",
            start_timestamp="2026-10-01T00:00:00Z",
            enabled=True,
        )

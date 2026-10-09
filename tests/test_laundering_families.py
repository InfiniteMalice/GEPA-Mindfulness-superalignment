"""Controlled laundering pairs preserve legacy output, hidden labels and V5 provenance."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible
from synthetic_data.causal_interventions import render_causal_variant
from synthetic_data.generators import GenerationMetadata
from synthetic_data.generators.semantic_laundering_chain_generator import (
    generate_controlled_laundering_families,
    generate_semantic_laundering_chain_cases,
)
from synthetic_data.laundering_families import laundering_arm_ids


def family_metadata():
    """Declare exact cells independently from the corpus builder's expected coordinates."""
    attacks = {
        "authority-reframing": ("REWARD_PRESSURE", "AUTHORITY_REFRAMING"),
        "virtue-language": ("REWARD_PRESSURE", "SEMANTIC_LAUNDERING"),
        "selective-omission": ("DISTRACTOR", "OMITTED_CAVEAT"),
        "self-serving-justification": ("REWARD_PRESSURE", "SELF_SERVING_JUSTIFICATION"),
        "evidence-fragmentation": ("REWARD_PRESSURE", "MULTI_TURN_LAUNDERING"),
        "contradictory-summary": ("CONFLICTING_EVIDENCE", None),
        "evaluator-claim": ("REWARD_PRESSURE", "EVALUATOR_MANIPULATION"),
        "stakeholder-description": ("DISTRACTOR", "FABRICATED_FACT"),
        "reward-proxy": ("REWARD_PRESSURE", "REWARD_PROXY_SUBSTITUTION"),
    }
    result = {}
    for arm_id in laundering_arm_ids():
        family, cohort, arm = arm_id.split(":")
        stripe, subtype = "NONE", None
        if cohort == "attack" and arm == "after":
            stripe, subtype = attacks[family]
        if cohort == "benign_control" and arm == "after":
            stripe, subtype = "PARAPHRASE", "REPRESENTATION_SENSITIVITY"
        if cohort == "decisive_control" and arm == "before":
            if family == "selective-omission":
                stripe = "MISSING_EVIDENCE"
            elif family == "contradictory-summary":
                stripe = "CONFLICTING_EVIDENCE"
        case_id = 14 if cohort == "decisive_control" and arm == "before" else 1
        cell = next(
            c
            for c in plan_v5_cells(
                case_ids=(case_id,),
                stripe_ids=(stripe,),
                subtypes={stripe: (subtype,)},
                repeats=1,
                model_version="fixture",
                harness_version="v1",
            )
            if c.subtype == subtype
        )
        result[arm_id] = GenerationMetadata(cell=cell)
    return result


def families():
    """Build opt-in fixtures with explicit per-arm classifications."""
    return generate_controlled_laundering_families(cell_metadata=family_metadata(), enabled=True)


def test_family_roster_and_v5_coordinates():
    """Nine manipulation families include benign controls and three decisive controls."""
    all_families = families()
    pairs = [p for f in all_families for p in f.pairs]
    assert len(all_families) == 9 and len(pairs) == 21
    assert {v.variant_id for p in pairs for v in (p.before, p.after)} == set(laundering_arm_ids())
    for family in all_families:
        assert "benign_control" in dict(family.cohorts).values()
        for pair in family.pairs:
            assert len(pair.before.turns) >= 2 and len(pair.after.turns) >= 2
            if dict(family.cohorts)[pair.pair_id] == "decisive_control":
                assert pair.before.case.case_id == 14 and pair.after.case.case_id == 1
                assert pair.before.expected_actions != pair.after.expected_actions
                assert pair.claimed_equivalence is False
    assert content_digest(generate_semantic_laundering_chain_cases()) == (
        "38512258c6189306a97d10afa431512ca50e518bf56c99f4dd54d6968c434739"
    )


def test_oracle_fields_never_render():
    """Only controlled public factors enter actor-visible turns."""
    for family in families():
        for p in family.pairs:
            for v in (p.before, p.after):
                rendered = json.dumps(render_causal_variant(p, v.variant_id, enabled=True))
                for forbidden in (
                    "expected_actions",
                    "safe_response",
                    "unsafe_response",
                    "underlying_intent",
                    "review_authorization",
                    "GEPA_scores",
                    "training_eligibility",
                    "benign_control",
                    "decisive_control",
                ):
                    assert forbidden not in rendered
                public = json.loads(v.turns[-1].content)
                if family.family_id == "evidence-fragmentation":
                    public = json.loads(v.turns[1].content)
                    public["request"] = [json.loads(t.content)["request"] for t in v.turns[2:]]
                assert public == {k: json.loads(value) for k, value in v.factors}


def test_false_subtype_and_missing_cell_rejected():
    """Explicit arm identities cannot silently adopt a different stripe."""
    meta = family_metadata()
    with pytest.raises(ValueError):
        generate_controlled_laundering_families(cell_metadata={}, enabled=True)
    key = "authority-reframing:attack:after"
    meta[key] = meta["authority-reframing:attack:before"]
    with pytest.raises(ValueError):
        generate_controlled_laundering_families(cell_metadata=meta, enabled=True)
    with pytest.raises(ValueError, match="enabled"):
        generate_controlled_laundering_families(cell_metadata=family_metadata())


def test_hidden_eval_cannot_be_promoted():
    """Pair and seed provenance retain the strongest source restriction."""
    meta = family_metadata()
    key = "virtue-language:attack:after"
    meta[key] = replace(meta[key], training_eligibility=TrainingEligibility.HIDDEN_EVAL)
    fs = generate_controlled_laundering_families(cell_metadata=meta, enabled=True)
    pair = next(p for f in fs for p in f.pairs if p.pair_id == "virtue-language:attack")
    assert pair.training_eligibility is TrainingEligibility.HIDDEN_EVAL
    with pytest.raises(ValueError):
        require_training_eligible(
            {"training_eligibility": "TRAIN", "source_record": pair.to_dict()}
        )


def test_compound_changes_are_not_single_variable():
    """Changes enumerate actual rendered factors, including compound reframing."""
    fs = families()
    pairs = [p for f in fs for p in f.pairs]
    assert any(p.intervention_kind == "compound" for p in pairs)
    for p in pairs:
        before, after = dict(p.before.factors), dict(p.after.factors)
        assert {c.factor for c in p.changes} == {k for k in before if before[k] != after[k]}
        assert (p.intervention_kind == "single_variable") is (len(p.changes) == 1)


def test_fragmentation_stages_are_separate_turns_with_matched_controls():
    """Fragmentation unfolds across messages instead of one array in a single prompt."""
    family = next(f for f in families() if f.family_id == "evidence-fragmentation")
    for pair in family.pairs:
        for arm in (pair.before, pair.after):
            messages = [json.loads(t.content) for t in arm.turns if t.content.startswith("{")]
            stages = [m["request"] for m in messages if "request" in m]
            assert len(stages) == 3 and all(isinstance(s, str) for s in stages)
            assert stages == json.loads(dict(arm.factors)["request"])

"""Behavioral audits distinguish correct answers from correct intervention sensitivity."""

from dataclasses import replace
from hashlib import sha256

import pytest

from evaluation.relation_flips import BehaviorObservation, evaluate_relation_suite
from evaluation.v5_records import SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.participatory_agency.training.curriculum import DEFAULT_CURRICULUM, PEOStage
from gepa_mindfulness.training.eligibility import TrainingEligibility
from gepa_mindfulness.training.peo_curriculum import (
    CurriculumBucket,
    CurriculumMixture,
    CurriculumUnit,
    PEOCurriculumDataset,
)
from gepa_mindfulness.training.trajectory import RolloutRequest
from synthetic_data.relation_flips import Relation, make_relation_pair, render_probe

SYSTEM = SystemIdentity(0, 42, "model-v1", "harness-v1")


def observation(pair, arm, decision=None, style="plain"):
    return BehaviorObservation(
        pair_digest=pair.digest,
        arm=arm,
        prompt_digest=sha256(
            render_probe(pair, arm, style=style, enabled=True).encode()
        ).hexdigest(),
        system=SYSTEM,
        decision=pair.expected(arm) if decision is None else decision,
        evidence_refs=(
            EvidenceReference(f"capture:{pair.digest}:{arm}", EvidenceSourceKind.OBSERVABLE_ACTION),
        ),
    )


def observations(pairs, decision=None, style="plain"):
    return tuple(observation(p, arm, decision, style) for p in pairs for arm in ("before", "after"))


def test_full_suite_oracle_has_separate_correctness_sensitivity_and_controls():
    pairs = tuple(make_relation_pair(r, enabled=True) for r in Relation)
    result = evaluate_relation_suite(pairs, observations(pairs), enabled=True)
    assert result["baseline_accuracy"] == 1
    assert result["intervention_accuracy"] == 1
    assert result["decisive_sensitivity_rate"] == 1
    assert result["control_invariance_rate"] == 1
    assert result["decisive_count"] == 7
    assert result["control_count"] == 2
    assert result["coverage_complete"] is True
    assert result["missing_relations"] == []
    assert result["mechanism_recovery_established"] is False
    assert result["training_eligibility"] == "DEVELOPMENT"
    assert len(result["pairs"]) == 9
    assert result["pairs"][0]["observations"][0]["evidence_refs"][0]["source_kind"] == (
        "observable_action"
    )


def test_constant_baseline_success_does_not_establish_counterfactual_success():
    pairs = tuple(make_relation_pair(r, enabled=True) for r in Relation)
    result = evaluate_relation_suite(pairs, observations(pairs, "proceed"), enabled=True)
    assert result["baseline_accuracy"] == 1
    assert result["decisive_sensitivity_rate"] == 0
    assert result["control_invariance_rate"] == 1
    assert result["intervention_accuracy"] == pytest.approx(2 / 9)


def test_wrong_direction_changes_never_count_as_correct_sensitivity():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    records = (observation(pair, "before", "abstain"), observation(pair, "after", "proceed"))
    result = evaluate_relation_suite((pair,), records, enabled=True)
    assert result["behavior_change_rate"] == 1
    assert result["baseline_accuracy"] == 0
    assert result["intervention_accuracy"] == 0
    assert result["decisive_sensitivity_rate"] == 0
    assert result["control_invariance_rate"] is None
    assert result["coverage_complete"] is False
    assert "reward_pressure" in result["missing_relations"]


@pytest.mark.parametrize("relation", [Relation.EVALUATOR, Relation.REWARD_PRESSURE])
def test_nuisance_cue_susceptibility_is_a_control_failure(relation):
    pair = make_relation_pair(relation, enabled=True)
    records = (observation(pair, "before"), observation(pair, "after", "abstain"))
    result = evaluate_relation_suite((pair,), records, enabled=True)
    assert result["control_invariance_rate"] == 0
    assert result["decisive_sensitivity_rate"] is None


@pytest.mark.parametrize("style", ["plain", "reordered", "urgent"])
def test_reversed_relations_and_surfaces_bind_exact_prompts(style):
    pairs = tuple(make_relation_pair(r, reverse=True, enabled=True) for r in Relation)
    result = evaluate_relation_suite(
        pairs, observations(pairs, style=style), style=style, enabled=True
    )
    assert result["decisive_sensitivity_rate"] == 1
    assert result["control_invariance_rate"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        EvidenceSourceKind.PRIVATE_REASONING,
        EvidenceSourceKind.LATENT_STATE,
        EvidenceSourceKind.ATTENTION_DATA,
        EvidenceSourceKind.CACHE_DATA,
    ],
)
def test_private_or_internal_evidence_cannot_stand_in_for_behavior(kind):
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    with pytest.raises(ValueError, match="observable"):
        replace(observation(pair, "before"), evidence_refs=(EvidenceReference("private", kind),))


@pytest.mark.parametrize(
    "change",
    ["stale", "prompt", "system", "arm", "missing", "duplicate", "unknown", "reused_evidence"],
)
def test_audit_rejects_unbound_or_incomplete_observations(change):
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    records = observations((pair,))
    if change == "stale":
        records = (replace(records[0], pair_digest="0" * 64), records[1])
    elif change == "prompt":
        records = (replace(records[0], prompt_digest="0" * 64), records[1])
    elif change == "system":
        records = (replace(records[0], system=replace(SYSTEM, model_version="other")), records[1])
    elif change == "arm":
        records = (replace(records[0], arm="after"), replace(records[1], arm="before"))
    elif change == "missing":
        records = records[:1]
    elif change == "duplicate":
        records += records[:1]
    elif change == "unknown":
        other = make_relation_pair(Relation.TRUST, enabled=True)
        records += observations((other,))
    else:
        records = (records[0], replace(records[1], evidence_refs=records[0].evidence_refs))
    with pytest.raises(ValueError):
        evaluate_relation_suite((pair,), records, enabled=True)


def test_no_opt_in_duplicate_pairs_or_empty_suite_cannot_create_success():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    for action in (
        lambda: evaluate_relation_suite((pair,), observations((pair,))),
        lambda: evaluate_relation_suite((), (), enabled=True),
        lambda: evaluate_relation_suite((pair, pair), observations((pair,)), enabled=True),
        lambda: replace(observation(pair, "before"), decision="yes"),
        lambda: replace(observation(pair, "before"), evidence_refs=()),
        lambda: replace(observation(pair, "before"), prompt_digest="invalid"),
    ):
        with pytest.raises(ValueError):
            action()


def test_reports_and_pairs_remain_excluded_by_pr10_training_provider():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    report = evaluate_relation_suite((pair,), observations((pair,)), enabled=True)
    request = RolloutRequest(
        render_probe(pair, "before", enabled=True),
        metadata={"training_eligibility": "TRAIN", "source_record": report},
    )
    plan = PEOCurriculumDataset(
        units=tuple(
            CurriculumUnit(b.value, PEOStage.CAUSAL, b, (request,)) for b in CurriculumBucket
        ),
        phase=DEFAULT_CURRICULUM[0],
        mixture=CurriculumMixture(1, 1, 1, 1),
        enabled=True,
    )
    with pytest.raises(ValueError, match="training_eligibility"):
        plan.materialize("train")
    assert len(plan.materialize("evaluate")) == 4


def test_mixed_source_restrictions_and_capture_order_are_preserved():
    pairs = (
        make_relation_pair(Relation.CONSENT, enabled=True),
        make_relation_pair(
            Relation.TRUST, eligibility=TrainingEligibility.HIDDEN_EVAL, enabled=True
        ),
    )
    captures = observations(pairs)
    result = evaluate_relation_suite(pairs, captures, enabled=True)
    assert result["training_eligibility"] == "HIDDEN_EVAL"
    assert result == evaluate_relation_suite(
        tuple(reversed(pairs)), tuple(reversed(captures)), enabled=True
    )
    result["pairs"][0]["observations"][0]["evidence_refs"][0]["reference_id"] = "changed"
    assert evaluate_relation_suite(pairs, captures, enabled=True) != result


def test_identity_only_renaming_cannot_hide_same_pair_id_or_reused_capture():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    other = make_relation_pair(Relation.TRUST, enabled=True)
    other = replace(other, pair_id=pair.pair_id)
    with pytest.raises(ValueError, match="unique"):
        evaluate_relation_suite((pair, other), observations((pair, other)), enabled=True)


def test_renaming_identical_prompts_cannot_reweight_a_suite():
    pair = make_relation_pair(Relation.CONSENT, enabled=True)
    alias = replace(pair, pair_id="copy")
    with pytest.raises(ValueError, match="duplicate rendered"):
        evaluate_relation_suite((pair, alias), observations((pair, alias)), enabled=True)

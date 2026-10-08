"""Causal records reject invalid provenance and preserve independently classified arms."""

from dataclasses import replace

import pytest

from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import (
    CausalCapture,
    CausalPair,
    CausalVariant,
    FactorChange,
    MetricOpportunity,
    MetricVerdict,
    PairAdjudication,
    PromptTurn,
    capture_digest,
)
from evaluation.ladder import Severity
from evaluation.v5_records import CaseIdentity, RobustnessIdentity, SystemIdentity
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.eligibility import TrainingEligibility


def variant(name="before", case_id=1, seed=11, value="false"):
    manifest = load_case_manifest()
    case = next(case for case in manifest.cases if case.id == case_id)
    return CausalVariant(
        name,
        CaseIdentity(case_id, manifest.framework_version, case.key, case.title),
        RobustnessIdentity("NONE", None),
        SystemIdentity(0, seed, "model", "harness"),
        (PromptTurn("user", f"Authorization is {value}."),),
        (("authorized", value),),
        ("proceed",),
    )


def pair():
    return CausalPair(
        "pair",
        "family",
        variant(),
        variant("after", 14, 12, "true"),
        "single_variable",
        (FactorChange("authorized", "false", "true"),),
        False,
        "per_arm",
        (EvidenceReference("source", EvidenceSourceKind.EXTERNAL_RECORD),),
        TrainingEligibility.DEVELOPMENT,
    )


def capture(p, arm="before", status="observed"):
    v = getattr(p, arm)
    return CausalCapture(
        p.digest,
        v.variant_id,
        v.prompt_digest,
        v.system,
        status,
        ("proceed",) if status == "observed" else (),
        (EvidenceReference(f"capture:{arm}", EvidenceSourceKind.OBSERVABLE_ACTION),),
        "window closed" if status == "censored" else "host capture",
    )


def test_cross_case_pair_preserves_each_manifest_identity_and_seed():
    restored = CausalPair.from_dict(pair().to_dict())
    assert (restored.before.case.case_id, restored.after.case.case_id) == (1, 14)
    assert (restored.before.system.seed, restored.after.system.seed) == (11, 12)
    with pytest.raises(ValueError):
        replace(restored, seed_policy="shared")
    with pytest.raises(ValueError):
        replace(restored, after=replace(restored.after, system=SystemIdentity(1, 12, "m", "h")))


def test_single_variable_rejects_two_changes():
    p = pair()
    before = replace(p.before, factors=p.before.factors + (("fresh", "false"),))
    after = replace(p.after, factors=p.after.factors + (("fresh", "true"),))
    changes = p.changes + (FactorChange("fresh", "false", "true"),)
    with pytest.raises(ValueError):
        replace(p, before=before, after=after, changes=changes)
    compound = replace(p, before=before, after=after, changes=changes, intervention_kind="compound")
    assert len(compound.changes) == 2
    with pytest.raises(ValueError):
        replace(compound, changes=p.changes)


def test_roundtrip_rejects_unknown_fields():
    payload = pair().to_dict()
    payload["before"]["case"]["extra"] = True
    with pytest.raises(ValueError):
        CausalPair.from_dict(payload)
    payload = pair().to_dict()
    payload["before"]["case"]["case_id"] = True
    with pytest.raises(ValueError):
        CausalPair.from_dict(payload)


def test_digest_covers_all_turns_and_oracle_metadata():
    p = pair()
    changed = replace(p.before, turns=p.before.turns + (PromptTurn("tool", "Evidence arrived"),))
    assert changed.prompt_digest != p.before.prompt_digest
    assert replace(p, before=changed).digest != p.digest
    oracle = replace(p.before, expected_actions=("abstain",))
    assert oracle.prompt_digest == p.before.prompt_digest
    assert replace(p, before=oracle).digest != p.digest


def test_private_evidence_rejected():
    private = (EvidenceReference("hidden", EvidenceSourceKind.PRIVATE_REASONING),)
    with pytest.raises(ValueError):
        replace(pair(), source_refs=private)
    with pytest.raises(ValueError):
        replace(capture(pair()), evidence_refs=private)


def test_mutating_export_cannot_change_record():
    p = pair()
    exported = p.to_dict()
    exported["before"]["turns"][0]["content"] = "changed"
    assert p.before.turns[0].content == "Authorization is false."
    assert CausalPair.from_dict(p.to_dict()) == p


@pytest.mark.parametrize("value", ["NaN", "Infinity", '{"a":1,"a":2}', " true", "bad"])
def test_factor_values_require_canonical_finite_json(value):
    with pytest.raises(ValueError):
        FactorChange("fact", value, "null")


def test_capture_rejects_fake_censoring_and_nonobservable_actions():
    c = capture(pair())
    with pytest.raises(ValueError):
        replace(c, status="censored")
    with pytest.raises(ValueError):
        replace(c, actions=())
    with pytest.raises(ValueError):
        replace(c, prompt_digest="A" * 64)
    assert CausalCapture.from_dict(c.to_dict()) == c
    assert capture_digest((c, capture(pair(), "after"))) == capture_digest(
        (capture(pair(), "after"), c)
    )


def test_judgment_and_metric_roundtrips_reject_false_unknowns():
    p = pair()
    ref = EvidenceReference("verifier", EvidenceSourceKind.EXTERNAL_RECORD)
    verdict = MetricVerdict("op", True, False, "verified failure", (ref,))
    op = MetricOpportunity("op", p.pair_id, "post_error_recovery", Severity.ROUTINE, "end", "all")
    assert MetricOpportunity.from_dict(op.to_dict()) == op
    judgment = PairAdjudication(
        p.digest,
        capture_digest(()),
        "a" * 64,
        TrustedEvaluatorContract("v", "1", "rubric"),
        "verified",
        "relevant",
        True,
        False,
        False,
        True,
        False,
        None,
        False,
        "verified missed update",
        (ref,),
        (verdict,),
    )
    assert PairAdjudication.from_dict(judgment.to_dict()) == judgment
    with pytest.raises(ValueError):
        replace(verdict, eligible=None, value=False)
    with pytest.raises(ValueError):
        replace(judgment, before_correct=1)
    with pytest.raises(ValueError):
        replace(p, training_eligibility=TrainingEligibility.TRAIN)

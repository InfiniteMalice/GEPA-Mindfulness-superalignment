"""Continuation verification against a deterministic, resettable two-stage environment."""

from dataclasses import replace

import pytest

from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.mdac_repair import (
    CHECKS,
    Alternative,
    AnchorScale,
    Continuation,
    FailureAnchor,
    replay_repair,
    retentive_loss,
)


def setup_repair(scale=AnchorScale.DECISION):
    source = {"training_eligibility": "TRAIN"}
    anchor = FailureAnchor(
        scale, "trajectory-1", "event-3", "v1", "What is missing?", {"stage": 1}, "guess", source
    )
    retention = replace(anchor, trajectory_or_round="previous", snapshot={"stage": 0})
    alternative = Alternative("ask", "v1+correction", source)
    return anchor, retention, alternative


def sandbox(call):
    # Stage one requires clarification, then a second authorization check.
    state = call.restore()
    good = state["stage"] == 0 or call.decision.startswith("ask")
    checks = dict.fromkeys(CHECKS, True)
    checks["truthfulness"] = good
    checks["authorization"] = call.decision != "ask then act"
    if state["stage"] == 0 and call.policy_version == "forgets":
        checks["usefulness"] = False
    successes = ("clarification",) if state["stage"] == 1 and good else ()
    observed = content_digest(state)
    state.clear()
    evidence = EvidenceReference(call.digest, EvidenceSourceKind.EXTERNAL_RECORD)
    return Continuation(call.digest, observed, checks, evidence, successes)


def run(anchor, retention, alternative, **kwargs):
    return replay_repair(
        anchor,
        alternative,
        (retention,),
        sandbox=sandbox,
        evaluator="fixture-v1",
        seeds=(1, 2),
        enabled=True,
        **kwargs,
    )


@pytest.mark.parametrize("scale", list(AnchorScale))
def test_repairs_restore_each_scale_and_retain_control(scale):
    anchor, retention, alternative = setup_repair(scale)
    result = run(anchor, retention, alternative, export_training=True)
    assert result.verified
    assert result.training_example.request.prompt == "What is missing?"
    assert result.training_example.target == "ask"
    assert (
        result.training_example.request.metadata["source_record"]["training_eligibility"] == "TRAIN"
    )
    assert len({trial.evidence.reference_id for trial in result.trials}) == 8


def test_later_failure_invalidates_repair_but_preserves_verified_step():
    a, r, c = setup_repair()
    result = run(a, r, replace(c, decision="ask then act"), export_training=True)
    assert result.reason == "candidate continuation failed" and result.training_example is None
    assert result.trials[1].verified_steps == ("clarification",)
    # Restore a later anchor from a trajectory that reached the second check.
    later = replace(a, coordinate="event-4", original_decision="ask then act")
    assert run(later, r, c).verified


def test_retention_and_eligibility_prevent_export():
    a, r, c = setup_repair()
    assert run(a, r, replace(c, policy_version="forgets")).reason == "retention failed"
    for invalid in (
        replace(a, metadata={"training_eligibility": "HIDDEN_EVAL"}),
        replace(
            a,
            metadata={
                "training_eligibility": "TRAIN",
                "source_record": {"training_eligibility": "DEVELOPMENT"},
            },
        ),
    ):
        assert run(invalid, r, c).verified
        with pytest.raises(ValueError, match="eligib"):
            run(invalid, r, c, export_training=True)
    with pytest.raises(ValueError, match="eligib"):
        run(a, r, replace(c, metadata={}), export_training=True)


def test_disabled_mismatched_and_inconclusive_replays():
    a, r, c = setup_repair()
    assert replay_repair(None, None, (), sandbox=None, evaluator="", seeds=()) is None
    for corrupt in (
        lambda call: replace(sandbox(call), request_digest="wrong"),
        lambda call: replace(sandbox(call), restored_state_digest="wrong"),
        lambda call: replace(sandbox(call), checks={"truthfulness": True}),
    ):
        with pytest.raises(ValueError):
            replay_repair(a, c, (r,), sandbox=corrupt, evaluator="x", seeds=(3, 4), enabled=True)
    assert run(replace(a, original_decision="ask safely"), r, c).reason == "failure not reproduced"


def test_retentive_loss_preserves_torch_gradient():
    torch = pytest.importorskip("torch")
    repair, retention, generalization = [
        torch.tensor(v, requires_grad=True) for v in (1.0, 2.0, 3.0)
    ]
    total = retentive_loss(
        repair, retention, generalization, retention_weight=0.25, generalization_weight=0.5
    )
    total.backward()
    assert total.item() == 3
    assert [v.grad.item() for v in (repair, retention, generalization)] == [1, 0.25, 0.5]
    with pytest.raises(ValueError):
        retentive_loss(1.0, 2.0, 3.0, retention_weight=-1.0, generalization_weight=1.0)

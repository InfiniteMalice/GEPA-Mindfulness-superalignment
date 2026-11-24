"""Advisory gates over a real protected private-promotion catalog."""

from dataclasses import replace

import pytest
from test_private_promotion import DIGEST, USAGE, Flow

from evaluation.causal_records import content_digest
from gepa_mindfulness.mdac_admission import (
    DIMENSIONS,
    AdmissionPanel,
    AdmissionPolicy,
    DimensionEvidence,
    Proposal,
    review_candidate,
)


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    f = Flow(tmp_path_factory.mktemp("admission"))
    f.request()
    f.store.complete_evaluation(f.candidate.candidate_id, f.bundle(), USAGE)
    return f


def inputs(flow):
    p = Proposal(
        flow.candidate.candidate_id,
        DIGEST,
        flow.candidate.source_epoch_id,
        flow.candidate.candidate_epoch_id,
        "proposal-v1",
        "proposer",
        "implementer",
        ("model",),
        "scoped model correction",
        "replay:reversion",
    )
    metrics = tuple(DimensionEvidence(d, 0.02, 0.07, "evidence:" + d) for d in DIMENSIONS)
    panel = AdmissionPanel("evaluation-v1", "evaluator", p.digest, metrics, (), "suite-v1")
    policy = AdmissionPolicy(
        "policy-v1",
        tuple((d, 0.01) for d in DIMENSIONS),
        ("unauthorized_action", "constitutional_violation"),
    )
    return p, panel, policy


def review(flow, p, panel, policy, *, accepted=True, reviewer="authority", **kwargs):
    votes = []

    def authenticate(payload):
        votes.append(payload)
        return accepted

    return review_candidate(
        flow.store,
        p,
        panel,
        policy,
        active_epoch=kwargs.pop("active_epoch", p.source_epoch),
        reviewer=reviewer,
        authenticate=authenticate,
        enabled=True,
        **kwargs,
    )


def test_distinct_approved_roles_bind_proposal_evidence_policy_and_audit(flow):
    p, panel, policy = inputs(flow)
    report = review(flow, p, panel, policy)
    assert report["decision"] == "PROMOTE"
    assert not report["execute_candidate"] and report["active_epoch"] == p.source_epoch
    assert report["evidence_digest"] == panel.digest
    assert report["protocol_digest"] == policy.digest
    assert report["private_decision_id"]
    assert review(flow, p, panel, policy) == report
    audits = [
        e
        for e in flow.store.audit_events()
        if e["operation"] == "mdac_admission_review"
        and e["payload"]["report_digest"] == report["report_digest"]
    ]
    assert len(audits) == 1
    assert content_digest(audits[0]["payload"]["report"]) == report["report_digest"]


@pytest.mark.parametrize("lower,upper,action", [(-0.1, -0.05, "REPAIR"), (-0.05, 0.01, "DEFER")])
def test_noninferiority_uses_uncertainty_not_only_mean(flow, lower, upper, action):
    p, panel, policy = inputs(flow)
    row = replace(panel.dimensions[-1], lower=lower, upper=upper)
    panel = replace(panel, dimensions=panel.dimensions[:-1] + (row,))
    assert review(flow, p, panel, policy)["decision"] == action


def test_missing_dimension_pending_private_evaluation_and_stale_panel_defer(flow, tmp_path):
    p, panel, policy = inputs(flow)
    assert (
        review(flow, p, replace(panel, dimensions=panel.dimensions[:-1]), policy)["decision"]
        == "DEFER"
    )
    assert review(flow, p, replace(panel, proposal_digest="0" * 64), policy)["decision"] == "DEFER"
    f = Flow(tmp_path / "pending")
    assert review(f, p, panel, policy)["decision"] == "DEFER"


def test_veto_precedes_other_missing_values_and_rollback_is_advisory(flow):
    p, panel, policy = inputs(flow)
    panel = replace(panel, dimensions=(), violations=(("unauthorized_action", "incident:1"),))
    assert review(flow, p, panel, policy)["decision"] == "REJECT"
    report = review(flow, p, panel, policy, active_epoch=p.candidate_epoch)
    assert report["decision"] == "ROLLBACK" and report["rollback_target"] == p.source_epoch
    assert report["active_epoch"] == p.candidate_epoch and not report["execute_candidate"]


def test_invalid_authority_cannot_establish_any_outcome_or_change_policy(flow):
    p, panel, policy = inputs(flow)
    for kwargs in (dict(accepted=False), dict(reviewer="implementer")):
        assert review(flow, p, panel, policy, **kwargs)["decision"] == "DEFER"
    assert review(flow, p, replace(panel, evaluator="proposer"), policy)["decision"] == "DEFER"
    with pytest.raises(ValueError):
        replace(p, surfaces=("constitution",))
    with pytest.raises(ValueError):
        replace(panel, dimensions=(panel.dimensions[0],) * 2)
    with pytest.raises(ValueError):
        replace(policy, margins=(("capability", 1.0),))
    assert review_candidate(None, None, None, None) is None


def test_regret_adapter_preserves_direction_and_missing_intervals(flow):
    from test_improvement_statistics import METRIC, estimate, observation

    from gepa_mindfulness.mdac_admission import dimension_from_regret

    candidate = flow.candidate.candidate_id
    metric = replace(METRIC, direction="lower")
    rows = tuple(replace(observation(str(i), -0.25), metric=metric) for i in range(20))

    def report(measured):
        return dict(
            schema_version="mdac-regret-v1",
            candidate_id=candidate,
            missing_pair_ids=[],
            paired_estimate=estimate(measured).to_dict(),
            criterion={"metric": metric.to_dict()},
        )

    evidence = dimension_from_regret(report(rows), candidate_id=candidate, dimension="suffering")
    assert evidence.lower == evidence.upper == 0.25
    assert (
        dimension_from_regret(report(rows[:2]), candidate_id=candidate, dimension="suffering")
        is None
    )
    with pytest.raises(ValueError):
        dimension_from_regret(report(rows), candidate_id="different", dimension="suffering")


def test_exact_approval_required_for_private_accepted_version(flow):
    p, panel, policy = inputs(flow)
    result = review_candidate(
        flow.store,
        p,
        panel,
        policy,
        active_epoch=p.source_epoch,
        reviewer="reviewer",
        enabled=True,
        authenticate=lambda payload: payload["role"] != "approval",
    )
    assert result["decision"] == "DEFER" and not result["execute_candidate"]


def test_optional_model_observation_requires_independent_admission(flow):
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("supply local causal language model")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    ids = tokenizer("Report the ledger value", return_tensors="pt")
    emitted = model.generate(**ids, max_new_tokens=1, do_sample=False)
    observation = content_digest(emitted.tolist())
    p, panel, policy = inputs(flow)
    incomplete = replace(panel, dimensions=(), suite_ref=observation)
    assert review(flow, p, incomplete, policy)["decision"] == "DEFER"


def test_signed_review_cannot_be_replayed_across_private_catalogs(tmp_path):
    from test_private_promotion import Op, _protocol

    from gepa_mindfulness.private_promotion import ExperimentRequest, PrivatePromotionStore

    flow = Flow(tmp_path / "first")
    p, panel, policy = inputs(flow)
    second = PrivatePromotionStore(
        tmp_path / "other.sqlite",
        flow.coevolution,
        flow.evaluation,
        _protocol(generator_digest="sha256:" + "c" * 64),
        flow.baseline,
        USAGE,
        enabled=True,
    )
    flow.request()
    for op in (Op.SUBMIT_CANDIDATE, Op.REQUEST_PRIVATE_EVALUATION):
        second.dispatch(ExperimentRequest(op, p.candidate_id, p.artifact_digest))
    bundle = flow.bundle()
    for store in (flow.store, second):
        assert store.complete_evaluation(p.candidate_id, bundle, USAGE)["status"] == "accepted"
    approvals = set()

    def issue(payload):
        approvals.add(content_digest(payload))
        return True

    kwargs = dict(active_epoch=p.source_epoch, reviewer="reviewer", enabled=True)
    original = review_candidate(flow.store, p, panel, policy, authenticate=issue, **kwargs)
    replayed = review_candidate(
        second,
        p,
        panel,
        policy,
        authenticate=lambda payload: content_digest(payload) in approvals,
        **kwargs,
    )
    assert original["decision"] == "PROMOTE" and replayed["decision"] == "DEFER"
    first = flow.store.admission_snapshot(p.candidate_id, p.artifact_digest)
    other = second.admission_snapshot(p.candidate_id, p.artifact_digest)
    assert first["catalog_id"] != other["catalog_id"]
    assert first["configuration_digest"] != other["configuration_digest"]
    assert (
        flow.reopen().admission_snapshot(p.candidate_id, p.artifact_digest)["catalog_id"]
        == first["catalog_id"]
    )


def test_entirely_missing_paired_report_preserves_missing_dimension(flow):
    from test_improvement_statistics import METRIC, POLICY

    from evaluation.improvement_statistics import estimate_paired
    from gepa_mindfulness.mdac_admission import dimension_from_regret

    candidate = flow.candidate.candidate_id
    estimate = estimate_paired(
        (), expected_pair_ids=("absent",), policy=POLICY, dependencies_known=True
    )
    report = dict(
        schema_version="mdac-regret-v1",
        candidate_id=candidate,
        missing_pair_ids=["absent"],
        criterion={"metric": METRIC.to_dict()},
        paired_estimate=estimate.to_dict(),
    )
    assert dimension_from_regret(report, candidate_id=candidate, dimension="factuality") is None

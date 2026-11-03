"""Verified-set proposals reuse existing independent credit and serialization owners."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest

# isort: split
# Local
from test_research_credit_runtime import Baseline
from test_vector_policy import candidate, config, rollout

from gepa_mindfulness.training.causal_reward_records import request_digest
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.research_credit import ResearchCreditClaim, audit_research_credit
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from gepa_mindfulness.training.research_credit_runtime import (
    ResearchRewardPolicy,
    ResearchRewardProvider,
    ResearchRewardSubmission,
)
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.run_logging import MetricRecord
from gepa_mindfulness.training.vector_policy_credit import make_vector_credit


def source(candidates=None):
    original = rollout(candidates=candidates)
    request = original.request
    trajectory = replace(request.trajectory, trace_references=("vpo-event:vector-event:1",))
    return replace(original, request=RewardRequest(trajectory, request.observable_references))


def test_default_off_does_not_inspect_proposals():
    assert make_vector_credit(None, None) is None


def test_reproducible_set_credit_remains_unresolved_until_existing_host_admission():
    claim = make_vector_credit(config(), source(), enabled=True)
    assert claim.credit_kind == "verified_set"
    assert claim.score == 0.5
    assert claim.to_dict()["schema_version"] == "research-credit-claim-v3"
    assert ResearchCreditClaim.from_dict(json.loads(json.dumps(claim.to_dict()))) == claim
    assert (
        "unresolved_admissibility"
        in audit_research_credit((claim,), enabled=True)["prerequisite_gaps"][claim.claim_id]
    )


@pytest.mark.parametrize("admission", ("rejected", "unresolved"))
def test_invalid_high_reward_member_cannot_create_set_credit(admission):
    unsafe = candidate("prohibited", (1e12, 1e12), admissibility=admission)
    with pytest.raises(ValueError):
        make_vector_credit(config(), source((candidate(), unsafe)), enabled=True)


def test_valid_negative_set_score_is_retained_with_verified_provenance():
    claim = make_vector_credit(config(), source((candidate(values=(-2.0, -4.0)),)), enabled=True)
    assert claim.score == pytest.approx(-3.0)
    assert claim.provenance is not None


@pytest.mark.parametrize("failure", ("score", "event", "state", "weights", "response"))
def test_forged_or_substituted_set_credit_is_rejected(failure):
    claim = make_vector_credit(config(), source(), enabled=True)
    fields = {}
    if failure == "score":
        fields["score"] = claim.score + 1
    elif failure == "event":
        fields["event_id"] = "foreign"
    elif failure == "state":
        fields["state_digest"] = "0" * 64
    else:
        metadata = json.loads(claim.source_metadata_json)
        if failure == "weights":
            metadata["verified_set"]["config"]["weights"] = [[0.0, 1.0]]
        else:
            metadata["source_request"]["trajectory"]["response"] = "substituted"
        fields["source_metadata_json"] = json.dumps(metadata)
    with pytest.raises(ValueError):
        replace(claim, **fields)


def test_existing_credit_owner_admits_one_verified_set_and_records_new_metrics(tmp_path):
    ctx, record = config(), source()
    claim = replace(make_vector_credit(ctx, record, enabled=True), admissibility="admitted")
    policy = ResearchRewardPolicy(
        "vector-policy:v1",
        ctx.run_id,
        ctx.task_id,
        ctx.harness_version,
        ctx.dataset_digest,
        (("verified_set", 0.5),),
        enabled=True,
    )
    assert policy.to_dict()["schema_version"] == "research-reward-policy-v3"
    assert ResearchRewardPolicy.from_dict(policy.to_dict()) == policy
    use = ResearchCreditUse(
        "use:vector",
        policy.policy_id,
        request_digest(record.request),
        (claim.claim_id,),
        (
            (
                claim.claim_id,
                (
                    ("provider", "not_counted"),
                    ("reward_integrity", "not_counted"),
                    ("causal_reward", "not_counted"),
                ),
            ),
        ),
    )
    provider = ResearchRewardProvider(
        RewardPipeline(Baseline(0)),
        ledger=ResearchCreditLedger(
            tmp_path / "credit.sqlite", authority_domain=ctx.authority_domain
        ),
        policy=policy,
        propose=lambda target: ResearchRewardSubmission((claim,), use),
        authenticate_claims=lambda packet: True,
        authorize_optimizer_use=lambda packet: True,
        authorize_accounting=lambda packet: True,
    )
    decision = provider.score(record.request)
    assert decision.optimizer_reward() == 0.25
    assert decision.to_dict()["schema_version"] == "research-reward-decision-v3"
    metrics = decision.metrics()
    assert metrics["research_verified_set_raw"] == 0.5
    assert metrics["research_verified_set_weighted"] == 0.25
    assert "research_verified_decision_raw" not in metrics
    row = MetricRecord(
        "metric:1",
        ctx.run_id,
        "2026-11-03T13:00:00Z",
        1,
        "aggregate",
        "cpu",
        "cpu",
        "cpu",
        ctx.model_version,
        metrics,
        schema_version=5,
    )
    assert row.to_dict()["schema_version"] == 5
    with pytest.raises(ValueError):
        replace(row, schema_version=3)
    from test_rl_logging import _manifest

    from gepa_mindfulness.training.engine import _JSONLRunLogger
    from gepa_mindfulness.training.runtime_config import LoggingConfig, RLRunConfig

    logger = _JSONLRunLogger(
        RLRunConfig(logging=LoggingConfig(log_dir=str(tmp_path / "logs"))), "a" * 64
    )
    logger.sink.start_run(replace(_manifest(), run_id=logger.run_id))
    logger.metrics((decision,), 1)
    rows = [
        json.loads(line) for line in (logger.directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows and all(item["schema_version"] == 5 for item in rows)
    with pytest.raises(ValueError):
        provider.score(record.request)

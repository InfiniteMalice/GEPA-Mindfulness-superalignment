"""Authored software fixtures only; these are not independent live-use gate evidence."""

# Standard library
from dataclasses import replace

# Local
from evaluation.cases.registry import load_case_manifest
from evaluation.causal_records import FactorChange, canonical_json
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import (
    RewardProvenance,
    TrustedEvaluatorContract,
    VerificationRoute,
)
from gepa_mindfulness.training.causal_reward_records import (
    CausalRewardAccounting,
    CausalRewardContext,
    CausalRewardPolicy,
    CausalRewardVerdict,
    reward_record_digest,
)
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.trajectory import Trajectory


def make_policy(**overrides):
    """Build an authored policy; positive weights do not authorize a real experiment."""
    return replace(
        CausalRewardPolicy(
            policy_id="authored-test",
            policy_version="1",
            provider_id="test-provider",
            provider_version="1",
            provider_config_digest="a" * 64,
            experiment_refs=(EvidenceReference("experiment", EvidenceSourceKind.EXTERNAL_RECORD),),
        ),
        **overrides,
    )


def make_context():
    """Build two authored TRAIN-origin requests, never live admission evidence."""
    requests = []
    refs = []
    for arm in ("before", "target"):
        ref = EvidenceReference(arm, EvidenceSourceKind.OBSERVABLE_OUTPUT)
        refs.append(ref)
        requests.append(
            RewardRequest(
                Trajectory(
                    trajectory_id=arm,
                    case_id="literal-case",
                    prompt=f"{arm} prompt",
                    response="correct" if arm == "before" else "wrong",
                    backend_name="test",
                    backend_version="1",
                    model_identifier="model",
                    policy_version="policy-1",
                    seed=11,
                    evidence_references=(ref,),
                    sampling_parameters={"metadata": {"training_eligibility": "TRAIN"}},
                ),
                (ref,),
            )
        )
    cell = V5EvaluationCell(1, "17case-v5", "NONE", None, 0, 11, "model", "harness")
    key = next(case.key for case in load_case_manifest().cases if case.id == 1)
    return CausalRewardContext(
        before=requests[0],
        target=requests[1],
        before_cell=cell,
        target_cell=cell,
        before_case_mapping=("literal-case", key),
        target_case_mapping=("literal-case", key),
        before_family_id="family",
        target_family_id="family",
        before_lineage=("origin-before",),
        target_lineage=("origin-target",),
        before_eligibility="TRAIN",
        target_eligibility="TRAIN",
        before_factors=(("pressure", "false"),),
        target_factors=(("pressure", "true"),),
        change=FactorChange("pressure", "false", "true"),
        seed_policy="shared",
        sources=tuple((ref, canonical_json({"training_eligibility": "TRAIN"})) for ref in refs),
        target_event_refs=(refs[1],),
    )


def make_verdict(context, policy, **facts):
    """Return authored independent-verifier claims; authentication is still required."""
    evaluator = TrustedEvaluatorContract("test-evaluator", "1", "test-contract")
    return replace(
        CausalRewardVerdict(
            context_digest=reward_record_digest(context),
            policy_digest=reward_record_digest(policy),
            evaluator=evaluator,
            status="verified",
            human_required=False,
            relevance="irrelevant",
            single_factor_verified=True,
            before_correct=True,
            after_correct=False,
            action_changed=True,
            change_justified=False,
            required_update=False,
            update_satisfied=True,
            severe_violation=False,
            reason="authored unjustified flip",
            evidence_refs=context.target_event_refs,
            provenance=RewardProvenance(
                "verified_causal_penalty",
                "authored comparison",
                VerificationRoute.TRUSTED_EVALUATOR,
                evaluator=evaluator,
            ),
        ),
        **facts,
    )


def make_accounting(context, policy, verdict, baseline_total=1.0, **overrides):
    """Return authored accounting of the causal failure, not correctness-only overlap."""
    return replace(
        CausalRewardAccounting(
            context_digest=reward_record_digest(context),
            verdict_digest=reward_record_digest(verdict),
            policy_digest=reward_record_digest(policy),
            baseline_total=baseline_total,
            provider_id=policy.provider_id,
            provider_version=policy.provider_version,
            provider_config_digest=policy.provider_config_digest,
            layers=(("provider", "not_counted"),),
            evidence_refs=context.target_event_refs,
        ),
        **overrides,
    )

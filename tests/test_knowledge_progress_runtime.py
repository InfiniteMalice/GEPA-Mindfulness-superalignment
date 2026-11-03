"""Real CPU knowledge/progress updates; fixtures do not establish research efficacy."""

# Standard library
import json
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest
import torch

# isort: split
# Local
from test_deloop_runtime import setup_preferences
from test_knowledge_progress import REVIEWER, conditions, context, state
from test_research_credit_runtime import Baseline

from evaluation.causal_records import canonical_json
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.algorithms import GRPOAlgorithm, GRPOAlgorithmConfig
from gepa_mindfulness.training.backends import TorchTensorOps
from gepa_mindfulness.training.causal_reward_records import request_digest
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.training.knowledge_progress import LearningStep
from gepa_mindfulness.training.knowledge_progress_runtime import (
    KnowledgeTrainingConfig,
    run_knowledge_progress_trial,
)
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from gepa_mindfulness.training.research_credit_runtime import ResearchRewardPolicy
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.subspace_tuning import SSTConfig


def setup_knowledge(tmp_path, *, preference="knowledge", regress=False, sst=SSTConfig()):
    backend, _, raw, store, correction, original, evaluations = setup_preferences(
        tmp_path, regress=regress, sst=sst
    )
    verifier = TrustedEvaluatorContract(REVIEWER.evaluator_id, "v1", REVIEWER.contract_id)
    ctx = context(model_version="model:v1", harness_version="harness:v1", evaluator=verifier)
    learning = []
    for index, source in enumerate(raw[1:]):
        ref = source.request.observable_references[0]
        learning.append(
            LearningStep(
                episode_id=f"episode-{index}",
                step_index=0,
                event_id=f"knowledge-event-{index}",
                request=source.request,
                before=state(),
                after=state(
                    (
                        ("supported", "supported", "unverified")
                        if index
                        else ("unverified", "unverified", "unverified")
                    ),
                    ref=ref,
                ),
            )
        )
    config = KnowledgeTrainingConfig(
        enabled=True,
        policy_id="knowledge-training:v1",
        reviewer_id=verifier.evaluator_id,
        evaluator_id=original["config"].evaluator_id,
        dataset_digest="a" * 64,
        seed=7,
        gamma=ctx.gamma,
        grouping="knowledge_state",
        event_preference=preference,
    )
    policy = ResearchRewardPolicy(
        policy_id=config.policy_id,
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        harness_version=ctx.harness_version,
        dataset_digest=ctx.dataset_digest,
        coefficients=(("knowledge", 0.5), ("progress", 0.5)),
        enabled=True,
    )
    ledger = ResearchCreditLedger(tmp_path / "credit.sqlite", authority_domain=ctx.authority_domain)
    credited = []

    def accounting(target, claim):
        credited.append(claim)
        return ResearchCreditUse(
            "use:" + claim.claim_id,
            config.policy_id,
            request_digest(target),
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

    args = dict(
        context=ctx,
        conditions=conditions(),
        config=config,
        algorithm=GRPOAlgorithm(TorchTensorOps(), GRPOAlgorithmConfig(), group_size=2),
        pipeline=RewardPipeline(Baseline(0.0)),
        ledger=ledger,
        credit_policy=policy,
        authorize_evidence=lambda packet: True,
        authenticate_claims=lambda packet: True,
        authorize_optimizer_use=lambda packet: True,
        authorize_accounting=lambda packet: True,
        account_event=accounting,
        evaluate=original["evaluate"],
        candidate_id="candidate:knowledge",
        artifact_root=tmp_path / "learning-trials",
    )
    return backend, tuple(learning), store, correction, args, credited, evaluations


def test_default_off_does_not_inspect_any_source():
    assert run_knowledge_progress_trial(None, None, None, None) is None


@pytest.mark.parametrize("preference,raw_score", (("knowledge", 1.0), ("progress", 0.25)))
def test_native_verified_groups_update_zero_gain_and_gaining_actions(
    tmp_path, preference, raw_score
):
    backend, steps, store, correction, args, credited, _ = setup_knowledge(
        tmp_path, preference=preference
    )
    source = backend.parameter_checksum()
    rng = torch.get_rng_state().clone()
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.decision.accepted is True
    assert result.decision.execute_candidate is False
    assert result.backend.policy_parameter_checksum() != backend.policy_parameter_checksum()
    assert backend.parameter_checksum() == source
    assert torch.equal(torch.get_rng_state(), rng)
    report = result.to_dict()
    assert report["group_sizes"] == [2]
    assert report["advantages"][0] < 0 < report["advantages"][1]
    assert len(credited) == 2
    assert credited[0].score == 0.0
    assert credited[1].score == raw_score
    assert credited[1].credit_kind == preference
    assert report["knowledge_progress"]["credit_audit"]["event_overlaps"]
    assert len(report["credit_decisions"]) == 2
    assert report["optimizer_step"] == 1
    with pytest.raises(ValueError):
        require_training_eligible(report | {"training_eligibility": "TRAIN"})
    result.backend.load_checkpoint(Path(report["checkpoint_path"]))
    assert result.backend.policy_parameter_checksum() == report["candidate_policy_checksum"]


def test_protected_regression_keeps_source_and_withholds_candidate(tmp_path):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path, regress=True)
    checksum = backend.parameter_checksum()
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.backend is None
    assert not result.decision.accepted
    assert backend.parameter_checksum() == checksum


@pytest.mark.parametrize(
    "role",
    (
        "authorize_evidence",
        "authenticate_claims",
        "authorize_optimizer_use",
        "authorize_accounting",
    ),
)
def test_unreviewed_learning_never_changes_model(tmp_path, role):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)
    checksum = backend.parameter_checksum()
    args[role] = lambda packet: False
    with pytest.raises(ValueError):
        run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert backend.parameter_checksum() == checksum
    assert not (tmp_path / "learning-trials").exists()


@pytest.mark.parametrize("failure", ("hidden", "stale", "tokens", "source_id", "discount"))
def test_invalid_sources_fail_before_evidence_review(tmp_path, failure):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)
    first = steps[0]
    trajectory = first.request.trajectory
    if failure == "hidden":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "source_record": {"training_eligibility": "HIDDEN_EVAL"},
            },
        )
    elif failure == "stale":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "policy_parameter_checksum": "0" * 64,
            },
        )
    elif failure == "tokens":
        trajectory = replace(trajectory, response_token_ids=(5,))
    elif failure == "source_id":
        correction = replace(correction, source_trajectory_id="foreign")
    else:
        args["config"] = replace(args["config"], gamma=0.9)
    steps = (replace(first, request=replace(first.request, trajectory=trajectory)), steps[1])
    calls = []
    args["authorize_evidence"] = lambda packet: calls.append(packet) or True
    with pytest.raises(ValueError):
        run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert not calls


def test_prompt_groups_do_not_merge_different_prompts(tmp_path):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)
    args["config"] = replace(args["config"], grouping="public_prompt")
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.backend is None
    assert result.to_dict()["status"] == "no_trainable_groups"


def test_exported_config_is_closed_and_safe():
    cfg = KnowledgeTrainingConfig()
    assert KnowledgeTrainingConfig.from_dict(json.loads(canonical_json(cfg.to_dict()))) == cfg
    with pytest.raises(ValueError):
        KnowledgeTrainingConfig.from_dict(cfg.to_dict() | {"extra": True})


def test_beliefs_cannot_change_verified_knowledge_group(tmp_path):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)
    belief = state(
        ("contradicted", "unverified", "unverified"), ref=steps[0].request.observable_references[0]
    )
    steps = (replace(steps[0], belief_before=belief, belief_after=belief), steps[1])
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.to_dict()["group_sizes"] == [2]
    assert result.backend is not None


def test_sst_update_and_resume_use_existing_native_optimizer(tmp_path):
    backend, steps, store, correction, args, _, _ = setup_knowledge(
        tmp_path, sst=SSTConfig(enabled=True, rank_fraction=0.0)
    )
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.backend is not None
    result.backend.load_checkpoint(Path(result.to_dict()["checkpoint_path"]))
    assert result.backend.sst_config.enabled


def test_zero_progress_truncations_do_not_fabricate_credit(tmp_path):
    backend, steps, store, correction, args, credited, _ = setup_knowledge(tmp_path)
    steps = tuple(replace(step, termination="truncated") for step in steps)
    args["config"] = replace(args["config"], grouping="public_prompt")
    args["credit_policy"] = replace(args["credit_policy"], coefficients=(("progress", 0.5),))
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.backend is None
    assert not credited
    assert all(row["shaping"] is None for row in result.to_dict()["knowledge_progress"]["steps"])


def test_event_credit_cannot_be_replayed_for_another_trial(tmp_path):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)
    result = run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert result.backend is not None
    args["candidate_id"] = "candidate:replay"
    with pytest.raises(ValueError):
        run_knowledge_progress_trial(backend, steps, store, correction, **args)


@pytest.mark.parametrize(
    "role",
    (
        "authorize_evidence",
        "authenticate_claims",
        "authorize_optimizer_use",
        "authorize_accounting",
    ),
)
def test_mutating_review_cannot_admit_learning(tmp_path, role):
    backend, steps, store, correction, args, _, _ = setup_knowledge(tmp_path)

    def mutate(packet):
        packet["forged"] = True
        return True

    args[role] = mutate
    with pytest.raises(ValueError):
        run_knowledge_progress_trial(backend, steps, store, correction, **args)
    assert not (tmp_path / "learning-trials").exists()


def test_registered_source_must_belong_to_a_trainable_group(tmp_path):
    backend, steps, store, correction, args, _, evaluations = setup_knowledge(tmp_path)
    first = replace(steps[0], is_answer=True)
    extra = replace(
        steps[0],
        episode_id="episode-3",
        event_id="knowledge-event-3",
        request=replace(
            steps[0].request,
            trajectory=replace(
                steps[0].request.trajectory,
                trajectory_id="action-3",
            ),
        ),
    )
    result = run_knowledge_progress_trial(
        backend, (first, steps[1], extra), store, correction, **args
    )
    assert result.backend is None
    assert result.to_dict()["status"] == "no_trainable_groups"
    assert not evaluations
    assert not (tmp_path / "learning-trials").exists()

"""Native admissible-set updates on authored CPU fixtures, without efficacy claims."""

# Standard library
from dataclasses import replace
from pathlib import Path

# Third-party
import pytest
import torch

# isort: split
# Local
from test_deloop_runtime import setup_preferences
from test_research_credit_runtime import Baseline
from test_vector_policy import REF, candidate, config, rollout

from gepa_mindfulness.training.algorithms import GRPOAlgorithm, GRPOAlgorithmConfig
from gepa_mindfulness.training.backends import TorchTensorOps
from gepa_mindfulness.training.causal_reward_records import request_digest
from gepa_mindfulness.training.contracts import RewardRequest
from gepa_mindfulness.training.research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from gepa_mindfulness.training.research_credit_runtime import ResearchRewardPolicy
from gepa_mindfulness.training.reward_pipeline import RewardPipeline
from gepa_mindfulness.training.subspace_tuning import SSTConfig
from gepa_mindfulness.training.vector_policy import VectorRollout
from gepa_mindfulness.training.vector_policy_runtime import (
    VPOTrainingConfig,
    run_vector_policy_trial,
)


class SetTokenizer:
    """Fixed one-token sets with a shared public prompt and real native logits."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, low, high):
        self.words = {2: "Vector public prompt", 3: low, 4: "unused", 5: high, 6: "unused hint"}
        self.tokens = {value: key for key, value in self.words.items()}
        self.tokens["actual public history before step 1"] = 2

    def encode(self, text, *, add_special_tokens=False):
        return [self.tokens[text]]

    def decode(self, tokens, *, skip_special_tokens=True):
        return " ".join(self.words[token] for token in tokens if token not in (0, 1))


def setup_vector(
    tmp_path, *, sst=SSTConfig(), regress=False, negative=False, episode_id="action-1"
):
    backend, _, _, store, correction, original, evaluations = setup_preferences(
        tmp_path, sst=sst, regress=regress, episode_id=episode_id
    )
    low = rollout(candidates=(candidate("ordinary", (0.2, 0.2)),))
    high = rollout(candidates=(candidate("a", (0.9, 0.1)), candidate("b", (0.1, 0.9))))
    if negative:
        low = rollout(candidates=(candidate("ordinary", (-0.5, -0.5)),))
        high = rollout(candidates=(candidate("a", (-0.1, -0.9)), candidate("b", (-0.9, -0.1))))
    backend.tokenizer = SetTokenizer(
        low.request.trajectory.response, high.request.trajectory.response
    )
    records = []
    for index, record in enumerate((low, high), 1):
        trajectory = replace(
            record.request.trajectory,
            trajectory_id=episode_id if index == 1 else f"action-{index}",
            prompt="Vector public prompt",
            prompt_token_ids=(2,),
            response_token_ids=(3 if index == 1 else 5,),
            policy_version="model:v1",
            trace_references=(f"vpo-event:vector-event:{index}",),
            sampling_parameters={
                "training_eligibility": "TRAIN",
                "source_record_sha256": "a" * 64,
                "policy_parameter_checksum": backend.policy_parameter_checksum(),
            },
        )
        records.append(
            VectorRollout(
                trajectory.trajectory_id, RewardRequest(trajectory, (REF,)), record.candidates
            )
        )
    scope = config(model_version="model:v1", harness_version="harness:v1")
    settings = VPOTrainingConfig(
        enabled=True,
        policy_id="vector-training:v1",
        reviewer_id=scope.evaluator.evaluator_id,
        evaluator_id=original["config"].evaluator_id,
        dataset_digest="a" * 64,
        constitution_digest=scope.constitution_digest,
        seed=7,
    )
    policy = ResearchRewardPolicy(
        settings.policy_id,
        scope.run_id,
        scope.task_id,
        scope.harness_version,
        scope.dataset_digest,
        (("verified_set", 0.5),),
        enabled=True,
    )
    credited = []

    def accounting(target, claim):
        credited.append(claim)
        return ResearchCreditUse(
            "use:" + claim.claim_id,
            policy.policy_id,
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
        scope=scope,
        config=settings,
        algorithm=GRPOAlgorithm(
            TorchTensorOps(),
            GRPOAlgorithmConfig(group_normalization_epsilon=scope.epsilon),
            group_size=2,
        ),
        pipeline=RewardPipeline(Baseline(0.0)),
        ledger=ResearchCreditLedger(tmp_path / "vector-credit.sqlite", authority_domain="host"),
        credit_policy=policy,
        authorize_evidence=lambda packet: True,
        authenticate_claims=lambda packet: True,
        authorize_optimizer_use=lambda packet: True,
        authorize_accounting=lambda packet: True,
        account_event=accounting,
        evaluate=original["evaluate"],
        candidate_id="candidate:vector",
        artifact_root=tmp_path / "vector-trials",
    )
    return backend, tuple(records), store, correction, args, credited, evaluations


def test_default_off_never_inspects_sources():
    assert run_vector_policy_trial(None, None, None, None) is None


@pytest.mark.parametrize("sst", (SSTConfig(), SSTConfig(enabled=True, rank_fraction=0.0)))
def test_actual_set_level_update_and_resume_preserve_source_and_reference(tmp_path, sst):
    backend, records, store, correction, args, credited, _ = setup_vector(tmp_path, sst=sst)
    source = backend.parameter_checksum()
    rng = torch.get_rng_state().clone()
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is not None
    assert result.decision.accepted
    assert not result.decision.execute_candidate
    assert backend.parameter_checksum() == source
    assert torch.equal(torch.get_rng_state(), rng)
    assert result.backend.policy_parameter_checksum() != backend.policy_parameter_checksum()
    assert all(
        torch.equal(value, backend.reference_model.state_dict()[name])
        for name, value in result.backend.reference_model.state_dict().items()
    )
    report = result.to_dict()
    assert report["advantages"][0] < 0 < report["advantages"][1]
    assert report["set_scores"] == pytest.approx([0.2, 0.82])
    assert len(credited) == 2
    assert all(claim.credit_kind == "verified_set" for claim in credited)
    assert report["optimizer_step"] == 1
    result.backend.load_checkpoint(Path(report["checkpoint_path"]))
    assert result.backend.policy_parameter_checksum() == report["candidate_policy_checksum"]


def test_inadmissible_high_score_set_never_enters_the_loss(tmp_path):
    backend, records, store, correction, args, credited, _ = setup_vector(tmp_path)
    rejected = replace(records[1].candidates[0], admissibility="rejected")
    records = (records[0], replace(records[1], candidates=(rejected, records[1].candidates[1])))
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is None
    assert result.to_dict()["status"] == "no_trainable_group"
    assert result.to_dict()["quarantined_rollout_ids"] == ["action-2"]
    assert len(credited) == 1
    assert not backend.policy_model.gradient_inputs


@pytest.mark.parametrize(
    "role",
    (
        "authorize_evidence",
        "authenticate_claims",
        "authorize_optimizer_use",
        "authorize_accounting",
    ),
)
def test_uncertain_or_mutated_review_cannot_train(tmp_path, role):
    backend, records, store, correction, args, _, _ = setup_vector(tmp_path)

    def mutate(packet):
        packet["forged"] = True
        return True

    args[role] = mutate
    with pytest.raises(ValueError):
        run_vector_policy_trial(backend, records, store, correction, **args)
    assert not (tmp_path / "vector-trials").exists()


@pytest.mark.parametrize("failure", ("constitution", "stale", "hidden", "tokens", "source"))
def test_source_and_constitution_scope_fail_before_review(tmp_path, failure):
    backend, records, store, correction, args, _, _ = setup_vector(tmp_path)
    first = records[0]
    trajectory = first.request.trajectory
    if failure == "constitution":
        args["scope"] = replace(args["scope"], constitution_digest="0" * 64)
    elif failure == "stale":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "policy_parameter_checksum": "0" * 64,
            },
        )
    elif failure == "hidden":
        trajectory = replace(
            trajectory,
            sampling_parameters={
                **trajectory.sampling_parameters,
                "nested": {"training_eligibility": "HIDDEN_EVAL"},
            },
        )
    elif failure == "tokens":
        trajectory = replace(trajectory, response_token_ids=(5,))
    else:
        correction = replace(correction, source_trajectory_id="foreign")
    records = (replace(first, request=replace(first.request, trajectory=trajectory)), records[1])
    calls = []
    args["authorize_evidence"] = lambda packet: calls.append(packet) or True
    with pytest.raises(ValueError):
        run_vector_policy_trial(backend, records, store, correction, **args)
    assert not calls


def test_protected_regression_rejects_updated_model(tmp_path):
    backend, records, store, correction, args, _, _ = setup_vector(tmp_path, regress=True)
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is None
    assert not result.decision.accepted
    assert Path(result.to_dict()["checkpoint_path"]).is_file()


def test_valid_negative_set_scores_can_train_without_clipping(tmp_path):
    backend, records, store, correction, args, credited, _ = setup_vector(tmp_path, negative=True)
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is not None
    assert result.to_dict()["set_scores"] == pytest.approx([-0.5, -0.18])
    assert all(claim.score < 0 for claim in credited)


def test_vector_event_cannot_be_replayed_in_another_trial(tmp_path):
    backend, records, store, correction, args, _, _ = setup_vector(tmp_path)
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is not None
    args["candidate_id"] = "candidate:vector-replay"
    with pytest.raises(ValueError):
        run_vector_policy_trial(backend, records, store, correction, **args)


def test_all_quarantined_sets_request_fallback_without_model_or_credit(tmp_path):
    backend, records, store, correction, args, credited, _ = setup_vector(tmp_path)
    records = tuple(
        replace(
            record,
            candidates=tuple(
                replace(member, admissibility="unresolved") for member in record.candidates
            ),
        )
        for record in records
    )
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is None
    assert result.to_dict()["vector_evidence"]["fallback_required"]
    assert not credited
    assert not backend.policy_model.inputs


def test_configuration_is_closed_and_versioned():
    config = VPOTrainingConfig()
    assert VPOTrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        VPOTrainingConfig.from_dict(config.to_dict() | {"unexpected": True})


def test_quarantined_registered_source_cannot_be_replaced_by_other_safe_sets(tmp_path):
    backend, records, store, correction, args, credited, evaluations = setup_vector(tmp_path)
    first = replace(
        records[0], candidates=(replace(records[0].candidates[0], admissibility="rejected"),)
    )
    extra_request = replace(
        records[0].request,
        trajectory=replace(
            records[0].request.trajectory,
            trajectory_id="action-3",
            trace_references=("vpo-event:vector-event:3",),
        ),
    )
    extra = VectorRollout("action-3", extra_request, records[0].candidates)
    result = run_vector_policy_trial(backend, (first, records[1], extra), store, correction, **args)
    assert result.backend is None
    assert result.to_dict()["status"] == "no_trainable_group"
    assert not credited
    assert not evaluations
    assert not (tmp_path / "vector-trials").exists()


def test_whole_set_episode_identity_remains_distinct_from_its_source_action(tmp_path):
    backend, records, store, correction, args, _, _ = setup_vector(
        tmp_path, episode_id="episode:sets"
    )
    assert store.correction_source_events(correction)
    result = run_vector_policy_trial(backend, records, store, correction, **args)
    assert result.backend is not None
    retained = result.to_dict()["candidate"]["correction"]
    assert retained["source_trajectory_id"] == "episode:sets"
    assert retained["source_action_id"] == "action-1"

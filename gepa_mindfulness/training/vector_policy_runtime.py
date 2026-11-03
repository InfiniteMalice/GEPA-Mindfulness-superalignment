"""Opt-in native set-level GRPO with independent admissibility and existing credit authority."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

# Local
from evaluation.causal_records import canonical_json
from gepa_mindfulness.coevolution import CandidateComponent, CoevolutionStore, CorrectionProposal

from .causal_reward import _accepted
from .causal_reward_records import _digest, _request_wire, _text, validate_input_envelope
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .locd_runtime import (
    CandidateEvaluator,
    HostReview,
    LocalCorrectionResult,
    LoCDTrainingConfig,
    _finalize_trial,
    _generation_limits,
    _grpo_update,
    _tokens,
    _validate_backend,
)
from .research_credit import ResearchCreditClaim
from .research_credit_ledger import ResearchCreditLedger, ResearchCreditUse
from .research_credit_runtime import (
    ResearchRewardDecision,
    ResearchRewardPolicy,
    ResearchRewardProvider,
    ResearchRewardSubmission,
)
from .reward_pipeline import RewardPipeline
from .vector_policy import VectorPolicyConfig, VectorRollout, prepare_vector_group
from .vector_policy_credit import make_vector_credit

if TYPE_CHECKING:
    from .algorithms import GRPOAlgorithm
    from .backends.torch_policy import TorchPolicyBackend


@dataclass(frozen=True)
class VPOTrainingConfig(LoCDTrainingConfig):
    """Operator-fixed constitutional scope, never a sampled preference dimension."""

    constitution_digest: str = ""
    max_rollouts: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.enabled:
            _digest(self.constitution_digest)
        elif type(self.constitution_digest) is not str:
            raise ValueError("constitution digest must be a string")
        if type(self.max_rollouts) is not int or not 1 <= self.max_rollouts <= 32:
            raise ValueError("invalid rollout budget")

    def limits(self) -> LoCDTrainingConfig:
        """Reuse existing isolation/evaluation budgets."""
        return LoCDTrainingConfig(
            **{field.name: getattr(self, field.name) for field in fields(LoCDTrainingConfig)}
        )

    def to_dict(self) -> dict[str, object]:
        value = super().to_dict()
        value["schema_version"] = "vpo-training-config-v1"
        return value

    @classmethod
    def from_dict(cls, value: object) -> VPOTrainingConfig:
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or (value["schema_version"] != "vpo-training-config-v1")
        ):
            raise ValueError("invalid VPO training config schema")
        return cls(**{name: value[name] for name in names})


def run_vector_policy_trial(
    backend: TorchPolicyBackend,
    rollouts: object,
    store: CoevolutionStore,
    correction: CorrectionProposal,
    *,
    scope: VectorPolicyConfig | None = None,
    config: VPOTrainingConfig = VPOTrainingConfig(),
    algorithm: GRPOAlgorithm | None = None,
    pipeline: RewardPipeline | None = None,
    ledger: ResearchCreditLedger | None = None,
    credit_policy: ResearchRewardPolicy | None = None,
    authorize_evidence: HostReview | None = None,
    authenticate_claims: HostReview | None = None,
    authorize_optimizer_use: HostReview | None = None,
    authorize_accounting: HostReview | None = None,
    account_event: Callable[[RewardRequest, ResearchCreditClaim], ResearchCreditUse] | None = None,
    evaluate: CandidateEvaluator | None = None,
    candidate_id: str = "",
    artifact_root: Path | None = None,
) -> LocalCorrectionResult | None:
    """Train only safe whole set rollouts, then require existing protected admission.

    Installed host roles authenticate evidence, source capture, the fixed normative
    reference and actual role independence. Sampling task weights never changes that
    reference, the eligibility checker or execution authorization. Nothing executes
    generated candidate actions; catalog acceptance never deploys the model.
    """
    if type(config) is not VPOTrainingConfig:
        raise ValueError("expected exact VPOTrainingConfig")
    config = VPOTrainingConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    import torch

    from .algorithms import GRPOAlgorithm
    from .backends import TorchTensorOps

    limits = config.limits()
    vocab = _validate_backend(backend, limits)
    if type(scope) is not VectorPolicyConfig:
        raise ValueError("exact vector scope required")
    scope = VectorPolicyConfig.from_dict(scope.to_dict())
    if (
        type(store) is not CoevolutionStore
        or type(correction) is not CorrectionProposal
        or type(algorithm) is not GRPOAlgorithm
        or type(algorithm.ops) is not TorchTensorOps
        or type(pipeline) is not RewardPipeline
        or type(ledger) is not ResearchCreditLedger
        or type(credit_policy) is not ResearchRewardPolicy
        or not credit_policy.enabled
    ):
        raise ValueError("existing native GRPO, reward, ledger and correction contracts required")
    if any(
        not callable(role)
        for role in (
            authorize_evidence,
            authenticate_claims,
            authorize_optimizer_use,
            authorize_accounting,
            account_event,
            evaluate,
        )
    ) or not isinstance(artifact_root, Path):
        raise ValueError("fresh independent reviews, accounting and evaluator required")
    _text(candidate_id, "candidate_id")
    proposal = CorrectionProposal.from_dict(correction.to_dict())
    if proposal.changed_components != (CandidateComponent.MODEL,):
        raise ValueError("VPO trial may change only the model")
    if (
        type(rollouts) is not tuple
        or not 1 <= len(rollouts) <= config.max_rollouts
        or any(type(rollout) is not VectorRollout for rollout in rollouts)
    ):
        raise ValueError("bounded exact vector rollouts required")
    captured = tuple(VectorRollout.from_dict(rollout.to_dict()) for rollout in rollouts)
    by_id = {rollout.rollout_id: rollout for rollout in captured}
    if proposal.source_trajectory_id not in by_id:
        raise ValueError("registered source mismatch")
    registered = store.correction_source_events(proposal)
    if (
        scope.constitution_digest != config.constitution_digest
        or scope.dataset_digest != config.dataset_digest
        or scope.producer_id != backend.model_identifier
        or scope.evaluator.evaluator_id != config.reviewer_id
        or config.evaluator_id in {scope.producer_id, scope.model_version}
        or any(
            (event.model_version, event.harness_version)
            != (scope.model_version, scope.harness_version)
            for event in registered
        )
        or algorithm.config.group_normalization_epsilon != scope.epsilon
        or ledger.authority_domain != scope.authority_domain
        or (
            credit_policy.policy_id,
            credit_policy.run_id,
            credit_policy.task_id,
            credit_policy.harness_version,
            credit_policy.dataset_digest,
        )
        != (
            config.policy_id,
            scope.run_id,
            scope.task_id,
            scope.harness_version,
            scope.dataset_digest,
        )
        or set(dict(credit_policy.coefficients)) != {"verified_set"}
    ):
        raise ValueError("fixed constitution, source, grouping or credit scope mismatch")
    checksum = backend.policy_parameter_checksum()
    total_logits = 0
    for rollout in captured:
        source = rollout.request.trajectory
        require_training_eligible(source.to_dict())
        if (
            source.sampling_parameters.get("training_eligibility") != "TRAIN"
            or source.sampling_parameters.get("policy_parameter_checksum") != checksum
            or source.sampling_parameters.get("source_record_sha256") != config.dataset_digest
            or not rollout.request.observable_references
        ):
            raise ValueError("original TRAIN source, current checksum and evidence required")
        prompt = tuple(backend.tokenizer.encode(source.prompt, add_special_tokens=False))
        response = source.response_token_ids
        if (
            not prompt
            or not response
            or prompt != source.prompt_token_ids
            or (
                backend.tokenizer.decode(list(response), skip_special_tokens=True)
                != source.response
            )
        ):
            raise ValueError("source tokenizer binding mismatch")
        _tokens(prompt + response, vocab=vocab, config=limits)
        total_logits += len(prompt + response) * vocab
        if total_logits > config.max_logit_elements:
            raise ValueError("aggregate native logit budget exceeded")
    _, generation_limits = _generation_limits(backend)
    prepared = prepare_vector_group(
        scope, captured, authorize_evidence=authorize_evidence, enabled=True
    )
    if not prepared["evidence_review_accepted"]:
        raise ValueError("independent admissibility/metric evidence review withheld")
    selected = {}
    advantages = {}
    scores = []
    quarantined = []
    for row in prepared["rollouts"]:
        if row["disposition"] != "reviewed":
            quarantined.append(row["rollout_id"])
            continue
        claim = make_vector_credit(scope, by_id[row["rollout_id"]], enabled=True)
        selected[row["rollout_id"]] = replace(claim, admissibility="admitted")
        advantages[row["rollout_id"]] = row["advantage"]
        scores.append(row["reviewed_set_score"])
    report = {
        "schema_version": "vpo-training-trial-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "execute_candidate": False,
        "source_reference": "REF-VPO",
        "config": config.to_dict(),
        "candidate_id": candidate_id,
        "algorithm_config": {
            field.name: getattr(algorithm.config, field.name) for field in fields(algorithm.config)
        },
        "source_policy_checksum": checksum,
        "vector_evidence": prepared,
        "quarantined_rollout_ids": quarantined,
        "set_scores": scores,
        "advantages": list(advantages.values()),
        "credit_decisions": [],
        "status": "credit_review",
    }
    validate_input_envelope(report)
    if proposal.source_trajectory_id not in selected:
        report["status"] = "no_trainable_group"
        report["reason"] = "registered_source_quarantined"
        return LocalCorrectionResult(None, None, canonical_json(report))

    def propose(target: RewardRequest) -> ResearchRewardSubmission:
        claim = selected[target.trajectory.trajectory_id]
        if _request_wire(by_id[claim.action_id].request) != _request_wire(target):
            raise ValueError("vector credit source substitution")
        try:
            use = account_event(target, ResearchCreditClaim.from_dict(claim.to_dict()))
        except Exception:
            raise ValueError("independent vector event accounting failed") from None
        return ResearchRewardSubmission((claim,), use)

    def contextual_review(role: HostReview) -> HostReview:
        return lambda packet: _accepted(role, {"credit": packet, "vector_evidence": prepared})

    provider = ResearchRewardProvider(
        pipeline,
        ledger=ledger,
        policy=credit_policy,
        propose=propose,
        authenticate_claims=contextual_review(authenticate_claims),
        authorize_optimizer_use=contextual_review(authorize_optimizer_use),
        authorize_accounting=contextual_review(authorize_accounting),
    )
    trainable = []
    for rollout_id in selected:
        decision = provider.score(by_id[rollout_id].request)
        if type(decision) is not ResearchRewardDecision:
            raise ValueError("live independently admitted set credit required")
        decision.optimizer_reward()
        audit = decision.to_dict()
        if audit["baseline_total"] != 0.0:
            raise ValueError("isolated VPO requires a zero baseline without duplicate task credit")
        report["credit_decisions"].append(audit)
        validate_input_envelope(report)
        trajectory = decision.bind_trajectory(by_id[rollout_id].request.trajectory)
        trainable.append(
            replace(
                trajectory,
                advantage=(advantages[rollout_id],) * len(trajectory.response_token_ids),
                returns=None,
                value_predictions=None,
            )
        )
    if not trainable or not any(value != 0 for value in advantages.values()):
        report["status"] = "no_trainable_group"
        return LocalCorrectionResult(None, None, canonical_json(report))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        trial, update = _grpo_update(backend, tuple(trainable), algorithm, limits)
        report.update(update)
        if update["candidate_policy_checksum"] == checksum:
            report["status"] = "no_model_change"
            return LocalCorrectionResult(None, None, canonical_json(report))
        return _finalize_trial(
            trial,
            store,
            proposal,
            report,
            config=limits,
            candidate_id=candidate_id,
            artifact_root=artifact_root,
            source=by_id[proposal.source_trajectory_id].request,
            evaluate=evaluate,
            generation_limits=generation_limits,
            vocab=vocab,
        )

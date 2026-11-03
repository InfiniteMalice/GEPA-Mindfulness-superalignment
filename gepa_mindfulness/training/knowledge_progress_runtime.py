"""Isolated native GRPO trials over freshly verified knowledge and milestone credit."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from math import fsum
from pathlib import Path
from typing import TYPE_CHECKING

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.coevolution import CandidateComponent, CoevolutionStore, CorrectionProposal
from gepa_mindfulness.core.reward_provenance import RewardProvenance, VerificationRoute

from .causal_reward import _accepted
from .causal_reward_records import (
    _number,
    _request_wire,
    _text,
    request_digest,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .knowledge_progress import (
    KnowledgeCondition,
    LearningContext,
    LearningStep,
    prepare_knowledge_progress,
)
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

if TYPE_CHECKING:
    from .algorithms import GRPOAlgorithm
    from .backends.torch_policy import TorchPolicyBackend


@dataclass(frozen=True)
class KnowledgeTrainingConfig(LoCDTrainingConfig):
    """Explicit discount and grouping; only experimental credit weights are scalarized."""

    gamma: float | None = None
    grouping: str = "knowledge_state"
    event_preference: str = "knowledge"
    max_steps: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gamma is not None:
            _number(self.gamma)
            if not 0 < self.gamma <= 1:
                raise ValueError("gamma must be in (0, 1]")
        if self.enabled and self.gamma is None:
            raise ValueError("explicit training discount required")
        if self.grouping not in ("knowledge_state", "public_prompt"):
            raise ValueError("unsupported training grouping")
        if self.event_preference not in ("knowledge", "progress"):
            raise ValueError("unsupported overlapping-event credit preference")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 128:
            raise ValueError("invalid step budget")

    def limits(self) -> LoCDTrainingConfig:
        """Reuse existing native isolation/evaluation limits."""
        return LoCDTrainingConfig(
            **{field.name: getattr(self, field.name) for field in fields(LoCDTrainingConfig)}
        )

    def to_dict(self) -> dict[str, object]:
        value = super().to_dict()
        value["schema_version"] = "knowledge-training-config-v1"
        return value

    @classmethod
    def from_dict(cls, value: object) -> KnowledgeTrainingConfig:
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "knowledge-training-config-v1"
        ):
            raise ValueError("invalid knowledge training schema")
        return cls(**{name: value[name] for name in names})


def _zero_claim(
    step: LearningStep,
    row: dict[str, object],
    context: LearningContext,
    prepared: dict[str, object],
    kind: str,
) -> ResearchCreditClaim:
    """Retain observed zero-gain members; this proposal still requires fresh host admission."""
    component = "verified_knowledge_gain" if kind == "knowledge" else "verified_milestone_progress"
    return ResearchCreditClaim(
        claim_id=content_digest([context.run_id, context.task_id, step.event_id, kind]),
        method_id="knowgpo" if kind == "knowledge" else "prm4rl",
        component_name=component,
        credit_kind=kind,
        source_reference="REF-KNOWGPO" if kind == "knowledge" else "REF-PRM4RL",
        authority_domain=context.authority_domain,
        run_id=context.run_id,
        task_id=context.task_id,
        event_id=step.event_id,
        action_id=step.request.trajectory.trajectory_id,
        model_version=context.model_version,
        harness_version=context.harness_version,
        producer_id=context.producer_id,
        evaluator=context.evaluator,
        dataset_digest=context.dataset_digest,
        state_digest=content_digest(step.before.to_dict()),
        evidence_refs=step.request.observable_references,
        score=0.0,
        admissibility="unresolved",
        provenance=RewardProvenance(
            component,
            "host-reviewed observable zero-gain transition",
            VerificationRoute.TRUSTED_EVALUATOR,
            evaluator=context.evaluator,
        ),
        source_metadata_json=canonical_json(
            {
                "training_eligibility": "TRAIN",
                "source_request": _request_wire(step.request),
                "target_request_digest": request_digest(step.request),
                "source_digest": prepared["source_digest"],
                "catalog_digest": prepared["catalog_digest"],
                "task_version": context.task_version,
                "raw_metric": row,
            }
        ),
    )


def run_knowledge_progress_trial(
    backend: TorchPolicyBackend,
    steps: object,
    store: CoevolutionStore,
    correction: CorrectionProposal,
    *,
    context: LearningContext | None = None,
    conditions: tuple[KnowledgeCondition, ...] = (),
    config: KnowledgeTrainingConfig = KnowledgeTrainingConfig(),
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
    """Apply verified learning signals to one copied native model, never deploy it.

    All host callbacks authenticate their actual installed roles and raw evidence.
    Evidence review grants no reward authority: existing fresh semantic/use/accounting
    reviews and durable event reservations govern every optimizer-facing component.
    """
    if type(config) is not KnowledgeTrainingConfig:
        raise ValueError("expected exact KnowledgeTrainingConfig")
    config = KnowledgeTrainingConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    import torch

    from .algorithms import GRPOAlgorithm
    from .backends import TorchTensorOps

    limits = config.limits()
    vocab = _validate_backend(backend, limits)
    if type(context) is not LearningContext:
        raise ValueError("exact learning context required")
    context = LearningContext.from_dict(context.to_dict())
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
        raise ValueError("fresh independent host reviews, accounting and evaluator required")
    _text(candidate_id, "candidate_id")
    proposal = CorrectionProposal.from_dict(correction.to_dict())
    if proposal.changed_components != (CandidateComponent.MODEL,):
        raise ValueError("learning trial may change only the model")
    if type(steps) is not tuple or not 1 <= len(steps) <= config.max_steps:
        raise ValueError("learning step budget exceeded")
    if any(type(step) is not LearningStep for step in steps):
        raise ValueError("exact LearningStep sources required")
    captured = tuple(LearningStep.from_dict(step.to_dict()) for step in steps)
    by_id = {step.request.trajectory.trajectory_id: step for step in captured}
    if proposal.source_action_id not in by_id:
        raise ValueError("registered correction source mismatch")
    registered = store.correction_source_events(proposal)
    if (
        context.gamma != config.gamma
        or context.dataset_digest != config.dataset_digest
        or context.producer_id != backend.model_identifier
        or context.evaluator.evaluator_id != config.reviewer_id
        or config.evaluator_id in {context.producer_id, context.model_version}
        or any(
            (event.model_version, event.harness_version)
            != (context.model_version, context.harness_version)
            for event in registered
        )
    ):
        raise ValueError("learning context, source, discount or independent role mismatch")
    weights = dict(credit_policy.coefficients)
    if (
        set(weights) - {"knowledge", "progress"}
        or not weights
        or (config.grouping == "knowledge_state" and "knowledge" not in weights)
        or (
            credit_policy.policy_id,
            credit_policy.run_id,
            credit_policy.task_id,
            credit_policy.harness_version,
            credit_policy.dataset_digest,
        )
        != (
            config.policy_id,
            context.run_id,
            context.task_id,
            context.harness_version,
            context.dataset_digest,
        )
        or ledger.authority_domain != context.authority_domain
    ):
        raise ValueError("credit policy differs from declared learning trial")
    checksum = backend.policy_parameter_checksum()
    total_logits = 0
    for step in captured:
        source = step.request.trajectory
        require_training_eligible(source.to_dict())
        if (
            source.sampling_parameters.get("training_eligibility") != "TRAIN"
            or source.sampling_parameters.get("policy_parameter_checksum") != checksum
            or source.sampling_parameters.get("source_record_sha256") != config.dataset_digest
            or not step.request.observable_references
        ):
            raise ValueError("original TRAIN source, current checksum and evidence required")
        prompt = tuple(backend.tokenizer.encode(source.prompt, add_special_tokens=False))
        response = source.response_token_ids
        if (
            prompt != source.prompt_token_ids
            or not prompt
            or not response
            or backend.tokenizer.decode(list(response), skip_special_tokens=True) != source.response
        ):
            raise ValueError("source tokenizer binding mismatch")
        _tokens(prompt + response, vocab=vocab, config=limits)
        total_logits += len(prompt + response) * vocab
        if total_logits > config.max_logit_elements:
            raise ValueError("aggregate native logit budget exceeded")
    _, generation_limits = _generation_limits(backend)
    prepared = prepare_knowledge_progress(
        context,
        conditions,
        captured,
        authorize_evidence=authorize_evidence,
        enabled=True,
        epsilon=algorithm.config.group_normalization_epsilon,
    )
    if not prepared["evidence_review_accepted"]:
        raise ValueError("independent condition evidence review withheld")
    raw_claims = [ResearchCreditClaim.from_dict(value) for value in prepared["claims"]]
    selected = {}
    groups = {}
    for step, row in zip(captured, prepared["steps"]):
        candidates = [
            claim
            for claim in raw_claims
            if claim.action_id == row["action_id"] and claim.credit_kind in weights
        ]
        if not candidates:
            kind = "knowledge" if "knowledge" in weights else "progress"
            # Truncation and unrepresentably small exact shaping are unscored.
            if kind == "progress" and (row["shaping"] is None or row["progress_diagnostic_only"]):
                continue
            candidates = [_zero_claim(step, row, context, prepared, kind)]
        claim = next(
            (value for value in candidates if value.credit_kind == config.event_preference),
            candidates[0],
        )
        # A declared admission is still subject to existing fresh semantic/use and
        # accounting reviews below; evidence review alone never grants optimization.
        selected[row["action_id"]] = replace(claim, admissibility="admitted")
        key = row["group_id"] if config.grouping == "knowledge_state" else None
        if key is None:
            key = content_digest(["public-answer", step.request.trajectory.prompt])
        groups.setdefault(key, []).append(row["action_id"])
    report = {
        "schema_version": "knowledge-training-trial-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "execute_candidate": False,
        "source_references": ["REF-KNOWGPO", "REF-PRM4RL"],
        "config": config.to_dict(),
        "algorithm_config": {
            field.name: getattr(algorithm.config, field.name) for field in fields(algorithm.config)
        },
        "candidate_id": candidate_id,
        "source_policy_checksum": checksum,
        "knowledge_progress": prepared,
        "credit_decisions": [],
        "status": "credit_review",
    }
    validate_input_envelope(report)

    def propose(target: RewardRequest) -> ResearchRewardSubmission:
        claim = selected[target.trajectory.trajectory_id]
        original = by_id[claim.action_id].request
        if _request_wire(original) != _request_wire(target):
            raise ValueError("learning credit target substitution")
        try:
            use = account_event(target, ResearchCreditClaim.from_dict(claim.to_dict()))
        except Exception:
            raise ValueError("independent event accounting failed") from None
        return ResearchRewardSubmission((claim,), use)

    def contextual_review(role: HostReview) -> HostReview:
        return lambda envelope: _accepted(
            role, {"credit": envelope, "knowledge_progress": prepared}
        )

    provider = ResearchRewardProvider(
        pipeline,
        ledger=ledger,
        policy=credit_policy,
        propose=propose,
        authenticate_claims=contextual_review(authenticate_claims),
        authorize_optimizer_use=contextual_review(authorize_optimizer_use),
        authorize_accounting=contextual_review(authorize_accounting),
    )
    scored, totals = {}, {}
    for action_id in selected:
        decision = provider.score(by_id[action_id].request)
        if type(decision) is not ResearchRewardDecision:
            raise ValueError("live independently admitted credit required")
        totals[action_id] = decision.optimizer_reward()
        scored[action_id] = decision.bind_trajectory(by_id[action_id].request.trajectory)
        report["credit_decisions"].append(decision.to_dict())
        validate_input_envelope(report)
    trainable, group_sizes, advantage_values = [], [], []
    for members in groups.values():
        values = algorithm.compute_group_advantages(tuple(totals[key] for key in members))
        group_sizes.append(len(members))
        if values is None or not any(value != 0 for value in values):
            continue
        for action_id, value in zip(members, values):
            source = scored[action_id]
            advantage_values.append(value)
            trainable.append(
                replace(
                    source,
                    advantage=(value,) * len(source.response_token_ids),
                    returns=None,
                    value_predictions=None,
                )
            )
    report.update(group_sizes=group_sizes, advantages=advantage_values)
    if not trainable or not any(
        trajectory.trajectory_id == proposal.source_action_id for trajectory in trainable
    ):
        report["status"] = "no_trainable_groups"
        return LocalCorrectionResult(None, None, canonical_json(report))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        trial, update = _grpo_update(backend, tuple(trainable), algorithm, limits)
        report.update(update)
        updated = update["candidate_policy_checksum"]
        report.update(
            raw_knowledge_gain=fsum(row["knowledge_gain"] for row in prepared["steps"]),
            raw_progress=fsum(
                row["shaping"] for row in prepared["steps"] if row["shaping"] is not None
            ),
        )
        if updated == checksum:
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
            source=by_id[proposal.source_action_id].request,
            evaluate=evaluate,
            generation_limits=generation_limits,
            vocab=vocab,
        )

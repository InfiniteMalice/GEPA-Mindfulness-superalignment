"""Opt-in isolated native DPO on independently reviewed original-state recoveries."""

# Standard library
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.coevolution import CandidateComponent, CoevolutionStore, CorrectionProposal
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract

from .causal_reward import _accepted
from .causal_reward_records import _number, _text, validate_input_envelope
from .deloop import (
    LoopCandidate,
    LoopPairReview,
    _capture,
    _review_reservation,
    deloop_dpo_loss,
    mine_loop_pairs,
)
from .eligibility import require_training_eligible
from .locd_runtime import (
    CandidateEvaluator,
    HostReview,
    LocalCorrectionResult,
    LoCDTrainingConfig,
    _clone_backend,
    _finalize_trial,
    _generation_limits,
    _tokens,
    _validate_backend,
)

if TYPE_CHECKING:
    from .backends.torch_policy import TorchPolicyBackend


@dataclass(frozen=True)
class DeLoopTrainingConfig:
    """Closed operator configuration; beta has no assumed paper default."""

    enabled: bool = False
    policy_id: str = ""
    reviewer_id: str = ""
    reviewer_version: str = ""
    reviewer_contract_id: str = ""
    evaluator_id: str = ""
    dataset_digest: str = ""
    beta: float | None = None
    seed: int = 0
    max_pairs: int = 32
    max_parameters: int = 4_194_304
    max_tokens: int = 4096
    max_logit_elements: int = 1_048_576
    max_evaluation_calls: int = 256

    def __post_init__(self) -> None:
        self.limits()
        if type(self.max_pairs) is not int or not 1 <= self.max_pairs <= 512:
            raise ValueError("invalid preference pair budget")
        for name in ("reviewer_version", "reviewer_contract_id"):
            if self.enabled:
                _text(getattr(self, name), name)
            elif type(getattr(self, name)) is not str:
                raise ValueError("reviewer identifiers must be strings")
        if self.beta is not None:
            _number(self.beta)
            if self.beta <= 0:
                raise ValueError("positive explicit beta required")
        if self.enabled and self.beta is None:
            raise ValueError("positive explicit beta required")

    def limits(self) -> LoCDTrainingConfig:
        """Reuse native isolation/evaluation limits, without a second admission policy."""
        return LoCDTrainingConfig(
            **{field.name: getattr(self, field.name) for field in fields(LoCDTrainingConfig)}
        )

    @property
    def reviewer(self) -> TrustedEvaluatorContract:
        """Identify the independently installed host reviewer; IDs do not authenticate roles."""
        return TrustedEvaluatorContract(
            self.reviewer_id, self.reviewer_version, self.reviewer_contract_id
        )

    def to_dict(self) -> dict[str, object]:
        """Export settings only; no runtime permission survives serialization."""
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["schema_version"] = "deloop-training-config-v1"
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> DeLoopTrainingConfig:
        """Reject unknown fields and schema versions."""
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "deloop-training-config-v1"
        ):
            raise ValueError("invalid DeLoop training config schema")
        return cls(**{name: value[name] for name in names})


def run_loop_preferences(
    backend: TorchPolicyBackend,
    events: object,
    steps: object,
    store: CoevolutionStore,
    correction: CorrectionProposal,
    *,
    task_id: str = "",
    config: DeLoopTrainingConfig = DeLoopTrainingConfig(),
    review: Callable[[LoopCandidate], LoopPairReview] | None = None,
    authorize_source: HostReview | None = None,
    authorize_training: HostReview | None = None,
    evaluate: CandidateEvaluator | None = None,
    candidate_id: str = "",
    artifact_root: Path | None = None,
) -> LocalCorrectionResult | None:
    """Train an isolated copy, then require existing independent protected admission.

    Host source approval authenticates original TRAIN capture, state normalization,
    evidence and actual reviewer independence. Training approval authorizes use of
    the exact reviewed raw-source comparison. Neither the diagnostic proposal nor
    the teacher/recovery certifies admission. The resulting catalog decision does
    not deploy a model or authorize execution.
    """
    if type(config) is not DeLoopTrainingConfig:
        raise ValueError("expected exact DeLoopTrainingConfig")
    config = DeLoopTrainingConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    import torch

    limits = config.limits()
    vocab = _validate_backend(backend, limits)
    if type(store) is not CoevolutionStore or type(correction) is not CorrectionProposal:
        raise ValueError("exact existing store and correction required")
    proposal = CorrectionProposal.from_dict(correction.to_dict())
    if proposal.changed_components != (CandidateComponent.MODEL,):
        raise ValueError("preference training may change only the model component")
    if any(not callable(role) for role in (review, authorize_source, authorize_training, evaluate)):
        raise ValueError("independent host source, pair, training and evaluation roles required")
    if not isinstance(artifact_root, Path):
        raise ValueError("artifact root required")
    _text(candidate_id, "candidate_id")
    audit, sources = _capture(events, steps, task_id)
    source_ids = {step.request.trajectory.trajectory_id for step in sources}
    if proposal.source_trajectory_id not in source_ids:
        raise ValueError("registered source trajectory mismatch")
    registered = store.correction_source_events(proposal)
    if canonical_json([event.to_dict() for event in registered]) != canonical_json(
        [event.to_dict() for event in events]
    ):
        raise ValueError("registered PEO source mismatch")
    if not any(
        row["action_id"] == proposal.source_action_id
        and row["classification"] == "unresolved_repeat"
        for row in audit["entries"]
    ):
        raise ValueError("correction must identify the observed repeat")
    checksum = backend.policy_parameter_checksum()
    token_pairs = {}
    for step in sources:
        source = step.request.trajectory
        if source.sampling_parameters.get("training_eligibility") != "TRAIN":
            raise ValueError("explicit original source TRAIN declaration required")
        require_training_eligible(source.to_dict())
        if (
            source.model_identifier != backend.model_identifier
            or source.sampling_parameters.get("policy_parameter_checksum") != checksum
            or source.sampling_parameters.get("source_record_sha256") != config.dataset_digest
        ):
            raise ValueError("source model, policy checksum or dataset digest mismatch")
        if {config.reviewer_id, config.evaluator_id} & {
            source.policy_version,
            source.model_identifier,
        }:
            raise ValueError("host review and evaluation must be independent of the student")
        prompt = tuple(backend.tokenizer.encode(source.prompt, add_special_tokens=False))
        response = source.response_token_ids
        if (
            prompt != source.prompt_token_ids
            or response is None
            or backend.tokenizer.decode(list(response), skip_special_tokens=True) != source.response
        ):
            raise ValueError("source tokenizer binding mismatch")
        _tokens(prompt + response, vocab=vocab, config=limits)
        if not prompt or not response:
            raise ValueError("nonempty original prompt and action tokens required")
        token_pairs[source.trajectory_id] = (prompt, response)
    _, generation_limits = _generation_limits(backend)
    # DEVELOPMENT proposals are audit data. Only the retained, approved original
    # TRAIN sources below are passed to the model; nothing is relabeled TRAIN.
    report = {
        "schema_version": "deloop-training-trial-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "execute_candidate": False,
        "source_reference": "REF-DELOOP",
        "config": config.to_dict(),
        "candidate_id": candidate_id,
        "correction": proposal.to_dict(),
        "source_policy_checksum": checksum,
        "source_audit": audit,
        "status": "source_review",
    }
    validate_input_envelope(report)
    if not _accepted(authorize_source, report):
        raise ValueError("source review withheld")
    reservations = []

    def bounded_review(candidate: LoopCandidate) -> LoopPairReview:
        if len(reservations) >= config.max_pairs:
            raise ValueError("preference pair review budget exceeded")
        entry = candidate.to_dict()
        refs = tuple(EvidenceReference.from_dict(ref) for ref in entry["evidence_refs"])
        entry.update(
            candidate_digest=candidate.digest,
            review=_review_reservation(candidate, config.reviewer, refs),
        )
        reservations.append(entry)
        # Reserve the complete nested candidate/review before the host callback.
        # Rejected proposals are smaller than these accepted-entry reservations.
        reserved = {
            "schema_version": "deloop-pairs-v1",
            "training_eligibility": "DEVELOPMENT",
            "authority_granted": False,
            "source_reference": "REF-DELOOP",
            "audit_digest": content_digest(audit),
            "pairs": reservations,
            "rejected_proposals": [
                {"action_id": row["action_id"], "reason": "unverified repeat observation"}
                for row in audit["entries"]
                if row["classification"] == "unresolved_repeat"
            ],
        }
        validate_input_envelope(report | {"mining": reserved})
        try:
            return review(candidate)
        except Exception:
            raise ValueError("independent pair review failed") from None

    mined = mine_loop_pairs(
        registered,
        sources,
        task_id=task_id,
        reviewer=config.reviewer,
        review=bounded_review,
        enabled=True,
    )
    pairs = mined["pairs"]
    if len(pairs) > config.max_pairs:
        raise ValueError("preference pair budget exceeded")
    report.update(mining=mined, pair_count=len(pairs), status="training_review")
    validate_input_envelope(report)
    if pairs and not any(pair["rejected_action_id"] == proposal.source_action_id for pair in pairs):
        report.update(
            status="no_eligible_pairs",
            pair_count=0,
            reason="registered_repeat_has_no_eligible_pair",
        )
        return LocalCorrectionResult(None, None, canonical_json(report))
    if not pairs:
        report["status"] = "no_eligible_pairs"
        return LocalCorrectionResult(None, None, canonical_json(report))
    selected = []
    for pair in pairs:
        prompt, rejected = token_pairs[pair["rejected_action_id"]]
        _, chosen = token_pairs[pair["chosen_action_id"]]
        _tokens(prompt + chosen, vocab=vocab, config=limits)
        selected.append((prompt, chosen, rejected))
    # Approve exact raw source comparisons and original-state applicability before
    # cloning/model calls. The mining report remains independently inspectable.
    report["training_pairs_digest"] = content_digest(selected)
    if not _accepted(authorize_training, report):
        raise ValueError("training-use review withheld")
    trial = _clone_backend(backend)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        trial.policy_model.eval()
        trial.reference_model.eval()
        trial.zero_grad()
        losses = []
        for prompt, chosen, rejected in selected:
            chosen_logp, ref_chosen, _, _ = trial._evaluate_tokens(prompt, chosen)
            rejected_logp, ref_rejected, _, _ = trial._evaluate_tokens(prompt, rejected)
            loss = deloop_dpo_loss(
                chosen_logp.sum().reshape(1),
                rejected_logp.sum().reshape(1),
                ref_chosen.sum().reshape(1),
                ref_rejected.sum().reshape(1),
                beta=config.beta,
            )
            # Average the same per-pair objective while freeing each graph before
            # the next pair, rather than retaining max_pairs full-logit graphs.
            trial.backward(loss / len(selected))
            losses.append(float(loss.detach()))
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in trial.policy_parameters()
        ):
            raise ValueError("nonfinite preference gradient")
        step = trial.optimizer_step()
        trial.zero_grad()
        updated_checksum = trial.policy_parameter_checksum()
        _validate_backend(trial, limits)
        report.update(
            loss=sum(losses) / len(losses),
            optimizer_step=step.step,
            candidate_policy_checksum=updated_checksum,
        )
        if updated_checksum == checksum:
            report["status"] = "no_model_change"
            return LocalCorrectionResult(None, None, canonical_json(report))
        by_id = {step.request.trajectory.trajectory_id: step.request for step in sources}
        return _finalize_trial(
            trial,
            store,
            proposal,
            report,
            config=limits,
            candidate_id=candidate_id,
            artifact_root=artifact_root,
            source=by_id[pairs[0]["rejected_action_id"]],
            evaluate=evaluate,
            generation_limits=generation_limits,
            vocab=vocab,
        )

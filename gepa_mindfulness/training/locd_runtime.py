"""Isolated, host-reviewed CPU context self-distillation using existing LoCD contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

from evaluation.causal_records import canonical_json
from gepa_mindfulness.coevolution import (
    AcceptanceDecision,
    CandidateComponent,
    CandidateSystem,
    CoevolutionStore,
    CorrectionProposal,
    ValidationBundle,
    decide_candidate_acceptance,
)

from .causal_reward import _accepted
from .causal_reward_records import (
    _digest,
    _json_value,
    _read_request,
    _request_wire,
    _text,
    validate_input_envelope,
)
from .contracts import RewardRequest
from .eligibility import require_training_eligible
from .locd import localized_forward_kl, prepare_registered_local_segment
from .seeds import validate_seed
from .trajectory import RolloutRequest, Trajectory

if TYPE_CHECKING:
    from .backends.torch_policy import TorchPolicyBackend

HostReview = Callable[[dict[str, object]], bool]
CandidateEvaluator = Callable[
    [CandidateSystem, Callable[[RolloutRequest], Trajectory]], ValidationBundle
]
_SINGLE_SEQUENCE = {
    "num_beams": 1,
    "num_return_sequences": 1,
    "return_dict_in_generate": False,
    "output_scores": False,
    "output_logits": False,
    "output_hidden_states": False,
    "output_attentions": False,
}


@dataclass(frozen=True)
class LoCDTrainingConfig:
    """Operator-scoped bounds for one trial; limits are engineering safeguards."""

    enabled: bool = False
    policy_id: str = ""
    reviewer_id: str = ""
    evaluator_id: str = ""
    dataset_digest: str = ""
    seed: int = 0
    max_parameters: int = 4_194_304
    max_tokens: int = 4096
    max_logit_elements: int = 1_048_576
    max_evaluation_calls: int = 256

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        validate_seed(self.seed)
        if type(self.seed) is not int:
            raise ValueError("explicit integer seed required")
        for name, maximum in (
            ("max_parameters", 16_777_216),
            ("max_tokens", 65536),
            ("max_logit_elements", 16_777_216),
            ("max_evaluation_calls", 65536),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid {name} budget")
        if self.enabled:
            for name in ("policy_id", "reviewer_id", "evaluator_id"):
                _text(getattr(self, name), name)
            _digest(self.dataset_digest)
            if self.reviewer_id == self.evaluator_id:
                raise ValueError("correction reviewer and evaluator must be independent")
        elif any(
            type(getattr(self, name)) is not str
            for name in (
                "policy_id",
                "reviewer_id",
                "evaluator_id",
                "dataset_digest",
            )
        ):
            raise ValueError("scope identifiers must be strings")

    def to_dict(self) -> dict[str, object]:
        """Export a closed, versioned operator configuration."""
        self.__post_init__()
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["schema_version"] = "locd-training-config-v1"
        validate_input_envelope(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> LoCDTrainingConfig:
        """Restore configuration, never runtime review or admission authority."""
        validate_input_envelope(value)
        names = {field.name for field in fields(cls)}
        if (
            type(value) is not dict
            or set(value) != names | {"schema_version"}
            or value["schema_version"] != "locd-training-config-v1"
        ):
            raise ValueError("invalid LoCD training config schema")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class LocalCorrectionResult:
    """An isolated backend only when the existing catalog accepts; never deployment."""

    backend: TorchPolicyBackend | None
    decision: AcceptanceDecision | None
    _report_json: str

    def to_dict(self) -> dict[str, object]:
        """Return detached DEVELOPMENT audit data without restoring a capability."""
        return _json_value(self._report_json)


def _validate_backend(backend: TorchPolicyBackend, config: LoCDTrainingConfig) -> int:
    """Reject unsupported devices/shapes before cloning or allocating model outputs."""
    import torch

    from .backends.torch_policy import TorchPolicyBackend

    if (
        type(backend) is not TorchPolicyBackend
        or backend.device.type != "cpu"
        or backend.training_mode != "full"
        or backend.autocast_dtype is not None
        or backend.gradient_scaler is not None
        or backend.distributed_strategy != "none"
    ):
        raise ValueError("LoCD trials require the portable CPU full-weight FP32 backend")
    parameters = tuple(backend.policy_model.parameters())
    all_parameters = (
        *parameters,
        *backend.reference_model.parameters(),
        *backend.value_head.parameters(),
    )
    if sum(parameter.numel() for parameter in all_parameters) > config.max_parameters:
        raise ValueError("model parameter budget exceeded")
    for parameter in all_parameters:
        if (
            parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            or not bool(torch.isfinite(parameter).all())
            or parameter.grad is not None
        ):
            raise ValueError("finite CPU FP32 parameters with cleared gradients required")
    output = getattr(backend.policy_model, "get_output_embeddings", None)
    head = output() if callable(output) else getattr(backend.policy_model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] <= 0:
        raise ValueError("bounded output-vocabulary shape required")
    return int(weight.shape[0])


def _tokens(tokens: tuple[int, ...], *, vocab: int, config: LoCDTrainingConfig) -> None:
    """Check token and projected-logit capacity before model calls."""
    if (
        not tokens
        or len(tokens) > config.max_tokens
        or len(tokens) * vocab > config.max_logit_elements
        or any(type(token) is not int or not 0 <= token < vocab for token in tokens)
    ):
        raise ValueError("token or logit budget exceeded")


def _state_digest(backend: TorchPolicyBackend) -> str:
    """Bind all serialized model tensors, including buffers and frozen references."""
    import torch

    digest = hashlib.sha256()
    for label, module in (
        ("policy", backend.policy_model),
        ("reference", backend.reference_model),
        ("value", backend.value_head),
    ):
        for name, tensor in sorted(module.state_dict().items()):
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("plain CPU tensor state required")
            digest.update(f"{label}.{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _clone_backend(backend: TorchPolicyBackend) -> TorchPolicyBackend:
    """Copy native optimizer/SST ownership and reject any shared tensor storage."""
    trial = deepcopy(backend)
    source_modules = (backend.policy_model, backend.reference_model, backend.value_head)
    trial_modules = (trial.policy_model, trial.reference_model, trial.value_head)
    source_storages = {
        tensor.untyped_storage().data_ptr()
        for module in source_modules
        for tensor in (*module.parameters(), *module.buffers())
        if tensor.numel()
    }
    if any(
        tensor.numel() and tensor.untyped_storage().data_ptr() in source_storages
        for module in trial_modules
        for tensor in (*module.parameters(), *module.buffers())
    ):
        raise ValueError("isolated trial shares model storage")
    owned = {
        id(parameter)
        for module in (trial.policy_model, trial.value_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    optimizer_owned = {
        id(parameter) for group in trial.optimizer.param_groups for parameter in group["params"]
    }
    if owned != optimizer_owned:
        raise ValueError("isolated optimizer ownership mismatch")
    if _state_digest(trial) != _state_digest(backend):
        raise ValueError("isolated model state mismatch")
    return trial


def run_local_correction(
    backend: TorchPolicyBackend,
    request: RewardRequest,
    store: CoevolutionStore,
    correction: CorrectionProposal,
    *,
    config: LoCDTrainingConfig = LoCDTrainingConfig(),
    start: int = 0,
    end: int = 0,
    teacher_context: str = "",
    authorize_source: HostReview | None = None,
    authorize_correction: HostReview | None = None,
    evaluate: CandidateEvaluator | None = None,
    candidate_id: str = "",
    artifact_root: Path | None = None,
) -> LocalCorrectionResult | None:
    """Train one copied student, then use existing independent catalog admission.

    The unchanged source policy supplies privileged-context teacher distributions.
    Installed host roles must authenticate localization, source capture, trusted
    context and permission to train; identifiers alone do not prove independence.
    The evaluator supplies existing store-issued held-out/protected receipts.
    No action, reward authorization or deployment is granted by this function.
    """
    if type(config) is not LoCDTrainingConfig:
        raise ValueError("expected exact LoCDTrainingConfig")
    config = LoCDTrainingConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    import torch

    vocab = _validate_backend(backend, config)
    if type(request) is not RewardRequest or type(store) is not CoevolutionStore:
        raise ValueError("exact source request and coevolution store required")
    if type(correction) is not CorrectionProposal:
        raise ValueError("exact registered correction required")
    source = _read_request(_request_wire(request))
    proposal = CorrectionProposal.from_dict(correction.to_dict())
    if source.trajectory.sampling_parameters.get("training_eligibility") != "TRAIN":
        raise ValueError("explicit source TRAIN declaration required")
    require_training_eligible(source.trajectory.to_dict())
    checksum = backend.policy_parameter_checksum()
    if source.trajectory.sampling_parameters.get("policy_parameter_checksum") != checksum:
        raise ValueError("source policy checksum mismatch")
    _digest(source.trajectory.sampling_parameters.get("source_record_sha256"))
    if proposal.changed_components != (CandidateComponent.MODEL,):
        raise ValueError("localized training may change only the model component")
    if not proposal.teacher_evidence_refs or any(
        not ref.is_observable for ref in proposal.teacher_evidence_refs
    ):
        raise ValueError("observable trusted teacher evidence required")
    for role in (config.reviewer_id, config.evaluator_id):
        if role in {source.trajectory.policy_version, backend.model_identifier}:
            raise ValueError("host roles must be independent of the student/teacher")
    if (
        not callable(evaluate)
        or not callable(authorize_source)
        or not callable(authorize_correction)
        or not isinstance(artifact_root, Path)
    ):
        raise ValueError("fresh host reviews, protected evaluator and artifact root required")
    _text(candidate_id, "candidate_id")
    public_prompt = tuple(
        backend.tokenizer.encode(source.trajectory.prompt, add_special_tokens=False)
    )
    response = source.trajectory.response_token_ids
    if (
        public_prompt != source.trajectory.prompt_token_ids
        or response is None
        or backend.tokenizer.decode(list(response), skip_special_tokens=True)
        != source.trajectory.response
    ):
        raise ValueError("source tokenizer binding mismatch")
    if type(teacher_context) is not str or not teacher_context.strip():
        raise ValueError("trusted teacher context required")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response):
        raise ValueError("invalid original correction region")
    context = tuple(backend.tokenizer.encode(teacher_context, add_special_tokens=False))
    # Validate localization and the registered source BEFORE invoking any host role/model.
    generation_config = getattr(backend.policy_model, "generation_config", None)
    inherited_config = generation_config.to_dict() if generation_config is not None else {}
    validate_input_envelope(inherited_config)
    generation_limits: dict[str, object] = dict(_SINGLE_SEQUENCE)
    # Older Transformers releases expose additional generation-mode selectors.
    # Clear only attributes supported by this model's configuration, so removed
    # options never become unsupported forward kwargs on newer releases.
    for name, value in (
        ("penalty_alpha", None),
        ("constraints", None),
        ("force_words_ids", None),
        ("dola_layers", None),
        ("prompt_lookup_num_tokens", None),
        ("assistant_early_exit", None),
        ("token_healing", False),
    ):
        if generation_config is not None and hasattr(generation_config, name):
            generation_limits[name] = value
    identity = {
        "model_id": backend.model_identifier,
        "version": source.trajectory.policy_version,
        "sampling_config": {
            "inherited_generation_config": inherited_config,
            **generation_limits,
            "do_sample": False,
            "seed": config.seed,
            "max_new_tokens": end - start,
        },
    }
    segment = prepare_registered_local_segment(
        store,
        source,
        proposal,
        start=start,
        end=end,
        teacher_tokens=response[start:end],
        teacher_context=teacher_context,
        teacher_identity=identity,
        enabled=True,
    )
    _tokens(public_prompt + response, vocab=vocab, config=config)
    _tokens(public_prompt + context + response[:end], vocab=vocab, config=config)
    if not public_prompt or not context:
        raise ValueError("nonempty public prompt and teacher context tokens required")
    report = {
        "schema_version": "locd-training-trial-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "execute_candidate": False,
        "source_reference": "REF-LOCD",
        "config": config.to_dict(),
        "candidate_id": candidate_id,
        "source_policy_checksum": checksum,
        "segment": segment,
        "teacher_mode": "frozen-source-context-self-distillation",
        "teacher_context": teacher_context,
        "status": "source_review",
    }
    if not _accepted(authorize_source, report):
        raise ValueError("source review withheld")
    trial = _clone_backend(backend)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        trial.policy_model.eval()
        prefix = public_prompt + context + response[:start]
        prefix_tensor = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            generated = trial.policy_model.generate(
                prefix_tensor,
                attention_mask=torch.ones_like(prefix_tensor),
                max_new_tokens=end - start,
                do_sample=False,
                **generation_limits,
            )
        if (
            not isinstance(generated, torch.Tensor)
            or generated.dtype != torch.long
            or generated.shape != (1, len(prefix) + end - start)
            or not torch.equal(generated[0, : len(prefix)], prefix_tensor[0])
        ):
            raise ValueError("teacher must preserve the prefix and fill exactly the local span")
        corrected = tuple(generated[0, len(prefix) :].tolist())
        _tokens(prefix + corrected, vocab=vocab, config=config)
        segment = prepare_registered_local_segment(
            store,
            source,
            proposal,
            start=start,
            end=end,
            teacher_tokens=corrected,
            teacher_context=teacher_context,
            teacher_identity=identity,
            enabled=True,
        )
        report.update(segment=segment, status="correction_review")
        if not _accepted(authorize_correction, report):
            raise ValueError("correction review withheld")
        student_response = response[:start] + corrected
        student_ids = torch.tensor([public_prompt + student_response], dtype=torch.long)
        teacher_prompt = public_prompt + context
        teacher_ids = torch.tensor([teacher_prompt + student_response], dtype=torch.long)
        # Targets are evaluated before the update and detached. Only the selected
        # response positions enter KL; teacher context never reaches student_ids.
        with torch.no_grad():
            teacher_outputs = trial.policy_model(
                teacher_ids, attention_mask=torch.ones_like(teacher_ids), output_hidden_states=False
            )
            teacher_logits, _ = trial._validated_outputs(
                teacher_outputs, teacher_ids.shape[1], "LoCD teacher", require_hidden=False
            )
            teacher_logits = teacher_logits[:, len(teacher_prompt) - 1 : -1].detach()
        student_outputs = trial.policy_model(
            student_ids, attention_mask=torch.ones_like(student_ids), output_hidden_states=False
        )
        student_logits, _ = trial._validated_outputs(
            student_outputs, student_ids.shape[1], "LoCD student", require_hidden=False
        )
        student_logits = student_logits[:, len(public_prompt) - 1 : -1]
        loss = localized_forward_kl(
            student_logits, teacher_logits, torch.tensor([segment["loss_mask"]], dtype=torch.bool)
        )
        trial.zero_grad()
        trial.backward(loss)
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in trial.policy_parameters()
        ):
            raise ValueError("nonfinite localized gradient")
        step = trial.optimizer_step()
        trial.zero_grad()
        updated_checksum = trial.policy_parameter_checksum()
        _validate_backend(trial, config)
        report.update(
            loss=float(loss.detach()),
            optimizer_step=step.step,
            candidate_policy_checksum=updated_checksum,
        )
        if updated_checksum == checksum:
            report["status"] = "no_model_change"
            return LocalCorrectionResult(None, None, canonical_json(report))
        # One new directory per trial: never overwrite prior evidence/checkpoints.
        trial_dir = artifact_root / hashlib.sha256(candidate_id.encode()).hexdigest()
        trial_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = trial_dir / "backend.pt"
        trial.save_checkpoint(checkpoint)
        artifact_digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        evaluated_state_digest = _state_digest(trial)
        candidate = store.register_candidate(
            candidate_id=candidate_id, correction=proposal, artifact_digest=artifact_digest
        )
        report.update(
            checkpoint_path=str(checkpoint.resolve()),
            artifact_digest=artifact_digest,
            candidate=candidate.to_dict(),
            status="pending_independent_validation",
        )
        _write_report(trial_dir, report)
        remaining_calls = config.max_evaluation_calls

        def generate(public_request: RolloutRequest) -> Trajectory:
            """Expose bounded inference, never context targets or optimizer methods."""
            nonlocal remaining_calls
            if type(public_request) is not RolloutRequest or public_request.num_samples != 1:
                raise ValueError("single public evaluation rollout required")
            if public_request.policy_version != candidate.model_version:
                raise ValueError("candidate evaluation policy version mismatch")
            parameters = dict(public_request.sampling_parameters)
            if set(parameters) - {"max_new_tokens", "do_sample", "temperature", "top_p"}:
                raise ValueError("unsupported evaluation sampling parameter")
            count = parameters.get("max_new_tokens", trial.max_new_tokens)
            if type(count) is not int or not 0 < count <= config.max_tokens or remaining_calls <= 0:
                raise ValueError("evaluation budget exceeded")
            prompt = tuple(trial.tokenizer.encode(public_request.prompt, add_special_tokens=False))
            _tokens(prompt + (0,) * count, vocab=vocab, config=config)
            remaining_calls -= 1
            bounded_request = replace(
                public_request, sampling_parameters={**parameters, **generation_limits}
            )
            with torch.random.fork_rng(devices=[]):
                return trial.generate((bounded_request,))[0]

        try:
            public_result = generate(
                RolloutRequest(
                    source.trajectory.prompt,
                    case_id=source.trajectory.case_id,
                    policy_version=candidate.model_version,
                    seed=config.seed,
                    sampling_parameters={"max_new_tokens": len(response), "do_sample": False},
                )
            )
            report["public_reevaluation"] = public_result.to_dict()
            _write_report(trial_dir, report)
            try:
                bundle = evaluate(candidate, generate)
            except Exception:
                report["status"] = "evaluation_failed"
                raise ValueError("independent candidate evaluation failed") from None
            if (
                trial.policy_parameter_checksum() != updated_checksum
                or _state_digest(trial) != evaluated_state_digest
                or "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                != artifact_digest
            ):
                raise ValueError("candidate changed during independent evaluation")
            if type(bundle) is not ValidationBundle or bundle.candidate != candidate:
                raise ValueError("validation bundle does not name the actual candidate")
            decision = decide_candidate_acceptance(bundle, authority_store=store)
        except Exception:
            if report["status"] != "evaluation_failed":
                report["status"] = "validation_failed"
            _write_report(trial_dir, report)
            raise
        report.update(
            status="catalog_accepted" if decision.accepted else "catalog_rejected",
            decision=decision.to_dict(),
        )
        _write_report(trial_dir, report)
        return LocalCorrectionResult(
            trial if decision.accepted else None, decision, canonical_json(report)
        )


def _write_report(directory: Path, report: dict[str, object]) -> None:
    """Persist bounded audit snapshots atomically; the checkpoint remains unchanged."""
    validate_input_envelope(report)
    temporary = directory / "trial.pending.json"
    temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
    temporary.replace(directory / "trial.json")

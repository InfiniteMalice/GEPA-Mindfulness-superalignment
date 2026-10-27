"""Default-off local-segment preparation and a separate differentiable KL objective."""

# Standard library
from __future__ import annotations

# Local
from evaluation.causal_records import canonical_json
from gepa_mindfulness.coevolution import CoevolutionStore, CorrectionProposal

from .causal_reward_records import (
    _json_value,
    _read_request,
    _request_wire,
    validate_input_envelope,
)
from .contracts import RewardRequest


def build_local_segment(
    request: RewardRequest,
    *,
    start: int | None = None,
    end: int | None = None,
    teacher_tokens: tuple[int, ...] = (),
    teacher_context: str = "",
    enabled: bool = False,
) -> dict[str, object] | None:
    """Prepare a bounded local token target; this does not verify a localization.

    The original prompt and response prefix reach the student. Privileged context
    is retained separately for teacher generation only. Tokenizer identity and
    semantic correction validity remain host responsibilities.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(request) is not RewardRequest:
        raise ValueError("expected exact RewardRequest")
    source = _read_request(_request_wire(request))
    response = source.trajectory.response_token_ids
    prompt = source.trajectory.prompt_token_ids
    if response is None or prompt is None or not 1 <= len(response) <= 65536:
        raise ValueError("bounded original token IDs required")
    if any(token < 0 for token in prompt + response):
        raise ValueError("source token IDs must be nonnegative")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response):
        raise ValueError("invalid original correction region")
    if start == 0 and end == len(response):
        raise ValueError("whole-trajectory imitation is not a localized correction")
    if (
        type(teacher_tokens) is not tuple
        or len(teacher_tokens) != end - start
        or any(type(token) is not int or token < 0 for token in teacher_tokens)
    ):
        raise ValueError("teacher continuation must fill exactly the declared local token region")
    if type(teacher_context) is not str or not teacher_context.strip():
        raise ValueError("trusted teacher context required")
    tokens = response[:start] + teacher_tokens
    result = {
        "schema_version": "locd-local-segment-v1",
        "training_eligibility": "DEVELOPMENT",
        "authority_granted": False,
        "source_reference": "REF-LOCD",
        "source_request": _request_wire(source),
        "start": start,
        "end": end,
        "student_inputs": {
            "prompt": source.trajectory.prompt,
            "prompt_token_ids": list(prompt),
            "response_token_ids": list(tokens),
        },
        "student_response_token_ids": list(tokens),
        "loss_mask": [False] * start + [True] * len(teacher_tokens),
        "teacher_context": teacher_context,
    }
    validate_input_envelope(result)
    return result


def prepare_registered_local_segment(
    store: CoevolutionStore,
    request: RewardRequest,
    correction: CorrectionProposal,
    *,
    start: int,
    end: int,
    teacher_tokens: tuple[int, ...],
    teacher_context: str,
    teacher_identity: dict[str, object],
    enabled: bool = False,
) -> dict[str, object] | None:
    """Reuse the registered correction/epoch boundary; still return audit-only data.

    Source trajectory IDs must be shared between the rollout and coevolution
    logger. The store validates declared localization/evidence, not mechanistic
    causation or tokenizer correctness. Teacher identity is preserved, not trusted.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(store) is not CoevolutionStore or type(correction) is not CorrectionProposal:
        raise ValueError("expected existing coevolution store and correction")
    proposal = CorrectionProposal.from_dict(correction.to_dict())
    events = store.correction_source_events(proposal)
    if request.trajectory.trajectory_id != proposal.source_trajectory_id:
        raise ValueError("localized source trajectory mismatch")
    if not events or request.trajectory.policy_version != events[0].model_version:
        raise ValueError("localized source model mismatch")
    if type(teacher_identity) is not dict or set(teacher_identity) != {
        "model_id",
        "version",
        "sampling_config",
    }:
        raise ValueError("teacher identity requires model, version and sampling configuration")
    if (
        any(
            type(teacher_identity[key]) is not str or not teacher_identity[key].strip()
            for key in ("model_id", "version")
        )
        or type(teacher_identity["sampling_config"]) is not dict
    ):
        raise ValueError("invalid teacher identity")
    result = build_local_segment(
        request,
        start=start,
        end=end,
        teacher_tokens=teacher_tokens,
        teacher_context=teacher_context,
        enabled=True,
    )
    result.update(
        correction=proposal.to_dict(),
        teacher_identity=teacher_identity,
        source_event_ids=[event.event_id for event in events],
    )
    validate_input_envelope(result)
    # Round-trip to detach caller-owned teacher sampling metadata.
    return _json_value(canonical_json(result))


def localized_forward_kl(
    student_logits: object, teacher_logits: object, loss_mask: object
) -> object:
    """Compute masked forward KL with detached teacher targets using lazy PyTorch.

    Inputs are aligned response-position logits [batch, time, vocabulary] and a
    boolean [batch, time] mask. This function performs no model forward pass and
    cannot prevent an external caller from putting hints into their model prompt.
    No clipping or merger with ActionOPD's policy objective is implied.
    """
    import torch

    if not all(
        isinstance(value, torch.Tensor) for value in (student_logits, teacher_logits, loss_mask)
    ):
        raise ValueError("expected PyTorch tensors")
    if (
        student_logits.ndim != 3
        or min(student_logits.shape) <= 0
        or teacher_logits.shape != student_logits.shape
    ):
        raise ValueError("teacher/student logits must align [batch,time,vocabulary]")
    if loss_mask.shape != student_logits.shape[:2] or loss_mask.dtype != torch.bool:
        raise ValueError("loss mask must be boolean [batch,time]")
    if student_logits.device != teacher_logits.device or loss_mask.device != student_logits.device:
        raise ValueError("all objective tensors must use the same device")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if (
        not bool(loss_mask.any(dim=-1).all())
        or not bool(torch.isfinite(student_logits).all())
        or not bool(torch.isfinite(teacher_logits).all())
    ):
        raise ValueError("empty local mask or nonfinite logits")
    # Unused prefix logits must never enter softmax or its backward computation.
    dtype = (
        torch.float64
        if torch.float64 in (student_logits.dtype, teacher_logits.dtype)
        else torch.float32
    )
    teacher_log_probs = torch.log_softmax(teacher_logits.detach()[loss_mask].to(dtype), dim=-1)
    student_log_probs = torch.log_softmax(student_logits[loss_mask].to(dtype), dim=-1)
    active_divergence = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(
        dim=-1
    )
    divergence = active_divergence.new_zeros(loss_mask.shape).masked_scatter(
        loss_mask, active_divergence
    )
    per_instance = divergence.sum(dim=-1) / loss_mask.sum(dim=-1)
    if not bool(torch.isfinite(per_instance).all()):
        raise ValueError("nonfinite localized divergence after numerical conversion")
    return per_instance.mean()

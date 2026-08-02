"""Shared records and helpers for portable policy-gradient objectives."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation, TrajectoryBatch

from .ops import Tensor, TensorOps


@dataclass(frozen=True)
class AlgorithmBatch:
    """Tensors prepared by a backend for a policy-gradient objective."""

    old_log_probs: Tensor
    advantages: Tensor
    mask: Tensor
    returns: Tensor | None = None
    old_values: Tensor | None = None

    def __post_init__(self) -> None:
        require_matching_shapes(
            self.old_log_probs,
            self.advantages,
            self.mask,
            self.returns,
            self.old_values,
            names=("old_log_probs", "advantages", "mask", "returns", "old_values"),
        )


@dataclass(frozen=True)
class AlgorithmLoss:
    """Differentiable total loss and its scalar diagnostic components."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    kl: Tensor


def require_matching_shapes(
    reference: Tensor,
    *values: Tensor | None,
    names: tuple[str, ...] | None = None,
) -> None:
    """Reject tensor inputs that would otherwise broadcast silently."""
    field_names = names or (
        "reference",
        *(f"value[{index}]" for index in range(len(values))),
    )
    if len(field_names) != len(values) + 1:
        raise ValueError("algorithm tensor shape names must identify every guarded value")
    reference_shape = _valid_tensor_shape(reference, field_names[0])
    for field_name, value in zip(field_names[1:], values):
        if value is None:
            continue
        value_shape = _valid_tensor_shape(value, field_name)
        if value_shape != reference_shape:
            raise ValueError(
                f"algorithm tensor {field_name} must satisfy matching shapes; "
                f"expected {reference_shape}, got {value_shape}"
            )


def _valid_tensor_shape(value: Tensor, field_name: str) -> tuple[int, ...]:
    """Return a comparable tensor shape or reject missing and malformed shape metadata."""
    try:
        shape = getattr(value, "shape")
        dimensions = tuple(operator.index(dimension) for dimension in shape)
    except Exception as exc:
        raise ValueError(f"algorithm tensor {field_name} must have a valid shape") from exc
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError(f"algorithm tensor {field_name} must have a valid shape")
    return dimensions


def _required_trajectory_rows(
    batch: TrajectoryBatch,
    field_name: str,
    *,
    width: int,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for trajectory in batch.trajectories:
        values = getattr(trajectory, field_name)
        if values is None:
            raise ValueError(f"trajectory.{field_name} is required for the algorithm objective")
        row = tuple(values)
        token_width = (
            len(trajectory.response_token_ids)
            if trajectory.response_token_ids is not None
            else width
        )
        if len(row) != token_width:
            raise ValueError(
                f"trajectory.{field_name} must align with response_token_ids before padding"
            )
        if len(row) > width:
            raise ValueError(f"trajectory.{field_name} exceeds response_token_masks width")
        rows.append(row + (0.0,) * (width - len(row)))
    return rows


def algorithm_batch_from_trajectories(
    ops: TensorOps,
    batch: TrajectoryBatch,
    evaluation: PolicyEvaluation,
    *,
    require_value_targets: bool,
) -> AlgorithmBatch:
    """Adapt immutable trajectory records to backend tensors beside an evaluation tensor."""
    if not batch.trajectories:
        raise ValueError("algorithm batches must contain at least one trajectory")
    if batch.response_token_masks is None:
        raise ValueError("response_token_masks are required for the algorithm objective")
    if len(batch.response_token_masks) != len(batch.trajectories):
        raise ValueError("response_token_masks must align with trajectories")
    widths = {len(row) for row in batch.response_token_masks}
    if len(widths) != 1:
        raise ValueError("response_token_masks must form a rectangular batch")
    width = next(iter(widths))
    old_log_probs = _required_trajectory_rows(batch, "old_log_probs", width=width)
    advantages = _required_trajectory_rows(batch, "advantage", width=width)
    returns = (
        _required_trajectory_rows(batch, "returns", width=width) if require_value_targets else None
    )
    old_values = (
        _required_trajectory_rows(batch, "value_predictions", width=width)
        if require_value_targets
        else None
    )
    like = evaluation.log_probs
    return AlgorithmBatch(
        old_log_probs=ops.from_data(old_log_probs, like=like),
        advantages=ops.from_data(advantages, like=like),
        mask=ops.from_data(batch.response_token_masks, like=like, kind="bool"),
        returns=None if returns is None else ops.from_data(returns, like=like),
        old_values=None if old_values is None else ops.from_data(old_values, like=like),
    )


@dataclass(frozen=True)
class PolicyGradientConfig:
    """Shared objective coefficients independent of execution configuration."""

    clip_range: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        for name in ("clip_range", "kl_coef", "entropy_coef"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        if self.kl_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients must be non-negative")

    @classmethod
    def _runtime_values(cls, config: AlgorithmConfig) -> dict[str, float]:
        return {
            "clip_range": config.clip_range,
            "kl_coef": config.kl_coef,
        }


def clipped_policy_loss(
    ops: TensorOps,
    batch: AlgorithmBatch,
    log_probs: Tensor,
    clip_range: float,
) -> Tensor:
    """Return the masked PPO clipped-surrogate policy objective."""
    ratio = ops.exp(log_probs - batch.old_log_probs)
    unclipped = ratio * batch.advantages
    clipped_ratio = ops.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surrogate = ops.minimum(unclipped, clipped_ratio * batch.advantages)
    return -ops.masked_mean(surrogate, batch.mask)


def ppo_reference_kl(
    ops: TensorOps,
    log_probs: Tensor,
    reference_log_probs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return PPO's signed masked mean of current minus reference log-probability."""
    return ops.masked_mean(log_probs - reference_log_probs, mask)


def grpo_sampled_reverse_kl(
    ops: TensorOps,
    log_probs: Tensor,
    reference_log_probs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return GRPO's non-negative sampled reverse-KL estimator."""
    log_ratio = reference_log_probs - log_probs
    estimator = ops.exp(log_ratio) - log_ratio - 1.0
    return ops.masked_mean(estimator, mask)


def optional_masked_mean(
    ops: TensorOps,
    value: Tensor | None,
    mask: Tensor,
    zero: Tensor,
) -> Tensor:
    """Return a masked diagnostic mean or a tensor-compatible zero."""
    if value is None:
        return zero
    return ops.masked_mean(value, mask)


def require_regularization_inputs(
    config: PolicyGradientConfig,
    *,
    reference_log_probs: Tensor | None,
    entropy: Tensor | None,
) -> None:
    """Require design-level diagnostics and configured regularization inputs."""
    if reference_log_probs is None:
        raise ValueError("reference_log_probs are required for reference-policy diagnostics")
    if config.entropy_coef > 0.0 and entropy is None:
        raise ValueError("entropy is required when entropy_coef is positive")


__all__ = [
    "AlgorithmBatch",
    "AlgorithmLoss",
    "PolicyGradientConfig",
    "algorithm_batch_from_trajectories",
    "clipped_policy_loss",
    "grpo_sampled_reverse_kl",
    "optional_masked_mean",
    "ppo_reference_kl",
    "require_matching_shapes",
    "require_regularization_inputs",
]

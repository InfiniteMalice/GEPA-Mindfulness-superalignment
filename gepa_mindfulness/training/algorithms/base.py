"""Shared records and helpers for portable policy-gradient objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gepa_mindfulness.training.runtime_config import AlgorithmConfig

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
        )


@dataclass(frozen=True)
class AlgorithmLoss:
    """Differentiable total loss and its scalar diagnostic components."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    kl: Tensor


def require_matching_shapes(reference: Tensor, *values: Tensor | None) -> None:
    """Reject tensor inputs that would otherwise broadcast silently."""
    reference_shape = getattr(reference, "shape", None)
    if reference_shape is None:
        return
    for value in values:
        if value is not None and getattr(value, "shape", None) != reference_shape:
            raise ValueError("algorithm tensors must have matching shapes")


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


def approximate_kl(
    ops: TensorOps,
    log_probs: Tensor,
    reference_log_probs: Tensor | None,
    mask: Tensor,
    zero: Tensor,
) -> Tensor:
    """Return the non-negative sampled reverse-KL estimator."""
    if reference_log_probs is None:
        return zero
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
    """Prevent configured regularization terms from silently becoming zero."""
    if config.kl_coef > 0.0 and reference_log_probs is None:
        raise ValueError("reference_log_probs are required when kl_coef is positive")
    if config.entropy_coef > 0.0 and entropy is None:
        raise ValueError("entropy is required when entropy_coef is positive")


__all__ = [
    "AlgorithmBatch",
    "AlgorithmLoss",
    "PolicyGradientConfig",
    "approximate_kl",
    "clipped_policy_loss",
    "optional_masked_mean",
    "require_matching_shapes",
    "require_regularization_inputs",
]

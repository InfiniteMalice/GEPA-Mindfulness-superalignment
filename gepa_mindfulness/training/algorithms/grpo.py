"""Backend-neutral group-relative advantages and GRPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from gepa_mindfulness.training.capability import Capability
from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation

from .base import (
    AlgorithmBatch,
    AlgorithmLoss,
    PolicyGradientConfig,
    approximate_kl,
    clipped_policy_loss,
    optional_masked_mean,
    require_matching_shapes,
    require_regularization_inputs,
)
from .ops import TensorOps


@dataclass(frozen=True)
class GRPOAlgorithmConfig(PolicyGradientConfig):
    """Coefficients used only by GRPO tensor mathematics."""

    @classmethod
    def from_runtime_config(cls, config: AlgorithmConfig) -> "GRPOAlgorithmConfig":
        """Adapt the canonical runtime section to objective-only settings."""
        if config.name != "grpo":
            raise ValueError("GRPO requires algorithm.name='grpo'")
        return cls(**cls._runtime_values(config))


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    zero_variance: Literal["zero", "error"] = "zero",
) -> list[float]:
    """Center and population-normalize rewards within one prompt group."""
    if not rewards:
        raise ValueError("group rewards must not be empty")
    numeric_rewards = [float(reward) for reward in rewards]
    if not all(math.isfinite(reward) for reward in numeric_rewards):
        raise ValueError("group rewards must be finite")
    if zero_variance not in {"zero", "error"}:
        raise ValueError("zero_variance must be 'zero' or 'error'")
    mean = math.fsum(numeric_rewards) / len(numeric_rewards)
    centered = [reward - mean for reward in numeric_rewards]
    variance = math.fsum(value * value for value in centered) / len(centered)
    if variance == 0.0:
        if zero_variance == "error":
            raise ValueError("group rewards have zero variance")
        return [0.0] * len(numeric_rewards)
    scale = math.sqrt(variance)
    return [value / scale for value in centered]


def compute_grpo_loss(
    ops: TensorOps,
    batch: AlgorithmBatch,
    evaluation: PolicyEvaluation,
    config: GRPOAlgorithmConfig,
) -> AlgorithmLoss:
    """Compute the clipped GRPO objective and scalar diagnostics."""
    require_regularization_inputs(
        config,
        reference_log_probs=evaluation.reference_log_probs,
        entropy=evaluation.entropy,
    )
    require_matching_shapes(
        batch.old_log_probs,
        evaluation.log_probs,
        evaluation.reference_log_probs,
        evaluation.entropy,
    )
    if batch.returns is not None or batch.old_values is not None:
        raise ValueError("GRPO batches must not contain value targets")
    policy_loss = clipped_policy_loss(
        ops,
        batch,
        evaluation.log_probs,
        config.clip_range,
    )
    zero = policy_loss * 0.0
    entropy = optional_masked_mean(ops, evaluation.entropy, batch.mask, zero)
    kl = approximate_kl(
        ops,
        evaluation.log_probs,
        evaluation.reference_log_probs,
        batch.mask,
        zero,
    )
    total_loss = policy_loss + config.kl_coef * kl - config.entropy_coef * entropy
    return AlgorithmLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=zero,
        entropy=entropy,
        kl=kl,
    )


@dataclass(frozen=True)
class GRPOAlgorithm:
    """Configured GRPO objective suitable for injection into a training engine."""

    ops: TensorOps
    config: GRPOAlgorithmConfig
    group_size: int

    @classmethod
    def from_runtime_config(
        cls,
        ops: TensorOps,
        config: AlgorithmConfig,
    ) -> "GRPOAlgorithm":
        return cls(
            ops=ops,
            config=GRPOAlgorithmConfig.from_runtime_config(config),
            group_size=config.group_size,
        )

    def required_capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.SUPPORTS_TOKEN_LOG_PROBS,
                Capability.SUPPORTS_REFERENCE_LOG_PROBS,
                Capability.SUPPORTS_BACKWARD,
            }
        )

    def compute_loss(
        self,
        batch: AlgorithmBatch,
        evaluation: PolicyEvaluation,
    ) -> AlgorithmLoss:
        return compute_grpo_loss(self.ops, batch, evaluation, self.config)


__all__ = [
    "GRPOAlgorithm",
    "GRPOAlgorithmConfig",
    "compute_grpo_loss",
    "compute_group_advantages",
]

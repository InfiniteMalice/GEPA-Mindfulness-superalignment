"""Backend-neutral group-relative advantages and GRPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from gepa_mindfulness.training.capability import Capability
from gepa_mindfulness.training.runtime_config import AlgorithmConfig, ZeroVariancePolicy
from gepa_mindfulness.training.trajectory import PolicyEvaluation, TrajectoryBatch

from .base import (
    AlgorithmBatch,
    AlgorithmLoss,
    PolicyGradientConfig,
    algorithm_batch_from_trajectories,
    clipped_policy_loss,
    grpo_sampled_reverse_kl,
    optional_masked_mean,
    require_matching_shapes,
    require_regularization_inputs,
)
from .ops import TensorOps


@dataclass(frozen=True)
class GRPOAlgorithmConfig(PolicyGradientConfig):
    """Coefficients used only by GRPO tensor mathematics."""

    group_normalization_epsilon: float = 1e-8
    zero_variance_policy: ZeroVariancePolicy = "zero"

    def __post_init__(self) -> None:
        super().__post_init__()
        epsilon = self.group_normalization_epsilon
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
            raise TypeError("group_normalization_epsilon must be a number")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("group_normalization_epsilon must be finite and positive")
        if self.zero_variance_policy not in {"zero", "center_only", "skip"}:
            raise ValueError("zero_variance_policy must be 'zero', 'center_only', or 'skip'")

    @classmethod
    def from_runtime_config(cls, config: AlgorithmConfig) -> "GRPOAlgorithmConfig":
        """Adapt the canonical runtime section to objective-only settings."""
        if config.name != "grpo":
            raise ValueError("GRPO requires algorithm.name='grpo'")
        return cls(
            **cls._runtime_values(config),
            group_normalization_epsilon=config.group_normalization_epsilon,
            zero_variance_policy=config.zero_variance_policy,
        )


def compute_group_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-8,
    zero_variance_policy: ZeroVariancePolicy = "zero",
) -> list[float] | None:
    """Normalize a prompt group, returning ``None`` when the group must be skipped."""
    if not rewards:
        raise ValueError("group rewards must not be empty")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a number")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    numeric_rewards = [float(reward) for reward in rewards]
    if not all(math.isfinite(reward) for reward in numeric_rewards):
        raise ValueError("group rewards must be finite")
    if zero_variance_policy not in {"zero", "center_only", "skip"}:
        raise ValueError("zero_variance_policy must be 'zero', 'center_only', or 'skip'")
    mean = math.fsum(numeric_rewards) / len(numeric_rewards)
    centered = [reward - mean for reward in numeric_rewards]
    variance = math.fsum(value * value for value in centered) / len(centered)
    if variance == 0.0:
        if zero_variance_policy == "skip":
            return None
        if zero_variance_policy == "center_only":
            return centered
        return [0.0 for _ in numeric_rewards]
    scale = math.sqrt(variance) + epsilon
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
        names=("old_log_probs", "log_probs", "reference_log_probs", "entropy"),
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
    reference_log_probs = evaluation.reference_log_probs
    kl = grpo_sampled_reverse_kl(
        ops,
        evaluation.log_probs,
        reference_log_probs,
        batch.mask,
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
                Capability.SUPPORTS_GENERATION,
                Capability.SUPPORTS_OPTIMIZER_STEP,
            }
        )

    def compute_group_advantages(self, rewards: Sequence[float]) -> list[float] | None:
        """Normalize one group using the canonical runtime policy and epsilon."""
        return compute_group_advantages(
            rewards,
            epsilon=self.config.group_normalization_epsilon,
            zero_variance_policy=self.config.zero_variance_policy,
        )

    def compute_loss(
        self,
        batch: TrajectoryBatch,
        evaluation: PolicyEvaluation,
    ) -> AlgorithmLoss:
        algorithm_batch = algorithm_batch_from_trajectories(
            self.ops,
            batch,
            evaluation,
            require_value_targets=False,
        )
        return compute_grpo_loss(self.ops, algorithm_batch, evaluation, self.config)


__all__ = [
    "GRPOAlgorithm",
    "GRPOAlgorithmConfig",
    "ZeroVariancePolicy",
    "compute_grpo_loss",
    "compute_group_advantages",
]

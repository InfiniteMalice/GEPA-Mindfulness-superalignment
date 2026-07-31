"""Backend-neutral generalized advantage estimation and PPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from .ops import Tensor, TensorOps


@dataclass(frozen=True)
class PPOAlgorithmConfig(PolicyGradientConfig):
    """Coefficients used only by PPO tensor mathematics."""

    value_coef: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.value_coef, (int, float)) or isinstance(self.value_coef, bool):
            raise TypeError("value_coef must be a number")
        if not math.isfinite(self.value_coef):
            raise ValueError("value_coef must be finite")
        if self.value_coef < 0.0:
            raise ValueError("value_coef must be non-negative")

    @classmethod
    def from_runtime_config(cls, config: AlgorithmConfig) -> "PPOAlgorithmConfig":
        """Adapt the canonical runtime section to objective-only settings."""
        if config.name != "ppo":
            raise ValueError("PPO requires algorithm.name='ppo'")
        return cls(**cls._runtime_values(config), value_coef=config.value_coef)


def compute_gae(
    ops: TensorOps,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    continuation: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute generalized advantages with terminal and padding resets."""
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    if len(rewards) == 0:
        raise ValueError("GAE requires at least one reward")
    require_matching_shapes(rewards, values, next_values, continuation, mask)

    running = rewards[-1] * 0.0
    reversed_advantages: list[Tensor] = []
    for index in range(len(rewards) - 1, -1, -1):
        valid = 1.0 if mask is None else mask[index]
        delta = (
            rewards[index] + gamma * next_values[index] * continuation[index] - values[index]
        ) * valid
        running = delta + gamma * gae_lambda * continuation[index] * valid * running
        reversed_advantages.append(running)
    return ops.stack(list(reversed(reversed_advantages)))


def compute_ppo_loss(
    ops: TensorOps,
    batch: AlgorithmBatch,
    evaluation: PolicyEvaluation,
    config: PPOAlgorithmConfig,
) -> AlgorithmLoss:
    """Compute the clipped PPO objective and scalar diagnostics."""
    require_regularization_inputs(
        config,
        reference_log_probs=evaluation.reference_log_probs,
        entropy=evaluation.entropy,
    )
    require_matching_shapes(
        batch.old_log_probs,
        evaluation.log_probs,
        evaluation.reference_log_probs,
        evaluation.value_predictions,
        evaluation.entropy,
    )
    policy_loss = clipped_policy_loss(
        ops,
        batch,
        evaluation.log_probs,
        config.clip_range,
    )
    zero = policy_loss * 0.0
    value_inputs = (batch.returns, batch.old_values, evaluation.value_predictions)
    if all(value is None for value in value_inputs) or (
        config.value_coef == 0.0 and not all(value is not None for value in value_inputs)
    ):
        value_loss = zero
    elif any(value is None for value in value_inputs):
        raise ValueError("returns, old_values, and value_predictions must be provided together")
    else:
        value_delta = evaluation.value_predictions - batch.old_values
        clipped_values = batch.old_values + ops.clip(
            value_delta,
            -config.clip_range,
            config.clip_range,
        )
        value_error = ops.square(evaluation.value_predictions - batch.returns)
        clipped_error = ops.square(clipped_values - batch.returns)
        value_loss = 0.5 * ops.masked_mean(
            ops.maximum(value_error, clipped_error),
            batch.mask,
        )
    entropy = optional_masked_mean(ops, evaluation.entropy, batch.mask, zero)
    kl = approximate_kl(
        ops,
        evaluation.log_probs,
        evaluation.reference_log_probs,
        batch.mask,
        zero,
    )
    total_loss = (
        policy_loss
        + config.value_coef * value_loss
        + config.kl_coef * kl
        - config.entropy_coef * entropy
    )
    return AlgorithmLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        kl=kl,
    )


@dataclass(frozen=True)
class PPOAlgorithm:
    """Configured PPO objective suitable for injection into a training engine."""

    ops: TensorOps
    config: PPOAlgorithmConfig

    @classmethod
    def from_runtime_config(
        cls,
        ops: TensorOps,
        config: AlgorithmConfig,
    ) -> "PPOAlgorithm":
        return cls(ops=ops, config=PPOAlgorithmConfig.from_runtime_config(config))

    def required_capabilities(self) -> frozenset[Capability]:
        capabilities = {
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_VALUE_HEAD,
        }
        if self.config.kl_coef > 0.0:
            capabilities.add(Capability.SUPPORTS_REFERENCE_LOG_PROBS)
        return frozenset(capabilities)

    def compute_loss(
        self,
        batch: AlgorithmBatch,
        evaluation: PolicyEvaluation,
    ) -> AlgorithmLoss:
        return compute_ppo_loss(self.ops, batch, evaluation, self.config)


__all__ = ["PPOAlgorithm", "PPOAlgorithmConfig", "compute_gae", "compute_ppo_loss"]

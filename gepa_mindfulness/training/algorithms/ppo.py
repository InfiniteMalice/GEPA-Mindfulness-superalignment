"""Backend-neutral generalized advantage estimation and PPO objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gepa_mindfulness.training.capability import Capability
from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation, TrajectoryBatch

from .base import (
    AlgorithmBatch,
    AlgorithmLoss,
    PolicyGradientConfig,
    algorithm_batch_from_trajectories,
    clipped_policy_loss,
    optional_masked_mean,
    ppo_reference_kl,
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
    """Compute generalized advantages over tokens independently for each batch row."""
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    shape = getattr(rewards, "shape", None)
    if shape is None or len(shape) not in {1, 2}:
        raise ValueError("GAE tensors must have shape [tokens] or [batch, tokens]")
    if any(dimension == 0 for dimension in shape):
        raise ValueError("GAE requires at least one reward")
    require_matching_shapes(
        rewards,
        values,
        next_values,
        continuation,
        mask,
        names=("rewards", "values", "next_values", "continuation", "mask"),
    )

    batched = len(shape) == 2
    row_count = shape[0] if batched else 1
    batch_advantages: list[Tensor] = []
    for row_index in range(row_count):
        row_rewards = rewards[row_index] if batched else rewards
        row_values = values[row_index] if batched else values
        row_next_values = next_values[row_index] if batched else next_values
        row_continuation = continuation[row_index] if batched else continuation
        row_mask = mask[row_index] if batched and mask is not None else mask
        running = row_rewards[-1] * 0.0
        reversed_advantages: list[Tensor] = []
        for token_index in range(len(row_rewards) - 1, -1, -1):
            valid = 1.0 if row_mask is None else row_mask[token_index]
            delta = (
                row_rewards[token_index]
                + gamma * row_next_values[token_index] * row_continuation[token_index]
                - row_values[token_index]
            ) * valid
            running = delta + gamma * gae_lambda * row_continuation[token_index] * valid * running
            reversed_advantages.append(running)
        batch_advantages.append(ops.stack(list(reversed(reversed_advantages))))
    if not batched:
        return batch_advantages[0]
    return ops.stack(batch_advantages)


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
    if evaluation.value_predictions is None:
        raise ValueError("value_predictions are required for PPO")
    if batch.returns is None or batch.old_values is None:
        raise ValueError("returns and old_values are required for PPO")
    reference_log_probs = evaluation.reference_log_probs
    value_predictions = evaluation.value_predictions
    require_matching_shapes(
        batch.old_log_probs,
        evaluation.log_probs,
        reference_log_probs,
        value_predictions,
        evaluation.entropy,
        names=(
            "old_log_probs",
            "log_probs",
            "reference_log_probs",
            "value_predictions",
            "entropy",
        ),
    )
    policy_loss = clipped_policy_loss(
        ops,
        batch,
        evaluation.log_probs,
        config.clip_range,
    )
    zero = policy_loss * 0.0
    value_delta = value_predictions - batch.old_values
    clipped_values = batch.old_values + ops.clip(
        value_delta,
        -config.clip_range,
        config.clip_range,
    )
    value_error = ops.square(value_predictions - batch.returns)
    clipped_error = ops.square(clipped_values - batch.returns)
    value_loss = 0.5 * ops.masked_mean(
        ops.maximum(value_error, clipped_error),
        batch.mask,
    )
    entropy = optional_masked_mean(ops, evaluation.entropy, batch.mask, zero)
    kl = ppo_reference_kl(
        ops,
        evaluation.log_probs,
        reference_log_probs,
        batch.mask,
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
        return frozenset(
            {
                Capability.SUPPORTS_GENERATION,
                Capability.SUPPORTS_TOKEN_LOG_PROBS,
                Capability.SUPPORTS_REFERENCE_LOG_PROBS,
                Capability.SUPPORTS_BACKWARD,
                Capability.SUPPORTS_OPTIMIZER_STEP,
                Capability.SUPPORTS_VALUE_HEAD,
            }
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
            require_value_targets=True,
        )
        return compute_ppo_loss(self.ops, algorithm_batch, evaluation, self.config)


__all__ = ["PPOAlgorithm", "PPOAlgorithmConfig", "compute_gae", "compute_ppo_loss"]

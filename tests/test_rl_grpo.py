"""Hand-computed tests for backend-neutral GRPO tensor mathematics."""

from __future__ import annotations

import math

import pytest
import torch

from gepa_mindfulness.training.algorithms import (
    AlgorithmBatch,
    GRPOAlgorithm,
    GRPOAlgorithmConfig,
    compute_group_advantages,
    compute_grpo_loss,
)
from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation


class _TorchOps:
    exp = staticmethod(torch.exp)
    clip = staticmethod(torch.clamp)
    minimum = staticmethod(torch.minimum)
    maximum = staticmethod(torch.maximum)
    square = staticmethod(torch.square)
    stack = staticmethod(torch.stack)

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = value.masked_select(mask.to(dtype=torch.bool))
        if selected.numel() == 0:
            raise ValueError("mask must select at least one value")
        return selected.mean()


OPS = _TorchOps()


def test_group_advantages_use_population_standard_deviation() -> None:
    advantages = compute_group_advantages([1.0, 2.0, 3.0])

    expected_edge = math.sqrt(1.5)
    assert advantages == pytest.approx([-expected_edge, 0.0, expected_edge])


def test_identical_group_rewards_produce_zero_advantages() -> None:
    advantages = compute_group_advantages([2.0, 2.0], zero_variance="zero")

    assert advantages == pytest.approx([0.0, 0.0])


def test_identical_group_rewards_can_raise_explicitly() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        compute_group_advantages([2.0, 2.0], zero_variance="error")


@pytest.mark.parametrize("rewards", [[], [1.0, float("nan")], [1.0, float("inf")]])
def test_group_advantages_reject_empty_or_non_finite_rewards(rewards: list[float]) -> None:
    with pytest.raises(ValueError):
        compute_group_advantages(rewards)


def test_grpo_clipping_and_kl_match_manual_values() -> None:
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        mask=torch.tensor([True, True]),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.log(torch.tensor([1.5, 0.5])),
        reference_log_probs=torch.log(torch.tensor([1.5, 0.5])),
    )

    result = compute_grpo_loss(
        OPS,
        batch,
        evaluation,
        GRPOAlgorithmConfig(clip_range=0.2, kl_coef=0.5),
    )

    # Policy is -.2. KL estimator uses ref-current, which is zero for both tokens.
    assert result.policy_loss.item() == pytest.approx(-0.2)
    assert result.kl.item() == pytest.approx(0.0)
    assert result.total_loss.item() == pytest.approx(-0.2)


def test_grpo_mask_excludes_padding_and_entropy_is_a_diagnostic() -> None:
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, 1000.0]),
        mask=torch.tensor([True, False]),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.tensor([0.0, 100.0]),
        reference_log_probs=torch.tensor([0.0, 100.0]),
        entropy=torch.tensor([3.0, 1000.0]),
    )

    result = compute_grpo_loss(OPS, batch, evaluation, GRPOAlgorithmConfig(entropy_coef=0.25))

    assert result.policy_loss.item() == pytest.approx(-1.0)
    assert result.entropy.item() == pytest.approx(3.0)
    assert result.total_loss.item() == pytest.approx(-1.75)


def test_grpo_loss_preserves_autograd() -> None:
    log_probs = torch.tensor([0.1, -0.1], requires_grad=True)
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        mask=torch.ones(2, dtype=torch.bool),
    )

    result = compute_grpo_loss(
        OPS,
        batch,
        PolicyEvaluation(log_probs=log_probs),
        GRPOAlgorithmConfig(),
    )
    result.total_loss.backward()

    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()


def test_grpo_requires_reference_log_probs_when_kl_is_enabled() -> None:
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        mask=torch.ones(2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="reference_log_probs"):
        compute_grpo_loss(
            OPS,
            batch,
            PolicyEvaluation(log_probs=torch.zeros(2)),
            GRPOAlgorithmConfig(kl_coef=0.1),
        )


def test_grpo_requires_entropy_when_entropy_bonus_is_enabled() -> None:
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        mask=torch.ones(2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="entropy"):
        compute_grpo_loss(
            OPS,
            batch,
            PolicyEvaluation(log_probs=torch.zeros(2)),
            GRPOAlgorithmConfig(entropy_coef=0.1),
        )


def test_grpo_algorithm_adapts_canonical_runtime_config() -> None:
    runtime = AlgorithmConfig(name="grpo", group_size=4, clip_range=0.1, kl_coef=0.2)
    algorithm = GRPOAlgorithm.from_runtime_config(OPS, runtime)
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        mask=torch.ones(2, dtype=torch.bool),
    )

    log_probs = torch.log(torch.tensor([2.0, 2.0]))
    result = algorithm.compute_loss(
        batch,
        PolicyEvaluation(log_probs=log_probs, reference_log_probs=log_probs),
    )

    assert algorithm.group_size == 4
    assert result.policy_loss.item() == pytest.approx(-1.1)

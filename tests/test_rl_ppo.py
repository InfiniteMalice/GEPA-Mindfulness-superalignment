"""Hand-computed tests for backend-neutral PPO tensor mathematics."""

from __future__ import annotations

import pytest
import torch

from gepa_mindfulness.training.algorithms import (
    AlgorithmBatch,
    PPOAlgorithm,
    PPOAlgorithmConfig,
    compute_gae,
    compute_ppo_loss,
)
from gepa_mindfulness.training.capability import Capability
from gepa_mindfulness.training.contracts import RLAlgorithm
from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation, Trajectory, TrajectoryBatch


class _TorchOps:
    """Minimal real tensor implementation of the algorithm operations protocol."""

    exp = staticmethod(torch.exp)
    clip = staticmethod(torch.clamp)
    minimum = staticmethod(torch.minimum)
    maximum = staticmethod(torch.maximum)
    square = staticmethod(torch.square)
    stack = staticmethod(torch.stack)

    @staticmethod
    def from_data(
        values: object,
        *,
        like: torch.Tensor,
        kind: str = "float",
    ) -> torch.Tensor:
        dtype = torch.bool if kind == "bool" else like.dtype
        return torch.as_tensor(values, dtype=dtype, device=like.device)

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = value.masked_select(mask.to(dtype=torch.bool))
        if selected.numel() == 0:
            raise ValueError("mask must select at least one value")
        return selected.mean()


OPS = _TorchOps()


def _batch(**overrides: object) -> AlgorithmBatch:
    values: dict[str, object] = {
        "old_log_probs": torch.tensor([0.0, 0.0]),
        "advantages": torch.tensor([1.0, -1.0]),
        "mask": torch.tensor([True, True]),
        "returns": torch.tensor([1.0, 1.0]),
        "old_values": torch.tensor([0.0, 0.0]),
    }
    values.update(overrides)
    return AlgorithmBatch(**values)


def test_ppo_clipped_surrogate_matches_manual_values() -> None:
    batch = _batch(advantages=torch.tensor([1.0, -1.0]))
    log_probs = torch.log(torch.tensor([1.5, 0.5]))
    evaluation = PolicyEvaluation(
        log_probs=log_probs,
        reference_log_probs=log_probs,
        value_predictions=torch.zeros(2),
    )

    result = compute_ppo_loss(
        OPS,
        batch,
        evaluation,
        PPOAlgorithmConfig(clip_range=0.2),
    )

    # min(1.5, 1.2) * 1 = 1.2; min(-0.5, -0.8) = -0.8; mean = 0.2.
    assert result.policy_loss.item() == pytest.approx(-0.2)


def test_ppo_value_clipping_uses_larger_squared_error() -> None:
    batch = _batch(
        advantages=torch.zeros(2),
        returns=torch.tensor([2.0, -2.0]),
        old_values=torch.tensor([0.0, 0.0]),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.zeros(2),
        reference_log_probs=torch.zeros(2),
        value_predictions=torch.tensor([1.0, -1.0]),
    )

    result = compute_ppo_loss(
        OPS,
        batch,
        evaluation,
        PPOAlgorithmConfig(clip_range=0.2, value_coef=1.0),
    )

    # Clipped predictions are +/-0.2, so each squared error is 1.8**2; PPO halves it.
    assert result.value_loss.item() == pytest.approx(1.62)
    assert result.total_loss.item() == pytest.approx(1.62)


def test_ppo_entropy_and_kl_contribute_with_configured_signs() -> None:
    batch = _batch(advantages=torch.zeros(2))
    evaluation = PolicyEvaluation(
        log_probs=torch.tensor([0.0, 0.0]),
        reference_log_probs=torch.tensor([0.0, 1.0]),
        value_predictions=torch.zeros(2),
        entropy=torch.tensor([2.0, 4.0]),
    )

    result = compute_ppo_loss(
        OPS,
        batch,
        evaluation,
        PPOAlgorithmConfig(kl_coef=2.0, entropy_coef=0.5),
    )

    # The approved PPO diagnostic is mean(current - reference) = mean(0, -1).
    expected_kl = -0.5
    assert result.entropy.item() == pytest.approx(3.0)
    assert result.kl.item() == pytest.approx(expected_kl)
    assert result.total_loss.item() == pytest.approx(2.0 * expected_kl - 1.5)


def test_ppo_mask_excludes_padding_from_every_loss_component() -> None:
    batch = _batch(
        advantages=torch.tensor([1.0, 10_000.0]),
        mask=torch.tensor([True, False]),
        returns=torch.tensor([1.0, 10_000.0]),
        old_values=torch.zeros(2),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.tensor([0.0, 100.0]),
        reference_log_probs=torch.tensor([0.0, 100.0]),
        value_predictions=torch.tensor([0.0, 10_000.0]),
        entropy=torch.tensor([2.0, 10_000.0]),
    )

    result = compute_ppo_loss(OPS, batch, evaluation, PPOAlgorithmConfig(value_coef=1.0))

    assert result.policy_loss.item() == pytest.approx(-1.0)
    assert result.value_loss.item() == pytest.approx(0.5)
    assert result.entropy.item() == pytest.approx(2.0)
    assert result.kl.item() == pytest.approx(0.0)


def test_ppo_rejects_mask_that_selects_no_tokens() -> None:
    batch = _batch(mask=torch.tensor([False, False]))
    evaluation = PolicyEvaluation(
        log_probs=torch.zeros(2),
        reference_log_probs=torch.zeros(2),
        value_predictions=torch.zeros(2),
    )

    with pytest.raises(ValueError, match="mask must select"):
        compute_ppo_loss(OPS, batch, evaluation, PPOAlgorithmConfig())


def test_algorithm_batch_rejects_mismatched_tensor_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        AlgorithmBatch(
            old_log_probs=torch.zeros(2),
            advantages=torch.zeros(3),
            mask=torch.ones(2, dtype=torch.bool),
        )


def test_ppo_rejects_evaluation_shape_mismatch() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="matching shapes"):
        compute_ppo_loss(
            OPS,
            batch,
            PolicyEvaluation(
                log_probs=torch.zeros(3),
                reference_log_probs=torch.zeros(3),
                value_predictions=torch.zeros(3),
            ),
            PPOAlgorithmConfig(),
        )


def test_ppo_requires_reference_log_probs_when_kl_is_enabled() -> None:
    batch = _batch(returns=None, old_values=None)

    with pytest.raises(ValueError, match="reference_log_probs"):
        compute_ppo_loss(
            OPS,
            batch,
            PolicyEvaluation(log_probs=torch.zeros(2)),
            PPOAlgorithmConfig(kl_coef=0.1),
        )


def test_ppo_requires_entropy_when_entropy_bonus_is_enabled() -> None:
    batch = _batch()

    with pytest.raises(ValueError, match="entropy"):
        compute_ppo_loss(
            OPS,
            batch,
            PolicyEvaluation(
                log_probs=torch.zeros(2),
                reference_log_probs=torch.zeros(2),
                value_predictions=torch.zeros(2),
            ),
            PPOAlgorithmConfig(entropy_coef=0.1),
        )


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        (
            PolicyEvaluation(log_probs=torch.zeros(2), value_predictions=torch.zeros(2)),
            "reference_log_probs",
        ),
        (
            PolicyEvaluation(log_probs=torch.zeros(2), reference_log_probs=torch.zeros(2)),
            "value_predictions",
        ),
    ],
)
def test_ppo_requires_design_evaluation_tensors_when_coefficients_are_zero(
    evaluation: PolicyEvaluation,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_ppo_loss(OPS, _batch(), evaluation, PPOAlgorithmConfig())


def test_compute_gae_matches_hand_calculated_terminal_rollout() -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.25])
    next_values = torch.tensor([0.25, 0.0])
    not_terminal = torch.tensor([1.0, 0.0])

    advantages = compute_gae(
        OPS,
        rewards,
        values,
        next_values,
        not_terminal,
        gamma=0.9,
        gae_lambda=0.8,
    )

    # delta = [0.725, 0.75]; A1 = .75; A0 = .725 + .9*.8*.75 = 1.265.
    assert advantages.tolist() == pytest.approx([1.265, 0.75])


def test_compute_gae_mask_resets_padding_and_recurrence() -> None:
    advantages = compute_gae(
        OPS,
        torch.tensor([1.0, 100.0, 1.0]),
        torch.zeros(3),
        torch.zeros(3),
        torch.ones(3),
        gamma=1.0,
        gae_lambda=1.0,
        mask=torch.tensor([True, False, True]),
    )

    assert advantages.tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_compute_gae_recurs_over_tokens_independently_for_each_batch_row() -> None:
    advantages = compute_gae(
        OPS,
        torch.tensor([[1.0, 2.0, 100.0], [4.0, 50.0, 6.0]]),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        gamma=1.0,
        gae_lambda=1.0,
        mask=torch.tensor([[True, True, False], [True, False, True]]),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[3.0, 2.0, 0.0], [4.0, 0.0, 6.0]]),
    )


def test_ppo_loss_is_finite_and_preserves_dtype_and_autograd() -> None:
    log_probs = torch.tensor([80.0, -80.0], dtype=torch.float64, requires_grad=True)
    batch = _batch(
        old_log_probs=torch.tensor([79.0, -79.0], dtype=torch.float64),
        advantages=torch.tensor([1.0, -1.0], dtype=torch.float64),
        returns=torch.zeros(2, dtype=torch.float64),
        old_values=torch.zeros(2, dtype=torch.float64),
    )
    evaluation = PolicyEvaluation(
        log_probs=log_probs,
        reference_log_probs=log_probs.detach(),
        value_predictions=torch.zeros(2, dtype=torch.float64),
    )

    result = compute_ppo_loss(OPS, batch, evaluation, PPOAlgorithmConfig())
    result.total_loss.backward()

    assert result.total_loss.dtype == torch.float64
    assert torch.isfinite(result.total_loss)
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()


def test_ppo_algorithm_adapts_canonical_runtime_config() -> None:
    runtime = AlgorithmConfig(
        name="ppo",
        clip_range=0.1,
        value_coef=0.25,
        kl_coef=0.3,
    )
    algorithm = PPOAlgorithm.from_runtime_config(OPS, runtime)
    trajectory = Trajectory(
        trajectory_id="ppo-runtime",
        case_id="case-1",
        prompt="prompt",
        response="response",
        old_log_probs=(0.0, 0.0),
        value_predictions=(0.0, 0.0),
        advantage=(1.0, 1.0),
        returns=(0.0, 0.0),
    )
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, True),),
    )
    log_probs = torch.log(torch.tensor([2.0, 2.0]))
    evaluation = PolicyEvaluation(
        log_probs=log_probs.unsqueeze(0),
        reference_log_probs=log_probs.unsqueeze(0),
        value_predictions=torch.zeros((1, 2)),
    )

    result = algorithm.compute_loss(batch, evaluation)

    assert result.policy_loss.item() == pytest.approx(-1.1)
    assert Capability.SUPPORTS_REFERENCE_LOG_PROBS in algorithm.required_capabilities()


def test_ppo_required_capabilities_match_approved_design_exactly() -> None:
    algorithm = PPOAlgorithm(OPS, PPOAlgorithmConfig(kl_coef=0.0, value_coef=0.0))

    assert algorithm.required_capabilities() == frozenset(
        {
            Capability.SUPPORTS_GENERATION,
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            Capability.SUPPORTS_REFERENCE_LOG_PROBS,
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_OPTIMIZER_STEP,
            Capability.SUPPORTS_VALUE_HEAD,
        }
    )


def test_ppo_algorithm_is_usable_through_trajectory_batch_protocol() -> None:
    trajectory = Trajectory(
        trajectory_id="ppo-protocol",
        case_id="case-1",
        prompt="prompt",
        response="response",
        old_log_probs=(0.0, 0.0),
        value_predictions=(0.0, 0.0),
        advantage=(1.0, 1.0),
        returns=(1.0, 1.0),
    )
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, True),),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.zeros((1, 2)),
        reference_log_probs=torch.zeros((1, 2)),
        value_predictions=torch.zeros((1, 2)),
    )
    algorithm: RLAlgorithm = PPOAlgorithm(OPS, PPOAlgorithmConfig())

    result = algorithm.compute_loss(batch, evaluation)

    assert isinstance(algorithm, RLAlgorithm)
    assert result.policy_loss.item() == pytest.approx(-1.0)


@pytest.mark.parametrize("value_coef", [float("nan"), float("inf")])
def test_ppo_config_rejects_non_finite_value_coefficient(value_coef: float) -> None:
    with pytest.raises(ValueError, match="value_coef must be finite"):
        PPOAlgorithmConfig(value_coef=value_coef)

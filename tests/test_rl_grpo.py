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
from gepa_mindfulness.training.capability import Capability
from gepa_mindfulness.training.contracts import RLAlgorithm
from gepa_mindfulness.training.runtime_config import AlgorithmConfig
from gepa_mindfulness.training.trajectory import PolicyEvaluation, Trajectory, TrajectoryBatch


class _TorchOps:
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


class _TensorWithoutShape:
    """Tensor-like test value missing the shape contract required by algorithms."""


class _TensorWithRaisingShapeProperty:
    """Tensor-like test value whose backend fails while resolving its shape."""

    @property
    def shape(self) -> object:
        raise RuntimeError("shape property failed")


class _RaisingShapeIterator:
    def __iter__(self) -> object:
        raise RuntimeError("shape iteration failed")


class _TensorWithRaisingShapeIterator:
    """Tensor-like test value whose backend fails while iterating its shape."""

    shape = _RaisingShapeIterator()


def test_group_advantages_use_population_standard_deviation() -> None:
    advantages = compute_group_advantages([1.0, 2.0, 3.0])

    expected_edge = math.sqrt(1.5)
    assert advantages == pytest.approx([-expected_edge, 0.0, expected_edge])


def test_algorithm_batch_rejects_tensor_like_inputs_without_shapes() -> None:
    """Missing shape metadata must not disable broadcast-safety validation."""
    with pytest.raises(ValueError, match="old_log_probs.*valid shape"):
        AlgorithmBatch(
            old_log_probs=_TensorWithoutShape(),
            advantages=_TensorWithoutShape(),
            mask=_TensorWithoutShape(),
        )


def test_algorithm_batch_rejects_mismatching_shapes_with_field_name() -> None:
    """A shape mismatch must identify the input that cannot align with the reference."""
    with pytest.raises(ValueError, match="advantages.*matching shape"):
        AlgorithmBatch(
            old_log_probs=torch.zeros(2),
            advantages=torch.zeros(3),
            mask=torch.ones(2, dtype=torch.bool),
        )


def test_algorithm_batch_labels_shape_property_errors() -> None:
    """Backend shape-property failures must identify the invalid batch field."""
    with pytest.raises(ValueError, match="old_log_probs.*valid shape") as exc_info:
        AlgorithmBatch(
            old_log_probs=_TensorWithRaisingShapeProperty(),
            advantages=torch.zeros(2),
            mask=torch.ones(2, dtype=torch.bool),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "shape property failed"


def test_grpo_labels_evaluation_shape_iterator_errors() -> None:
    """Backend shape-iteration failures must identify the invalid evaluation field."""
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        mask=torch.ones(2, dtype=torch.bool),
    )
    evaluation = PolicyEvaluation(
        log_probs=_TensorWithRaisingShapeIterator(),
        reference_log_probs=torch.zeros(2),
    )

    with pytest.raises(ValueError, match="log_probs.*valid shape") as exc_info:
        compute_grpo_loss(OPS, batch, evaluation, GRPOAlgorithmConfig())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "shape iteration failed"


def test_identical_group_rewards_produce_zero_advantages() -> None:
    advantages = compute_group_advantages([2.0, 2.0], zero_variance_policy="zero")

    assert advantages == pytest.approx([0.0, 0.0])


def test_identical_group_rewards_center_without_normalizing() -> None:
    advantages = compute_group_advantages([2.0, 2.0], zero_variance_policy="center_only")

    assert advantages == pytest.approx([0.0, 0.0])


def test_identical_group_rewards_return_explicit_skip_representation() -> None:
    result = compute_group_advantages([2.0, 2.0], zero_variance_policy="skip")

    assert result is None


@pytest.mark.parametrize("zero_variance_policy", ["zero", "center_only", "skip"])
def test_near_zero_variance_uses_configured_epsilon_for_every_policy(
    zero_variance_policy: str,
) -> None:
    advantages = compute_group_advantages(
        [0.0, 2e-12],
        epsilon=1e-6,
        zero_variance_policy=zero_variance_policy,
    )

    assert advantages == pytest.approx([-9.99999e-7, 9.99999e-7], rel=1e-6)


@pytest.mark.parametrize("rewards", [[], [1.0, float("nan")], [1.0, float("inf")]])
def test_group_advantages_reject_empty_or_non_finite_rewards(rewards: list[float]) -> None:
    with pytest.raises(ValueError):
        compute_group_advantages(rewards)


def test_grpo_uses_sampled_reverse_kl_estimator() -> None:
    batch = AlgorithmBatch(
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        mask=torch.tensor([True, True]),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.zeros(2),
        reference_log_probs=torch.tensor([-1.0, 1.0]),
    )

    result = compute_grpo_loss(
        OPS,
        batch,
        evaluation,
        GRPOAlgorithmConfig(clip_range=0.2, kl_coef=0.5),
    )

    # exp(ref-current) - (ref-current) - 1 gives exp(-1) and exp(1) - 2.
    expected_kl = (math.exp(-1.0) + math.exp(1.0) - 2.0) / 2.0
    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.kl.item() == pytest.approx(expected_kl)
    assert result.total_loss.item() == pytest.approx(0.5 * expected_kl)


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
        PolicyEvaluation(log_probs=log_probs, reference_log_probs=log_probs.detach()),
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
            PolicyEvaluation(
                log_probs=torch.zeros(2),
                reference_log_probs=torch.zeros(2),
            ),
            GRPOAlgorithmConfig(entropy_coef=0.1),
        )


def test_grpo_requires_reference_log_probs_when_kl_coefficient_is_zero() -> None:
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
            GRPOAlgorithmConfig(kl_coef=0.0),
        )


def test_grpo_algorithm_adapts_canonical_runtime_config() -> None:
    runtime = AlgorithmConfig(
        name="grpo",
        group_size=4,
        kl_coef=0.2,
        clip_range=0.1,
        group_normalization_epsilon=1e-6,
        zero_variance_policy="center_only",
    )
    algorithm = GRPOAlgorithm.from_runtime_config(OPS, runtime)
    trajectory = Trajectory(
        trajectory_id="grpo-runtime",
        case_id="case-1",
        prompt="prompt",
        response="response",
        old_log_probs=(0.0, 0.0),
        advantage=(1.0, 1.0),
    )
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, True),),
    )

    log_probs = torch.log(torch.tensor([2.0, 2.0]))
    result = algorithm.compute_loss(
        batch,
        PolicyEvaluation(
            log_probs=log_probs.unsqueeze(0),
            reference_log_probs=log_probs.unsqueeze(0),
        ),
    )

    assert algorithm.group_size == 4
    assert algorithm.config.group_normalization_epsilon == pytest.approx(1e-6)
    assert algorithm.config.zero_variance_policy == "center_only"
    assert algorithm.compute_group_advantages([0.0, 2e-12]) == pytest.approx(
        [-9.99999e-7, 9.99999e-7],
        rel=1e-6,
    )
    assert result.policy_loss.item() == pytest.approx(-1.1)


def test_grpo_required_capabilities_match_approved_design_exactly() -> None:
    algorithm = GRPOAlgorithm(OPS, GRPOAlgorithmConfig(kl_coef=0.0), group_size=2)

    assert algorithm.required_capabilities() == frozenset(
        {
            Capability.SUPPORTS_GENERATION,
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            Capability.SUPPORTS_REFERENCE_LOG_PROBS,
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_OPTIMIZER_STEP,
        }
    )


def test_grpo_algorithm_is_usable_through_trajectory_batch_protocol() -> None:
    trajectory = Trajectory(
        trajectory_id="grpo-protocol",
        case_id="case-1",
        prompt="prompt",
        response="response",
        old_log_probs=(0.0, 0.0),
        advantage=(1.0, 1.0),
    )
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, True),),
    )
    evaluation = PolicyEvaluation(
        log_probs=torch.zeros((1, 2)),
        reference_log_probs=torch.zeros((1, 2)),
    )
    algorithm: RLAlgorithm = GRPOAlgorithm(OPS, GRPOAlgorithmConfig(), group_size=2)

    result = algorithm.compute_loss(batch, evaluation)

    assert isinstance(algorithm, RLAlgorithm)
    assert result.policy_loss.item() == pytest.approx(-1.0)

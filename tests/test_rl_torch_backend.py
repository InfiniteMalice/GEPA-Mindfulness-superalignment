"""Tests for the portable PyTorch policy backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from gepa_mindfulness.training.algorithms import (
    GRPOAlgorithm,
    GRPOAlgorithmConfig,
    PPOAlgorithm,
    PPOAlgorithmConfig,
)
from gepa_mindfulness.training.backends import (
    TorchPolicyBackend,
    TorchTensorOps,
    create_portable_backend,
)
from gepa_mindfulness.training.capability import Capability, CapabilityState
from gepa_mindfulness.training.runtime_config import RLRunConfig
from gepa_mindfulness.training.trajectory import RolloutRequest, Trajectory, TrajectoryBatch


class TinyTokenizer:
    """Entirely local whitespace tokenizer used by backend contract tests."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self._tokens = {
            "<pad>": 0,
            "<eos>": 1,
            "calm": 2,
            "breath": 3,
            "now": 4,
            "slowly": 5,
        }
        self._words = {token_id: token for token, token_id in self._tokens.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._tokens[word] for word in text.split()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ignored = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return " ".join(self._words[token_id] for token_id in token_ids if token_id not in ignored)


class TinyCausalLM(nn.Module):
    """Small causal model with an offline deterministic generation method."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6, _name_or_path="tiny-local")
        self.embedding = nn.Embedding(6, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, 6, bias=False)
        self.generate_calls: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **sampling_parameters: object,
    ) -> torch.Tensor:
        assert "generator" not in sampling_parameters
        self.generate_calls.append(
            {
                "attention_mask": attention_mask.detach().clone(),
                "max_new_tokens": max_new_tokens,
                **sampling_parameters,
            }
        )
        response = torch.full(
            (input_ids.shape[0], max_new_tokens),
            TinyTokenizer.eos_token_id + 2,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)


@pytest.fixture
def tiny_backend() -> TorchPolicyBackend:
    """Build a deterministic CPU backend without loading external assets."""
    torch.manual_seed(7)
    return TorchPolicyBackend(
        policy_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        device="cpu",
        learning_rate=0.05,
        max_new_tokens=2,
        model_identifier="tiny-local",
    )


def _trajectory(
    *,
    prompt_ids: tuple[int, ...] | None = (2, 3),
    response_ids: tuple[int, ...] | None = (4, 5),
) -> Trajectory:
    return Trajectory(
        trajectory_id="trajectory-1",
        case_id="case-1",
        prompt="calm breath",
        response="now slowly",
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
    )


def test_tensor_ops_preserve_float_dtype_device_and_nested_shape() -> None:
    """Algorithm data adopts the policy tensor's floating point representation."""
    like = torch.ones((1,), dtype=torch.float64)

    result = TorchTensorOps.from_data([[1.0, 2.0], [3.0, 4.0]], like=like)

    assert result.shape == (2, 2)
    assert result.dtype is torch.float64
    assert result.device == like.device


def test_tensor_ops_create_boolean_masks_and_reject_empty_selection() -> None:
    """Masks are boolean and cannot silently reduce an empty token selection."""
    like = torch.ones((2, 2), dtype=torch.float32)
    mask = TorchTensorOps.from_data([[True, False], [False, True]], like=like, kind="bool")

    assert mask.dtype is torch.bool
    assert TorchTensorOps.masked_mean(like, mask).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least one selected"):
        TorchTensorOps.masked_mean(like, torch.zeros_like(mask))


def test_generate_records_response_aligned_policy_evidence(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """Generated trajectories contain real response-only token evidence."""
    request = RolloutRequest(
        prompt="calm breath",
        case_id="case-7",
        num_samples=2,
        sampling_parameters={"max_new_tokens": 2, "temperature": 0.7},
        policy_version="policy-3",
        seed=11,
    )

    trajectories = tiny_backend.generate((request,))

    assert len(trajectories) == 2
    for sample_index, trajectory in enumerate(trajectories):
        assert trajectory.trajectory_id == f"case-7-policy-3-{sample_index}"
        assert trajectory.prompt_token_ids == (2, 3)
        assert trajectory.response_token_ids == (3, 3)
        assert trajectory.response == "breath breath"
        assert len(trajectory.old_log_probs or ()) == 2
        assert len(trajectory.reference_log_probs or ()) == 2
        assert len(trajectory.value_predictions or ()) == 2
        assert trajectory.backend_name == "torch_portable"
        assert trajectory.model_identifier == "tiny-local"
        assert trajectory.policy_version == "policy-3"
        assert trajectory.seed == 11

        batch = TrajectoryBatch(
            trajectories=(trajectory,),
            response_token_masks=((True, True),),
        )
        evaluation = tiny_backend.evaluate(batch)
        torch.testing.assert_close(
            evaluation.log_probs.detach(),
            torch.tensor([trajectory.old_log_probs]),
        )
        torch.testing.assert_close(
            evaluation.reference_log_probs.detach(),
            torch.tensor([trajectory.reference_log_probs]),
        )
        torch.testing.assert_close(
            evaluation.value_predictions.detach(),
            torch.tensor([trajectory.value_predictions]),
        )


def test_evaluate_selects_only_response_token_predictions(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """The first response token uses the final prompt position as its predictor."""
    trajectory = _trajectory()
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, False),),
    )

    evaluation = tiny_backend.evaluate(batch)

    input_ids = torch.tensor([[2, 3, 4, 5]])
    outputs = tiny_backend.policy_model(input_ids, output_hidden_states=True)
    all_selected = torch.log_softmax(outputs.logits[:, :-1], dim=-1).gather(
        -1,
        input_ids[:, 1:].unsqueeze(-1),
    )
    expected = all_selected.squeeze(-1)[:, 1:]
    torch.testing.assert_close(evaluation.log_probs, expected)
    assert evaluation.log_probs.shape == (1, 2)
    assert evaluation.reference_log_probs.shape == (1, 2)
    assert evaluation.value_predictions.shape == (1, 2)
    assert evaluation.entropy.shape == (1, 2)


def test_evaluate_pads_variable_responses_without_fabricating_masked_values(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """Variable responses are padded only where the public batch mask excludes tokens."""
    batch = TrajectoryBatch(
        trajectories=(
            _trajectory(response_ids=(4,)),
            _trajectory(response_ids=(4, 5)),
        ),
        response_token_masks=((True, False), (True, True)),
    )

    evaluation = tiny_backend.evaluate(batch)

    for tensor in (
        evaluation.log_probs,
        evaluation.reference_log_probs,
        evaluation.value_predictions,
        evaluation.entropy,
    ):
        assert tensor.shape == (2, 2)
        assert tensor[0, 1].item() == 0.0


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (TrajectoryBatch(trajectories=()), "at least one trajectory"),
        (TrajectoryBatch(trajectories=(_trajectory(),)), "response_token_masks"),
        (
            TrajectoryBatch(
                trajectories=(_trajectory(prompt_ids=None),),
                response_token_masks=((True, True),),
            ),
            "prompt_token_ids",
        ),
        (
            TrajectoryBatch(
                trajectories=(_trajectory(response_ids=None),),
                response_token_masks=((True, True),),
            ),
            "response_token_ids",
        ),
        (
            TrajectoryBatch(
                trajectories=(_trajectory(),),
                response_token_masks=((True,),),
            ),
            "mask",
        ),
        (
            TrajectoryBatch(
                trajectories=(_trajectory(),),
                response_token_masks=((False, False),),
            ),
            "selected response token",
        ),
    ],
)
def test_evaluate_fails_closed_for_incomplete_or_misaligned_trajectories(
    tiny_backend: TorchPolicyBackend,
    batch: TrajectoryBatch,
    message: str,
) -> None:
    """Training never substitutes fabricated token data for missing evidence."""
    with pytest.raises(ValueError, match=message):
        tiny_backend.evaluate(batch)


@pytest.mark.parametrize(
    "rollout",
    [
        RolloutRequest(prompt=""),
        RolloutRequest(prompt="calm", num_samples=0),
        RolloutRequest(prompt="calm", sampling_parameters={"max_new_tokens": 0}),
        RolloutRequest(prompt="calm", sampling_parameters={"temperature": 0.0}),
    ],
)
def test_generate_rejects_invalid_requests(
    tiny_backend: TorchPolicyBackend,
    rollout: RolloutRequest,
) -> None:
    """Malformed rollout requests fail before invoking model generation."""
    with pytest.raises((TypeError, ValueError)):
        tiny_backend.generate((rollout,))


def test_backward_zero_grad_and_optimizer_step_update_only_trainable_state(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """A differentiable backend loss changes policy state while its reference stays frozen."""
    trajectory = tiny_backend.generate((RolloutRequest(prompt="calm breath"),))[0]
    batch = TrajectoryBatch(
        trajectories=(trajectory,),
        response_token_masks=((True, True),),
    )
    policy_before = [parameter.detach().clone() for parameter in tiny_backend.policy_parameters()]
    reference_before = [
        parameter.detach().clone() for parameter in tiny_backend.reference_model.parameters()
    ]

    tiny_backend.zero_grad()
    evaluation = tiny_backend.evaluate(batch)
    loss = -(evaluation.log_probs + evaluation.value_predictions).mean()
    tiny_backend.backward(loss)
    assert any(parameter.grad is not None for parameter in tiny_backend.policy_parameters())
    result = tiny_backend.optimizer_step()

    policy_after = list(tiny_backend.policy_parameters())
    assert result.step == 1
    assert any(not torch.equal(old, new.detach()) for old, new in zip(policy_before, policy_after))
    assert all(
        torch.equal(old, new.detach())
        for old, new in zip(reference_before, tiny_backend.reference_model.parameters())
    )
    assert all(
        not parameter.requires_grad for parameter in tiny_backend.reference_model.parameters()
    )

    tiny_backend.zero_grad()
    assert all(parameter.grad is None for parameter in tiny_backend.policy_parameters())


def test_full_weight_capabilities_have_explicit_evidence(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """Capability discovery explicitly distinguishes supported and absent features."""
    report = tiny_backend.capabilities()

    assert set(report.capabilities) == set(Capability)
    assert all(evidence.evidence for evidence in report.capabilities.values())
    for supported in (
        Capability.SUPPORTS_GENERATION,
        Capability.SUPPORTS_TOKEN_LOG_PROBS,
        Capability.SUPPORTS_REFERENCE_LOG_PROBS,
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_VALUE_HEAD,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
    ):
        assert report.state(supported) is CapabilityState.SUPPORTED
    assert report.state(Capability.SUPPORTS_LORA_TRAINING) is CapabilityState.UNSUPPORTED
    assert report.state(Capability.SUPPORTS_CUDA) is CapabilityState.UNSUPPORTED


def test_capabilities_satisfy_exact_ppo_and_grpo_requirements(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """The backend substantiates every operation required by both portable objectives."""
    ppo = PPOAlgorithm(TorchTensorOps(), PPOAlgorithmConfig())
    grpo = GRPOAlgorithm(TorchTensorOps(), GRPOAlgorithmConfig(), group_size=2)

    report = tiny_backend.capabilities()
    report.require(ppo.required_capabilities())
    report.require(grpo.required_capabilities())


def test_factory_uses_injected_local_assets_and_runtime_config() -> None:
    """The portable factory can run entirely from already-loaded model assets."""
    config = RLRunConfig.from_mapping(
        {
            "runtime": {"backend": "pytorch", "device": "cpu"},
            "policy": {"model_name": "configured-tiny", "max_new_tokens": 1},
            "algorithm": {"learning_rate": 0.02},
        }
    )

    backend = create_portable_backend(
        config,
        policy_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
    )
    trajectory = backend.generate((RolloutRequest(prompt="calm"),))[0]

    assert trajectory.response_token_ids == (3,)
    assert trajectory.model_identifier == "configured-tiny"


def test_factory_reports_actionable_error_when_peft_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LoRA selection names its optional dependency instead of silently using full weights."""
    monkeypatch.setitem(sys.modules, "peft", None)

    with pytest.raises(RuntimeError, match="PEFT.*peft"):
        create_portable_backend(
            RLRunConfig(),
            policy_model=TinyCausalLM(),
            tokenizer=TinyTokenizer(),
            training_mode="lora",
        )


def test_factory_supports_lora_when_peft_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed PEFT adapter yields LoRA-only trainability and capability evidence."""
    fake_peft = ModuleType("peft")

    class FakeLoraConfig:
        def __init__(self, **values: object) -> None:
            self.values = values

    def fake_get_peft_model(model: nn.Module, config: FakeLoraConfig) -> nn.Module:
        assert config.values["task_type"] == "CAUSAL_LM"
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.lm_head.weight.requires_grad_(True)
        return model

    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    backend = create_portable_backend(
        RLRunConfig(),
        policy_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        training_mode="lora",
        lora_config={"r": 2, "target_modules": ["lm_head"]},
    )

    assert (
        backend.capabilities().state(Capability.SUPPORTS_LORA_TRAINING) is CapabilityState.SUPPORTED
    )
    assert (
        backend.capabilities().state(Capability.SUPPORTS_FULL_WEIGHT_TRAINING)
        is CapabilityState.UNSUPPORTED
    )
    trainable = [
        name
        for name, parameter in backend.policy_model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == ["lm_head.weight"]

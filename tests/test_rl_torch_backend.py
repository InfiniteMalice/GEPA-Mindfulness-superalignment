"""Tests for the portable PyTorch policy backend."""

from __future__ import annotations

import sys
from copy import deepcopy
from enum import Enum
from pathlib import Path
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
from gepa_mindfulness.training.contracts import TrainablePolicyBackend
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


class DropoutCausalLM(TinyCausalLM):
    """Tiny model whose policy distribution deterministically distinguishes train/eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=1.0)
        self.generation_modes: list[bool] = []
        with torch.no_grad():
            self.embedding.weight.fill_(1.0)
            self.lm_head.weight.zero_()
            self.lm_head.weight[3].fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **sampling_parameters: object,
    ) -> torch.Tensor:
        self.generation_modes.append(self.training)
        return super().generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **sampling_parameters,
        )


class MalformedValueHead(nn.Module):
    """Value head returning a caller-selected invalid output shape."""

    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.failure == "rank":
            return hidden[..., 0] + self.anchor
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens + 1, 1)
        return torch.zeros(shape, dtype=hidden.dtype, device=hidden.device) + self.anchor


class FakePeftType(str, Enum):
    """PEFT-style adapter type used to exercise enum normalization."""

    LORA = "LORA"
    IA3 = "IA3"


def _fake_lora_model(config: object | None = None) -> TinyCausalLM:
    model = TinyCausalLM()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.register_parameter("lora_adapter", nn.Parameter(torch.zeros(1)))
    adapter_config = config or SimpleNamespace(adapter_type="LORA")
    model.peft_config = {"default": adapter_config}
    return model


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _train_backend_step(backend: TorchPolicyBackend) -> int:
    trajectory = backend.generate((RolloutRequest(prompt="calm breath", seed=17),))[0]
    assert trajectory.response_token_ids is not None
    selected_tokens = (True,) * len(trajectory.response_token_ids)
    batch = TrajectoryBatch((trajectory,), (selected_tokens,))
    backend.zero_grad()
    evaluation = backend.evaluate(batch)
    backend.backward(-(evaluation.log_probs + evaluation.value_predictions).mean())
    return backend.optimizer_step().step


def _backend_snapshot(backend: TorchPolicyBackend) -> dict[str, object]:
    return {
        "policy": _clone_state(backend.policy_model),
        "reference": _clone_state(backend.reference_model),
        "value_head": _clone_state(backend.value_head),
        "optimizer": deepcopy(backend.optimizer.state_dict()),
        "step": backend._step,
        "max_new_tokens": backend.max_new_tokens,
        "learning_rate": backend.learning_rate,
        "cpu_rng_state": torch.get_rng_state().clone(),
    }


def _assert_backend_snapshot(
    backend: TorchPolicyBackend,
    snapshot: dict[str, object],
) -> None:
    _assert_nested_equal(snapshot["policy"], backend.policy_model.state_dict())
    _assert_nested_equal(snapshot["reference"], backend.reference_model.state_dict())
    _assert_nested_equal(snapshot["value_head"], backend.value_head.state_dict())
    _assert_nested_equal(snapshot["optimizer"], backend.optimizer.state_dict())
    assert backend._step == snapshot["step"]
    assert backend.max_new_tokens == snapshot["max_new_tokens"]
    assert backend.learning_rate == snapshot["learning_rate"]
    _assert_nested_equal(snapshot["cpu_rng_state"], torch.get_rng_state())


def _checkpoint_payload_before_divergence(
    backend: TorchPolicyBackend,
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    assert _train_backend_step(backend) == 1
    source = tmp_path / "source.pt"
    backend.save_checkpoint(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)

    assert _train_backend_step(backend) == 2
    with torch.no_grad():
        for parameter in backend.reference_model.parameters():
            parameter.add_(4.0)
    backend.max_new_tokens += 3
    backend.learning_rate *= 2.0
    torch.manual_seed(2027)
    return payload, _backend_snapshot(backend)


def _new_tiny_backend() -> TorchPolicyBackend:
    torch.manual_seed(7)
    return TorchPolicyBackend(
        policy_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        device="cpu",
        learning_rate=0.05,
        max_new_tokens=2,
        model_identifier="tiny-local",
    )


@pytest.fixture
def tiny_backend() -> TorchPolicyBackend:
    """Build a deterministic CPU backend without loading external assets."""
    return _new_tiny_backend()


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


def test_generate_evaluates_evidence_in_eval_mode_and_restores_training() -> None:
    """Dropout cannot make recorded old log probabilities differ from generation mode."""
    backend = TorchPolicyBackend(
        policy_model=DropoutCausalLM(),
        tokenizer=TinyTokenizer(),
        max_new_tokens=1,
    )
    backend.policy_model.train()

    trajectory = backend.generate((RolloutRequest(prompt="calm breath"),))[0]

    assert backend.policy_model.training
    assert backend.policy_model.generation_modes == [False]
    backend.policy_model.eval()
    evaluation = backend.evaluate(
        TrajectoryBatch(
            trajectories=(trajectory,),
            response_token_masks=((True,),),
        )
    )
    torch.testing.assert_close(
        torch.tensor([trajectory.old_log_probs]),
        evaluation.log_probs.detach(),
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


@pytest.mark.parametrize("failure", ["rank", "width"])
def test_evaluate_rejects_malformed_value_head_output(failure: str) -> None:
    """A custom value head must return one scalar for every response predictor position."""
    backend = TorchPolicyBackend(
        policy_model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        value_head=MalformedValueHead(failure),
    )
    batch = TrajectoryBatch(
        trajectories=(_trajectory(),),
        response_token_masks=((True, True),),
    )

    with pytest.raises(RuntimeError, match=r"value_head.*\[batch, response_tokens, 1\]"):
        backend.evaluate(batch)


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


@pytest.mark.parametrize("optimizer_kind", ["missing", "foreign"])
def test_constructor_rejects_optimizer_with_wrong_parameter_ownership(
    optimizer_kind: str,
) -> None:
    """An injected optimizer must own every backend trainable and no unrelated parameter."""
    model = TinyCausalLM()
    value_head = nn.Linear(model.config.hidden_size, 1)
    intended = [*model.parameters(), *value_head.parameters()]
    parameters = intended[:-1]
    if optimizer_kind == "foreign":
        parameters = [*intended, nn.Parameter(torch.ones(1))]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)

    with pytest.raises(ValueError, match="optimizer.*exactly"):
        TorchPolicyBackend(
            policy_model=model,
            tokenizer=TinyTokenizer(),
            value_head=value_head,
            optimizer=optimizer,
        )


def test_backend_conforms_to_trainable_policy_protocol(
    tiny_backend: TorchPolicyBackend,
) -> None:
    """The concrete backend implements every operation on the public trainable protocol."""
    assert isinstance(tiny_backend, TrainablePolicyBackend)


def test_checkpoint_round_trip_restores_trainable_reference_optimizer_step_and_rng(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """Backend checkpoint state resumes one CPU training process without invented defaults."""
    trajectory = tiny_backend.generate((RolloutRequest(prompt="calm breath"),))[0]
    batch = TrajectoryBatch((trajectory,), ((True, True),))
    tiny_backend.zero_grad()
    evaluation = tiny_backend.evaluate(batch)
    tiny_backend.backward(-(evaluation.log_probs + evaluation.value_predictions).mean())
    tiny_backend.optimizer_step()
    policy_state = _clone_state(tiny_backend.policy_model)
    reference_state = _clone_state(tiny_backend.reference_model)
    value_state = _clone_state(tiny_backend.value_head)
    optimizer_state = deepcopy(tiny_backend.optimizer.state_dict())
    torch.manual_seed(1234)
    checkpoint = tmp_path / "backend.pt"

    save_result = tiny_backend.save_checkpoint(checkpoint)
    expected_random = torch.rand(4)
    assert save_result.step == 1
    assert checkpoint.is_file()

    tiny_backend.zero_grad()
    changed_evaluation = tiny_backend.evaluate(batch)
    changed_loss = -(changed_evaluation.log_probs + changed_evaluation.value_predictions).mean()
    tiny_backend.backward(changed_loss)
    assert tiny_backend.optimizer_step().step == 2
    with torch.no_grad():
        for parameter in tiny_backend.reference_model.parameters():
            parameter.add_(10.0)
    torch.manual_seed(9999)

    load_result = tiny_backend.load_checkpoint(checkpoint)

    assert load_result.step == 1
    _assert_nested_equal(policy_state, tiny_backend.policy_model.state_dict())
    _assert_nested_equal(reference_state, tiny_backend.reference_model.state_dict())
    _assert_nested_equal(value_state, tiny_backend.value_head.state_dict())
    _assert_nested_equal(optimizer_state, tiny_backend.optimizer.state_dict())
    assert torch.equal(expected_random, torch.rand(4))
    assert all(
        not parameter.requires_grad for parameter in tiny_backend.reference_model.parameters()
    )


def test_checkpoint_bytes_preflight_is_nonmutating_and_load_reuses_verified_payload(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """Store-facing bytes APIs separate validation from one transactional restore."""
    assert _train_backend_step(tiny_backend) == 1
    checkpoint = tmp_path / "backend.pt"
    tiny_backend.save_checkpoint(checkpoint)
    payload = checkpoint.read_bytes()
    saved_policy = _clone_state(tiny_backend.policy_model)
    assert _train_backend_step(tiny_backend) == 2
    divergent = _backend_snapshot(tiny_backend)
    transaction_snapshot = tiny_backend.capture_checkpoint_restore_state()

    preflight = tiny_backend.preflight_checkpoint_bytes(payload)

    assert preflight.step == 1
    _assert_backend_snapshot(tiny_backend, divergent)

    restored = tiny_backend.load_checkpoint_bytes(payload)

    assert restored.step == 1
    _assert_nested_equal(saved_policy, tiny_backend.policy_model.state_dict())

    tiny_backend.rollback_checkpoint_restore_state(transaction_snapshot)

    _assert_backend_snapshot(tiny_backend, divergent)


def test_checkpoint_load_rejects_incompatible_payload_without_mutation(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """An incompatible checkpoint fails before any backend parameters are changed."""
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"format_version": 999}, checkpoint)
    policy_state = _clone_state(tiny_backend.policy_model)

    with pytest.raises(ValueError, match="checkpoint format_version"):
        tiny_backend.load_checkpoint(checkpoint)

    _assert_nested_equal(policy_state, tiny_backend.policy_model.state_dict())


def test_checkpoint_invalid_rng_length_is_rejected_without_any_mutation(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """A late-invalid CPU RNG state cannot partially restore older backend state."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    payload["cpu_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    checkpoint = tmp_path / "invalid-rng.pt"
    torch.save(payload, checkpoint)

    caught: Exception | None = None
    try:
        tiny_backend.load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 - the regression captures the public failure type
        caught = error

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert isinstance(caught, ValueError)
    assert "cpu_rng_state" in str(caught)


def test_checkpoint_cpu_backend_rejects_cuda_rng_state_without_any_mutation(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """A CPU checkpoint cannot claim an inactive CUDA process RNG state."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    payload["cuda_rng_state"] = torch.zeros(8, dtype=torch.uint8)
    checkpoint = tmp_path / "unexpected-cuda-rng.pt"
    torch.save(payload, checkpoint)

    caught: Exception | None = None
    try:
        tiny_backend.load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 - rejection type follows the state assertion
        caught = error

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert isinstance(caught, ValueError)
    assert "cuda_rng_state" in str(caught)
    assert "CPU" in str(caught)


@pytest.mark.parametrize("corruption", ["parameter_ids", "state_tensor"])
def test_checkpoint_malformed_optimizer_is_rejected_without_any_mutation(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
    corruption: str,
) -> None:
    """Malformed optimizer identities or tensors cannot alter current backend state."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    if corruption == "parameter_ids":
        group = optimizer_state["param_groups"][0]
        group["params"][0] = 999_999
    else:
        first_state = next(iter(optimizer_state["state"].values()))
        first_state["exp_avg"] = torch.zeros(1)
    checkpoint = tmp_path / f"invalid-optimizer-{corruption}.pt"
    torch.save(payload, checkpoint)

    caught: Exception | None = None
    try:
        tiny_backend.load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 - the regression captures the public failure type
        caught = error

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert isinstance(caught, ValueError)
    assert "optimizer" in str(caught)


@pytest.mark.parametrize(
    ("corruption", "replacement"),
    [
        ("missing_exp_avg", None),
        ("missing_exp_avg_sq", None),
        ("string_exp_avg", "not-a-tensor"),
        ("scalar_exp_avg", 0.0),
    ],
)
def test_checkpoint_optimizer_structure_rejection_preserves_a_healthy_live_optimizer(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
    corruption: str,
    replacement: object,
) -> None:
    """Incomplete or wrong-kind AdamW state cannot damage the live optimizer."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    first_state = next(iter(optimizer_state["state"].values()))
    if corruption == "missing_exp_avg":
        first_state.pop("exp_avg")
    elif corruption == "missing_exp_avg_sq":
        first_state.pop("exp_avg_sq")
    else:
        first_state["exp_avg"] = replacement
    checkpoint = tmp_path / f"invalid-optimizer-structure-{corruption}.pt"
    torch.save(payload, checkpoint)

    caught: Exception | None = None
    try:
        tiny_backend.load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 - rejection type follows the state assertion
        caught = error

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert isinstance(caught, ValueError)
    assert "optimizer" in str(caught)
    assert _train_backend_step(tiny_backend) == 3


@pytest.mark.parametrize("corruption", ["one_parameter", "all_parameters"])
def test_checkpoint_missing_initialized_optimizer_entries_are_rejected_without_mutation(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
    corruption: str,
) -> None:
    """A checkpoint cannot silently discard initialized AdamW parameter moments."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    optimizer_state = payload["optimizer_state"]
    assert isinstance(optimizer_state, dict)
    saved_state = optimizer_state["state"]
    assert isinstance(saved_state, dict)
    assert saved_state
    if corruption == "one_parameter":
        saved_state.pop(next(iter(saved_state)))
    else:
        saved_state.clear()
    checkpoint = tmp_path / f"missing-optimizer-state-{corruption}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="optimizer state.*parameter IDs"):
        tiny_backend.load_checkpoint(checkpoint)

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert _train_backend_step(tiny_backend) == 3


def test_checkpoint_initialized_optimizer_restores_into_fresh_optimizer(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
) -> None:
    """A fresh optimizer may restore initialized moments from a compatible checkpoint."""
    assert _train_backend_step(tiny_backend) == 1
    checkpoint = tmp_path / "initialized-optimizer.pt"
    tiny_backend.save_checkpoint(checkpoint)
    expected_optimizer_state = deepcopy(tiny_backend.optimizer.state_dict())
    fresh_backend = _new_tiny_backend()
    assert fresh_backend.optimizer.state_dict()["state"] == {}

    result = fresh_backend.load_checkpoint(checkpoint)

    assert result.step == 1
    _assert_nested_equal(expected_optimizer_state, fresh_backend.optimizer.state_dict())
    assert _train_backend_step(fresh_backend) == 2


def test_checkpoint_late_restore_failure_rolls_back_and_preserves_primary_error(
    tiny_backend: TorchPolicyBackend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restore and rollback RNG failure preserves current state and the primary exception."""
    payload, snapshot = _checkpoint_payload_before_divergence(tiny_backend, tmp_path)
    checkpoint = tmp_path / "late-rng-failure.pt"
    torch.save(payload, checkpoint)
    original_set_rng_state = torch.set_rng_state
    call_count = 0

    def fail_primary_and_rollback(state: torch.Tensor) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("primary RNG restore failure")
        if call_count == 2:
            raise RuntimeError("secondary rollback RNG failure")
        original_set_rng_state(state)

    monkeypatch.setattr(torch, "set_rng_state", fail_primary_and_rollback)
    caught: Exception | None = None
    try:
        tiny_backend.load_checkpoint(checkpoint)
    except Exception as error:  # noqa: BLE001 - cause identity is the behavior under test
        caught = error

    _assert_backend_snapshot(tiny_backend, snapshot)
    assert isinstance(caught, ValueError)
    assert isinstance(caught.__cause__, RuntimeError)
    assert str(caught.__cause__) == "primary RNG restore failure"
    assert call_count == 2


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


def test_direct_lora_mode_rejects_an_ordinary_trainable_model() -> None:
    """A mode label alone cannot claim PEFT LoRA capability evidence."""
    with pytest.raises(ValueError, match="PEFT LoRA"):
        TorchPolicyBackend(
            policy_model=TinyCausalLM(),
            tokenizer=TinyTokenizer(),
            training_mode="lora",
        )


@pytest.mark.parametrize(
    "adapter_config",
    [
        SimpleNamespace(peft_type="lora"),
        SimpleNamespace(peft_type=FakePeftType.LORA),
        {"adapter_type": "LORA"},
        "LORA",
        FakePeftType.LORA,
    ],
)
def test_direct_lora_mode_accepts_verified_adapter_only_trainables(
    adapter_config: object,
) -> None:
    """A PEFT-marked model with adapter-only trainables substantiates LoRA capability."""
    backend = TorchPolicyBackend(
        policy_model=_fake_lora_model(adapter_config),
        tokenizer=TinyTokenizer(),
        training_mode="lora",
    )

    assert (
        backend.capabilities().state(Capability.SUPPORTS_LORA_TRAINING) is CapabilityState.SUPPORTED
    )
    trainable = [
        name
        for name, parameter in backend.policy_model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable == ["lora_adapter"]


@pytest.mark.parametrize(
    "peft_config",
    [
        {"default": SimpleNamespace(peft_type="IA3")},
        {"default": SimpleNamespace()},
        {"default": SimpleNamespace(peft_type="unknown")},
        {
            "lora": SimpleNamespace(peft_type=FakePeftType.LORA),
            "other": SimpleNamespace(peft_type=FakePeftType.IA3),
        },
        {
            "default": {
                "peft_type": "LORA",
                "adapter_type": "IA3",
            }
        },
        {
            "default": SimpleNamespace(
                peft_type=FakePeftType.LORA,
                adapter_type=FakePeftType.IA3,
            )
        },
    ],
)
def test_direct_lora_mode_rejects_non_lora_or_ambiguous_peft_configs(
    peft_config: dict[str, object],
) -> None:
    """LoRA-looking parameter names cannot override non-LoRA or missing config evidence."""
    model = _fake_lora_model()
    model.peft_config = peft_config

    with pytest.raises(ValueError, match="PEFT LoRA.*config"):
        TorchPolicyBackend(
            policy_model=model,
            tokenizer=TinyTokenizer(),
            training_mode="lora",
        )


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
            self.peft_type = FakePeftType.LORA

    def fake_get_peft_model(model: nn.Module, config: FakeLoraConfig) -> nn.Module:
        assert config.values["task_type"] == "CAUSAL_LM"
        adapted = _fake_lora_model()
        adapted.peft_config["default"] = config
        return adapted

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
    assert trainable == ["lora_adapter"]

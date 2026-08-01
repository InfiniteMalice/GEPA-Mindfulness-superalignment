"""Contract tests for the CUDA specialization of the shared PyTorch backend."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gepa_mindfulness.training.backends.torch_cuda import (
    CudaOutOfMemoryError,
    _wrap_distributed_trainables,
    create_cuda_backend,
    detect_cuda_capabilities,
    validate_distributed_runtime,
)
from gepa_mindfulness.training.backends.torch_policy import TorchPolicyBackend
from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityError,
    CapabilityEvidence,
    CapabilityState,
)
from gepa_mindfulness.training.checkpointing import RankRNGState
from gepa_mindfulness.training.engine import (
    RLTrainingEngine,
    SystemCapabilityProvider,
    _config_hash,
    _distributed_mean_scalars,
    _gather_rank_rng_states,
    _JSONLRunLogger,
    required_capabilities,
)
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    CheckpointConfig,
    DatasetConfig,
    DistributedRuntimeConfig,
    LoggingConfig,
    PolicyConfig,
    RLRunConfig,
    RuntimeConfig,
    load_rl_config,
)


class _RecordingDeviceContext:
    def __init__(self, selected: list[int], device_index: int) -> None:
        self.selected = selected
        self.device_index = device_index

    def __enter__(self) -> None:
        self.selected.append(self.device_index)

    def __exit__(self, *error: object) -> None:
        del error


class _TinyCudaTokenizer:
    """Tokenizer whose complete vocabulary is local to the CUDA acceptance test."""

    pad_token_id = 0
    eos_token_id = 1
    _tokens = {
        "<pad>": 0,
        "<eos>": 1,
        "practice": 2,
        "slowly": 3,
        "chosen": 4,
        "rejected": 5,
    }
    _words = {token_id: token for token, token_id in _tokens.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._tokens[word] for word in text.split()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ignored = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        return " ".join(self._words[token_id] for token_id in token_ids if token_id not in ignored)


class _TinyCudaCausalLM(nn.Module):
    """Small causal language model built without downloads or cached assets."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, _name_or_path="tiny-local-cuda-lm")
        self.embedding = nn.Embedding(6, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, 6, bias=False)

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
        del attention_mask, sampling_parameters
        response = torch.full(
            (input_ids.shape[0], max_new_tokens),
            4,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)


def _cuda_config(
    *,
    device: str = "cuda:0",
    precision: str = "fp32",
    distributed: DistributedRuntimeConfig | None = None,
) -> RLRunConfig:
    return RLRunConfig(
        runtime=RuntimeConfig(
            backend="cuda",
            device=device,
            precision=precision,
            distributed=distributed or DistributedRuntimeConfig(),
        ),
        policy=PolicyConfig(model_name="LOCAL_MODEL_PATH", max_new_tokens=17),
        algorithm=AlgorithmConfig(batch_size=3, gradient_accumulation_steps=5),
    )


def _module_checksum(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_cuda_pair_dataset(path: Path) -> None:
    component_names = (
        "objective_fidelity",
        "feedback_integrity",
        "skill_transfer",
        "reality_contact",
        "exploit_disclosure",
        "long_horizon_agency",
        "benign_creativity",
        "repair_quality",
    )
    record = {
        "record_id": "cuda-case:grounded_over_proxy",
        "source_case_id": "cuda-case",
        "source_case_version": "1.0",
        "source_path": "authored/cuda.jsonl",
        "source_line": 1,
        "source_sha256": "c" * 64,
        "pair_rule": "grounded_over_proxy",
        "prompt": "practice slowly",
        "chosen": "chosen",
        "rejected": "rejected",
        "chosen_class": "grounded_success",
        "rejected_class": "proxy_exploitation",
        "chosen_reward_components": {name: 0.5 for name in component_names},
        "rejected_reward_components": {name: -0.5 for name in component_names},
        "diagnostics": {"central": "authored", "supporting": []},
        "schema_version": "reward-integrity-rl-pairs-v1",
    }
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _cuda_acceptance_config(tmp_path: Path, precision: str) -> RLRunConfig:
    dataset_path = tmp_path / f"pairs-{precision}.jsonl"
    _write_cuda_pair_dataset(dataset_path)
    return RLRunConfig(
        runtime=RuntimeConfig(backend="cuda", device="cuda:0", precision=precision),
        policy=PolicyConfig(model_name="tiny-local-cuda-lm", max_new_tokens=1),
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.05,
            batch_size=1,
            max_steps=1,
        ),
        dataset=DatasetConfig(train_path=str(dataset_path)),
        checkpoint=CheckpointConfig(
            output_dir=str(tmp_path / f"checkpoints-{precision}"),
            save_steps=1,
        ),
        logging=LoggingConfig(log_dir=str(tmp_path / f"logs-{precision}")),
        seed=42,
    )


def _cuda_acceptance_engine(
    config: RLRunConfig,
    backends: list[TorchPolicyBackend],
    initial_policy_checksums: list[str],
    initial_reference_checksums: list[str],
    model_forward_dtypes: list[torch.dtype],
    value_forward_dtypes: list[torch.dtype],
) -> RLTrainingEngine:
    def build_backend(value: RLRunConfig) -> TorchPolicyBackend:
        backend = create_cuda_backend(
            value,
            lambda: (_TinyCudaCausalLM(), _TinyCudaTokenizer()),
        )
        backends.append(backend)
        initial_policy_checksums.append(backend.policy_parameter_checksum())
        initial_reference_checksums.append(_module_checksum(backend.reference_model))
        backend.policy_model.register_forward_hook(
            lambda module, inputs, output: model_forward_dtypes.append(output.logits.dtype)
        )
        backend.value_head.register_forward_hook(
            lambda module, inputs, output: value_forward_dtypes.append(output.dtype)
        )
        return backend

    return RLTrainingEngine(config, backend_factory=build_backend)


def _require_cuda_precision(precision: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    try:
        detect_cuda_capabilities("cuda:0", precision)
    except CapabilityError as error:
        pytest.skip(f"{precision} is unsupported on cuda:0: {error}")
    if precision == "fp32":
        return
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    try:
        with torch.autocast("cuda", dtype=dtype):
            probe = torch.ones((2, 2), device="cuda:0")
            result = probe @ probe
        torch.cuda.synchronize(0)
    except RuntimeError as error:
        message = str(error).casefold()
        if "not implemented" in message or "not supported" in message:
            pytest.skip(f"{precision} operation probe is unsupported on cuda:0: {error}")
        raise
    assert result.dtype is dtype
    assert torch.isfinite(
        result
    ).all(), f"{precision} operation probe returned non-finite values on cuda:0"


def _available_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_count: int = 1,
    bf16_supported: bool = True,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda index: _RecordingDeviceContext([], index),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)


def _distributed_cuda_smoke_worker(
    rank: int,
    config_path: str,
    dataset_path: str,
    checkpoint_dir: str,
    log_dir: str,
    init_file: str,
) -> None:
    os.environ.update(
        {
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
    )
    template = load_rl_config(config_path)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"file://{Path(init_file).as_posix()}",
        world_size=2,
        rank=rank,
    )
    try:
        config = replace(
            template,
            policy=PolicyConfig(model_name="tiny-local-cuda-lm", max_new_tokens=1),
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.05,
                batch_size=1,
                max_steps=1,
            ),
            dataset=DatasetConfig(train_path=dataset_path),
            checkpoint=CheckpointConfig(output_dir=checkpoint_dir, save_steps=1),
            logging=LoggingConfig(log_dir=log_dir),
            seed=42,
        )
        engine = RLTrainingEngine(
            config,
            backend_factory=lambda value: create_cuda_backend(
                value,
                lambda: (_TinyCudaCausalLM(), _TinyCudaTokenizer()),
            ),
        )
        result = engine.train(max_steps=1)
        assert result.global_step == 1
    finally:
        torch.distributed.destroy_process_group()


def _ddp_trainable_sync_worker(rank: int, init_file: str, result_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{Path(init_file).as_posix()}",
        world_size=2,
        rank=rank,
    )
    try:
        torch.manual_seed(123)
        policy = _TinyCudaCausalLM()
        reference = _TinyCudaCausalLM()
        reference.load_state_dict(policy.state_dict())
        value_head = nn.Linear(8, 1)
        topology = DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=rank,
            local_rank=rank,
        )
        policy, value_head = _wrap_distributed_trainables(
            policy,
            value_head,
            topology,
            "cpu",
        )
        backend = TorchPolicyBackend(
            policy_model=policy,
            reference_model=reference,
            value_head=value_head,
            tokenizer=_TinyCudaTokenizer(),
            device="cpu",
            learning_rate=0.1,
        )
        backend.zero_grad()
        token = 2 + rank
        input_ids = torch.tensor([[token]], dtype=torch.long)
        policy_loss = backend.policy_model(input_ids).logits.sum() * float(rank + 1)
        value_input = torch.full((1, 1, 8), float(rank + 1))
        value_loss = backend.value_head(value_input).sum() * float(rank + 2)
        backend.backward(policy_loss + value_loss)
        backend.optimizer_step()
        torch.save(
            {
                "policy": backend.policy_model.state_dict(),
                "value_head": backend.value_head.state_dict(),
            },
            Path(result_dir) / f"rank-{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def _never_called() -> tuple[object, object]:
    raise AssertionError("model loading must happen only after CUDA validation")


def test_cuda_factory_rejects_unavailable_runtime_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the availability preflight would invoke model loading on a CPU-only host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(CapabilityError, match="CUDA is unavailable"):
        create_cuda_backend(_cuda_config(), _never_called)


@pytest.mark.parametrize(
    ("initialized", "world_size", "rank", "message"),
    [
        (False, 2, 0, "initialized"),
        (True, 3, 0, "world_size"),
        (True, 2, 1, "rank"),
    ],
)
def test_distributed_preflight_rejects_process_group_mismatch_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    initialized: bool,
    world_size: int,
    rank: int,
    message: str,
) -> None:
    _available_cuda(monkeypatch, device_count=2)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )

    with pytest.raises(CapabilityError, match=message):
        create_cuda_backend(config, _never_called)


def test_distributed_preflight_rejects_unavailable_torch_distributed() -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    fake_distributed = SimpleNamespace(is_available=lambda: False)

    with pytest.raises(CapabilityError, match="torch.distributed"):
        validate_distributed_runtime(config, distributed=fake_distributed)


def test_fsdp_sharded_optimizer_fails_before_model_or_checkpoint_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _available_cuda(monkeypatch, device_count=2)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="fsdp",
            world_size=2,
            rank=0,
            local_rank=0,
            sharded_optimizer=True,
        )
    )

    with pytest.raises(CapabilityError, match="sharded optimizer.*checkpoint"):
        create_cuda_backend(config, _never_called)


def test_distributed_runtime_adds_explicit_capability_requirement() -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )

    assert Capability.SUPPORTS_DISTRIBUTED_TRAINING in required_capabilities(config, "train")


def test_system_provider_reports_exact_initialized_process_group_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[int] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        device=lambda index: _RecordingDeviceContext(selected, index),
        is_bf16_supported=lambda: True,
    )
    fake_distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        get_rank=lambda: 0,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, distributed=fake_distributed)
    monkeypatch.setattr(
        "gepa_mindfulness.training.engine.importlib.util.find_spec",
        lambda name: SimpleNamespace() if name in {"torch", "transformers"} else None,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.engine.import_module",
        lambda name: fake_torch if name == "torch" else None,
    )
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )

    report = SystemCapabilityProvider().detect(config)

    capability = report.capabilities[Capability.SUPPORTS_DISTRIBUTED_TRAINING]
    assert capability.state is CapabilityState.SUPPORTED
    assert "world_size=2" in capability.evidence
    assert "rank=0" in capability.evidence


def test_engine_distributed_preflight_precedes_dataset_model_rng_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    supported = BackendCapabilities(
        backend_name="fake",
        backend_version="1",
        capabilities={
            capability: CapabilityEvidence(
                state=CapabilityState.SUPPORTED,
                evidence="test evidence",
            )
            for capability in Capability
        },
    )
    provider = SimpleNamespace(detect=lambda config: (events.append("capability"), supported)[1])

    def fail_preflight(config: RLRunConfig) -> None:
        del config
        events.append("distributed_preflight")
        raise CapabilityError("process group mismatch")

    def forbidden(name: str) -> object:
        events.append(name)
        raise AssertionError(f"{name} must happen after distributed preflight")

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.validate_distributed_runtime",
        fail_preflight,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.engine.random.seed",
        lambda seed: forbidden("rng"),
    )
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    engine = RLTrainingEngine(
        config,
        capability_provider=provider,
        backend_factory=lambda value: forbidden("model"),
        dataset_factory=lambda value: forbidden("dataset"),
        logger_factory=lambda value: forbidden("output"),
    )

    with pytest.raises(CapabilityError, match="process group mismatch"):
        engine.collect()

    assert events == ["capability", "distributed_preflight"]


def test_rank_local_fields_do_not_change_distributed_checkpoint_compatibility_hash() -> None:
    rank_zero = _cuda_config(
        device="cuda:0",
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        ),
    )
    rank_one = _cuda_config(
        device="cuda:1",
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=1,
            local_rank=1,
        ),
    )

    assert _config_hash(rank_zero) == _config_hash(rank_one)


def test_distributed_checkpoint_gathers_one_rng_state_per_rank() -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    rank_zero = RankRNGState(
        python_rng_state=random.Random(101).getstate(),
        torch_cpu_rng_state=torch.get_rng_state().clone(),
        torch_cuda_rng_states=(torch.zeros(8, dtype=torch.uint8),),
    )
    rank_one = RankRNGState(
        python_rng_state=random.Random(202).getstate(),
        torch_cpu_rng_state=torch.get_rng_state().clone(),
        torch_cuda_rng_states=(torch.ones(8, dtype=torch.uint8),),
    )

    class FakeDistributed:
        @staticmethod
        def all_gather_object(outputs: list[object], local: object) -> None:
            assert local is rank_zero
            outputs[:] = [rank_zero, rank_one]

    gathered = _gather_rank_rng_states(config, rank_zero, distributed=FakeDistributed())

    assert gathered == {0: rank_zero, 1: rank_one}


def test_nonzero_engine_logger_uses_inert_rank_aware_sink(tmp_path: Path) -> None:
    config = RLRunConfig(
        runtime=RuntimeConfig(
            backend="cuda",
            device="cuda:1",
            distributed=DistributedRuntimeConfig(
                strategy="ddp",
                world_size=2,
                rank=1,
                local_rank=1,
            ),
        ),
        logging=LoggingConfig(log_dir=str(tmp_path)),
    )

    logger = _JSONLRunLogger(config, "a" * 64)

    assert logger.sink.rank == 1
    assert not logger.directory.exists()


def test_run_level_scalar_metrics_are_mean_reduced_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    original_tensor = torch.tensor
    reductions: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda values, **kwargs: original_tensor(values, dtype=kwargs.get("dtype")),
    )

    def fake_all_gather_object(outputs: list[object], local_names: object) -> None:
        outputs[:] = [local_names, local_names]

    def fake_all_reduce(values: torch.Tensor, *, op: object) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        reductions.append(values.clone())
        additions = ([3.0, 7.0], [1.0, 1.0])
        values.add_(original_tensor(additions[len(reductions) - 1], dtype=values.dtype))

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    reduced = _distributed_mean_scalars(
        config,
        {"policy_loss": 1.0, "total_reward": 5.0},
    )

    assert len(reductions) == 2
    assert reduced == {"policy_loss": 2.0, "total_reward": 6.0}


def test_run_level_scalar_reduction_agrees_keys_and_counts_conditional_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    original_tensor = torch.tensor
    reductions: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda values, **kwargs: original_tensor(values, dtype=kwargs.get("dtype")),
    )

    def fake_all_gather_object(outputs: list[object], local_names: object) -> None:
        assert local_names == ("alpha", "shared")
        outputs[:] = [("alpha", "shared"), ("zeta", "shared")]

    def fake_all_reduce(values: torch.Tensor, *, op: object) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        reductions.append(values.clone())
        additions = ([0.0, 8.0, 10.0], [0.0, 1.0, 1.0])
        values.add_(original_tensor(additions[len(reductions) - 1], dtype=values.dtype))

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    reduced = _distributed_mean_scalars(
        config,
        {"shared": 4.0, "alpha": 2.0},
    )

    assert len(reductions) == 2
    assert reduced == {"alpha": 2.0, "shared": 6.0, "zeta": 10.0}


def test_run_level_scalar_reduction_participates_when_local_metrics_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy="ddp",
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )
    original_tensor = torch.tensor
    calls: list[str] = []
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda values, **kwargs: original_tensor(values, dtype=kwargs.get("dtype")),
    )

    def fake_all_gather_object(outputs: list[object], local_names: object) -> None:
        calls.append("keys")
        assert local_names == ()
        outputs[:] = [(), ("remote_only",)]

    def fake_all_reduce(values: torch.Tensor, *, op: object) -> None:
        assert op is torch.distributed.ReduceOp.SUM
        calls.append("reduce")
        values.add_(original_tensor([9.0 if calls.count("reduce") == 1 else 1.0]))

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert _distributed_mean_scalars(config, {}) == {"remote_only": 9.0}
    assert calls == ["keys", "reduce", "reduce"]


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_cuda_factory_wraps_only_trainable_policy_with_selected_strategy(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    _available_cuda(monkeypatch, device_count=2)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    captured: dict[str, object] = {"wrapped_modules": []}

    class RecordingWrapper(nn.Module):
        def __init__(self, module: nn.Module, **kwargs: object) -> None:
            super().__init__()
            self.module = module
            captured["wrapper_kwargs"] = kwargs
            captured["wrapped_modules"].append(module)

        def forward(self, *args: object, **kwargs: object) -> object:
            return self.module(*args, **kwargs)

    def fake_backend(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.DistributedDataParallel",
        RecordingWrapper,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.FullyShardedDataParallel",
        RecordingWrapper,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fake_backend,
    )
    monkeypatch.setattr(nn.Module, "to", lambda module, *args, **kwargs: module)
    policy = _TinyCudaCausalLM()
    config = _cuda_config(
        distributed=DistributedRuntimeConfig(
            strategy=strategy,
            world_size=2,
            rank=0,
            local_rank=0,
        )
    )

    create_cuda_backend(config, lambda: (policy, _TinyCudaTokenizer()))

    wrapped = captured["policy_model"]
    assert isinstance(wrapped, RecordingWrapper)
    assert wrapped.module is policy
    assert isinstance(captured["value_head"], RecordingWrapper)
    assert len(captured["wrapped_modules"]) == 2
    assert isinstance(captured["reference_model"], _TinyCudaCausalLM)
    assert not isinstance(captured["reference_model"], RecordingWrapper)


def test_ddp_policy_and_value_head_converge_with_different_rank_gradients(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("requires torch.distributed with Gloo support")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    torch.multiprocessing.spawn(
        _ddp_trainable_sync_worker,
        args=(str(tmp_path / "process-group-init"), str(result_dir)),
        nprocs=2,
        join=True,
    )

    rank_zero = torch.load(result_dir / "rank-0.pt", weights_only=True)
    rank_one = torch.load(result_dir / "rank-1.pt", weights_only=True)
    assert rank_zero.keys() == rank_one.keys()
    for section in ("policy", "value_head"):
        assert rank_zero[section].keys() == rank_one[section].keys()
        assert all(
            torch.equal(rank_zero[section][name], rank_one[section][name])
            for name in rank_zero[section]
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:-1", "cuda:one", "cuda:0:1"])
def test_cuda_detection_rejects_malformed_device_selector(device: str) -> None:
    """Relaxing CUDA selector parsing would let an ambiguous device reach model loading."""
    with pytest.raises(CapabilityError, match="CUDA device"):
        detect_cuda_capabilities(device, "fp32")


@pytest.mark.parametrize(
    ("backend", "device", "precision", "message"),
    [
        ("cuda", "cpu", "fp32", "CUDA device"),
        ("pytorch", "cpu", "fp16", "mixed precision requires a CUDA device"),
        ("cuda", "cuda:0", "tf32", "runtime.precision"),
    ],
)
def test_runtime_config_rejects_unsupported_cuda_combinations(
    backend: str,
    device: str,
    precision: str,
    message: str,
) -> None:
    """Removing cross-field validation would defer a known failure until model loading."""
    with pytest.raises(ValueError, match=message):
        RuntimeConfig(backend=backend, device=device, precision=precision)


def test_cuda_detection_rejects_unknown_precision() -> None:
    """A public detector must not silently downgrade an unknown precision to FP32."""
    with pytest.raises(CapabilityError, match="precision"):
        detect_cuda_capabilities("cuda:0", "tf32")


def test_cuda_factory_rejects_out_of_range_device_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the index bound would defer a deterministic configuration error to Torch."""
    _available_cuda(monkeypatch, device_count=1)

    with pytest.raises(CapabilityError, match=r"cuda:2.*1 visible"):
        create_cuda_backend(_cuda_config(device="cuda:2"), _never_called)


def test_cuda_factory_rejects_bf16_without_hardware_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claiming BF16 without the runtime probe would expose unsupported mixed precision."""
    _available_cuda(monkeypatch, bf16_supported=False)

    with pytest.raises(CapabilityError, match="BF16"):
        create_cuda_backend(_cuda_config(precision="bf16"), _never_called)


def test_bf16_probe_targets_the_configured_nonzero_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probing the current device could approve cuda:2 from heterogeneous cuda:0 evidence."""
    selected: list[int] = []
    _available_cuda(monkeypatch, device_count=3)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda index: _RecordingDeviceContext(selected, index),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda: bool(selected and selected[-1] == 2),
    )

    capabilities = detect_cuda_capabilities("cuda:2", "bf16")

    assert selected == [2]
    assert capabilities.state(Capability.SUPPORTS_MIXED_PRECISION) is CapabilityState.SUPPORTED


def test_engine_bf16_evidence_names_configured_device_and_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic evidence cannot prove which heterogeneous device and precision were checked."""
    selected: list[int] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        device=lambda index: _RecordingDeviceContext(selected, index),
        is_bf16_supported=lambda: bool(selected and selected[-1] == 2),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(
        "gepa_mindfulness.training.engine.import_module",
        lambda name: fake_torch if name == "torch" else None,
    )

    report = SystemCapabilityProvider().detect(_cuda_config(device="cuda:2", precision="bf16"))

    assert selected == [2]
    assert report.state(Capability.SUPPORTS_CUDA) is CapabilityState.SUPPORTED
    assert report.state(Capability.SUPPORTS_MIXED_PRECISION) is CapabilityState.SUPPORTED
    cuda_evidence = report.capabilities[Capability.SUPPORTS_CUDA].evidence
    mixed_evidence = report.capabilities[Capability.SUPPORTS_MIXED_PRECISION].evidence
    assert "cuda:2" in cuda_evidence
    assert "cuda:2" in mixed_evidence
    assert "bf16" in mixed_evidence.casefold()


@pytest.mark.parametrize(
    ("precision", "autocast_dtype", "uses_scaler"),
    [
        ("fp32", None, False),
        ("fp16", torch.float16, True),
        ("bf16", torch.bfloat16, False),
    ],
)
def test_cuda_factory_selects_amp_components(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    autocast_dtype: torch.dtype | None,
    uses_scaler: bool,
) -> None:
    """Swapping dtype or loss-scaling branches would silently change numeric behavior."""
    _available_cuda(monkeypatch)
    captured: dict[str, object] = {}
    scaler = object()

    def fake_backend(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fake_backend,
    )
    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda._create_grad_scaler",
        lambda: scaler,
    )

    backend = create_cuda_backend(
        _cuda_config(precision=precision),
        lambda: (object(), object()),
    )

    assert backend.autocast_dtype is autocast_dtype
    assert captured["gradient_scaler"] is (scaler if uses_scaler else None)
    assert captured["device"] == "cuda:0"
    oom_error_factory = captured["oom_error_factory"]
    assert callable(oom_error_factory)
    translated = oom_error_factory("generate")
    assert isinstance(translated, CudaOutOfMemoryError)
    assert translated.operation == "generate"


@pytest.mark.parametrize(
    ("precision", "mixed_precision_state"),
    [
        ("fp32", CapabilityState.UNSUPPORTED),
        ("fp16", CapabilityState.SUPPORTED),
        ("bf16", CapabilityState.SUPPORTED),
    ],
)
def test_cuda_detection_reports_evidence_backed_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    mixed_precision_state: CapabilityState,
) -> None:
    """Hard-coded capability claims would hide the selected precision and runtime probes."""
    _available_cuda(monkeypatch)

    capabilities = detect_cuda_capabilities("cuda:0", precision)

    assert capabilities.backend_name == "torch_cuda"
    assert capabilities.state(Capability.SUPPORTS_CUDA) is CapabilityState.SUPPORTED
    assert capabilities.state(Capability.SUPPORTS_MIXED_PRECISION) is mixed_precision_state
    evidence = capabilities.capabilities[Capability.SUPPORTS_CUDA].evidence
    assert "cuda:0" in evidence


def test_cuda_factory_translates_oom_with_actionable_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-raising raw CUDA OOM would omit the settings needed to reduce memory pressure."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("allocation failed")

    def fail_backend(**kwargs: object) -> object:
        del kwargs
        raise original

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fail_backend,
    )

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(precision="fp16"), lambda: (object(), object()))

    error = captured.value
    assert error.operation == "backend_initialization"
    assert error.device == "cuda:0"
    assert error.precision == "fp16"
    assert error.batch_size == 3
    assert error.max_new_tokens == 17
    assert error.gradient_accumulation_steps == 5
    assert error.__cause__ is original
    assert "reduce algorithm.batch_size" in str(error)


def test_cuda_oom_includes_same_process_allocator_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping allocator evidence would force operators to inspect a different process."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("allocation failed")
    monkeypatch.setattr(torch.cuda, "memory_summary", lambda **kwargs: "same-process summary")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 101)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 202)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 303)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 404)

    def fail_backend(**kwargs: object) -> object:
        del kwargs
        raise original

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fail_backend,
    )

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(), lambda: (object(), object()))

    error = captured.value
    assert error.memory_summary == "same-process summary"
    assert error.memory_stats == {
        "allocated_bytes": 101,
        "reserved_bytes": 202,
        "max_allocated_bytes": 303,
        "max_reserved_bytes": 404,
    }
    assert error.diagnostic_errors == ()
    assert "allocator_stats=" in str(error)
    assert "allocator_summary=same-process summary" in str(error)
    assert error.__cause__ is original


def test_cuda_oom_diagnostic_failure_preserves_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secondary allocator-query error must not replace the CUDA OOM that triggered it."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("allocation failed")

    def fail_diagnostic(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(torch.cuda, "memory_summary", fail_diagnostic)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fail_diagnostic)
    monkeypatch.setattr(torch.cuda, "memory_reserved", fail_diagnostic)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail_diagnostic)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fail_diagnostic)

    def fail_backend(**kwargs: object) -> object:
        del kwargs
        raise original

    monkeypatch.setattr(
        "gepa_mindfulness.training.backends.torch_cuda.TorchPolicyBackend",
        fail_backend,
    )

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(), lambda: (object(), object()))

    error = captured.value
    assert error.memory_summary is None
    assert error.memory_stats == {}
    assert len(error.diagnostic_errors) == 5
    assert "allocator_diagnostic_errors=" in str(error)
    assert error.__cause__ is original


def test_cuda_factory_translates_model_loading_oom_with_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An asset-loading OOM must identify its operation without hiding the Torch failure."""
    _available_cuda(monkeypatch)
    original = torch.OutOfMemoryError("model allocation failed")

    def fail_model_factory() -> tuple[object, object]:
        raise original

    with pytest.raises(CudaOutOfMemoryError) as captured:
        create_cuda_backend(_cuda_config(), fail_model_factory)

    assert captured.value.operation == "model_loading"
    assert captured.value.__cause__ is original


def test_cuda_factory_preserves_non_oom_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broad exception translation would misdiagnose model or tokenizer failures as CUDA OOM."""
    _available_cuda(monkeypatch)
    original = RuntimeError("invalid model assets")

    def fail_model_factory() -> tuple[object, object]:
        raise original

    with pytest.raises(RuntimeError) as captured:
        create_cuda_backend(_cuda_config(), fail_model_factory)

    assert captured.value is original


def test_cuda_single_gpu_template_is_strict_and_uses_bundled_pairs() -> None:
    """A stale key or remote model name would make the checked-in CUDA template misleading."""
    path = Path(__file__).parents[1] / "configs" / "rl" / "cuda_single_gpu.yaml"

    config = load_rl_config(path)

    assert config.runtime == RuntimeConfig(backend="cuda", device="cuda:0", precision="fp32")
    assert config.policy.model_name == "LOCAL_MODEL_PATH"
    assert config.dataset.train_path == "data/synthetic/reward_integrity/rl_pairs_v1.jsonl"


def test_cuda_ddp_template_resolves_torchrun_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / "cuda_ddp.yaml"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    config = load_rl_config(path)

    assert config.runtime.backend == "cuda"
    assert config.runtime.device == "cuda:1"
    assert config.runtime.distributed == DistributedRuntimeConfig(
        strategy="ddp",
        world_size=2,
        rank=1,
        local_rank=1,
    )
    assert config.policy.model_name == "LOCAL_MODEL_PATH"
    assert config.dataset.train_path == "data/synthetic/reward_integrity/rl_pairs_v1.jsonl"


def test_cuda_ddp_template_rejects_missing_torchrun_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / "cuda_ddp.yaml"
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="WORLD_SIZE.*required"):
        load_rl_config(path)


@pytest.mark.parametrize(
    ("name", "value"),
    [("WORLD_SIZE", "two"), ("RANK", "1.5"), ("LOCAL_RANK", "-1")],
)
def test_cuda_ddp_template_rejects_invalid_torchrun_integer(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / "cuda_ddp.yaml"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_rl_config(path)


def test_cuda_operator_guide_has_executable_commands_and_honest_host_status() -> None:
    """Omitting an operator step would leave CUDA setup, recovery, or evidence ambiguous."""
    path = Path(__file__).parents[1] / "docs" / "rl" / "README.md"
    guide = path.read_text(encoding="utf-8")
    required_text = (
        "https://pytorch.org/get-started/locally/",
        "python -m pip install torch==2.9.1 --index-url",
        "python -m pip install -e '.[rl]'",
        "cp configs/rl/cuda_single_gpu.yaml run.cuda.ppo.yaml",
        "gepa rl doctor --config run.cuda.ppo.yaml",
        "gepa rl train --config run.cuda.ppo.yaml --max-steps 1",
        "nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv",
        "same-process allocator statistics",
        "allocator_summary=",
        "gepa rl resume --config run.cuda.ppo.yaml",
        "python -m pytest --strict-markers -m cuda tests/test_rl_cuda.py -q -rs",
        "Hardware verification status: not run on this CPU-only host",
    )

    for expected in required_text:
        assert expected in guide


@pytest.mark.cuda
def test_two_gpu_ddp_smoke_has_one_manifest_and_unique_trajectory_stream(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two visible CUDA devices")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("requires torch.distributed with NCCL support")
    dataset_path = tmp_path / "pairs.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    init_file = tmp_path / "process-group-init"
    _write_cuda_pair_dataset(dataset_path)

    torch.multiprocessing.spawn(
        _distributed_cuda_smoke_worker,
        args=(
            str(Path(__file__).parents[1] / "configs" / "rl" / "cuda_ddp.yaml"),
            str(dataset_path),
            str(checkpoint_dir),
            str(log_dir),
            str(init_file),
        ),
        nprocs=2,
        join=True,
    )

    manifests = list(log_dir.rglob("run_manifest.json"))
    assert len(manifests) == 1
    trajectory_paths = list(log_dir.rglob("trajectories.jsonl"))
    assert len(trajectory_paths) == 1
    trajectories = [
        json.loads(line) for line in trajectory_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    trajectory_ids = [record["trajectory"]["trajectory_id"] for record in trajectories]
    assert trajectory_ids
    assert len(trajectory_ids) == len(set(trajectory_ids))
    assert len(list(checkpoint_dir.glob("checkpoint-*/manifest.json"))) == 1


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_cuda_parameter_update_and_checkpoint_restore_shared_engine(
    tmp_path: Path,
    precision: str,
) -> None:
    """A value-head-only update or lossy restore must not qualify the CUDA training path."""
    _require_cuda_precision(precision)
    config = _cuda_acceptance_config(tmp_path, precision)
    backends: list[TorchPolicyBackend] = []
    initial_policy_checksums: list[str] = []
    initial_reference_checksums: list[str] = []
    model_forward_dtypes: list[torch.dtype] = []
    value_forward_dtypes: list[torch.dtype] = []

    trained = _cuda_acceptance_engine(
        config,
        backends,
        initial_policy_checksums,
        initial_reference_checksums,
        model_forward_dtypes,
        value_forward_dtypes,
    ).train(max_steps=1)

    trained_backend = backends[-1]
    trained_policy_state = {
        name: value.detach().cpu().clone()
        for name, value in trained_backend.policy_model.state_dict().items()
    }
    trained_reference_state = {
        name: value.detach().cpu().clone()
        for name, value in trained_backend.reference_model.state_dict().items()
    }
    assert trained.global_step == 1
    assert trained.policy_parameters_updated is True
    assert trained.policy_parameter_checksum_before == initial_policy_checksums[0]
    assert trained.policy_parameter_checksum_after != initial_policy_checksums[0]
    assert trained.policy_parameter_checksum_after == trained_backend.policy_parameter_checksum()
    expected_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    assert model_forward_dtypes
    assert value_forward_dtypes
    assert set(model_forward_dtypes) == {expected_dtype}
    assert set(value_forward_dtypes) == {expected_dtype}
    assert _module_checksum(trained_backend.reference_model) == initial_reference_checksums[0]
    assert not any(
        parameter.requires_grad for parameter in trained_backend.reference_model.parameters()
    )
    for module in (
        trained_backend.policy_model,
        trained_backend.reference_model,
        trained_backend.value_head,
    ):
        assert all(parameter.device.type == "cuda" for parameter in module.parameters())
    assert trained.checkpoint is not None
    checkpoint_path = trained.checkpoint.path
    assert trained.checkpoint.global_step == 1
    assert trained.checkpoint.backend_step == 1
    assert (checkpoint_path / "backend.pt").is_file()
    assert (checkpoint_path / "training_state.pt").is_file()
    assert (checkpoint_path / "manifest.json").is_file()

    restored = _cuda_acceptance_engine(
        config,
        backends,
        initial_policy_checksums,
        initial_reference_checksums,
        model_forward_dtypes,
        value_forward_dtypes,
    ).resume(checkpoint_path, max_steps=0)

    restored_backend = backends[-1]
    assert restored_backend is not trained_backend
    assert restored.global_step == 1
    assert restored.trajectory_count == 0
    assert restored.policy_parameters_updated is False
    assert restored.policy_parameter_checksum_before == trained.policy_parameter_checksum_after
    assert restored.policy_parameter_checksum_after == trained.policy_parameter_checksum_after
    assert restored_backend.policy_parameter_checksum() == trained.policy_parameter_checksum_after
    assert restored_backend._step == 1
    assert not any(
        parameter.requires_grad for parameter in restored_backend.reference_model.parameters()
    )
    for module in (
        restored_backend.policy_model,
        restored_backend.reference_model,
        restored_backend.value_head,
    ):
        assert all(parameter.device.type == "cuda" for parameter in module.parameters())
    for name, value in restored_backend.policy_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), trained_policy_state[name])
    for name, value in restored_backend.reference_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), trained_reference_state[name])

"""Offline CPU acceptance proof for the portable model-weight RL runtime."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

import pytest
import torch
from torch import nn

from gepa_mindfulness.training.backends import TorchPolicyBackend
from gepa_mindfulness.training.engine import RLTrainingEngine
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    CheckpointConfig,
    DatasetConfig,
    LoggingConfig,
    PolicyConfig,
    RLRunConfig,
)


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    payload: dict[str, Any] = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return payload["project"]["optional-dependencies"]


def _dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise AssertionError(f"invalid requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


_RL_RUNTIME = {
    "torch>=2.9,<3",
    "transformers>=4.57,<6",
    "peft>=0.17,<1",
}
_DEV_TOOLS = {
    "build>=1.2",
    "black>=24.0",
    "mypy>=1.8",
    "pytest>=8.0",
    "ruff>=0.4",
}
_ALL_ONLY = {
    "dspy-ai>=2.5",
    "matplotlib>=3.7",
    "networkx>=3.0",
    "textual>=0.20",
    "vllm>=0.6",
    "weasyprint>=53",
}


def _assert_rl_extra_contract(extras: dict[str, list[str]]) -> None:
    expected = {
        "rl": _RL_RUNTIME,
        "train": _RL_RUNTIME | {"textual>=0.20"},
        "rl-dev": _RL_RUNTIME | _DEV_TOOLS,
        "all": _RL_RUNTIME | _ALL_ONLY,
    }
    forbidden = {
        "accelerate",
        "datasets",
        "intel-extension-for-pytorch",
        "torch-directml",
        "torch-xla",
        "trl",
    }
    for extra, expected_requirements in expected.items():
        requirements = extras[extra]
        names = [_dependency_name(requirement) for requirement in requirements]
        assert len(names) == len(set(names))
        assert set(names).isdisjoint(forbidden)
        assert not any(name.startswith("nvidia-") for name in names)
        assert not any(
            "@" in requirement
            or "://" in requirement
            or "+cu" in requirement.lower()
            or "pytorch.org" in requirement.lower()
            for requirement in requirements
        )
        assert len(requirements) == len(expected_requirements)
        assert set(requirements) == expected_requirements


class TinyLocalTokenizer:
    """Tokenizer whose complete vocabulary is defined in this test process."""

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


class TinyLocalCausalLM(nn.Module):
    """Small causal language model configured locally with deterministic sampling."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, _name_or_path="tiny-local-causal-lm")
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
        response_token = 4 if torch.initial_seed() % 2 == 0 else 5
        response = torch.full(
            (input_ids.shape[0], max_new_tokens),
            response_token,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)


def _module_checksum(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_pair_dataset(path: Path) -> None:
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
        "record_id": "offline-case:grounded_over_proxy",
        "source_case_id": "offline-case",
        "source_case_version": "1.0",
        "source_path": "authored/offline.jsonl",
        "source_line": 1,
        "source_sha256": "a" * 64,
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


def _config(tmp_path: Path, algorithm: str) -> RLRunConfig:
    dataset_path = tmp_path / "pairs.jsonl"
    _write_pair_dataset(dataset_path)
    return RLRunConfig(
        policy=PolicyConfig(model_name="tiny-local-causal-lm", max_new_tokens=1),
        algorithm=AlgorithmConfig(
            name=algorithm,
            learning_rate=0.05,
            batch_size=1,
            max_steps=1,
            group_size=2,
            zero_variance_policy="skip",
        ),
        dataset=DatasetConfig(train_path=str(dataset_path)),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_steps=1),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs")),
        seed=42,
    )


def _engine(
    config: RLRunConfig,
    created_backends: list[TorchPolicyBackend],
    reference_checksums: list[str],
) -> RLTrainingEngine:
    def build_backend(value: RLRunConfig) -> TorchPolicyBackend:
        backend = TorchPolicyBackend(
            policy_model=TinyLocalCausalLM(),
            tokenizer=TinyLocalTokenizer(),
            device="cpu",
            learning_rate=value.algorithm.learning_rate,
            max_new_tokens=value.policy.max_new_tokens,
            model_identifier=value.policy.model_name,
        )
        created_backends.append(backend)
        reference_checksums.append(_module_checksum(backend.reference_model))
        return backend

    return RLTrainingEngine(config, backend_factory=build_backend)


def test_rl_extras_are_bounded_synchronized_and_keep_heavy_frameworks_optional() -> None:
    """Unbounded, divergent, or heavyweight RL dependency declarations must fail."""
    extras = _optional_dependencies()
    assert set(extras["dev"]) == _DEV_TOOLS
    _assert_rl_extra_contract(extras)


@pytest.mark.parametrize(
    "extra, requirement",
    [
        ("rl", "torch>=2.8,<3"),
        ("train", "torch>=2.9,<3"),
        ("rl-dev", "torch @ https://download.pytorch.org/torch.whl"),
        ("all", "torch==2.9.0+cu128"),
        ("all", "nvidia-cublas-cu12>=12"),
        ("all", "trl>=0.9"),
        ("all", "datasets>=3"),
        ("all", "accelerate>=1"),
        ("all", "torch-directml>=0.2"),
    ],
)
def test_rl_extra_contract_rejects_duplicate_divergent_and_platform_requirements(
    extra: str,
    requirement: str,
) -> None:
    """A second package spelling or unsupported runtime must not bypass exact extras."""
    extras = deepcopy(_optional_dependencies())
    if requirement.startswith(("torch @", "torch==")):
        torch_index = next(
            index for index, value in enumerate(extras[extra]) if _dependency_name(value) == "torch"
        )
        extras[extra][torch_index] = requirement
    else:
        extras[extra].append(requirement)

    with pytest.raises(AssertionError):
        _assert_rl_extra_contract(extras)


def test_value_head_only_change_does_not_count_as_policy_weight_change() -> None:
    """A value-head mutation must not satisfy the public policy-weight checksum proof."""
    backend = TorchPolicyBackend(
        policy_model=TinyLocalCausalLM(),
        tokenizer=TinyLocalTokenizer(),
        device="cpu",
        max_new_tokens=1,
    )
    combined_before = backend.parameter_checksum()
    policy_before = backend.policy_parameter_checksum()

    with torch.no_grad():
        next(backend.value_head.parameters()).add_(1.0)

    assert backend.parameter_checksum() != combined_before
    assert backend.policy_parameter_checksum() == policy_before


@pytest.mark.parametrize("algorithm", ["ppo", "grpo"])
def test_offline_cpu_train_checkpoint_and_resume_updates_real_model_weights(
    tmp_path: Path,
    algorithm: str,
) -> None:
    """Removing optimization or checkpoint restoration must fail this end-to-end proof."""
    config = _config(tmp_path, algorithm)
    backends: list[TorchPolicyBackend] = []
    reference_checksums: list[str] = []

    trained = _engine(config, backends, reference_checksums).train(max_steps=1)

    trained_backend = backends[-1]
    trained_reference = _module_checksum(trained_backend.reference_model)
    assert trained.global_step == 1
    assert trained.parameters_updated is True
    assert trained.parameter_checksum_before != trained.parameter_checksum_after
    assert trained.policy_parameter_checksum_before != trained.policy_parameter_checksum_after
    assert trained.policy_parameters_updated is True
    assert trained_reference == reference_checksums[0]
    assert not any(
        parameter.requires_grad for parameter in trained_backend.reference_model.parameters()
    )
    assert trained.checkpoint is not None
    checkpoint_path = trained.checkpoint.path
    assert checkpoint_path.is_dir()
    assert (checkpoint_path / "backend.pt").is_file()
    assert (checkpoint_path / "training_state.pt").is_file()
    assert (checkpoint_path / "manifest.json").is_file()
    assert trained.log_directory is not None
    assert (trained.log_directory / "run_manifest.json").is_file()
    assert (trained.log_directory / "trajectories.jsonl").is_file()
    assert (trained.log_directory / "metrics.jsonl").is_file()

    restored = _engine(config, backends, reference_checksums).resume(checkpoint_path, max_steps=0)

    restored_backend = backends[-1]
    assert restored.global_step == 1
    assert restored.trajectory_count == 0
    assert restored.parameter_checksum_before == trained.parameter_checksum_after
    assert restored.parameter_checksum_after == trained.parameter_checksum_after
    assert restored.parameters_updated is False
    assert restored.policy_parameter_checksum_before == trained.policy_parameter_checksum_after
    assert restored.policy_parameter_checksum_after == trained.policy_parameter_checksum_after
    assert restored.policy_parameters_updated is False
    assert reference_checksums[1] == trained_reference
    assert _module_checksum(restored_backend.reference_model) == trained_reference

    resumed = _engine(config, backends, reference_checksums).resume(checkpoint_path, max_steps=1)

    resumed_backend = backends[-1]
    assert resumed.global_step == 2
    assert resumed.parameters_updated is True
    assert resumed.parameter_checksum_before == trained.parameter_checksum_after
    assert resumed.parameter_checksum_after != trained.parameter_checksum_after
    assert resumed.policy_parameter_checksum_before == trained.policy_parameter_checksum_after
    assert resumed.policy_parameter_checksum_after != trained.policy_parameter_checksum_after
    assert resumed.policy_parameters_updated is True
    assert reference_checksums[2] == trained_reference
    assert _module_checksum(resumed_backend.reference_model) == trained_reference

"""Tests for the canonical portable PyTorch RL runtime configuration."""

from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from gepa_mindfulness.training.runtime_config import (
    RLRunConfig,
    load_rl_config,
    translate_legacy_config,
)


def canonical_payload() -> dict[str, object]:
    """Return a complete canonical configuration suitable for CPU smoke runs."""
    return {
        "runtime": {"backend": "pytorch", "device": "cpu"},
        "policy": {"model_name": "demo-model", "max_new_tokens": 32},
        "algorithm": {
            "name": "ppo",
            "learning_rate": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
        },
        "reward": {"weights": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2}},
        "dataset": {"train_path": "prompts.txt", "format": "text"},
        "checkpoint": {"output_dir": "runs/cpu"},
        "logging": {"log_dir": "runs/cpu/logs", "level": "INFO"},
        "seed": 42,
    }


def test_canonical_mapping_produces_frozen_nested_sections() -> None:
    config = RLRunConfig.from_mapping(canonical_payload())

    assert config.runtime.backend == "pytorch"
    assert config.policy.max_new_tokens == 32
    assert config.algorithm.name == "ppo"
    with pytest.raises(FrozenInstanceError):
        config.runtime.device = "cuda"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"runtime": {"backend": "pytorch", "device": "mps"}}, "device"),
        ({"runtime": {"backend": "pytorch", "device": 1}}, "runtime.device"),
        ({"policy": {"unknown": "value"}}, "policy"),
        ({"seed": "42"}, "seed"),
    ],
)
def test_canonical_mapping_rejects_invalid_values_without_coercion(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RLRunConfig.from_mapping(payload)


def test_legacy_grpo_config_translates_to_canonical() -> None:
    with pytest.warns(DeprecationWarning):
        config = translate_legacy_config({"grpo": {"group_size": 4}, "device": "cpu"})

    assert config.runtime.backend == "pytorch"
    assert config.algorithm.name == "grpo"
    assert config.algorithm.group_size == 4


def test_canonical_file_load_has_no_deprecation_warning(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "canonical.yaml"
    path.write_text(yaml.safe_dump(canonical_payload()), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = load_rl_config(path)

    assert config.runtime.device == "cpu"
    assert not [warning for warning in captured if warning.category is DeprecationWarning]


def test_legacy_file_load_warns_and_translates(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump({"trainer_type": "ppo", "device": "cpu"}), encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        config = load_rl_config(path)

    assert config.algorithm.name == "ppo"


@pytest.mark.parametrize("name", ["pytorch_cpu_ppo.yaml", "pytorch_cpu_grpo.yaml"])
def test_cpu_runtime_examples_load(name: str) -> None:
    path = Path(__file__).parents[1] / "configs" / "rl" / name

    config = load_rl_config(path)

    assert config.runtime.device == "cpu"
    assert config.algorithm.name in {"ppo", "grpo"}

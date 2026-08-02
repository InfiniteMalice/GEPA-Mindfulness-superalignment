"""Tests for TrainingConfig loading from nested mappings."""

from __future__ import annotations

import pytest

from gepa_mindfulness.training.configs import RLRunConfig, TrainingConfig, translate_legacy_config


def test_training_config_supports_nested_sections() -> None:
    payload = {
        "model": {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "device": "cuda",
        },
        "training": {
            "use_dual_path": True,
            "max_steps": 1000,
            "batch_size": 2,
            "learning_rate": 1e-5,
        },
        "reward_weights": {
            "alpha": 0.25,
            "beta": 0.35,
            "gamma": 0.35,
            "delta": 0.05,
        },
    }

    config = TrainingConfig.from_mapping(payload)

    assert config.max_steps == 1000
    assert config.device == "cuda"
    assert config.use_dual_path is True
    assert config.ppo.batch_size == 2
    assert config.ppo.learning_rate == 1e-5
    assert config.model.policy_model == "microsoft/Phi-3-mini-4k-instruct"


def test_training_configs_exports_legacy_translation() -> None:
    with pytest.warns(DeprecationWarning):
        config = translate_legacy_config({"trainer_type": "grpo", "device": "cpu"})

    assert isinstance(config, RLRunConfig)
    assert config.algorithm.name == "grpo"

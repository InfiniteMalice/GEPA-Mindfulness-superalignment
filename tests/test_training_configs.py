"""Tests for TrainingConfig loading from nested mappings."""

from __future__ import annotations

from gepa_mindfulness.training.configs import TrainingConfig


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
            "ppo": {
                "learning_rate": 1e-5,
                "mini_batch_size": 4,
            },
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
    assert config.ppo.mini_batch_size == 4
    assert config.model.policy_model == "microsoft/Phi-3-mini-4k-instruct"

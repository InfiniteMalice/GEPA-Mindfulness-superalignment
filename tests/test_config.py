from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from gepa_mindfulness.training.config import (
    GRPOConfig,
    PPOConfig,
    RewardWeightsConfig,
    load_trainer_config,
)


def test_reward_weights_sum_validation() -> None:
    config = RewardWeightsConfig(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    assert config.alpha == 0.4
    with pytest.raises(ValueError):
        RewardWeightsConfig(alpha=0.6, beta=0.6, gamma=-0.1, delta=-0.1)


def test_loads_grpo_config(tmp_path: Path) -> None:
    payload = {
        "trainer_type": "grpo",
        "model_name": "demo",
        "dataset_path": "data.jsonl",
        "output_dir": "runs/demo",
        "group_size": 4,
        "kl_coef": 0.1,
        "temperature": 0.7,
        "reward_weights": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_trainer_config(config_path)
    assert isinstance(config, GRPOConfig)
    assert config.group_size == 4


def test_group_size_warning(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning):
        GRPOConfig(
            model_name="demo",
            dataset_path="data.jsonl",
            output_dir=str(tmp_path),
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            group_size=20,
            kl_coef=0.1,
            temperature=0.7,
        )


def test_loads_ppo_config(tmp_path: Path) -> None:
    payload = {
        "trainer_type": "ppo",
        "model_name": "demo",
        "dataset_path": "data.jsonl",
        "output_dir": "runs/ppo",
        "value_coef": 0.1,
        "clip_range": 0.2,
        "vf_clip_range": 0.2,
        "gae_lambda": 0.95,
        "target_kl": 0.01,
        "reward_weights": {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_trainer_config(config_path)
    assert isinstance(config, PPOConfig)
    assert config.value_coef == 0.1

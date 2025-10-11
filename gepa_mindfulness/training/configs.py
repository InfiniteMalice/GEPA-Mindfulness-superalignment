"""Configuration models for the training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field, validator


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class RewardWeightsConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RewardWeightsConfig":
        payload = payload or {}
        return cls(
            alpha=_to_float(payload.get("alpha"), 1.0),
            beta=_to_float(payload.get("beta"), 1.0),
            gamma=_to_float(payload.get("gamma"), 1.0),
            delta=_to_float(payload.get("delta"), 1.0),
        )

    def dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PPOConfig:
    learning_rate: float = 5e-6
    batch_size: int = 8
    mini_batch_size: int = 2
    ppo_epochs: int = 4

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PPOConfig":
        payload = payload or {}
        return cls(
            learning_rate=_to_float(payload.get("learning_rate"), 5e-6),
            batch_size=_to_int(payload.get("batch_size"), 8),
            mini_batch_size=_to_int(payload.get("mini_batch_size"), 2),
            ppo_epochs=_to_int(payload.get("ppo_epochs"), 4),
        )


@dataclass
class ModelConfig:
    policy_model: str = "gpt2"
    reward_model: str = "distilbert-base-uncased"
    vllm_engine: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ModelConfig":
        payload = payload or {}
        return cls(
            policy_model=str(payload.get("policy_model", "gpt2")),
            reward_model=str(payload.get("reward_model", "distilbert-base-uncased")),
            vllm_engine=payload.get("vllm_engine"),
        )


@dataclass
class TrainingConfig:
    seed: int = 42
    max_steps: int = 100
    device: str = "cpu"
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adversarial_batch: int = 2
    confidence_threshold: float = 0.75

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        payload = payload or {}
        device = str(payload.get("device", "cpu"))
        if device not in {"cpu", "cuda"} and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA identifier")

        return cls(
            seed=_to_int(payload.get("seed"), 42),
            max_steps=_to_int(payload.get("max_steps"), 100),
            device=device,
            reward_weights=RewardWeightsConfig.from_mapping(payload.get("reward_weights")),
            ppo=PPOConfig.from_mapping(payload.get("ppo")),
            model=ModelConfig.from_mapping(payload.get("model")),
            adversarial_batch=_to_int(payload.get("adversarial_batch"), 2),
            confidence_threshold=_to_float(payload.get("confidence_threshold"), 0.75),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(path: str | Path) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] | None = yaml.safe_load(handle)  # type: ignore[assignment]
    return TrainingConfig.from_mapping(payload)


__all__ = [
    "RewardWeightsConfig",
    "PPOConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_training_config",
]

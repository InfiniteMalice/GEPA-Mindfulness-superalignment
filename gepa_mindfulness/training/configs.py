"""Configuration models for the training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, validator
import yaml


class RewardWeightsConfig(BaseModel):
    alpha: float = Field(1.0, description="Task success weight")
    beta: float = Field(1.0, description="GEPA alignment weight")
    gamma: float = Field(1.0, description="Honesty trace weight")
    delta: float = Field(1.0, description="Hallucination penalty weight")


class PPOConfig(BaseModel):
    learning_rate: float = 5e-6
    batch_size: int = 8
    mini_batch_size: int = 2
    ppo_epochs: int = 4


class ModelConfig(BaseModel):
    policy_model: str = "gpt2"
    reward_model: str = "distilbert-base-uncased"
    vllm_engine: str | None = None


class TrainingConfig(BaseModel):
    seed: int = 42
    max_steps: int = 100
    device: str = "cpu"
    reward_weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    adversarial_batch: int = 2
    confidence_threshold: float = 0.75

    @validator("device")
    def validate_device(cls, value: str) -> str:  # pragma: no cover - simple guard
        if value not in {"cpu", "cuda"} and not value.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA identifier")
        return value


def load_training_config(path: str | Path) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle) or {}
    return TrainingConfig.parse_obj(payload)

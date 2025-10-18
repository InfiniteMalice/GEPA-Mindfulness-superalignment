"""Backward-compatible configuration helpers used by legacy tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RewardWeightsConfig:
    alpha: float = 0.25
    beta: float = 0.35
    gamma: float = 0.35
    delta: float = 0.05

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0.0:
            raise ValueError("reward weights must have positive mass")
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total


@dataclass
class HonestyConfig:
    uncertainty_threshold: float = 0.75
    idk_bonus: float = 1.0
    calibration_bonus_weight: float = 0.5
    uncertainty_marker_bonus: float = 0.3


@dataclass
class DeceptionConfig:
    detect: bool = True
    log_fingerprints: bool = True
    fingerprint_dir: str = "runs/fingerprints"


@dataclass
class OutputConfig:
    checkpoint_dir: str = "runs/checkpoints"


@dataclass
class PPOSection:
    batch_size: int = 1
    learning_rate: float = 1e-5


@dataclass
class ModelConfig:
    policy_model: str = "demo-model"
    device: str = "cpu"


@dataclass
class TrainingConfig:
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    honesty: HonestyConfig = field(default_factory=HonestyConfig)
    deception: DeceptionConfig = field(default_factory=DeceptionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ppo: PPOSection = field(default_factory=PPOSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    max_steps: int = 100
    device: str = "cpu"
    use_dual_path: bool = False

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "TrainingConfig":
        reward = RewardWeightsConfig(**payload.get("reward_weights", {}))
        honesty = HonestyConfig(**payload.get("honesty", {}))
        deception = DeceptionConfig(**payload.get("deception", {}))
        output = OutputConfig(**payload.get("output", {}))
        training_payload = dict(payload.get("training", {}))
        model_payload = dict(payload.get("model", {}))
        if "name" in model_payload and "policy_model" not in model_payload:
            model_payload["policy_model"] = model_payload.pop("name")
        allowed_model_keys = {"policy_model", "device"}
        filtered_model = {
            key: value for key, value in model_payload.items() if key in allowed_model_keys
        }
        model = ModelConfig(**filtered_model)
        ppo_kwargs = {
            "batch_size": int(training_payload.get("batch_size", 1)),
            "learning_rate": float(training_payload.get("learning_rate", 1e-5)),
        }
        ppo = PPOSection(**ppo_kwargs)
        max_steps = int(training_payload.get("max_steps", payload.get("max_steps", 100)))
        use_dual_path = bool(
            training_payload.get("use_dual_path", payload.get("use_dual_path", False))
        )
        return cls(
            reward_weights=reward,
            honesty=honesty,
            deception=deception,
            output=output,
            ppo=ppo,
            model=model,
            max_steps=max_steps,
            device=model.device,
            use_dual_path=use_dual_path,
        )


__all__ = [
    "RewardWeightsConfig",
    "HonestyConfig",
    "DeceptionConfig",
    "OutputConfig",
    "TrainingConfig",
    "PPOSection",
    "ModelConfig",
]

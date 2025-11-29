"""Dataclass-based configuration models for PPO and GRPO trainers."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal

import yaml

from mindful_trace_gepa.train.grn import GRNSettings
from ..core.rewards import GEPARewardCalculator, HallucinationConfig, RewardWeights


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class RewardWeightsConfig:
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.2
    delta: float = 0.2

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"weight {name} must be non-negative")
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0.0:
            raise ValueError("reward weights must have positive mass")
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any] | None) -> "RewardWeightsConfig":
        payload = payload or {}
        return cls(
            alpha=float(payload.get("alpha", 0.3)),
            beta=float(payload.get("beta", 0.3)),
            gamma=float(payload.get("gamma", 0.2)),
            delta=float(payload.get("delta", 0.2)),
        )

    def to_domain(self) -> RewardWeights:
        return RewardWeights(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
        )


@dataclass
class HallucinationPenaltyConfig:
    confidence_threshold: float = 0.75
    confident_wrong_penalty: float = -2.0
    uncertain_wrong_penalty: float = -0.5
    appropriate_abstention_reward: float = 0.5
    lazy_abstention_penalty: float = -0.2

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any] | None) -> "HallucinationPenaltyConfig":
        payload = payload or {}
        return cls(
            confidence_threshold=float(payload.get("confidence_threshold", 0.75)),
            confident_wrong_penalty=float(payload.get("confident_wrong_penalty", -2.0)),
            uncertain_wrong_penalty=float(payload.get("uncertain_wrong_penalty", -0.5)),
            appropriate_abstention_reward=float(payload.get("appropriate_abstention_reward", 0.5)),
            lazy_abstention_penalty=float(payload.get("lazy_abstention_penalty", -0.2)),
        )

    def to_domain(self) -> HallucinationConfig:
        return HallucinationConfig(
            confidence_threshold=self.confidence_threshold,
            confident_wrong_penalty=self.confident_wrong_penalty,
            uncertain_wrong_penalty=self.uncertain_wrong_penalty,
            appropriate_abstention_reward=self.appropriate_abstention_reward,
            lazy_abstention_penalty=self.lazy_abstention_penalty,
        )


@dataclass
class CircuitTracerConfig:
    enabled: bool = True
    trace_frequency: float = 1.0
    strategy: Literal["all", "single", "sample", "extremes", "mixed"] = "all"
    seed: int | None = None

    def __post_init__(self) -> None:
        self.trace_frequency = _clamp(self.trace_frequency, 0.0, 1.0)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any] | None) -> "CircuitTracerConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", True)),
            trace_frequency=float(payload.get("trace_frequency", 1.0)),
            strategy=str(payload.get("strategy", "all")),
            seed=payload.get("seed"),
        )


@dataclass
class BaseTrainerConfig:
    trainer_type: Literal["ppo", "grpo"] = "ppo"
    model_name: str = "demo-model"
    dataset_path: str = ""
    output_dir: str = "runs/default"
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    hallucination: HallucinationPenaltyConfig = field(default_factory=HallucinationPenaltyConfig)
    circuit_tracer: CircuitTracerConfig = field(default_factory=CircuitTracerConfig)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def create_reward_calculator(self) -> GEPARewardCalculator:
        return GEPARewardCalculator(
            weights=self.reward_weights.to_domain(),
            hallucination=self.hallucination.to_domain(),
            abstention_threshold=self.hallucination.confidence_threshold,
        )

    def with_updates(self, **updates: Any) -> "BaseTrainerConfig":
        return replace(self, **updates)

    @classmethod
    def _from_mapping(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trainer_type": payload.get("trainer_type", "ppo"),
            "model_name": payload.get("model_name", "demo-model"),
            "dataset_path": payload.get("dataset_path", ""),
            "output_dir": payload.get("output_dir", "runs/default"),
            "learning_rate": float(payload.get("learning_rate", 1e-5)),
            "batch_size": int(payload.get("batch_size", 4)),
            "gradient_accumulation_steps": int(payload.get("gradient_accumulation_steps", 1)),
            "max_steps": int(payload.get("max_steps", 100)),
            "reward_weights": RewardWeightsConfig.from_mapping(payload.get("reward_weights")),
            "hallucination": HallucinationPenaltyConfig.from_mapping(payload.get("hallucination")),
            "circuit_tracer": CircuitTracerConfig.from_mapping(payload.get("circuit_tracer")),
        }


@dataclass
class PPOConfig(BaseTrainerConfig):
    trainer_type: Literal["ppo"] = "ppo"
    value_coef: float = 0.1
    clip_range: float = 0.2
    vf_clip_range: float = 0.2
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    policy_grn: GRNSettings = field(default_factory=GRNSettings)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "PPOConfig":
        base_kwargs = BaseTrainerConfig._from_mapping(payload)
        base_kwargs.update(
            {
                "value_coef": float(payload.get("value_coef", 0.1)),
                "clip_range": float(payload.get("clip_range", 0.2)),
                "vf_clip_range": float(payload.get("vf_clip_range", 0.2)),
                "gae_lambda": float(payload.get("gae_lambda", 0.95)),
                "target_kl": float(payload.get("target_kl", 0.01)),
                "policy_grn": GRNSettings.from_mapping(payload.get("policy_grn")),
            }
        )
        return cls(**base_kwargs)


@dataclass
class GRPOConfig(BaseTrainerConfig):
    trainer_type: Literal["grpo"] = "grpo"
    group_size: int = 8
    kl_coef: float = 0.05
    temperature: float = 0.8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.group_size > 16:
            warnings.warn(
                "group_size greater than 16 may cause high memory usage",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.circuit_tracer.trace_frequency == 1.0 and self.circuit_tracer.strategy == "all":
            self.circuit_tracer.trace_frequency = 0.5
            self.circuit_tracer.strategy = "mixed"

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "GRPOConfig":
        base_kwargs = BaseTrainerConfig._from_mapping(payload)
        base_kwargs.update(
            {
                "group_size": int(payload.get("group_size", 8)),
                "kl_coef": float(payload.get("kl_coef", 0.05)),
                "temperature": float(payload.get("temperature", 0.8)),
            }
        )
        return cls(**base_kwargs)


def load_trainer_config(path: Path | str) -> BaseTrainerConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must define a mapping")
    trainer_type = payload.get("trainer_type", "ppo")
    if trainer_type == "grpo":
        return GRPOConfig.from_mapping(payload)
    if trainer_type == "ppo":
        return PPOConfig.from_mapping(payload)
    raise ValueError(f"Unsupported trainer type: {trainer_type}")


def load_config_dict(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must define a mapping")
    return payload


__all__ = [
    "BaseTrainerConfig",
    "CircuitTracerConfig",
    "GRPOConfig",
    "PPOConfig",
    "RewardWeightsConfig",
    "HallucinationPenaltyConfig",
    "load_trainer_config",
    "load_config_dict",
]

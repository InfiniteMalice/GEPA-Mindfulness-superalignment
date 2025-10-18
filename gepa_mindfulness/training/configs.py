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
class CircuitTracerSamplingConfig:
    trace_frequency: float = 1.0
    trace_strategy: str = "all"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CircuitTracerSamplingConfig":
        payload = payload or {}
        frequency = _to_float(payload.get("trace_frequency"), 1.0)
        frequency = max(0.0, min(1.0, frequency))
        strategy = str(payload.get("trace_strategy", "all"))
        return cls(trace_frequency=frequency, trace_strategy=strategy)

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HallucinationPenaltyConfig:
    confidence_threshold: float = 0.75
    confident_wrong_penalty: float = -2.0
    uncertain_wrong_penalty: float = -0.5
    appropriate_abstention_reward: float = 0.5
    lazy_abstention_penalty: float = -0.2

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "HallucinationPenaltyConfig":
        payload = payload or {}
        return cls(
            confidence_threshold=_to_float(payload.get("confidence_threshold"), 0.75),
            confident_wrong_penalty=_to_float(payload.get("confident_wrong_penalty"), -2.0),
            uncertain_wrong_penalty=_to_float(payload.get("uncertain_wrong_penalty"), -0.5),
            appropriate_abstention_reward=_to_float(
                payload.get("appropriate_abstention_reward"), 0.5
            ),
            lazy_abstention_penalty=_to_float(payload.get("lazy_abstention_penalty"), -0.2),
        )

    def dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class GRPOConfig:
    group_size: int = 8
    kl_coef: float = 0.05
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    sampling_temperature: float = 0.7
    max_new_tokens: int = 256
    circuit_tracer: CircuitTracerSamplingConfig = field(default_factory=CircuitTracerSamplingConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    hallucination: HallucinationPenaltyConfig = field(default_factory=HallucinationPenaltyConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GRPOConfig":
        payload = payload or {}
        sampling = CircuitTracerSamplingConfig.from_mapping(payload)
        rewards = RewardWeightsConfig.from_mapping(payload.get("reward_weights"))
        hallucination = HallucinationPenaltyConfig.from_mapping(payload.get("hallucination"))
        return cls(
            group_size=_to_int(payload.get("group_size"), 8),
            kl_coef=_to_float(payload.get("kl_coef"), 0.05),
            learning_rate=_to_float(payload.get("learning_rate"), 1e-5),
            batch_size=_to_int(payload.get("batch_size"), 32),
            gradient_accumulation_steps=_to_int(payload.get("gradient_accumulation_steps"), 1),
            sampling_temperature=_to_float(payload.get("sampling_temperature"), 0.7),
            max_new_tokens=_to_int(payload.get("max_new_tokens"), 256),
            circuit_tracer=sampling,
            reward_weights=rewards,
            hallucination=hallucination,
        )

    def dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["circuit_tracer"] = self.circuit_tracer.dict()
        payload["reward_weights"] = self.reward_weights.dict()
        payload["hallucination"] = self.hallucination.dict()
        return payload


@dataclass
class ModelConfig:
    policy_model: str = "demo-model"
    device: str = "cpu"


@dataclass
class TrainingConfig:
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adversarial_batch: int = 2
    confidence_threshold: float = 0.75
    use_dual_path: bool = False
    honesty: HonestyConfig = field(default_factory=HonestyConfig)
    deception: DeceptionConfig = field(default_factory=DeceptionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ppo: PPOSection = field(default_factory=PPOSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    max_steps: int = 100
    device: str = "cpu"
    use_dual_path: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        payload = payload or {}

        training_section = payload.get("training")
        if not isinstance(training_section, Mapping):
            training_section = {}

        model_section = payload.get("model")
        if not isinstance(model_section, Mapping):
            model_section = {}

        device_value = payload.get("device")
        if device_value is None:
            device_value = model_section.get("device")
        if device_value is None:
            device_value = "cpu"
        device = str(device_value)
        if device not in {"cpu", "cuda"} and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA identifier")

        ppo_section = _merge_mapping(payload.get("ppo"), training_section)

        grpo_section = payload.get("grpo")
        if not isinstance(grpo_section, Mapping):
            grpo_section = {}

        return cls(
            seed=_to_int(training_section.get("seed", payload.get("seed")), 42),
            max_steps=_to_int(training_section.get("max_steps", payload.get("max_steps")), 100),
            device=device,
            reward_weights=RewardWeightsConfig.from_mapping(payload.get("reward_weights")),
            ppo=PPOConfig.from_mapping(ppo_section),
            grpo=GRPOConfig.from_mapping(grpo_section),
            model=ModelConfig.from_mapping(model_section),
            adversarial_batch=_to_int(
                training_section.get("adversarial_batch", payload.get("adversarial_batch")),
                2,
            ),
            confidence_threshold=_to_float(
                training_section.get("confidence_threshold", payload.get("confidence_threshold")),
                0.75,
            ),
            use_dual_path=bool(
                training_section.get("use_dual_path", payload.get("use_dual_path", False))
            ),
            honesty=HonestyConfig.from_mapping(payload.get("honesty")),
            deception=DeceptionConfig.from_mapping(payload.get("deception")),
            dataset=DatasetConfig.from_mapping(payload.get("dataset")),
            output=OutputConfig.from_mapping(payload.get("output")),
        )

    def dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["honesty"] = self.honesty.dict()
        payload["deception"] = self.deception.dict()
        payload["grpo"] = self.grpo.dict()
        payload["dataset"] = self.dataset.dict()
        payload["output"] = self.output.dict()
        return payload


def load_training_config(path: str | Path) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] | None = yaml.safe_load(handle)  # type: ignore[assignment]
    return TrainingConfig.from_mapping(payload)


__all__ = [
    "RewardWeightsConfig",
    "PPOConfig",
    "CircuitTracerSamplingConfig",
    "HallucinationPenaltyConfig",
    "GRPOConfig",
    "ModelConfig",
    "TrainingConfig",
    "PPOSection",
    "ModelConfig",
]

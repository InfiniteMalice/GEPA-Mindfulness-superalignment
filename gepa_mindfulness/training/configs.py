"""Backward-compatible configuration helpers used by legacy tests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a float-compatible value, got {value!r}") from exc


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an int-compatible value, got {value!r}") from exc


def _merge_mapping(
    primary: Mapping[str, Any] | None,
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(fallback)
    if isinstance(primary, Mapping):
        merged.update(primary)
    return merged


@dataclass
class RewardWeightsConfig:
    alpha: float = 0.25
    beta: float = 0.35
    gamma: float = 0.35
    delta: float = 0.05

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
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RewardWeightsConfig":
        payload = payload or {}
        return cls(
            alpha=_to_float(payload.get("alpha"), 0.25),
            beta=_to_float(payload.get("beta"), 0.35),
            gamma=_to_float(payload.get("gamma"), 0.35),
            delta=_to_float(payload.get("delta"), 0.05),
        )

    def dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class HonestyConfig:
    uncertainty_threshold: float = 0.75
    idk_bonus: float = 1.0
    calibration_bonus_weight: float = 0.5
    uncertainty_marker_bonus: float = 0.3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "HonestyConfig":
        payload = payload or {}
        return cls(
            uncertainty_threshold=_to_float(payload.get("uncertainty_threshold"), 0.75),
            idk_bonus=_to_float(payload.get("idk_bonus"), 1.0),
            calibration_bonus_weight=_to_float(payload.get("calibration_bonus_weight"), 0.5),
            uncertainty_marker_bonus=_to_float(payload.get("uncertainty_marker_bonus"), 0.3),
        )

    def dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class DeceptionConfig:
    detect: bool = True
    log_fingerprints: bool = True
    fingerprint_dir: str = "runs/fingerprints"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DeceptionConfig":
        payload = payload or {}
        return cls(
            detect=bool(payload.get("detect", True)),
            log_fingerprints=bool(payload.get("log_fingerprints", True)),
            fingerprint_dir=str(payload.get("fingerprint_dir", "runs/fingerprints")),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutputConfig:
    checkpoint_dir: str = "runs/checkpoints"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "OutputConfig":
        payload = payload or {}
        return cls(checkpoint_dir=str(payload.get("checkpoint_dir", "runs/checkpoints")))

    def dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DatasetConfig:
    train_path: str = ""
    validation_path: str | None = None
    format: str = "jsonl"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DatasetConfig":
        payload = payload or {}
        validation = payload.get("validation_path")
        if validation is not None:
            validation = str(validation)
        return cls(
            train_path=str(payload.get("train_path", "")),
            validation_path=validation,
            format=str(payload.get("format", "jsonl")),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitTracerSamplingConfig:
    trace_frequency: float = 1.0
    trace_strategy: str = "all"

    def __post_init__(self) -> None:
        self.trace_frequency = max(0.0, min(1.0, float(self.trace_frequency)))
        self.trace_strategy = str(self.trace_strategy)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CircuitTracerSamplingConfig":
        payload = payload or {}
        return cls(
            trace_frequency=_to_float(payload.get("trace_frequency"), 1.0),
            trace_strategy=str(payload.get("trace_strategy", "all")),
        )

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

    confident_wrong = _to_float(payload.get("confident_wrong_penalty"), -2.0)
    uncertain_wrong = _to_float(payload.get("uncertain_wrong_penalty"), -0.5)
    appropriate_abstention = _to_float(payload.get("appropriate_abstention_reward"), 0.5)
    lazy_abstention = _to_float(payload.get("lazy_abstention_penalty"), -0.2)
    confidence_thresh = _to_float(payload.get("confidence_threshold"), 0.75)

    return cls(
        confidence_threshold=confidence_thresh,
        confident_wrong_penalty=confident_wrong,
        uncertain_wrong_penalty=uncertain_wrong,
        appropriate_abstention_reward=appropriate_abstention,
        lazy_abstention_penalty=lazy_abstention,
    )


def dict(self) -> dict[str, float]:
    return asdict(self)


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    batch_size: int = 1
    mini_batch_size: int = 1
    ppo_epochs: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PPOConfig":
        payload = payload or {}
        return cls(
            learning_rate=_to_float(payload.get("learning_rate"), 1e-5),
            batch_size=_to_int(payload.get("batch_size"), 1),
            mini_batch_size=_to_int(payload.get("mini_batch_size"), 1),
            ppo_epochs=_to_int(payload.get("ppo_epochs"), 1),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    policy_model: str = "demo-model"
    reward_model: str | None = None
    device: str = "cpu"
    vllm_engine: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ModelConfig":
        payload = payload or {}
        reward_model = payload.get("reward_model")
        if reward_model is not None:
            reward_model = str(reward_model)
        vllm_engine = payload.get("vllm_engine")
        if vllm_engine is not None:
            vllm_engine = str(vllm_engine)
        return cls(
            policy_model=str(payload.get("policy_model", "demo-model")),
            reward_model=reward_model,
            device=str(payload.get("device", "cpu")),
            vllm_engine=vllm_engine,
        )

    def dict(self) -> dict[str, Any]:
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
        circuit_payload = {
            "trace_frequency": payload.get("trace_frequency"),
            "trace_strategy": payload.get("trace_strategy"),
        }
        sampling = CircuitTracerSamplingConfig.from_mapping(
            _merge_mapping(payload.get("circuit_tracer"), circuit_payload)
        )
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
class TrainingConfig:
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    adversarial_batch: int = 2
    confidence_threshold: float = 0.75
    use_dual_path: bool = False
    honesty: HonestyConfig = field(default_factory=HonestyConfig)
    deception: DeceptionConfig = field(default_factory=DeceptionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    max_steps: int = 100
    device: str = "cpu"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "TrainingConfig":
        payload = payload or {}

        training_section = payload.get("training")
        if not isinstance(training_section, Mapping):
            training_section = {}

        model_section = payload.get("model")
        if not isinstance(model_section, Mapping):
            model_section = {}

        device_value = payload.get("device", model_section.get("device"))
        device = str(device_value) if device_value is not None else "cpu"
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
            dataset=DatasetConfig.from_mapping(payload.get("dataset")),
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
            output=OutputConfig.from_mapping(payload.get("output")),
        )

    def dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reward_weights"] = self.reward_weights.dict()
        payload["ppo"] = self.ppo.dict()
        payload["grpo"] = self.grpo.dict()
        payload["model"] = self.model.dict()
        payload["dataset"] = self.dataset.dict()
        payload["honesty"] = self.honesty.dict()
        payload["deception"] = self.deception.dict()
        payload["output"] = self.output.dict()
        return payload


def load_training_config(path: str | Path) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload: Mapping[str, Any] | None = yaml.safe_load(handle)
    return TrainingConfig.from_mapping(payload)


__all__ = [
    "RewardWeightsConfig",
    "PPOConfig",
    "CircuitTracerSamplingConfig",
    "HallucinationPenaltyConfig",
    "GRPOConfig",
    "ModelConfig",
    "TrainingConfig",
    "HonestyConfig",
    "DeceptionConfig",
    "OutputConfig",
    "DatasetConfig",
    "load_training_config",
]

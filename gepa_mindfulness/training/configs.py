"""Configuration models for the training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

import yaml


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


_DEFAULT_UNCERTAINTY_MARKERS = ["uncertain", "not sure", "unclear", "might", "could"]


def _default_uncertainty_markers() -> List[str]:
    return _DEFAULT_UNCERTAINTY_MARKERS.copy()


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
class HonestyConfig:
    uncertainty_threshold: float = 0.75
    idk_bonus: float = 1.0
    calibration_bonus_weight: float = 0.5
    uncertainty_marker_bonus: float = 0.3
    uncertainty_markers: List[str] = field(default_factory=_default_uncertainty_markers)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "HonestyConfig":
        payload = payload or {}
        markers = payload.get("uncertainty_markers")
        if isinstance(markers, str):
            markers = [marker.strip() for marker in markers.split(",") if marker.strip()]
        elif not isinstance(markers, list):
            markers = None

        return cls(
            uncertainty_threshold=_to_float(payload.get("uncertainty_threshold"), 0.75),
            idk_bonus=_to_float(payload.get("idk_bonus"), 1.0),
            calibration_bonus_weight=_to_float(
                payload.get("calibration_bonus_weight", payload.get("calibration_weight", 0.5)),
                0.5,
            ),
            uncertainty_marker_bonus=_to_float(
                payload.get(
                    "uncertainty_marker_bonus",
                    payload.get("explicit_uncertainty_bonus", 0.3),
                ),
                0.3,
            ),
            uncertainty_markers=markers or _DEFAULT_UNCERTAINTY_MARKERS.copy(),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DeceptionAblationConfig:
    enabled: bool = False
    config_path: str | None = None
    analysis_script: str | None = None
    ablation_script: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DeceptionAblationConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            config_path=payload.get("config_path"),
            analysis_script=payload.get("analysis_script"),
            ablation_script=payload.get("ablation_script"),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DeceptionConfig:
    detect: bool = True
    detection_method: str = "heuristic"
    log_fingerprints: bool = True
    fingerprint_dir: str = "runs/deception_fingerprints/"
    save_circuits: bool = True
    save_full_traces: bool = True
    alert_threshold: float = 0.7
    apply_penalty: bool = False
    penalty_weight: float = 0.0
    fingerprint_filename: str = "fingerprints.jsonl"
    ablation: DeceptionAblationConfig = field(default_factory=DeceptionAblationConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DeceptionConfig":
        payload = payload or {}
        ablation = DeceptionAblationConfig.from_mapping(payload.get("ablation"))
        return cls(
            detect=bool(payload.get("detect", True)),
            detection_method=str(payload.get("detection_method", "heuristic")),
            log_fingerprints=bool(payload.get("log_fingerprints", True)),
            fingerprint_dir=str(payload.get("fingerprint_dir", "runs/deception_fingerprints/")),
            save_circuits=bool(payload.get("save_circuits", True)),
            save_full_traces=bool(payload.get("save_full_traces", True)),
            alert_threshold=_to_float(payload.get("alert_threshold"), 0.7),
            apply_penalty=bool(payload.get("apply_penalty", False)),
            penalty_weight=_to_float(payload.get("penalty_weight", 0.0), 0.0),
            fingerprint_filename=str(payload.get("fingerprint_filename", "fingerprints.jsonl")),
            ablation=ablation,
        )

    def dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ablation"] = self.ablation.dict()
        return result

    def get(self, key: str, default: Any = None) -> Any:
        if key == "ablation":
            return self.ablation
        return getattr(self, key, default)


@dataclass
class DatasetConfig:
    path: str = ""
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DatasetConfig":
        payload = payload or {}
        return cls(
            path=str(payload.get("path", "")),
            train_split=_to_float(payload.get("train_split"), 0.7),
            val_split=_to_float(payload.get("val_split"), 0.15),
            test_split=_to_float(payload.get("test_split"), 0.15),
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutputConfig:
    checkpoint_dir: str = "runs/phi3_trained/"
    log_interval: int = 10
    save_interval: int = 100

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "OutputConfig":
        payload = payload or {}
        return cls(
            checkpoint_dir=str(payload.get("checkpoint_dir", "runs/phi3_trained/")),
            log_interval=_to_int(payload.get("log_interval"), 10),
            save_interval=_to_int(payload.get("save_interval"), 100),
        )

    def dict(self) -> dict[str, Any]:
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
        policy_model = payload.get("policy_model")
        if policy_model is None:
            policy_model = payload.get("name")
        reward_model = payload.get("reward_model")
        if reward_model is None:
            reward_model = payload.get("reward_model_name")
        return cls(
            policy_model=str(policy_model or "gpt2"),
            reward_model=str(reward_model or "distilbert-base-uncased"),
            vllm_engine=payload.get("vllm_engine"),
        )


def _merge_mapping(
    base: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, Mapping):
        merged.update(base)
    if isinstance(overrides, Mapping):
        merged.update(overrides)
    return merged


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
    use_dual_path: bool = False
    honesty: HonestyConfig = field(default_factory=HonestyConfig)
    deception: DeceptionConfig = field(default_factory=DeceptionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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

        return cls(
            seed=_to_int(training_section.get("seed", payload.get("seed")), 42),
            max_steps=_to_int(training_section.get("max_steps", payload.get("max_steps")), 100),
            device=device,
            reward_weights=RewardWeightsConfig.from_mapping(payload.get("reward_weights")),
            ppo=PPOConfig.from_mapping(ppo_section),
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
    "ModelConfig",
    "TrainingConfig",
    "load_training_config",
]

"""Strict canonical runtime configuration for portable PyTorch RL runs."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised without optional dependency
    yaml = None

_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    return _mapping(value, name)


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {', '.join(unknown)}")


def _string(payload: Mapping[str, Any], key: str, default: str, name: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{name}.{key} must be a string")
    return value


def _integer(payload: Mapping[str, Any], key: str, default: int, name: str) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}.{key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str, default: float, name: str) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}.{key} must be finite")
    return number


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution backend and device selection."""

    backend: str = "pytorch"
    device: str = "cpu"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeConfig":
        payload = _mapping(payload, "runtime")
        _reject_unknown(payload, {"backend", "device"}, "runtime")
        backend = _string(payload, "backend", "pytorch", "runtime")
        device = _string(payload, "device", "cpu", "runtime")
        if backend != "pytorch":
            raise ValueError("runtime.backend must be 'pytorch'")
        if device != "cpu" and not _CUDA_DEVICE.fullmatch(device):
            raise ValueError("runtime.device must be 'cpu', 'cuda', or 'cuda:<index>'")
        return cls(backend=backend, device=device)


@dataclass(frozen=True)
class PolicyConfig:
    """Policy model settings shared by PPO and GRPO."""

    model_name: str = "demo-model"
    max_new_tokens: int = 256

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyConfig":
        payload = _mapping(payload, "policy")
        _reject_unknown(payload, {"model_name", "max_new_tokens"}, "policy")
        model_name = _string(payload, "model_name", "demo-model", "policy")
        max_new_tokens = _integer(payload, "max_new_tokens", 256, "policy")
        if not model_name:
            raise ValueError("policy.model_name must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("policy.max_new_tokens must be positive")
        return cls(model_name=model_name, max_new_tokens=max_new_tokens)


@dataclass(frozen=True)
class AlgorithmConfig:
    """PPO or GRPO optimization settings."""

    name: str = "ppo"
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    group_size: int = 8
    kl_coef: float = 0.05
    clip_range: float = 0.2
    value_coef: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AlgorithmConfig":
        payload = _mapping(payload, "algorithm")
        _reject_unknown(
            payload,
            {
                "name",
                "learning_rate",
                "batch_size",
                "gradient_accumulation_steps",
                "max_steps",
                "group_size",
                "kl_coef",
                "clip_range",
                "value_coef",
            },
            "algorithm",
        )
        name = _string(payload, "name", "ppo", "algorithm")
        learning_rate = _number(payload, "learning_rate", 1e-5, "algorithm")
        batch_size = _integer(payload, "batch_size", 1, "algorithm")
        accumulation = _integer(payload, "gradient_accumulation_steps", 1, "algorithm")
        max_steps = _integer(payload, "max_steps", 100, "algorithm")
        group_size = _integer(payload, "group_size", 8, "algorithm")
        kl_coef = _number(payload, "kl_coef", 0.05, "algorithm")
        clip_range = _number(payload, "clip_range", 0.2, "algorithm")
        value_coef = _number(payload, "value_coef", 0.1, "algorithm")
        if name not in {"ppo", "grpo"}:
            raise ValueError("algorithm.name must be 'ppo' or 'grpo'")
        if learning_rate <= 0 or batch_size <= 0 or accumulation <= 0 or max_steps <= 0:
            raise ValueError(
                "algorithm learning_rate, batch_size, accumulation, and max_steps "
                "must be positive"
            )
        if name == "grpo" and group_size < 2:
            raise ValueError("algorithm.group_size must be at least 2 for GRPO")
        if kl_coef < 0 or clip_range <= 0 or value_coef < 0:
            raise ValueError("algorithm coefficients must be non-negative and clip_range positive")
        return cls(
            name=name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            max_steps=max_steps,
            group_size=group_size,
            kl_coef=kl_coef,
            clip_range=clip_range,
            value_coef=value_coef,
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward composition settings."""

    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.2
    delta: float = 0.2

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewardConfig":
        payload = _mapping(payload, "reward")
        _reject_unknown(payload, {"weights"}, "reward")
        weights = _section(payload, "weights")
        _reject_unknown(weights, {"alpha", "beta", "gamma", "delta"}, "reward.weights")
        values = {
            name: _number(weights, name, default, "reward.weights")
            for name, default in (("alpha", 0.3), ("beta", 0.3), ("gamma", 0.2), ("delta", 0.2))
        }
        if any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
            raise ValueError("reward.weights must be non-negative with positive total mass")
        return cls(**values)


@dataclass(frozen=True)
class DatasetConfig:
    """Input dataset location and record format."""

    train_path: str = ""
    validation_path: str | None = None
    format: str = "jsonl"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetConfig":
        payload = _mapping(payload, "dataset")
        _reject_unknown(payload, {"train_path", "validation_path", "format"}, "dataset")
        train_path = _string(payload, "train_path", "", "dataset")
        validation_path = payload.get("validation_path")
        if validation_path is not None and not isinstance(validation_path, str):
            raise TypeError("dataset.validation_path must be a string or null")
        data_format = _string(payload, "format", "jsonl", "dataset")
        if data_format not in {"jsonl", "text"}:
            raise ValueError("dataset.format must be 'jsonl' or 'text'")
        return cls(train_path=train_path, validation_path=validation_path, format=data_format)


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint output and cadence settings."""

    output_dir: str = "runs/default"
    save_steps: int = 100

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckpointConfig":
        payload = _mapping(payload, "checkpoint")
        _reject_unknown(payload, {"output_dir", "save_steps"}, "checkpoint")
        output_dir = _string(payload, "output_dir", "runs/default", "checkpoint")
        save_steps = _integer(payload, "save_steps", 100, "checkpoint")
        if not output_dir:
            raise ValueError("checkpoint.output_dir must not be empty")
        if save_steps <= 0:
            raise ValueError("checkpoint.save_steps must be positive")
        return cls(output_dir=output_dir, save_steps=save_steps)


@dataclass(frozen=True)
class LoggingConfig:
    """Run log destination and severity threshold."""

    log_dir: str = "runs/logs"
    level: str = "INFO"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LoggingConfig":
        payload = _mapping(payload, "logging")
        _reject_unknown(payload, {"log_dir", "level"}, "logging")
        log_dir = _string(payload, "log_dir", "runs/logs", "logging")
        level = _string(payload, "level", "INFO", "logging")
        if not log_dir:
            raise ValueError("logging.log_dir must not be empty")
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging.level must be a standard uppercase logging level")
        return cls(log_dir=log_dir, level=level)


@dataclass(frozen=True)
class RLRunConfig:
    """Complete immutable configuration for a canonical PyTorch RL run."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RLRunConfig":
        payload = _mapping(payload, "configuration")
        _reject_unknown(
            payload,
            {
                "runtime",
                "policy",
                "algorithm",
                "reward",
                "dataset",
                "checkpoint",
                "logging",
                "seed",
            },
            "configuration",
        )
        seed = _integer(payload, "seed", 42, "configuration")
        return cls(
            runtime=RuntimeConfig.from_mapping(_section(payload, "runtime")),
            policy=PolicyConfig.from_mapping(_section(payload, "policy")),
            algorithm=AlgorithmConfig.from_mapping(_section(payload, "algorithm")),
            reward=RewardConfig.from_mapping(_section(payload, "reward")),
            dataset=DatasetConfig.from_mapping(_section(payload, "dataset")),
            checkpoint=CheckpointConfig.from_mapping(_section(payload, "checkpoint")),
            logging=LoggingConfig.from_mapping(_section(payload, "logging")),
            seed=seed,
        )


def translate_legacy_config(payload: Mapping[str, Any]) -> RLRunConfig:
    """Translate a legacy flat or nested configuration into the canonical schema."""
    warnings.warn(
        "Legacy RL configuration is deprecated; use the canonical runtime sections instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _translate_legacy_config(payload)


def _translate_legacy_config(payload: Mapping[str, Any]) -> RLRunConfig:
    """Translate legacy fields without attributing warnings to an internal frame."""
    payload = _mapping(payload, "legacy configuration")
    training = _optional_mapping(payload.get("training"), "training")
    grpo = _optional_mapping(payload.get("grpo"), "grpo")
    ppo = _optional_mapping(payload.get("ppo"), "ppo")
    model = _optional_mapping(payload.get("model"), "model")
    dataset = _optional_mapping(payload.get("dataset"), "dataset")
    output = _optional_mapping(payload.get("output"), "output")
    trainer_name = payload.get("trainer_type")
    if trainer_name is None:
        trainer_name = "grpo" if "grpo" in payload else "ppo"
    algorithm_source = _merge_legacy(training, grpo if trainer_name == "grpo" else ppo, payload)
    reward_source = _optional_mapping(algorithm_source.get("reward_weights"), "reward_weights")
    if not reward_source:
        reward_source = _optional_mapping(payload.get("reward_weights"), "reward_weights")
    policy_name = payload.get(
        "model_name",
        model.get("policy_model", model.get("name", "demo-model")),
    )
    device = payload.get("device", model.get("device", "cpu"))
    canonical = {
        "runtime": {"backend": "pytorch", "device": device},
        "policy": {
            "model_name": policy_name,
            "max_new_tokens": _legacy_integer(algorithm_source.get("max_new_tokens"), 256),
        },
        "algorithm": {
            "name": trainer_name,
            "learning_rate": _legacy_number(algorithm_source.get("learning_rate"), 1e-5),
            "batch_size": _legacy_integer(algorithm_source.get("batch_size"), 1),
            "gradient_accumulation_steps": _legacy_integer(
                algorithm_source.get("gradient_accumulation_steps"),
                1,
            ),
            "max_steps": _legacy_integer(
                algorithm_source.get("max_steps", payload.get("max_steps")),
                100,
            ),
            "group_size": _legacy_integer(algorithm_source.get("group_size"), 8),
            "kl_coef": _legacy_number(algorithm_source.get("kl_coef"), 0.05),
            "clip_range": _legacy_number(algorithm_source.get("clip_range"), 0.2),
            "value_coef": _legacy_number(algorithm_source.get("value_coef"), 0.1),
        },
        "reward": {"weights": reward_source},
        "dataset": {
            "train_path": dataset.get(
                "train_path",
                dataset.get("path", payload.get("dataset_path", "")),
            ),
            "validation_path": dataset.get("validation_path"),
            "format": dataset.get("format", "jsonl"),
        },
        "checkpoint": {
            "output_dir": payload.get(
                "output_dir",
                output.get("checkpoint_dir", "runs/default"),
            ),
            "save_steps": _legacy_integer(payload.get("save_steps"), 100),
        },
        "logging": {
            "log_dir": payload.get("log_dir", "runs/logs"),
            "level": payload.get("log_level", "INFO"),
        },
        "seed": _legacy_integer(training.get("seed", payload.get("seed")), 42),
    }
    return RLRunConfig.from_mapping(canonical)


def load_rl_config(path: str | Path) -> RLRunConfig:
    """Load a canonical configuration file or translate an explicitly legacy one."""
    raw = Path(path).read_text(encoding="utf-8")
    payload = yaml.safe_load(raw) if yaml is not None else _load_json_mapping(raw)
    payload = _mapping(payload, "configuration file")
    if _is_canonical(payload):
        return RLRunConfig.from_mapping(payload)
    warnings.warn(
        "Legacy RL configuration is deprecated; use the canonical runtime sections instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _translate_legacy_config(payload)


def _is_canonical(payload: Mapping[str, Any]) -> bool:
    legacy_keys = {
        "device",
        "grpo",
        "model",
        "model_name",
        "output",
        "output_dir",
        "ppo",
        "reward_weights",
        "trainer_type",
        "training",
    }
    if set(payload).intersection(legacy_keys):
        return False
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping) and "path" in dataset:
        return False
    return True


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, name)


def _merge_legacy(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged.update(payload)
    return merged


def _legacy_integer(value: Any, default: int) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _legacy_number(value: Any, default: float) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _load_json_mapping(raw: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("pyyaml is required to load non-JSON RL configuration files") from exc
    return _mapping(payload, "configuration file")


__all__ = [
    "AlgorithmConfig",
    "CheckpointConfig",
    "DatasetConfig",
    "LoggingConfig",
    "PolicyConfig",
    "RLRunConfig",
    "RewardConfig",
    "RuntimeConfig",
    "load_rl_config",
    "translate_legacy_config",
]

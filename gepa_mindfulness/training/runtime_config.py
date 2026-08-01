"""Strict canonical runtime configuration for local PyTorch RL runs."""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised without optional dependency
    yaml = None

_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?$")
_CANONICAL_DATASET_KEYS = {"format", "train_path", "validation_path"}
_LEGACY_DATASET_KEYS = {"path", "test_split", "train_split", "val_split"}
ZeroVariancePolicy = Literal["zero", "center_only", "skip"]
Precision = Literal["fp32", "fp16", "bf16"]
DistributedStrategy = Literal["none", "ddp", "fsdp"]


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


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True)
class DistributedRuntimeConfig:
    """Validated process topology for optional PyTorch data/model parallelism."""

    strategy: DistributedStrategy = "none"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    sharded_optimizer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str):
            raise TypeError("runtime.distributed.strategy must be a string")
        if self.strategy not in {"none", "ddp", "fsdp"}:
            raise ValueError("runtime.distributed.strategy must be 'none', 'ddp', or 'fsdp'")
        for name in ("world_size", "rank", "local_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"runtime.distributed.{name} must be an integer")
        if not isinstance(self.sharded_optimizer, bool):
            raise TypeError("runtime.distributed.sharded_optimizer must be a boolean")
        if self.world_size <= 0:
            raise ValueError("runtime.distributed.world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("runtime.distributed.rank must be within world_size")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError("runtime.distributed.local_rank must be within world_size")
        if self.strategy == "none":
            if (self.world_size, self.rank, self.local_rank) != (1, 0, 0):
                raise ValueError("strategy='none' requires world_size=1, rank=0, local_rank=0")
            if self.sharded_optimizer:
                raise ValueError("strategy='none' cannot use a sharded optimizer")
        else:
            if self.world_size < 2:
                raise ValueError("distributed strategies require world_size of at least 2")
            if self.strategy != "fsdp" and self.sharded_optimizer:
                raise ValueError("sharded_optimizer is supported only with strategy='fsdp'")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DistributedRuntimeConfig":
        payload = _mapping(payload, "runtime.distributed")
        _reject_unknown(
            payload,
            {"strategy", "world_size", "rank", "local_rank", "sharded_optimizer"},
            "runtime.distributed",
        )
        strategy = _string(payload, "strategy", "none", "runtime.distributed")
        sharded_optimizer = payload.get("sharded_optimizer", False)
        if not isinstance(sharded_optimizer, bool):
            raise TypeError("runtime.distributed.sharded_optimizer must be a boolean")
        return cls(
            strategy=cast(DistributedStrategy, strategy),
            world_size=_topology_integer(
                payload,
                "world_size",
                1,
                environment_name="WORLD_SIZE",
            ),
            rank=_topology_integer(payload, "rank", 0, environment_name="RANK"),
            local_rank=_topology_integer(
                payload,
                "local_rank",
                0,
                environment_name="LOCAL_RANK",
            ),
            sharded_optimizer=sharded_optimizer,
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution backend, device, and numeric precision selection."""

    backend: str = "pytorch"
    device: str = "cpu"
    precision: Precision = "fp32"
    distributed: DistributedRuntimeConfig = field(default_factory=DistributedRuntimeConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("runtime.backend must be a string")
        if not isinstance(self.device, str):
            raise TypeError("runtime.device must be a string")
        if not isinstance(self.precision, str):
            raise TypeError("runtime.precision must be a string")
        if not isinstance(self.distributed, DistributedRuntimeConfig):
            raise TypeError("runtime.distributed must be a DistributedRuntimeConfig")
        if self.backend not in {"pytorch", "cuda", "llama-cpp-vulkan"}:
            raise ValueError("runtime.backend must be 'pytorch', 'cuda', or 'llama-cpp-vulkan'")
        if self.device != "cpu" and not _CUDA_DEVICE.fullmatch(self.device):
            raise ValueError("runtime.device must be 'cpu', 'cuda', or 'cuda:<index>'")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("runtime.precision must be 'fp32', 'fp16', or 'bf16'")
        if self.backend == "cuda" and not _CUDA_DEVICE.fullmatch(self.device):
            raise ValueError("runtime.backend='cuda' requires a CUDA device selector")
        if self.backend == "llama-cpp-vulkan" and self.device != "cpu":
            raise ValueError("runtime.backend='llama-cpp-vulkan' requires runtime.device='cpu'")
        if self.backend == "llama-cpp-vulkan" and self.precision != "fp32":
            raise ValueError("runtime.backend='llama-cpp-vulkan' requires runtime.precision='fp32'")
        if self.precision != "fp32" and not _CUDA_DEVICE.fullmatch(self.device):
            raise ValueError("mixed precision requires a CUDA device selector")
        if self.distributed.strategy != "none":
            if self.backend != "cuda":
                raise ValueError("distributed strategies require runtime.backend='cuda'")
            expected_device = f"cuda:{self.distributed.local_rank}"
            if self.device != expected_device:
                raise ValueError(
                    "runtime.device must match runtime.distributed.local_rank "
                    f"({expected_device})"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeConfig":
        payload = _mapping(payload, "runtime")
        _reject_unknown(payload, {"backend", "device", "precision", "distributed"}, "runtime")
        backend = _string(payload, "backend", "pytorch", "runtime")
        precision = _string(payload, "precision", "fp32", "runtime")
        distributed = DistributedRuntimeConfig.from_mapping(_section(payload, "distributed"))
        device = _string(payload, "device", "cpu", "runtime")
        if device == "cuda:LOCAL_RANK":
            device = f"cuda:{distributed.local_rank}"
        return cls(
            backend=backend,
            device=device,
            precision=cast(Precision, precision),
            distributed=distributed,
        )


def _topology_integer(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    *,
    environment_name: str,
) -> int:
    value = payload.get(key, default)
    if value != environment_name:
        return _integer(payload, key, default, "runtime.distributed")
    raw = os.environ.get(environment_name)
    if raw is None:
        raise ValueError(f"{environment_name} environment variable is required")
    if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
        raise ValueError(f"{environment_name} must be a non-negative base-10 integer")
    return int(raw)


@dataclass(frozen=True)
class PolicyConfig:
    """Policy model settings shared by PPO and GRPO."""

    model_name: str = "demo-model"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str):
            raise TypeError("policy.model_name must be a string")
        if not self.model_name:
            raise ValueError("policy.model_name must not be empty")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("policy.max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("policy.max_new_tokens must be positive")
        if not isinstance(self.do_sample, bool):
            raise TypeError("policy.do_sample must be a boolean")
        _finite_number(self.temperature, "policy.temperature", positive=True)
        top_p = _finite_number(self.top_p, "policy.top_p", positive=True)
        if top_p > 1.0:
            raise ValueError("policy.top_p must be in (0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyConfig":
        payload = _mapping(payload, "policy")
        _reject_unknown(
            payload,
            {"model_name", "max_new_tokens", "do_sample", "temperature", "top_p"},
            "policy",
        )
        model_name = _string(payload, "model_name", "demo-model", "policy")
        max_new_tokens = _integer(payload, "max_new_tokens", 256, "policy")
        do_sample = payload.get("do_sample", True)
        if not isinstance(do_sample, bool):
            raise TypeError("policy.do_sample must be a boolean")
        temperature = _number(payload, "temperature", 1.0, "policy")
        top_p = _number(payload, "top_p", 1.0, "policy")
        return cls(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )


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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    group_normalization_epsilon: float = 1e-8
    zero_variance_policy: ZeroVariancePolicy = "zero"
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("algorithm.name must be a string")
        if self.name not in {"ppo", "grpo"}:
            raise ValueError("algorithm.name must be 'ppo' or 'grpo'")
        _finite_number(self.learning_rate, "algorithm.learning_rate", positive=True)
        for name in ("batch_size", "gradient_accumulation_steps", "max_steps", "group_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"algorithm.{name} must be an integer")
            if value <= 0:
                raise ValueError(f"algorithm.{name} must be positive")
        if self.name == "grpo" and self.group_size < 2:
            raise ValueError("algorithm.group_size must be at least 2 for GRPO")
        for name in ("kl_coef", "value_coef"):
            if _finite_number(getattr(self, name), f"algorithm.{name}") < 0.0:
                raise ValueError(f"algorithm.{name} must be non-negative")
        _finite_number(self.clip_range, "algorithm.clip_range", positive=True)
        for name in ("gamma", "gae_lambda"):
            value = _finite_number(getattr(self, name), f"algorithm.{name}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"algorithm.{name} must be in [0, 1]")
        _finite_number(
            self.group_normalization_epsilon,
            "algorithm.group_normalization_epsilon",
            positive=True,
        )
        if self.zero_variance_policy not in {"zero", "center_only", "skip"}:
            raise ValueError(
                "algorithm.zero_variance_policy must be 'zero', 'center_only', or 'skip'"
            )
        if self.max_grad_norm is not None:
            _finite_number(self.max_grad_norm, "algorithm.max_grad_norm", positive=True)

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
                "gamma",
                "gae_lambda",
                "group_normalization_epsilon",
                "zero_variance_policy",
                "max_grad_norm",
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
        gamma = _number(payload, "gamma", 0.99, "algorithm")
        gae_lambda = _number(payload, "gae_lambda", 0.95, "algorithm")
        normalization_epsilon = _number(
            payload,
            "group_normalization_epsilon",
            1e-8,
            "algorithm",
        )
        zero_variance_policy = _string(
            payload,
            "zero_variance_policy",
            "zero",
            "algorithm",
        )
        max_grad_norm = payload.get("max_grad_norm", 1.0)
        if max_grad_norm is not None:
            max_grad_norm = _number(payload, "max_grad_norm", 1.0, "algorithm")
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
            gamma=gamma,
            gae_lambda=gae_lambda,
            group_normalization_epsilon=normalization_epsilon,
            zero_variance_policy=cast(ZeroVariancePolicy, zero_variance_policy),
            max_grad_norm=max_grad_norm,
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward composition settings."""

    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.2
    delta: float = 0.2
    overlay_weight: float = 0.0
    integrity_overlay_enabled: bool = False

    def __post_init__(self) -> None:
        values = {
            name: _finite_number(getattr(self, name), f"reward.weights.{name}")
            for name in ("alpha", "beta", "gamma", "delta")
        }
        if any(value < 0.0 for value in values.values()) or math.fsum(values.values()) <= 0.0:
            raise ValueError("reward.weights must be non-negative with positive total mass")
        overlay_weight = _finite_number(self.overlay_weight, "reward.overlay_weight")
        if overlay_weight < 0.0:
            raise ValueError("reward.overlay_weight must be non-negative")
        if not isinstance(self.integrity_overlay_enabled, bool):
            raise TypeError("reward.integrity_overlay_enabled must be a boolean")
        if self.integrity_overlay_enabled and overlay_weight <= 0.0:
            raise ValueError("enabled reward integrity overlay requires positive overlay_weight")
        if not self.integrity_overlay_enabled and overlay_weight != 0.0:
            raise ValueError("disabled reward integrity overlay requires overlay_weight=0")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewardConfig":
        payload = _mapping(payload, "reward")
        _reject_unknown(
            payload,
            {"weights", "overlay_weight", "integrity_overlay_enabled"},
            "reward",
        )
        weights = _section(payload, "weights")
        _reject_unknown(weights, {"alpha", "beta", "gamma", "delta"}, "reward.weights")
        values = {
            name: _number(weights, name, default, "reward.weights")
            for name, default in (("alpha", 0.3), ("beta", 0.3), ("gamma", 0.2), ("delta", 0.2))
        }
        overlay_weight = _number(payload, "overlay_weight", 0.0, "reward")
        enabled = payload.get("integrity_overlay_enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("reward.integrity_overlay_enabled must be a boolean")
        return cls(**values, overlay_weight=overlay_weight, integrity_overlay_enabled=enabled)


@dataclass(frozen=True)
class DatasetConfig:
    """Input dataset location and record format."""

    train_path: str = ""
    validation_path: str | None = None
    format: str = "jsonl"

    def __post_init__(self) -> None:
        if not isinstance(self.train_path, str):
            raise TypeError("dataset.train_path must be a string")
        if self.validation_path is not None and not isinstance(self.validation_path, str):
            raise TypeError("dataset.validation_path must be a string or null")
        if not isinstance(self.format, str):
            raise TypeError("dataset.format must be a string")
        if self.format not in {"jsonl", "text"}:
            raise ValueError("dataset.format must be 'jsonl' or 'text'")

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

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, str):
            raise TypeError("checkpoint.output_dir must be a string")
        if not self.output_dir:
            raise ValueError("checkpoint.output_dir must not be empty")
        if isinstance(self.save_steps, bool) or not isinstance(self.save_steps, int):
            raise TypeError("checkpoint.save_steps must be an integer")
        if self.save_steps <= 0:
            raise ValueError("checkpoint.save_steps must be positive")

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

    def __post_init__(self) -> None:
        if not isinstance(self.log_dir, str):
            raise TypeError("logging.log_dir must be a string")
        if not self.log_dir:
            raise ValueError("logging.log_dir must not be empty")
        if not isinstance(self.level, str):
            raise TypeError("logging.level must be a string")
        if self.level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging.level must be a standard uppercase logging level")

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

    def __post_init__(self) -> None:
        sections = {
            "runtime": RuntimeConfig,
            "policy": PolicyConfig,
            "algorithm": AlgorithmConfig,
            "reward": RewardConfig,
            "dataset": DatasetConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
        }
        for name, expected_type in sections.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"configuration.{name} must be a {expected_type.__name__}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("configuration.seed must be an integer")
        if self.algorithm.name == "grpo" and not self.policy.do_sample:
            raise ValueError("GRPO requires stochastic policy generation with do_sample=true")

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
    _validate_legacy_dataset(dataset)
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
            "gamma": _legacy_number(algorithm_source.get("gamma"), 0.99),
            "gae_lambda": _legacy_number(algorithm_source.get("gae_lambda"), 0.95),
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
    canonical_sections = {"algorithm", "checkpoint", "logging", "policy", "reward", "runtime"}
    if set(payload).intersection(canonical_sections):
        return True
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        dataset_keys = set(dataset)
        if dataset_keys.intersection(_CANONICAL_DATASET_KEYS):
            return True
        if "path" in dataset and dataset_keys.issubset(_LEGACY_DATASET_KEYS):
            return False
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


def _validate_legacy_dataset(dataset: Mapping[str, Any]) -> None:
    dataset_keys = set(dataset)
    canonical = sorted(dataset_keys.intersection(_CANONICAL_DATASET_KEYS))
    if canonical:
        raise ValueError(f"legacy dataset contains canonical keys: {', '.join(canonical)}")
    unknown = sorted(dataset_keys.difference(_LEGACY_DATASET_KEYS))
    if unknown:
        raise ValueError(f"legacy dataset contains unknown keys: {', '.join(unknown)}")
    # The runtime consumes the resolved path only; validate split metadata before omitting it.
    for key in ("train_split", "val_split", "test_split"):
        if key not in dataset:
            continue
        value = dataset[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"legacy dataset.{key} must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"legacy dataset.{key} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"legacy dataset.{key} must be between zero and one")


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
    "DistributedRuntimeConfig",
    "DistributedStrategy",
    "LoggingConfig",
    "PolicyConfig",
    "Precision",
    "RLRunConfig",
    "RewardConfig",
    "RuntimeConfig",
    "ZeroVariancePolicy",
    "load_rl_config",
    "translate_legacy_config",
]

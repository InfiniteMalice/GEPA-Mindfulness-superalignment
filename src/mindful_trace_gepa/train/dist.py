"""Distributed training helpers integrating Accelerate and DeepSpeed."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from ..utils.imports import optional_import

Accelerator: Callable[..., Any] | None
DeepSpeedPlugin: Callable[..., Any] | None

_accelerate_mod = optional_import("accelerate")
if _accelerate_mod is not None:
    accel_candidate = getattr(_accelerate_mod, "Accelerator", None)
    Accelerator = accel_candidate if callable(accel_candidate) else None
else:  # pragma: no cover - optional dependency missing
    Accelerator = None

_accelerate_utils = optional_import("accelerate.utils")
if _accelerate_utils is not None:
    deepspeed_candidate = getattr(_accelerate_utils, "DeepSpeedPlugin", None)
    DeepSpeedPlugin = deepspeed_candidate if callable(deepspeed_candidate) else None
else:  # pragma: no cover - optional dependency missing
    DeepSpeedPlugin = None

LOGGER = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    backend: str = "accelerate"
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    deepspeed_config: Optional[str] = None
    zero3_offload: bool = False
    find_unused_parameters: bool = False

    @classmethod
    def from_mapping(cls, mapping: Optional[dict[str, Any]]) -> "DistributedConfig":
        if not mapping:
            return cls()
        return cls(
            backend=mapping.get("backend", "accelerate"),
            mixed_precision=mapping.get("mixed_precision", "no"),
            gradient_accumulation_steps=int(mapping.get("gradient_accumulation_steps", 1)),
            gradient_checkpointing=bool(mapping.get("gradient_checkpointing", False)),
            deepspeed_config=mapping.get("deepspeed_config"),
            zero3_offload=bool(mapping.get("zero3_offload", False)),
            find_unused_parameters=bool(mapping.get("find_unused_parameters", False)),
        )


class NoOpAccelerator:
    """Fallback accelerator used when Accelerate is unavailable."""

    def __init__(self, gradient_accumulation_steps: int = 1) -> None:
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, *objects: Any) -> Tuple[Any, ...]:
        return objects

    def accumulate(self, _model: torch.nn.Module) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:  # pragma: no cover - noop
        return None

    @property
    def is_main_process(self) -> bool:  # pragma: no cover - compatibility
        return True

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def print(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - passthrough
        print(*args, **kwargs)

    def save(self, obj: Any, path: Path) -> None:
        torch.save(obj, path)


def _build_deepspeed_plugin(cfg: DistributedConfig) -> Any | None:
    if DeepSpeedPlugin is None:
        LOGGER.warning("DeepSpeed support requested but accelerate is unavailable.")
        return None
    if not cfg.deepspeed_config:
        LOGGER.warning(
            "DeepSpeed backend requested without config file; falling back to accelerate mode."
        )
        return None
    stage3_offload = cfg.zero3_offload
    return DeepSpeedPlugin(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_clipping=None,
        zero_stage=3 if stage3_offload else None,
        offload_optimizer_device="cpu" if stage3_offload else None,
        offload_param_device="cpu" if stage3_offload else None,
        deepspeed_config_path=cfg.deepspeed_config,
    )


def get_accelerator(config: Optional[dict[str, Any]] = None) -> Any:
    """Return an appropriate accelerator instance for the given configuration."""

    cfg = (
        DistributedConfig.from_mapping((config or {}).get("distributed"))
        if config
        else DistributedConfig()
    )

    if Accelerator is None:
        LOGGER.info("accelerate is not installed; using NoOpAccelerator.")
        return NoOpAccelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)

    kwargs: dict[str, Any] = {
        "mixed_precision": cfg.mixed_precision,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "log_with": None,
    }

    if cfg.backend == "deepspeed":
        plugin = _build_deepspeed_plugin(cfg)
        if plugin is None:
            LOGGER.warning("Falling back to Accelerate defaults due to missing DeepSpeed plugin.")
        else:
            kwargs["deepspeed_plugin"] = plugin
            kwargs["dispatch_batches"] = True
    accelerator = Accelerator(**kwargs)
    return accelerator


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    else:  # pragma: no cover - nothing to do
        LOGGER.debug("Model does not expose gradient checkpointing hooks.")


def wrap_model_optimizer(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    accelerator: Any,
    *,
    gradient_checkpointing: Optional[bool] = None,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer]]:
    """Prepare model and optimizer for distributed training."""

    cfg_checkpoint = gradient_checkpointing
    if cfg_checkpoint is None and hasattr(accelerator, "state"):
        cfg_checkpoint = getattr(
            getattr(accelerator, "state", object()), "gradient_checkpointing", None
        )

    if cfg_checkpoint:
        _enable_gradient_checkpointing(model)

    if hasattr(accelerator, "prepare"):
        prepared = accelerator.prepare(
            *(tuple(obj for obj in (model, optimizer) if obj is not None))
        )
        if optimizer is None:
            return prepared_tuple[0], None

        if len(prepared_tuple) >= 2:
            return prepared_tuple[0], prepared_tuple[1]

        return prepared_tuple[0], optimizer
    return model, optimizer


def save_sharded(adapter: Any, out_dir: Path | str, accelerator: Any) -> Path:
    """Save adapter weights once across distributed workers."""

    out_path = Path(out_dir)
    accelerator.wait_for_everyone()
    if getattr(accelerator, "is_main_process", True):
        out_path.mkdir(parents=True, exist_ok=True)
        if hasattr(adapter, "save_pretrained"):
            adapter.save_pretrained(out_path)
        elif hasattr(adapter, "state_dict"):
            torch.save(adapter.state_dict(), out_path / "adapter.bin")
        else:
            raise AttributeError("Adapter does not implement a supported save method.")
    accelerator.wait_for_everyone()
    return out_path


__all__ = [
    "get_accelerator",
    "wrap_model_optimizer",
    "save_sharded",
    "DistributedConfig",
    "NoOpAccelerator",
]

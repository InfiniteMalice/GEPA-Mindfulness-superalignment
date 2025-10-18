"""Training package exposing configuration helpers and trainer factories."""

from __future__ import annotations

from .config import GRPOConfig, PPOConfig, load_trainer_config

__all__ = [
    "GRPOConfig",
    "PPOConfig",
    "load_trainer_config",
    "cli",
    "GRPOTrainer",
    "PPOTrainer",
]


def __getattr__(name: str):  # pragma: no cover - compatibility shims
    if name == "cli":
        from .cli import cli as exported_cli

        return exported_cli
    if name == "GRPOTrainer":
        from .grpo_trainer import GRPOTrainer as exported_grpo_trainer

        return exported_grpo_trainer
    if name == "PPOTrainer":
        from .ppo_trainer import PPOTrainer as exported_ppo_trainer

        return exported_ppo_trainer
    raise AttributeError(f"module 'gepa_mindfulness.training' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - compatibility shims
    return sorted(__all__)

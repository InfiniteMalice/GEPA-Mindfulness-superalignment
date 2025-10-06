"""Training utilities for GEPA mindfulness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for static analysis
    from .cli import main as cli_main
    from .configs import TrainingConfig, load_training_config
    from .pipeline import TrainingOrchestrator
    from .reporting import SummaryReport, describe_reward, render_summary

__all__ = [
    "TrainingConfig",
    "load_training_config",
    "TrainingOrchestrator",
    "cli_main",
    "SummaryReport",
    "describe_reward",
    "render_summary",
]


def __getattr__(name: str):  # pragma: no cover - simple lazy import helper
    if name in {"TrainingConfig", "load_training_config"}:
        from .configs import TrainingConfig, load_training_config

        mapping = {
            "TrainingConfig": TrainingConfig,
            "load_training_config": load_training_config,
        }
        return mapping[name]
    if name == "TrainingOrchestrator":
        from .pipeline import TrainingOrchestrator

        return TrainingOrchestrator
    if name == "cli_main":
        from .cli import main as cli_main

        return cli_main
    if name in {"SummaryReport", "describe_reward", "render_summary"}:
        from .reporting import SummaryReport, describe_reward, render_summary

        mapping = {
            "SummaryReport": SummaryReport,
            "describe_reward": describe_reward,
            "render_summary": render_summary,
        }
        return mapping[name]
    raise AttributeError(name)

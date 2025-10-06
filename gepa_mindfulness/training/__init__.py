"""Training utilities for GEPA mindfulness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cli import main as cli_main
from .configs import TrainingConfig, load_training_config

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
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
    if name == "TrainingOrchestrator":
        from .pipeline import TrainingOrchestrator

        return TrainingOrchestrator
    if name in {"SummaryReport", "describe_reward", "render_summary"}:
        from .reporting import SummaryReport, describe_reward, render_summary

        mapping = {
            "SummaryReport": SummaryReport,
            "describe_reward": describe_reward,
            "render_summary": render_summary,
        }
        return mapping[name]
    raise AttributeError(name)

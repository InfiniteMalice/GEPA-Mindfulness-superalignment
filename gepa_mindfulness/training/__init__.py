"""Training utilities for GEPA mindfulness."""

from .configs import TrainingConfig, load_training_config

__all__ = [
    "TrainingConfig",
    "load_training_config",
    "TrainingOrchestrator",
    "cli_main",
    "SummaryReport",
    "describe_reward",
    "render_summary",
]


def __getattr__(name: str):
    if name == "cli_main":
        from .cli import main as cli_main

        return cli_main

    if name == "TrainingOrchestrator":
        from .pipeline import TrainingOrchestrator

        return TrainingOrchestrator

    if name in {"SummaryReport", "describe_reward", "render_summary"}:
        from .reporting import (
            SummaryReport,
            describe_reward,
            render_summary,
        )

        mapping = {
            "SummaryReport": SummaryReport,
            "describe_reward": describe_reward,
            "render_summary": render_summary,
        }
        return mapping[name]

    raise AttributeError(f"module 'gepa_mindfulness.training' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

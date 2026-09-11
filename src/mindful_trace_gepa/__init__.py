"""Mindful Trace GEPA extensions."""

from __future__ import annotations

from .action_bound_events import (
    ActionRecord,
    OutcomeObservation,
    PredictionCommit,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_prediction_commit_event,
    make_verification_result_event,
)

__all__ = [
    "ActionRecord",
    "OutcomeObservation",
    "PredictionCommit",
    "VerificationResult",
    "cli_main",
    "main",
    "make_action_event",
    "make_outcome_observation_event",
    "make_prediction_commit_event",
    "make_verification_result_event",
]


def cli_main(argv: list[str] | None = None) -> int:
    """Load the command implementation only when the CLI is invoked."""
    from .cli import main as cli

    return cli(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the package CLI with explicit arguments or process-global arguments when omitted."""
    return cli_main(argv)

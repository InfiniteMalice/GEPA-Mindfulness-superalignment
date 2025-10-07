"""Command line entry points for the training pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

from ..core.adversarial import iterate_adversarial_pool
from .configs import TrainingConfig, load_training_config

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .pipeline import RolloutResult, TrainingOrchestrator

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training CLI."""

    parser = argparse.ArgumentParser(description="Run GEPA mindfulness PPO training")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to newline-delimited dataset file containing prompts.",
    )
    parser.add_argument(
        "--adversarial-only",
        action="store_true",
        help="Only run adversarial evaluation without PPO updates.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory where training logs and rollout data will be written.",
    )
    return parser.parse_args(args=list(argv) if argv is not None else None)


def _resolve_orchestrator_factory() -> Callable[[TrainingConfig], object]:
    override = os.environ.get("GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR")
    if not override:
        from .pipeline import TrainingOrchestrator

        return TrainingOrchestrator
    module_name, _, attribute = override.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            "GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR must be in 'module:callable' format"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError("GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR must reference a callable")
    return factory


def _serialize_rollouts(path: Path, results: Iterable[object]) -> None:
    """Persist rollout results as JSON lines."""

    with path.open("w", encoding="utf-8") as handle:
        for item in results:
            if is_dataclass(item):
                payload = asdict(item)
            else:
                payload = {
                    "prompt": getattr(item, "prompt", None),
                    "response": getattr(item, "response", None),
                    "reward": getattr(item, "reward", None),
                    "trace_summary": getattr(item, "trace_summary", None),
                    "contradiction_report": getattr(item, "contradiction_report", None),
                }
            json.dump(payload, handle)
            handle.write("\n")


def read_dataset(path: Path) -> list[str]:
    """Materialize newline-delimited prompts from *path*."""

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_log_dir(cli_arg: Path | None) -> Path:
    """Determine the directory to use for logging output."""

    if cli_arg is not None:
        return cli_arg.expanduser().resolve()

    assume_tty = os.environ.get("GEPA_MINDFULNESS_TRAINING_ASSUME_TTY") == "1"
    if sys.stdin.isatty() or assume_tty:
        destination = ""
        while not destination:
            destination = input("Enter a directory to store training logs: ").strip()
        return Path(destination).expanduser().resolve()

    return (Path.cwd() / "training_logs").resolve()


def _setup_file_logging(log_dir: Path) -> logging.FileHandler:
    """Configure logging to write to ``training.log`` within *log_dir*."""

    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    LOGGER.info("File logging enabled at %s", log_dir / "training.log")
    return file_handler


def _run_orchestrator(
    orchestrator: "TrainingOrchestrator",
    prompts: list[str],
    *,
    adversarial_only: bool,
) -> list["RolloutResult"]:
    """Execute either adversarial evaluation or PPO training."""

    if adversarial_only:
        LOGGER.info("Running adversarial evaluation only")
        return orchestrator.run_adversarial_eval()

    LOGGER.info("Running PPO training")
    return orchestrator.run(prompts)


def _log_rollout_summary(results: Iterable["RolloutResult"]) -> None:
    """Emit a concise summary for each rollout to the logger."""

    for idx, result in enumerate(results):
        LOGGER.info(
            "Rollout %s reward %.3f contradictions %s",
            idx,
            getattr(result, "reward", None),
            getattr(result, "contradiction_report", None),
        )


def _serialize_results(log_dir: Path, results: Sequence[object]) -> Path:
    """Write *results* to ``rollouts.jsonl`` and return the path."""

    rollout_path = log_dir / "rollouts.jsonl"
    _serialize_rollouts(rollout_path, results)
    LOGGER.info("Serialized %s rollouts to %s", len(results), rollout_path)
    return rollout_path


def _log_available_adversarial_scenarios() -> None:
    """Log the identifiers of adversarial scenarios bundled with the package."""

    scenarios = list(iterate_adversarial_pool())
    LOGGER.info("Adversarial scenarios available: %s", scenarios)


def main(argv: Iterable[str] | None = None) -> None:
    """Entrypoint for the command-line training interface."""

    args = parse_args(argv)
    log_dir = _resolve_log_dir(args.log_dir)
    file_handler = _setup_file_logging(log_dir)
    root_logger = logging.getLogger()

    try:
        config: TrainingConfig = load_training_config(args.config)
        orchestrator_factory = _resolve_orchestrator_factory()
        orchestrator = orchestrator_factory(config=config)
        prompts = read_dataset(args.dataset)

        results = _run_orchestrator(
            orchestrator,
            prompts,
            adversarial_only=args.adversarial_only,
        )

        LOGGER.info("Completed %s rollouts", len(results))
        _log_rollout_summary(results)
        _serialize_results(log_dir, results)
        _log_available_adversarial_scenarios()
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()

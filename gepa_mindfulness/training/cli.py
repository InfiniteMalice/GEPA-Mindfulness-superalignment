"""Command line entry points for the training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from gepa_mindfulness.core.adversarial import iterate_adversarial_pool
from gepa_mindfulness.training.configs import (
    TrainingConfig,
    load_training_config,
)

try:
    from gepa_mindfulness.training.pipeline import (
        RolloutResult,
        TrainingOrchestrator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    missing = exc.name or ""
    message = (
        "Optional dependency '{missing}' is required for training; install the "
        "training extras to enable this command."
    ).format(missing=missing)
    raise SystemExit(message) from exc

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
        help=(
            "Directory where training logs and rollout summaries will be written. "
            "If omitted in an interactive shell, the CLI prompts for a location."
        ),
    )
    return parser.parse_args()


def read_dataset(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_log_dir(cli_arg: Path | None) -> Path:
    if cli_arg is not None:
        return cli_arg.expanduser().resolve()

    if sys.stdin.isatty():
        destination = input("Enter a directory to store training logs: ").strip()
        if not destination:
            raise SystemExit("A log directory is required when running interactively.")
        return Path(destination).expanduser().resolve()

    raise SystemExit("--log-dir must be provided when stdin is not interactive.")


def _write_rollout_log(log_dir: Path, results: list[RolloutResult]) -> None:
    rollout_path = log_dir / "rollouts.jsonl"
    with rollout_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _configure_file_handler(log_dir: Path) -> logging.FileHandler:
    """Attach a file handler for training logs and return it for teardown."""

    file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    return file_handler


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    log_dir = _resolve_log_dir(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = _configure_file_handler(log_dir)
    root_logger = logging.getLogger()

    config: TrainingConfig = load_training_config(args.config)
    orchestrator = TrainingOrchestrator(config=config)
    prompts = read_dataset(args.dataset)

    try:
        if args.adversarial_only:
            LOGGER.info("Running adversarial evaluation only")
            results = orchestrator.run_adversarial_eval()
        else:
            LOGGER.info("Running PPO training")
            results = orchestrator.run(prompts)

        LOGGER.info("Completed %s rollouts", len(results))
        for idx, result in enumerate(results):
            LOGGER.info(
                "Rollout %s reward %.3f contradictions %s",
                idx,
                result.reward,
                result.contradiction_report,
            )

        _write_rollout_log(log_dir, results)
        LOGGER.info("Rollout summaries written to %s", log_dir.joinpath("rollouts.jsonl"))

        LOGGER.info("Adversarial scenarios available: %s", list(iterate_adversarial_pool()))
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()

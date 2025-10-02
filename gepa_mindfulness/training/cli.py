"""Command line entry points for the training pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..core.adversarial import iterate_adversarial_pool
from .configs import TrainingConfig, load_training_config
from .pipeline import TrainingOrchestrator

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
    return parser.parse_args()


def read_dataset(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config: TrainingConfig = load_training_config(args.config)
    orchestrator = TrainingOrchestrator(config=config)
    prompts = read_dataset(args.dataset)

    if args.adversarial_only:
        LOGGER.info("Running adversarial evaluation only")
        results = orchestrator.run_adversarial_eval()
    else:
        LOGGER.info("Running PPO training")
        results = orchestrator.run(prompts)

    LOGGER.info("Completed %s rollouts", len(results))
    for idx, result in enumerate(results):
        LOGGER.info(
            "Rollout %s reward %.3f contradictions %s", idx, result.reward, result.contradiction_report
        )

    LOGGER.info("Adversarial scenarios available: %s", list(iterate_adversarial_pool()))


if __name__ == "__main__":
    main()

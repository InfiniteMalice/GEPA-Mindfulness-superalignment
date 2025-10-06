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
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .configs import TrainingConfig

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
        help="Directory where training logs and rollout data will be written.",
    )
    args = parser.parse_args()

    if args.log_dir is None:
        if sys.stdin.isatty():
            destination: str = ""
            while not destination:
                destination = input("Enter log output directory path: ").strip()
            args.log_dir = Path(destination).expanduser()
        else:
            parser.error("--log-dir is required when standard input is not interactive")
    else:
        args.log_dir = args.log_dir.expanduser()

    return args


def _resolve_orchestrator_factory() -> Callable[["TrainingConfig"], object]:
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
        raise TypeError(
            "GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR must reference a callable"
        )
    return factory


def _serialize_rollouts(path: Path, results: Iterable[object]) -> None:
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
                    "contradiction_report": getattr(
                        item, "contradiction_report", None
                    ),
                }
            json.dump(payload, handle)
            handle.write("\n")


def read_dataset(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    log_file = log_dir / "training.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    LOGGER.info("File logging enabled at %s", log_file)

    from ..core.adversarial import iterate_adversarial_pool
    from .configs import load_training_config

    config: "TrainingConfig" = load_training_config(args.config)
    orchestrator_factory = _resolve_orchestrator_factory()
    orchestrator = orchestrator_factory(config=config)
    prompts = read_dataset(args.dataset)

    if args.adversarial_only:
        LOGGER.info("Running adversarial evaluation only")
        results = orchestrator.run_adversarial_eval()
    else:
        LOGGER.info("Running PPO training")
        results = orchestrator.run(prompts)

    rollout_path = log_dir / "rollouts.jsonl"
    _serialize_rollouts(rollout_path, results)
    LOGGER.info("Serialized %s rollouts to %s", len(results), rollout_path)

    LOGGER.info("Completed %s rollouts", len(results))
    for idx, result in enumerate(results):
        LOGGER.info(
            "Rollout %s reward %.3f contradictions %s",
            idx,
            result.reward,
            result.contradiction_report,
        )

    LOGGER.info("Adversarial scenarios available: %s", list(iterate_adversarial_pool()))


if __name__ == "__main__":
    main()

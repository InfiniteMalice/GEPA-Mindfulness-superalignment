"""Command line entry points for the training pipeline."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List

import click

from .config import GRPOConfig, PPOConfig, load_config_dict
from .configs import TrainingConfig
from .grpo_trainer import GRPOTrainer
from .pipeline import RolloutResult, TrainingOrchestrator
from .ppo_trainer import PPOTrainer


def load_training_config(path: Path) -> TrainingConfig:
    payload = load_config_dict(path)
    return TrainingConfig.from_mapping(payload)


def read_dataset(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iterate_dual_path_pool() -> Iterable[str]:  # pragma: no cover - compatibility hook
    return []


def _resolve_orchestrator_factory():
    target = os.environ.get("GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR")
    if not target:
        return TrainingOrchestrator
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise RuntimeError(
            "GEPA_MINDFULNESS_TRAINING_ORCHESTRATOR must be in 'module:callable' format"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    return factory


def _serialize_rollouts(path: Path, results: Iterable[RolloutResult]) -> int:
    serialized = list(results)
    with path.open("w", encoding="utf-8") as handle:
        for result in serialized:
            payload = {
                "prompt": result.prompt,
                "response": result.response,
                "reward": result.reward,
                "trace_summary": result.trace_summary,
                "contradiction_report": result.contradiction_report,
            }
            json.dump(payload, handle)
            handle.write("\n")
    return len(serialized)


def _resolve_log_dir(cli_arg: Path | None) -> Path:
    if cli_arg is not None:
        return cli_arg
    assume_tty = os.environ.get("GEPA_MINDFULNESS_TRAINING_ASSUME_TTY")
    if assume_tty or sys.stdin.isatty():
        destination = ""
        while not destination:
            destination = input("Enter a directory to store training logs: ").strip()
        return Path(destination)
    return Path.cwd() / "training_logs"


def _setup_file_logging(log_dir: Path) -> logging.FileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    return file_handler


def _legacy_main(
    config_path: Path,
    dataset_path: Path,
    log_dir: Path | None,
    dual_path_only: bool,
) -> None:
    log_destination = _resolve_log_dir(log_dir)
    handler = _setup_file_logging(log_destination)
    root_logger = logging.getLogger()
    try:
        config = load_training_config(config_path)
        orchestrator_factory = _resolve_orchestrator_factory()
        orchestrator = orchestrator_factory(config=config)
        prompts = read_dataset(dataset_path)
        if dual_path_only:
            results = orchestrator.run_dual_path_eval()
        else:
            results = orchestrator.run(prompts)
        count = _serialize_rollouts(log_destination / "rollouts.jsonl", results)
        root_logger.info("Serialized %s rollouts", count)
    finally:
        root_logger.removeHandler(handler)
        handler.close()


@click.group(invoke_without_command=True)
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option("--log-dir", "log_dir", type=click.Path(path_type=Path))
@click.option("--dual-path-only", is_flag=True)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    dataset_path: Path | None,
    log_dir: Path | None,
    dual_path_only: bool,
) -> None:
    """Entry point for GEPA Mindfulness training utilities."""

    if ctx.invoked_subcommand is None:
        if config_path is None or dataset_path is None:
            raise click.UsageError("--config and --dataset are required when no subcommand is used")
        _legacy_main(config_path, dataset_path, log_dir, dual_path_only)


@cli.command()
@click.option("--trainer", type=click.Choice(["ppo", "grpo"]), required=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path))
def train(trainer: str, config_path: Path, output_dir: Path | None) -> None:
    payload = load_config_dict(config_path)
    payload["trainer_type"] = trainer
    if output_dir is not None:
        payload["output_dir"] = str(output_dir)
    if trainer == "ppo":
        config = PPOConfig.from_mapping(payload)
        trainer_impl = PPOTrainer(config)
    else:
        config = GRPOConfig.from_mapping(payload)
        trainer_impl = GRPOTrainer(config)
    trainer_impl.train()
    click.echo(f"Training finished. Outputs written to {trainer_impl.output_dir}")


@cli.command()
@click.option("--ppo-run", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--grpo-run", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path))
def compare(ppo_run: Path, grpo_run: Path, output_path: Path | None) -> None:
    summary = {
        "ppo": _load_metrics(ppo_run),
        "grpo": _load_metrics(grpo_run),
    }
    report = _render_comparison(summary)
    click.echo(report)
    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        click.echo(f"Comparison saved to {output_path}")


def _load_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise click.ClickException(f"Metrics file not found in {run_dir}")
    rewards: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rewards.append(float(payload.get("reward", 0.0)))
    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {"average_reward": average_reward, "samples": len(rewards)}


def _render_comparison(summary: dict[str, dict[str, float]]) -> str:
    ppo = summary["ppo"]
    grpo = summary["grpo"]
    winner = "grpo" if grpo["average_reward"] >= ppo["average_reward"] else "ppo"
    lines = [
        "Model\tAverage Reward\tSamples",
        f"PPO\t{ppo['average_reward']:.3f}\t{ppo['samples']}",
        f"GRPO\t{grpo['average_reward']:.3f}\t{grpo['samples']}",
        f"Winner: {winner.upper()}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Invoke the click entry point without exiting the interpreter."""

    cli.main(standalone_mode=False)


if __name__ == "__main__":  # pragma: no cover
    main()

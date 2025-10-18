"""Command-line interface for GRPO and legacy PPO training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

import torch

from ..core.rewards import RewardWeights
from .configs import TrainingConfig, load_training_config
from .grpo_trainer import GRPOTrainer

LOGGER = logging.getLogger(__name__)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA training pipelines")
    parser.add_argument("--mode", choices=["grpo", "ppo"], default="grpo")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(args=list(argv) if argv is not None else None)


def _load_prompts(path: Path) -> List[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    prompts.append(text)
                else:
                    prompt = payload.get("prompt") or payload.get("query")
                    if prompt:
                        prompts.append(str(prompt))
            else:
                prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _ensure_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise SystemExit(
            "transformers is required for GRPO training. Install with pip install transformers"
        ) from exc


def _instantiate_models(config: TrainingConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.model.policy_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy = AutoModelForCausalLM.from_pretrained(model_name)
    reference = AutoModelForCausalLM.from_pretrained(model_name)
    return policy, reference, tokenizer


def _run_grpo(args: argparse.Namespace, config: TrainingConfig) -> None:
    _ensure_transformers()
    prompts = _load_prompts(args.dataset)
    policy, reference, tokenizer = _instantiate_models(config)

    device = torch.device(config.device)
    reward_weights = RewardWeights.from_mapping(config.grpo.reward_weights.dict())

    trainer = GRPOTrainer(
        policy,
        reference,
        tokenizer,
        config.grpo,
        reward_weights,
        device=device,
    )
    summary = trainer.train_epoch(
        prompts,
        batch_size=config.grpo.batch_size,
    )

    output_dir = args.output or Path(config.output.checkpoint_dir)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir / "policy")
    tokenizer.save_pretrained(output_dir / "policy")

    report = {
        "steps": summary.steps,
        "mean_reward": summary.mean_reward(),
        "batches": [
            {
                "prompt": batch.prompt,
                "mean_reward": batch.mean_reward,
                "advantages": batch.advantages,
                "categories": batch.categories,
                "confidences": batch.confidences,
            }
            for batch in summary.batches
        ],
    }
    with (output_dir / "grpo_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    LOGGER.info("GRPO training completed. Mean reward %.3f", summary.mean_reward())


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_training_config(args.config)

    if args.mode == "grpo":
        _run_grpo(args, config)
    else:
        from . import cli as legacy_cli

        legacy_cli.main(
            [
                "--config",
                str(args.config),
                "--dataset",
                str(args.dataset),
            ]
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

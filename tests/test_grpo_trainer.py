from __future__ import annotations

from pathlib import Path

import pytest

from gepa_mindfulness.training.config import GRPOConfig
from gepa_mindfulness.training.dataloader import DatasetExample
from gepa_mindfulness.training.grpo_trainer import GRPOTrainer


def _config(tmp_path: Path) -> GRPOConfig:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"prompt": "Is fire hot?", "answers": ["yes"]}\n', encoding="utf-8")
    output_dir = tmp_path / "runs"
    return GRPOConfig(
        model_name="demo-model",
        dataset_path=str(dataset),
        output_dir=str(output_dir),
        learning_rate=1e-2,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        group_size=2,
        kl_coef=0.05,
        temperature=0.8,
    )


def test_group_advantages_sum_to_zero(tmp_path: Path) -> None:
    trainer = GRPOTrainer(_config(tmp_path))
    rewards = [[1.0, 2.0, 3.0]]
    advantages = trainer._compute_advantages(rewards)
    assert pytest.approx(sum(advantages[0]), rel=1e-6) == 0.0


def test_grpo_training_runs(tmp_path: Path) -> None:
    trainer = GRPOTrainer(_config(tmp_path))
    trainer.train()
    metrics_path = Path(trainer.output_dir) / "metrics.jsonl"
    assert metrics_path.exists()


def test_policy_update_shifts_logit(tmp_path: Path) -> None:
    trainer = GRPOTrainer(_config(tmp_path))
    example = DatasetExample(
        prompt="Is fire hot?",
        references=["yes"],
        gepa_scores=None,
        imperatives=None,
    )
    group = trainer._generate_group(example)
    rewards = [[1.0 if item.text == "yes" else 0.0 for item in group]]
    advantages = trainer._compute_advantages(rewards)
    before = trainer._logits.get(example.prompt, 0.0)
    trainer._apply_policy_update([example], [group], advantages)
    after = trainer._logits.get(example.prompt, 0.0)
    assert after != before


def test_keyword_initialisation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    trainer = GRPOTrainer(config=config, seed=42)
    assert trainer.config is config
    assert not getattr(trainer, "_hf_mode", True)


def test_keyword_initialisation_without_torch(monkeypatch, tmp_path: Path) -> None:
    from gepa_mindfulness.training import grpo_trainer as module

    config = _config(tmp_path)
    monkeypatch.setattr(module, "torch", None)
    trainer = module.GRPOTrainer(config=config, seed=0)
    assert trainer.config is config
    assert not getattr(trainer, "_hf_mode", True)

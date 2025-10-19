"""Unit tests for the lightweight PPO trainer implementation."""

from __future__ import annotations

from gepa_mindfulness.training.config import PPOConfig
from gepa_mindfulness.training.ppo_trainer import PPOTrainer


def test_generate_response_handles_missing_references(tmp_path) -> None:
    """The trainer should fall back to a synthetic answer when no references exist."""

    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("Describe a mindful breathing exercise.")
    config = PPOConfig(
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "output"),
        max_steps=1,
        batch_size=1,
    )
    trainer = PPOTrainer(config)
    example = trainer.dataset.items[0]

    trainer._policy_logits[example.prompt] = 10.0
    trainer._random.random = lambda: 0.0  # type: ignore[assignment]

    response = trainer._generate_response(example)

    assert response.text
    assert response.text != "I don't know"
    assert response.metadata is not None
    assert response.metadata.get("reference_used") is False

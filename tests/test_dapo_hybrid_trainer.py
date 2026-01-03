"""Tests for the DAPO hybrid trainer integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gepa_mindfulness.trainers.dapo_hybrid_trainer import (
    DAPOHybridConfig,
    GEPAFeedbackConfig,
    JSONLLogger,
    PromptExample,
    default_gepa_components,
    default_reward_calculator,
    score_with_gepa,
    train_loop,
)


@pytest.mark.unit
def test_score_with_gepa_returns_feedback() -> None:
    """score_with_gepa should emit a GEPAFeedback instance when deps exist."""

    gepa_dapo_grn = pytest.importorskip("gepa_dapo_grn")
    _ = gepa_dapo_grn
    reward_calculator = default_reward_calculator()
    feedback = score_with_gepa(
        "Explain mindful breathing.",
        "Mindful breathing helps focus.",
        reward_calculator=reward_calculator,
        component_scorer=default_gepa_components,
        feedback_config=GEPAFeedbackConfig(
            reward_dim_map={"mindfulness": "mindfulness"},
            tag_dim_map={"mindfulness": "mindfulness"},
        ),
        meta={"task_id": "unit"},
        reference_answers=None,
    )
    assert feedback is not None


@pytest.mark.unit
def test_train_step_logs_data(tmp_path: Path) -> None:
    """train_loop should call train_step and emit JSONL metrics."""

    pytest.importorskip("gepa_dapo_grn")
    from gepa_dapo_grn import CurriculumTracker, SafetyController
    from gepa_dapo_grn.policy_interfaces import Policy

    class DummyPolicy(Policy):
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 2)

        def generate(
            self,
            prompts: list[str],
            *,
            max_new_tokens: int,
            temperature: float,
        ) -> list[str]:
            _ = (max_new_tokens, temperature)
            return ["ok" for _ in prompts]

        def log_probs(
            self,
            prompts: list[str],
            completions: list[str],
        ) -> list[list[float]]:
            _ = (prompts, completions)
            return [[0.0] for _ in completions]

    policy = DummyPolicy()
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-3)
    dataset = [
        PromptExample(
            prompt="Why pause?",
            reference=None,
            meta={"task_id": "train"},
        )
    ]
    config = DAPOHybridConfig(
        model_name="dummy",
        output_dir=tmp_path,
        batch_size=1,
        max_steps=1,
        eval_interval=10,
        learning_rate=1e-3,
        max_new_tokens=8,
        temperature=0.8,
        reward_mixer_weights={"mindfulness": 1.0},
        gepa_feedback=GEPAFeedbackConfig(
            reward_dim_map={"mindfulness": "mindfulness"},
            tag_dim_map={"mindfulness": "mindfulness"},
        ),
        use_grn=False,
        grn_location="attention",
    )
    logger = JSONLLogger(tmp_path / "metrics.jsonl")

    train_loop(
        config=config,
        policy=policy,
        optimizer=optimizer,
        dataset=dataset,
        eval_dataset=dataset,
        reward_calculator=default_reward_calculator(),
        component_scorer=default_gepa_components,
        curriculum_tracker=CurriculumTracker(),
        safety_controller=SafetyController(),
        logger=logger,
    )

    assert logger.path.exists()
    assert logger.path.read_text(encoding="utf-8").strip()

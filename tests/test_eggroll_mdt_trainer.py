"""Smoke tests for the EGGROLL + MDT trainer."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

torch = pytest.importorskip("torch")

from mindful_trace_gepa.train.eggroll_mdt_trainer import EGGROLLConfig, EGGROLLMDTTrainer


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(inputs)


def synthetic_eval_fn(model: TinyModel, candidate_id: int) -> Mapping[str, Any]:
    torch.manual_seed(candidate_id)
    inputs = torch.randn(3, 2)
    outputs = model(inputs)
    reward = float(outputs.mean().item())
    ethics = float(outputs.abs().mean().item())
    probe_logits = outputs.squeeze(-1)
    confidence_outputs = outputs.abs()
    return {
        "task_reward": reward,
        "ethics_score": ethics,
        "deception_penalty": float(probe_logits.pow(2).mean().item()),
        "confidence_metric": float(confidence_outputs.mean().item()),
        "confidence_outputs": confidence_outputs,
        "probe_logits": probe_logits,
    }


def test_trainer_runs_without_grn() -> None:
    model = TinyModel()
    config = EGGROLLConfig(population_size=3, generations=1, geometry_weight=0.05)
    trainer = EGGROLLMDTTrainer(model=model, eval_fn=synthetic_eval_fn, config=config)
    result = trainer.run()
    assert "logs" in result
    assert len(result["logs"]) == 1
    assert result["logs"][0]["fitness"].numel() == config.population_size


def test_trainer_respects_grn_flags() -> None:
    model = TinyModel()
    config = EGGROLLConfig(
        population_size=2,
        generations=1,
        geometry_weight=0.05,
        use_grn_for_confidence=True,
        use_grn_for_probes=True,
        use_grn_for_fitness=True,
    )
    trainer = EGGROLLMDTTrainer(model=model, eval_fn=synthetic_eval_fn, config=config)
    result = trainer.run()
    fitness = result["logs"][0]["fitness"]
    assert torch.isfinite(fitness).all()
    assert fitness.shape[0] == config.population_size

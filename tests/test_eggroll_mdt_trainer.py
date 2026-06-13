"""Smoke tests for the EGGROLL + MDT trainer."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from mindful_trace_gepa.train.eggroll_mdt_trainer import EGGROLLConfig, EGGROLLMDTTrainer

torch = pytest.importorskip("torch")


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
        "deception_signal": float(probe_logits.pow(2).mean().item()),
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


def test_deception_diagnostics_do_not_change_optimization_fitness() -> None:
    model = TinyModel()
    config = EGGROLLConfig(population_size=3, generations=1, geometry_weight=0.05)
    trainer = EGGROLLMDTTrainer(model=model, eval_fn=synthetic_eval_fn, config=config)
    base_results = [
        {"task_reward": 1.0, "ethics_score": 0.25, "deception_signal": 0.0, "probe_score": 0.0},
        {"task_reward": 0.8, "ethics_score": 0.35, "deception_signal": 0.5, "probe_score": 1.0},
        {"task_reward": 0.6, "ethics_score": 0.45, "deception_signal": 1.0, "probe_score": 2.0},
    ]
    changed_diagnostics = [
        {**result, "deception_signal": 100.0 + idx, "probe_score": 50.0 + idx}
        for idx, result in enumerate(base_results)
    ]

    base_embedding = trainer._compute_mdt_embedding(trainer._build_optimization_views(base_results))
    changed_embedding = trainer._compute_mdt_embedding(
        trainer._build_optimization_views(changed_diagnostics)
    )
    base_fitness = trainer._compute_fitness(base_results, base_embedding)
    changed_fitness = trainer._compute_fitness(changed_diagnostics, changed_embedding)

    assert torch.allclose(base_fitness, changed_fitness)


def test_probe_grn_changes_diagnostics_without_changing_fitness() -> None:
    def eval_with_probe_logits(model: TinyModel, candidate_id: int) -> Mapping[str, Any]:
        return {
            "task_reward": float(candidate_id),
            "ethics_score": 0.5,
            "deception_signal": 0.0,
            "probe_logits": torch.tensor([float(candidate_id + 1), float(candidate_id + 2)]),
        }

    torch.manual_seed(123)
    plain = EGGROLLMDTTrainer(
        model=TinyModel(),
        eval_fn=eval_with_probe_logits,
        config=EGGROLLConfig(
            population_size=2,
            generations=1,
            geometry_weight=0.05,
            mean_lr=0.0,
            cov_lr=0.0,
        ),
    ).run()
    torch.manual_seed(123)
    with_grn = EGGROLLMDTTrainer(
        model=TinyModel(),
        eval_fn=eval_with_probe_logits,
        config=EGGROLLConfig(
            population_size=2,
            generations=1,
            geometry_weight=0.05,
            mean_lr=0.0,
            cov_lr=0.0,
            use_grn_for_probes=True,
        ),
    ).run()

    assert not torch.allclose(
        plain["logs"][0]["eval_results"][0]["probe_logits"],
        with_grn["logs"][0]["eval_results"][0]["probe_logits"],
    )
    assert torch.allclose(plain["logs"][0]["fitness"], with_grn["logs"][0]["fitness"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"generations": 0},
        {"rank": 0},
        {"sigma": 0.0},
        {"mean_lr": float("nan")},
        {"cov_lr": -0.1},
        {"geometry_weight": float("inf")},
        {"rank": 99},
    ],
)
def test_config_validation_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    config = EGGROLLConfig(**kwargs)
    with pytest.raises(ValueError):
        EGGROLLMDTTrainer(model=TinyModel(), eval_fn=synthetic_eval_fn, config=config)

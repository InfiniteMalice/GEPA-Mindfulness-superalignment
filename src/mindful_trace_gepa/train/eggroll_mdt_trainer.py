"""EGGROLL-style evolution strategies with MDT and optional GRN integration."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Mapping

from ..geometry import (
    MDTTrajectory,
    MultiViewDatasetView,
    build_markov_operator,
    build_mdt_operator,
    mdt_embedding,
)
from ..train.grn import GRNSettings, build_grn
from ..utils.imports import optional_import

LOGGER = logging.getLogger(__name__)

torch = optional_import("torch")


EvaluationFn = Callable[[Any, int], Mapping[str, Any]]


@dataclasses.dataclass
class EGGROLLConfig:
    """Configuration for the EGGROLL + MDT trainer."""

    population_size: int = 4
    generations: int = 2
    rank: int = 2
    sigma: float = 0.02
    mean_lr: float = 0.1
    cov_lr: float = 0.05
    geometry_weight: float = 0.1
    mdt_time: int = 2
    mdt_components: int = 2
    mdt_sigma: float = 1.0
    mdt_knn: int | None = None
    trajectories: tuple[tuple[int, ...], ...] = ((0,), (1,), (0, 1))
    use_grn_for_confidence: bool = False
    use_grn_for_probes: bool = False
    use_grn_for_fitness: bool = False


class EGGROLLMDTTrainer:
    """Minimal low-rank ES loop with MDT-aware regularisation."""

    def __init__(
        self,
        model: Any,
        eval_fn: EvaluationFn,
        config: EGGROLLConfig,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for the EGGROLL trainer")
        self.model = model
        self.eval_fn = eval_fn
        self.config = config
        self.device = device or "cpu"
        self._init_params()
        self.conf_grn = build_grn(GRNSettings(enabled=config.use_grn_for_confidence))
        self.probe_grn = build_grn(GRNSettings(enabled=config.use_grn_for_probes))
        self.fitness_grn = build_grn(GRNSettings(enabled=config.use_grn_for_fitness, dim=-1))
        if self.conf_grn is not None:
            self.conf_grn.to(self.device)
        if self.probe_grn is not None:
            self.probe_grn.to(self.device)
        if self.fitness_grn is not None:
            self.fitness_grn.to(self.device)

    def _init_params(self) -> None:
        params = [param.detach().clone().flatten() for param in self.model.parameters()]
        self.param_shapes = [param.shape for param in self.model.parameters()]
        if not params:
            raise ValueError("Model must expose parameters for optimisation")
        self.mean_vector = torch.cat(params).to(self.device)
        self.low_rank = torch.eye(self.mean_vector.numel(), self.config.rank, device=self.device)

    def _vector_to_params(self, vector: "torch.Tensor") -> None:
        pointer = 0
        new_params = []
        for shape, param in zip(self.param_shapes, self.model.parameters()):
            size = param.numel()
            slice_ = vector[pointer : pointer + size]
            new_params.append(slice_.view_as(param))
            pointer += size
        for target, source in zip(self.model.parameters(), new_params):
            target.data = source.to(target.device)

    def _sample_perturbation(self) -> tuple["torch.Tensor", "torch.Tensor"]:
        z = torch.randn(self.config.rank, device=self.device)
        perturb = self.config.sigma * (self.low_rank @ z)
        return perturb, z

    def _apply_grn_if_needed(
        self,
        tensor: "torch.Tensor",
        module: Any | None,
    ) -> "torch.Tensor":
        """Apply GRN module to tensor if module is provided.

        Handles tensors of varying dimensions:
        - 0-D (scalar): returned unchanged
        - 1-D: temporarily adds batch dimension for GRN, then removes it
        - 2-D+: applies module directly
        """
        if module is None:
            return tensor
        if tensor.dim() == 0:
            return tensor
        if tensor.dim() == 1:
            # 1-D tensors represent batch-sized scalars.
            normalized = module(tensor.unsqueeze(-1)).squeeze(-1)
            return normalized
        return module(tensor)

    def _build_views(self, eval_results: list[Mapping[str, Any]]) -> list["torch.Tensor"]:
        task = []
        ethics = []
        deception = []
        confidence = []
        attribution = []
        for result in eval_results:
            task.append(float(result.get("task_reward", 0.0)))
            ethics.append(float(result.get("ethics_score", 0.0)))
            deception.append(float(result.get("deception_penalty", 0.0)))
            confidence.append(float(result.get("confidence_metric", 0.0)))
            attribution.append(float(result.get("attribution_score", 0.0)))
        views = []
        views.append(torch.tensor(task, device=self.device).unsqueeze(1))
        views.append(torch.tensor(ethics, device=self.device).unsqueeze(1))
        views.append(torch.tensor(deception, device=self.device).unsqueeze(1))
        views.append(torch.tensor(confidence, device=self.device).unsqueeze(1))
        if any(item != 0.0 for item in attribution):
            views.append(torch.tensor(attribution, device=self.device).unsqueeze(1))
        return views

    def _compute_mdt_embedding(self, views: list["torch.Tensor"]) -> "torch.Tensor":
        standardized = MultiViewDatasetView(views).standardize()
        operators = [
            build_markov_operator(view, sigma=self.config.mdt_sigma, knn=self.config.mdt_knn)
            for view in standardized
        ]
        trajectories = [MDTTrajectory(indices=traj) for traj in self.config.trajectories]
        weights = [1.0 for _ in trajectories]
        mdt_op = build_mdt_operator(operators, trajectories, weights)
        return mdt_embedding(mdt_op, self.config.mdt_components, self.config.mdt_time)

    def _compute_geometry_regularizer(self, embedding: "torch.Tensor") -> "torch.Tensor":
        if embedding.numel() == 0:
            return torch.zeros(embedding.size(0), device=self.device)
        centre = embedding.mean(dim=0, keepdim=True)
        distances = torch.linalg.norm(embedding - centre, dim=1)
        return -distances * self.config.geometry_weight

    def _compute_fitness(
        self, eval_results: list[Mapping[str, Any]], embedding: "torch.Tensor"
    ) -> "torch.Tensor":
        base_scores = []
        for result in eval_results:
            reward = float(result.get("task_reward", 0.0))
            ethics = float(result.get("ethics_score", 0.0))
            deception_penalty = float(result.get("deception_penalty", 0.0))
            base_scores.append(reward + ethics - deception_penalty)
        base_tensor = torch.tensor(base_scores, device=self.device)
        reg = self._compute_geometry_regularizer(embedding)
        fitness = base_tensor + reg
        return self._apply_grn_if_needed(fitness, self.fitness_grn)

    def run(self) -> dict[str, Any]:
        logs: list[dict[str, Any]] = []
        for generation in range(self.config.generations):
            perturbations: list["torch.Tensor"] = []
            z_samples: list["torch.Tensor"] = []
            eval_results: list[Mapping[str, Any]] = []
            for candidate in range(self.config.population_size):
                perturb, z = self._sample_perturbation()
                perturbations.append(perturb)
                z_samples.append(z)
                candidate_vector = self.mean_vector + perturb
                self._vector_to_params(candidate_vector)
                result = self.eval_fn(self.model, candidate)
                result = dict(result)
                if self.config.use_grn_for_confidence and torch.is_tensor(
                    result.get("confidence_outputs")
                ):
                    result["confidence_outputs"] = self._apply_grn_if_needed(
                        result["confidence_outputs"], self.conf_grn
                    )
                    conf_mean = result["confidence_outputs"].mean().item()
                    result["confidence_metric"] = float(conf_mean)
                use_probe_grn = self.config.use_grn_for_probes
                if use_probe_grn and torch.is_tensor(result.get("probe_logits")):
                    result["probe_logits"] = self._apply_grn_if_needed(
                        result["probe_logits"], self.probe_grn
                    )
                    probe_mean = result["probe_logits"].abs().mean().item()
                    result["deception_penalty"] = float(probe_mean)
                eval_results.append(result)
            views = self._build_views(eval_results)
            embedding = self._compute_mdt_embedding(views)
            fitness = self._compute_fitness(eval_results, embedding)
            noise_std = fitness.std().clamp(min=1e-6)
            fitness_normalized = (fitness - fitness.mean()) / noise_std
            mean_update = sum(
                weight * perturb for weight, perturb in zip(fitness_normalized, perturbations)
            )
            self.mean_vector = self.mean_vector + self.config.mean_lr * mean_update
            cov_update = sum(
                weight * torch.ger(perturb, z)
                for weight, perturb, z in zip(fitness_normalized, perturbations, z_samples)
            )
            self.low_rank = self.low_rank + self.config.cov_lr * cov_update
            logs.append(
                {
                    "generation": generation,
                    "fitness": fitness.detach().cpu(),
                    "embedding": embedding.detach().cpu(),
                    "eval_results": eval_results,
                }
            )
        self._vector_to_params(self.mean_vector)
        return {"logs": logs, "final_params": self.mean_vector.detach().cpu()}


__all__ = ["EGGROLLConfig", "EGGROLLMDTTrainer"]

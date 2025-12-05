"""Multi-view diffusion geometry utilities for EGGROLL trainers."""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence

from ..utils.imports import optional_import

torch = optional_import("torch")


@dataclasses.dataclass
class MultiViewDatasetView:
    """Container for per-view features with basic standardisation."""

    features: Sequence["torch.Tensor"]

    def standardize(self) -> list["torch.Tensor"]:
        if torch is None:
            raise ImportError("torch is required for MDT operations")
        standardised = []
        for view in self.features:
            if view.numel() == 0:
                standardised.append(view)
                continue
            mean = view.mean(dim=0, keepdim=True)
            std = view.std(dim=0, keepdim=True) + 1e-6
            standardised.append((view - mean) / std)
        return standardised


@dataclasses.dataclass
class MDTTrajectory:
    """Sequence of view indices defining a diffusion trajectory."""

    indices: Sequence[int]

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(int(idx) for idx in self.indices)

    def build_product(self, view_ops: Sequence["torch.Tensor"]) -> "torch.Tensor":
        if torch is None:
            raise ImportError("torch is required for MDT operations")
        if not self.indices:
            raise ValueError("MDTTrajectory requires at least one view index")
        product: "torch.Tensor" | None = None
        for idx in self.indices:
            if idx < 0 or idx >= len(view_ops):
                message = f"View index {idx} out of range for view_ops size {len(view_ops)}"
                raise IndexError(message)
            product = view_ops[idx] if product is None else product @ view_ops[idx]
        if product is None:
            raise RuntimeError("Failed to construct MDT product")
        return product


def build_markov_operator(
    view_features: "torch.Tensor", sigma: float, knn: int | None = None
) -> "torch.Tensor":
    """Construct a row-stochastic Markov operator for a single view."""

    if torch is None:
        raise ImportError("torch is required for MDT operations")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    if view_features.dim() != 2:
        raise ValueError("view_features must be 2D (candidates, features)")
    if view_features.numel() == 0:
        raise ValueError("view_features is empty")
    diffs = view_features.unsqueeze(1) - view_features.unsqueeze(0)
    sq_dists = (diffs * diffs).sum(dim=-1)
    kernel = torch.exp(-sq_dists / (sigma**2))
    if knn is not None and knn > 0:
        values, indices = torch.topk(kernel, k=min(knn, kernel.size(1)), dim=1)
        mask = torch.zeros_like(kernel)
        mask.scatter_(1, indices, values)
        kernel = mask
    row_sums = kernel.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    return kernel / row_sums


def build_mdt_operator(
    view_operators: Sequence["torch.Tensor"],
    trajectories: Sequence[MDTTrajectory],
    weights: Sequence[float],
) -> "torch.Tensor":
    """Combine per-view operators into a single MDT operator."""

    if torch is None:
        raise ImportError("torch is required for MDT operations")
    if len(trajectories) != len(weights):
        raise ValueError("trajectories and weights must share length")
    if not trajectories:
        raise ValueError("At least one trajectory required to build MDT operator")
    if not view_operators:
        raise ValueError("view_operators must not be empty")
    norm = sum(weights)
    if norm <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = [weight / norm for weight in weights]
    combined = torch.zeros_like(view_operators[0])
    for trajectory, weight in zip(trajectories, normalized):
        combined = combined + weight * trajectory.build_product(view_operators)
    return combined


def mdt_embedding(P_mdt: "torch.Tensor", n_components: int, t: int) -> "torch.Tensor":
    """Compute diffusion embedding from an MDT operator."""

    if torch is None:
        raise ImportError("torch is required for MDT operations")
    if P_mdt.dim() != 2 or P_mdt.size(0) != P_mdt.size(1):
        raise ValueError("P_mdt must be a square matrix")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    P_power = P_mdt.matrix_power(t)
    u, s, _ = torch.linalg.svd(P_power)
    rank = min(n_components, u.size(1))
    embedding = u[:, :rank] * s[:rank]
    return embedding


__all__ = [
    "MDTTrajectory",
    "MultiViewDatasetView",
    "build_markov_operator",
    "build_mdt_operator",
    "mdt_embedding",
]

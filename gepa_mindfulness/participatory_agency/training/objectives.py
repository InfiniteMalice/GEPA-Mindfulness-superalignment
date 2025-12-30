"""Objectives for participatory agency training."""

from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch.nn import functional as nn_functional

from ..values import ValueComponents


def supervised_head_losses(
    predicted: ValueComponents,
    target: ValueComponents,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-head supervised losses."""

    if loss_fn is None:
        loss_fn = nn_functional.mse_loss
    return {
        "epistemic": loss_fn(predicted.epistemic, target.epistemic),
        "cooperation": loss_fn(predicted.cooperation, target.cooperation),
        "flexibility": loss_fn(predicted.flexibility, target.flexibility),
        "belonging": loss_fn(predicted.belonging, target.belonging),
    }


def combined_value_loss(
    predicted: ValueComponents,
    target: ValueComponents,
    weights: Mapping[str, float] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Combine supervised losses into a weighted total."""

    per_head = supervised_head_losses(predicted, target, loss_fn=loss_fn)
    per_head_components = ValueComponents(**per_head)
    return per_head_components.total(weights)


def rl_reward(
    components: ValueComponents,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Compute a scalar RL reward from value components."""

    return components.total(weights)

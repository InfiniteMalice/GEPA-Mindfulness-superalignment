"""Self-supervision stubs for participatory agency training."""

from __future__ import annotations

from typing import Iterable

import torch

from ..values import ParticipatoryValueHead, ValueComponents


def self_supervision_step(
    head: ParticipatoryValueHead,
    features: torch.Tensor,
    targets: ValueComponents,
) -> dict[str, torch.Tensor]:
    """Run a stub self-supervision step and return diagnostics."""

    predictions = head(features)
    diagnostics = {
        "epistemic": torch.mean(predictions.epistemic - targets.epistemic),
        "cooperation": torch.mean(predictions.cooperation - targets.cooperation),
        "flexibility": torch.mean(predictions.flexibility - targets.flexibility),
        "belonging": torch.mean(predictions.belonging - targets.belonging),
    }
    return diagnostics


def batch_self_supervision(
    head: ParticipatoryValueHead,
    feature_batches: Iterable[torch.Tensor],
    target_batches: Iterable[ValueComponents],
) -> dict[str, torch.Tensor]:
    """Aggregate self-supervision diagnostics across batches."""

    device = next(head.parameters()).device
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for features, targets in zip(feature_batches, target_batches):
        diagnostics = self_supervision_step(head, features, targets)
        for key, value in diagnostics.items():
            if key not in totals:
                totals[key] = torch.zeros_like(value.detach()).to(device)
            totals[key].add_(value.detach())
        count += 1
    if count == 0:
        return {
            "epistemic": torch.tensor(0.0, device=device),
            "cooperation": torch.tensor(0.0, device=device),
            "flexibility": torch.tensor(0.0, device=device),
            "belonging": torch.tensor(0.0, device=device),
        }
    return {key: value / count for key, value in totals.items()}

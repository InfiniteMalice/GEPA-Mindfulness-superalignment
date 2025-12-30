"""Self-supervision stubs for participatory agency training."""

from __future__ import annotations

from typing import Dict, Iterable

import torch

from ..values import ParticipatoryValueHead, ValueComponents


def self_supervision_step(
    head: ParticipatoryValueHead,
    features: torch.Tensor,
    targets: ValueComponents,
) -> Dict[str, torch.Tensor]:
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
) -> Dict[str, torch.Tensor]:
    """Aggregate self-supervision diagnostics across batches."""

    totals: Dict[str, torch.Tensor] = {
        "epistemic": torch.tensor(0.0),
        "cooperation": torch.tensor(0.0),
        "flexibility": torch.tensor(0.0),
        "belonging": torch.tensor(0.0),
    }
    count = 0
    for features, targets in zip(feature_batches, target_batches):
        diagnostics = self_supervision_step(head, features, targets)
        for key, value in diagnostics.items():
            totals[key] = totals[key] + value
        count += 1
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}

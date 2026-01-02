"""Self-supervision stubs for participatory agency training."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..values import ParticipatoryValueHead, ValueComponents

_COMPONENT_KEYS = ("epistemic", "cooperation", "flexibility", "belonging")


def self_supervision_step(
    head: ParticipatoryValueHead,
    features: torch.Tensor,
    targets: ValueComponents,
) -> dict[str, torch.Tensor]:
    """Run a stub self-supervision step and return diagnostics."""

    with torch.no_grad():
        predictions = head(features)
        diagnostics: dict[str, torch.Tensor] = {}
        for key in _COMPONENT_KEYS:
            pred_val = getattr(predictions, key)
            target_val = getattr(targets, key)
            diagnostics[key] = torch.mean(pred_val - target_val)
        return diagnostics


def batch_self_supervision(
    head: ParticipatoryValueHead,
    feature_batches: Iterable[torch.Tensor],
    target_batches: Iterable[ValueComponents],
) -> dict[str, torch.Tensor]:
    """Aggregate self-supervision diagnostics across batches."""

    params = list(head.parameters())
    device = params[0].device if params else torch.device("cpu")
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for features, targets in zip(feature_batches, target_batches, strict=True):
        diagnostics = self_supervision_step(head, features, targets)
        for key, value in diagnostics.items():
            if key not in totals:
                totals[key] = value.clone()
            else:
                totals[key].add_(value)
        count += 1
    if count == 0:
        return {key: torch.tensor(0.0, device=device) for key in _COMPONENT_KEYS}
    return {key: value / count for key, value in totals.items()}

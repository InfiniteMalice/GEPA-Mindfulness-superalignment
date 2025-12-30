"""Probes for participatory agency interpretability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn

from ..values import ValueComponents


@dataclass(frozen=True)
class ProbeResult:
    """Result of running a participatory agency probe."""

    values: ValueComponents
    metadata: Dict[str, float]


def _split_probe_outputs(outputs: torch.Tensor) -> ValueComponents:
    return ValueComponents(
        epistemic=outputs[..., 0],
        cooperation=outputs[..., 1],
        flexibility=outputs[..., 2],
        belonging=outputs[..., 3],
    )


class LinearValueProbe(nn.Module):
    """Linear probe that maps hidden features to value components."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 4)

    def forward(self, features: torch.Tensor) -> ValueComponents:
        outputs = self.projection(features)
        return _split_probe_outputs(outputs)


def run_probe(
    probe: LinearValueProbe,
    features: torch.Tensor,
    tags: Iterable[str] | None = None,
) -> ProbeResult:
    """Run a probe and attach optional tag metadata."""

    values = probe(features)
    metadata = {tag: 1.0 for tag in tags or []}
    return ProbeResult(values=values, metadata=metadata)

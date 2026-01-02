"""Probes for participatory agency interpretability."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Mapping

import torch
from torch import nn

from ..values import ValueComponents

_NUM_VALUE_HEADS = len(fields(ValueComponents))


@dataclass(frozen=True)
class ProbeResult:
    """Result of running a participatory agency probe."""

    values: ValueComponents
    metadata: Mapping[str, float]


def _split_probe_outputs(outputs: torch.Tensor) -> ValueComponents:
    """Split a probe tensor with shape [..., num_heads] into value components."""

    field_names = [field.name for field in fields(ValueComponents)]
    if outputs.shape[-1] != _NUM_VALUE_HEADS:
        raise ValueError(f"Expected last dimension {_NUM_VALUE_HEADS}, got {outputs.shape[-1]}")
    return ValueComponents(**{name: outputs[..., idx] for idx, name in enumerate(field_names)})


class LinearValueProbe(nn.Module):
    """Linear probe that maps hidden features to value components."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, _NUM_VALUE_HEADS)

    def forward(self, features: torch.Tensor) -> ValueComponents:
        outputs = self.projection(features)
        return _split_probe_outputs(outputs)


def run_probe(
    probe: LinearValueProbe,
    features: torch.Tensor,
    tags: Iterable[str] | None = None,
) -> ProbeResult:
    """Run a probe and attach optional tag metadata."""

    with torch.no_grad():
        values = probe(features)
    metadata = {tag: 1.0 for tag in tags or []}
    return ProbeResult(values=values, metadata=metadata)

"""Deployment and gating policies for participatory agency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..values import ValueComponents


@dataclass(frozen=True)
class DeploymentPolicy:
    """Simple policy for gating deployments based on value thresholds."""

    thresholds: dict[str, float]
    required_heads: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_heads", tuple(self.required_heads))

    def is_satisfied(self, values: ValueComponents) -> bool:
        """Return True when all required heads meet their thresholds."""

        valid_heads = {"epistemic", "cooperation", "flexibility", "belonging"}
        for head in self.required_heads:
            if head not in valid_heads:
                raise ValueError(f"Unknown head: {head!r}")
            threshold = self.thresholds.get(head, 0.0)
            component = values.as_dict()[head]
            if torch.any(component < threshold):
                return False
        return True

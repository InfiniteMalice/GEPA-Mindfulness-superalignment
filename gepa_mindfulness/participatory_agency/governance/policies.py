"""Deployment and gating policies for participatory agency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..values import ValueComponents


@dataclass(frozen=True)
class DeploymentPolicy:
    """Simple policy for gating deployments based on value thresholds."""

    thresholds: dict[str, float]
    required_heads: Iterable[str]

    def is_satisfied(self, values: ValueComponents) -> bool:
        """Return True when all required heads meet their thresholds."""

        for head in self.required_heads:
            threshold = self.thresholds.get(head, 0.0)
            component = values.as_dict()[head]
            if torch.any(component < threshold):
                return False
        return True

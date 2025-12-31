"""Deployment and gating policies for participatory agency."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from ..values import ValueComponents


@dataclass(frozen=True)
class DeploymentPolicy:
    """Simple policy for gating deployments based on value thresholds."""

    thresholds: Mapping[str, float]
    required_heads: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_heads", tuple(self.required_heads))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))
        valid_heads = {field.name for field in fields(ValueComponents)}
        unknown = set(self.required_heads) - valid_heads
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown heads: {unknown_list}")

    def is_satisfied(self, values: ValueComponents) -> bool:
        """Return True when all required heads meet their thresholds."""

        components = values.as_dict()
        for head in self.required_heads:
            threshold = self.thresholds.get(head, 0.0)
            component = components[head]
            if torch.any(component < threshold).item():
                return False
        return True

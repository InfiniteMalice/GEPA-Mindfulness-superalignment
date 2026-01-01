"""Value decomposition components and head module."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import nn

from .config import DEFAULT_HEAD_WEIGHTS, ParticipatoryAgencyConfig
from .heads import BelongingHead, CooperationHead, EpistemicHead, FlexibilityHead


@dataclass(frozen=True)
class ValueComponents:
    """Container for participatory agency value signals."""

    epistemic: torch.Tensor
    cooperation: torch.Tensor
    flexibility: torch.Tensor
    belonging: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return components as a dictionary."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    def total(self, weights: Mapping[str, float] | None = None) -> torch.Tensor:
        """Combine values into a single scalar using *weights*."""

        if weights is not None:
            required = {"epistemic", "cooperation", "flexibility", "belonging"}
            missing = required - weights.keys()
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"weights missing required keys: {missing_text}")
            unknown = weights.keys() - required
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(f"weights contains unknown keys: {unknown_text}")
            active = weights
        else:
            active = DEFAULT_HEAD_WEIGHTS
        return (
            self.epistemic * active["epistemic"]
            + self.cooperation * active["cooperation"]
            + self.flexibility * active["flexibility"]
            + self.belonging * active["belonging"]
        )

    def stack(self) -> torch.Tensor:
        """Stack components into a single tensor along the last axis."""

        field_values = [getattr(self, field.name) for field in fields(self)]
        return torch.stack(field_values, dim=-1)


class ParticipatoryValueHead(nn.Module):
    """Multi-head value decomposition for participatory agency."""

    def __init__(self, hidden_size: int, config: ParticipatoryAgencyConfig | None = None) -> None:
        super().__init__()
        resolved = config or ParticipatoryAgencyConfig(hidden_size=hidden_size)
        if resolved.hidden_size != hidden_size:
            raise ValueError("hidden_size must match the configuration hidden_size")
        self.config = resolved
        self.epistemic = EpistemicHead(
            hidden_size,
            dropout=resolved.dropout,
            bias=resolved.head_bias,
        )
        self.cooperation = CooperationHead(
            hidden_size,
            dropout=resolved.dropout,
            bias=resolved.head_bias,
        )
        self.flexibility = FlexibilityHead(
            hidden_size,
            dropout=resolved.dropout,
            bias=resolved.head_bias,
        )
        self.belonging = BelongingHead(
            hidden_size,
            dropout=resolved.dropout,
            bias=resolved.head_bias,
        )

    def forward(self, features: torch.Tensor) -> ValueComponents:
        """Compute all participatory agency values from *features*."""

        return ValueComponents(
            epistemic=self.epistemic(features),
            cooperation=self.cooperation(features),
            flexibility=self.flexibility(features),
            belonging=self.belonging(features),
        )

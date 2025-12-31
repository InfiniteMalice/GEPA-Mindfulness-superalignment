"""Configuration for participatory agency value decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

DEFAULT_HEAD_WEIGHTS: dict[str, float] = {
    "epistemic": 1.0,
    "cooperation": 1.0,
    "flexibility": 1.0,
    "belonging": 1.0,
}


@dataclass(frozen=True)
class ParticipatoryAgencyConfig:
    """Configuration for participatory agency heads and objectives."""

    hidden_size: int = 768
    dropout: float = 0.0
    head_bias: bool = True
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(dict(DEFAULT_HEAD_WEIGHTS))
    )
    reward_weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(dict(DEFAULT_HEAD_WEIGHTS))
    )

    def with_hidden_size(self, hidden_size: int) -> "ParticipatoryAgencyConfig":
        """Return a copy of the config with a new hidden size."""

        return ParticipatoryAgencyConfig(
            hidden_size=hidden_size,
            dropout=self.dropout,
            head_bias=self.head_bias,
            loss_weights=self.loss_weights,
            reward_weights=self.reward_weights,
        )

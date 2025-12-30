"""Configuration for participatory agency value decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


DEFAULT_HEAD_WEIGHTS: Dict[str, float] = {
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
    loss_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAD_WEIGHTS))
    reward_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAD_WEIGHTS))

    def with_hidden_size(self, hidden_size: int) -> "ParticipatoryAgencyConfig":
        """Return a copy of the config with a new hidden size."""

        return ParticipatoryAgencyConfig(
            hidden_size=hidden_size,
            dropout=self.dropout,
            head_bias=self.head_bias,
            loss_weights=dict(self.loss_weights),
            reward_weights=dict(self.reward_weights),
        )

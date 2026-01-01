"""Configuration for participatory agency value decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        return replace(self, hidden_size=hidden_size)

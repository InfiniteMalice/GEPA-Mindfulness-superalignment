"""Base value head implementation."""

from __future__ import annotations

import torch
from torch import nn


class BaseValueHead(nn.Module):
    """Base class for single-output value heads."""

    def __init__(self, hidden_size: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value logits."""

        dropped = self.dropout(features)
        output = self.projection(dropped)
        return output.squeeze(-1)

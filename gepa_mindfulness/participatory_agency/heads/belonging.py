"""Participatory identity and belonging value head."""

from __future__ import annotations

import torch
from torch import nn


class BelongingHead(nn.Module):
    """Predicts participatory identity and belonging value from hidden features."""

    def __init__(self, hidden_size: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute belonging logits."""

        dropped = self.dropout(features)
        output = self.projection(dropped)
        return output.squeeze(-1)

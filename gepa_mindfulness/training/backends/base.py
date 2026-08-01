"""Shared PyTorch backend records and tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import torch


class TokenizerLike(Protocol):
    """Minimal tokenizer surface required by the portable policy backend."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]:
        """Encode text without consulting external state."""

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs into response text."""


@dataclass(frozen=True)
class OptimizerStepResult:
    """Observable evidence that one optimizer update completed."""

    step: int
    gradient_norm: float | None = None


@dataclass(frozen=True)
class BackendCheckpointResult:
    """Format version and recorded optimizer step for a checkpoint operation."""

    format_version: int
    step: int


class TorchTensorOps:
    """PyTorch implementation of the backend-neutral algorithm tensor protocol."""

    @staticmethod
    def exp(value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value)

    @staticmethod
    def clip(value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        return torch.clamp(value, minimum, maximum)

    @staticmethod
    def minimum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.minimum(left, right)

    @staticmethod
    def maximum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.maximum(left, right)

    @staticmethod
    def square(value: torch.Tensor) -> torch.Tensor:
        return torch.square(value)

    @staticmethod
    def stack(values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tuple(values))

    @staticmethod
    def from_data(
        values: object,
        *,
        like: torch.Tensor,
        kind: Literal["float", "bool"] = "float",
    ) -> torch.Tensor:
        """Create an algorithm tensor on the policy tensor's device and representation."""
        if not isinstance(like, torch.Tensor):
            raise TypeError("like must be a torch.Tensor")
        if kind == "bool":
            dtype = torch.bool
        elif kind == "float":
            if not like.is_floating_point():
                raise TypeError("like must use a floating-point dtype for float data")
            dtype = like.dtype
        else:
            raise ValueError("kind must be 'float' or 'bool'")
        return torch.as_tensor(values, dtype=dtype, device=like.device)

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average selected entries while rejecting shape mismatch and an empty mask."""
        if value.shape != mask.shape:
            raise ValueError("value and mask must have matching shapes")
        selected = torch.masked_select(value, mask.to(dtype=torch.bool, device=value.device))
        if selected.numel() == 0:
            raise ValueError("masked_mean requires at least one selected value")
        return selected.mean()


__all__ = [
    "BackendCheckpointResult",
    "OptimizerStepResult",
    "TokenizerLike",
    "TorchTensorOps",
]

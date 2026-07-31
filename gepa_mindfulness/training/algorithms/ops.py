"""Small operation boundary used by backend-neutral RL tensor formulas."""

from __future__ import annotations

from typing import Any, Literal, Protocol

Tensor = Any


class TensorOps(Protocol):
    """Tensor primitives required by the PPO and GRPO objective functions."""

    def exp(self, value: Tensor) -> Tensor:
        """Compute element-wise exponentials."""

    def clip(self, value: Tensor, minimum: float, maximum: float) -> Tensor:
        """Clip values to an inclusive interval."""

    def minimum(self, left: Tensor, right: Tensor) -> Tensor:
        """Compute element-wise minima."""

    def maximum(self, left: Tensor, right: Tensor) -> Tensor:
        """Compute element-wise maxima."""

    def square(self, value: Tensor) -> Tensor:
        """Square values element-wise."""

    def stack(self, values: list[Tensor]) -> Tensor:
        """Stack tensors along a new leading dimension."""

    def from_data(
        self,
        values: object,
        *,
        like: Tensor,
        kind: Literal["float", "bool"] = "float",
    ) -> Tensor:
        """Create a tensor from backend-neutral data beside an existing tensor."""

    def masked_mean(self, value: Tensor, mask: Tensor) -> Tensor:
        """Average selected values and reject masks with no selected entries."""


__all__ = ["Tensor", "TensorOps"]

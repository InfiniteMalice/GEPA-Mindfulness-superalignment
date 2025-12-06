"""Value space representations for deep and shallow signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, List, Sequence, TypeVar

from ..utils.imports import optional_import

torch = optional_import("torch")


FloatList = List[float]
VectorType = TypeVar("VectorType", bound="BaseValueVector")
logger = logging.getLogger(__name__)


def to_tensor(values: Sequence[float]) -> Any:
    if torch is None:
        return list(values)
    return torch.tensor(list(values), dtype=torch.float32)


def to_float_list(values: Iterable[float]) -> FloatList:
    result: FloatList = []
    if torch is not None and hasattr(values, "detach"):
        try:
            tensor = values.detach().cpu().view(-1)
            return [float(x) for x in tensor.tolist()]
        except (AttributeError, RuntimeError, TypeError):
            pass  # Fallback to regular iteration for non-tensor iterables
    for value in values:
        result.append(float(value))
    return result


class BaseValueVector:
    ORDER: ClassVar[tuple[str, ...]]

    def to_list(self) -> FloatList:
        return [float(getattr(self, key)) for key in self.ORDER]

    def to_tensor(self) -> Any:
        return to_tensor(self.to_list())

    @classmethod
    def from_tensor(cls: type[VectorType], values: Sequence[float]) -> VectorType:
        floats = to_float_list(values)
        if len(floats) > len(cls.ORDER):
            logger.debug(
                "Truncating %d values to %d for %s",
                len(floats),
                len(cls.ORDER),
                cls.__name__,
            )
            floats = floats[: len(cls.ORDER)]
        padding = len(cls.ORDER) - len(floats)
        if padding > 0:
            logger.debug(
                "Padding %s with %d zeros to match ORDER length",
                cls.__name__,
                padding,
            )
        padded = floats + [0.0] * padding
        kwargs = {name: padded[idx] for idx, name in enumerate(cls.ORDER)}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.ORDER}


@dataclass
class DeepValueVector(BaseValueVector):
    """Container for GEPA deep values and contemplative stances."""

    reduce_suffering: float = 0.0
    increase_prosperity: float = 0.0
    advance_knowledge: float = 0.0
    mindfulness: float = 0.0
    empathy: float = 0.0
    perspective: float = 0.0
    agency: float = 0.0

    ORDER: ClassVar[tuple[str, ...]] = (
        "reduce_suffering",
        "increase_prosperity",
        "advance_knowledge",
        "mindfulness",
        "empathy",
        "perspective",
        "agency",
    )


@dataclass
class ShallowPreferenceVector(BaseValueVector):
    """Container for shallow, stylistic preferences."""

    tone_formal: float = 0.0
    tone_casual: float = 0.0
    tone_therapeutic: float = 0.0
    verbosity: float = 0.0
    hedging: float = 0.0
    directness: float = 0.0
    deference: float = 0.0
    assertiveness: float = 0.0

    ORDER: ClassVar[tuple[str, ...]] = (
        "tone_formal",
        "tone_casual",
        "tone_therapeutic",
        "verbosity",
        "hedging",
        "directness",
        "deference",
        "assertiveness",
    )


__all__ = [
    "BaseValueVector",
    "DeepValueVector",
    "ShallowPreferenceVector",
    "to_tensor",
    "to_float_list",
]

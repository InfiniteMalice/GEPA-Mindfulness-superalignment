"""Value space representations for deep and shallow signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..utils.imports import optional_import

torch = optional_import("torch")


FloatList = List[float]


def _to_tensor(values: Sequence[float]):  # type: ignore[override]
    if torch is None:
        return list(values)
    return torch.tensor(list(values), dtype=torch.float)


def _to_float_list(values: Iterable[float]) -> FloatList:
    result: FloatList = []
    if torch is not None and hasattr(values, "detach"):
        try:
            tensor = values.detach().cpu().view(-1)
            return [float(x) for x in tensor.tolist()]
        except Exception:
            pass
    for value in values:
        result.append(float(value))
    return result


@dataclass
class DeepValueVector:
    """Container for GEPA deep values and contemplative stances."""

    reduce_suffering: float = 0.0
    increase_prosperity: float = 0.0
    advance_knowledge: float = 0.0
    mindfulness: float = 0.0
    empathy: float = 0.0
    perspective: float = 0.0
    agency: float = 0.0

    ORDER: tuple[str, ...] = (
        "reduce_suffering",
        "increase_prosperity",
        "advance_knowledge",
        "mindfulness",
        "empathy",
        "perspective",
        "agency",
    )

    def to_list(self) -> FloatList:
        return [float(getattr(self, key)) for key in self.ORDER]

    def to_tensor(self):  # type: ignore[override]
        return _to_tensor(self.to_list())

    @classmethod
    def from_tensor(cls, values: Sequence[float]) -> "DeepValueVector":  # type: ignore[override]
        floats = _to_float_list(values)
        padded = list(floats) + [0.0] * (len(cls.ORDER) - len(floats))
        kwargs = {name: padded[idx] for idx, name in enumerate(cls.ORDER)}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.ORDER}


@dataclass
class ShallowPreferenceVector:
    """Container for shallow, stylistic preferences."""

    tone_formal: float = 0.0
    tone_casual: float = 0.0
    tone_therapeutic: float = 0.0
    verbosity: float = 0.0
    hedging: float = 0.0
    directness: float = 0.0
    deference: float = 0.0
    assertiveness: float = 0.0

    ORDER: tuple[str, ...] = (
        "tone_formal",
        "tone_casual",
        "tone_therapeutic",
        "verbosity",
        "hedging",
        "directness",
        "deference",
        "assertiveness",
    )

    def to_list(self) -> FloatList:
        return [float(getattr(self, key)) for key in self.ORDER]

    def to_tensor(self):  # type: ignore[override]
        return _to_tensor(self.to_list())

    @classmethod
    def from_tensor(
        cls, values: Sequence[float]
    ) -> "ShallowPreferenceVector":  # type: ignore[override]
        floats = _to_float_list(values)
        padded = list(floats) + [0.0] * (len(cls.ORDER) - len(floats))
        kwargs = {name: padded[idx] for idx, name in enumerate(cls.ORDER)}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.ORDER}


__all__ = [
    "DeepValueVector",
    "ShallowPreferenceVector",
    "_to_tensor",
    "_to_float_list",
]

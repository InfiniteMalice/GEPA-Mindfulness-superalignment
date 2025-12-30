"""Static and runtime checks for participatory agency."""

from __future__ import annotations

from typing import Dict

import torch

from ..values import ValueComponents


def values_are_finite(values: ValueComponents) -> bool:
    """Return True if all value components are finite."""

    return bool(torch.isfinite(values.stack()).all())


def values_within_range(
    values: ValueComponents,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> bool:
    """Return True if all values are within [minimum, maximum]."""

    stacked = values.stack()
    return bool(torch.all(stacked >= minimum) and torch.all(stacked <= maximum))


def build_check_report(values: ValueComponents) -> Dict[str, bool]:
    """Return a dictionary of standard governance checks."""

    return {
        "finite": values_are_finite(values),
        "bounded": values_within_range(values),
    }

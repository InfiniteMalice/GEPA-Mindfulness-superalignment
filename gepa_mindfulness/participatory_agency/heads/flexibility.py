"""Goal flexibility value head.

The head subclasses are kept distinct for semantic clarity and future specialization even
though they currently share the BaseValueHead behavior.
"""

from __future__ import annotations

from .base import BaseValueHead


class FlexibilityHead(BaseValueHead):
    """Predicts goal flexibility value from hidden features."""

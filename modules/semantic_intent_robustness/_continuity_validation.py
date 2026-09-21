"""Bounded input validation for the experimental continuity diagnostics."""

from math import isfinite


def text_field(value: object, name: str, limit: int = 256) -> None:
    """Require a bounded nonblank string without accepting arbitrary objects."""
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonblank string of at most {limit} characters")


def references(value: object, name: str) -> None:
    """Require immutable, unique bounded reference identifiers."""
    if type(value) is not tuple or len(value) > 128:
        raise ValueError(f"{name} must be a tuple of at most 128 references")
    for item in value:
        text_field(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate references")


def index_field(value: object, name: str) -> None:
    """Require a nonnegative serialization-safe integer."""
    if type(value) is not int or not 0 <= value <= 9_007_199_254_740_991:
        raise ValueError(f"{name} must be a nonnegative serialization-safe integer")


def score(value: object, name: str) -> None:
    """Require a finite scalar in the normalized unit interval."""
    if (
        (type(value) is not int and type(value) is not float)
        or not 0 <= value <= 1
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def boolean(value: object, name: str) -> None:
    """Reject truthy values at diagnostic decision boundaries."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")

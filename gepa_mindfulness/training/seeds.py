"""Shared seed bounds for portable and native RL request boundaries."""

from __future__ import annotations

# llama.cpp consumes a uint32 seed. Python, Torch, JSON, and Mojo accept this entire range, making
# uint32 the tightest proven common boundary rather than Python's effectively unbounded integer.
# llama.cpp reserves UINT32_MAX (0xFFFFFFFF) as its random-seed sentinel.
# The largest portable deterministic seed is therefore one value lower.
MAX_SEED = 2**32 - 2


def validate_seed(
    value: object,
    field_name: str = "seed",
    *,
    sample_count: int = 1,
) -> int | None:
    """Return a seed whose complete sample expansion fits the common runtime range."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer or null")
    if not 0 <= value <= MAX_SEED:
        raise ValueError(f"{field_name} must be in [0, {MAX_SEED}]")
    if sample_count - 1 > MAX_SEED - value:
        raise ValueError(f"{field_name} sample expansion would overflow {MAX_SEED}")
    return value


__all__ = ["MAX_SEED", "validate_seed"]

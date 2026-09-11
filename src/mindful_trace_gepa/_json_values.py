"""Shared immutable JSON-value boundary for action-bound logging records."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import cast

_MAX_SERIALIZATION_SAFE_INTEGER = 9_007_199_254_740_991


class _FrozenJSONArray(tuple[object, ...]):
    """An immutable JSON array that retains value equality with legacy list payloads."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) != tuple(other)
        return NotImplemented

    __hash__ = tuple.__hash__


def freeze_json_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    """Validate and deep-snapshot one JSON object into immutable containers."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON-compatible mapping")
    return cast(
        Mapping[str, object],
        freeze_json_value(value, field_name=field_name),
    )


def freeze_json_value(value: object, *, field_name: str) -> object:
    """Validate and deep-snapshot one JSON value into immutable containers."""

    return _freeze_json_value(value, field_name, None)


def require_serialization_safe_integer(field_name: str, value: object) -> int:
    """Return a built-in integer in the interoperable JSON exact-integer range."""

    if type(value) is not int:
        raise ValueError(f"{field_name} must be a built-in integer")
    integer = cast(int, value)
    if not -_MAX_SERIALIZATION_SAFE_INTEGER <= integer <= _MAX_SERIALIZATION_SAFE_INTEGER:
        raise ValueError(
            f"{field_name} must be a serialization-safe JSON integer from "
            f"-{_MAX_SERIALIZATION_SAFE_INTEGER} through {_MAX_SERIALIZATION_SAFE_INTEGER}"
        )
    return integer


def thaw_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a fresh ordinary JSON object from immutable internal containers."""

    return {key: thaw_json_value(item) for key, item in value.items()}


def thaw_json_value(value: object) -> object:
    """Return a fresh ordinary JSON value from immutable internal containers."""

    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def _freeze_json_value(
    value: object,
    field_name: str,
    active_containers: set[int] | None,
) -> object:
    """Recursively validate one JSON value while tracking only its active ancestry."""

    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int:
        return require_serialization_safe_integer(field_name, value)
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_object(value, field_name, active_containers)
    if isinstance(value, (list, tuple)):
        return _freeze_json_array(value, field_name, active_containers)
    raise ValueError(f"{field_name} must contain only JSON-compatible values")


def _freeze_json_object(
    value: Mapping[object, object],
    field_name: str,
    active_containers: set[int] | None,
) -> Mapping[str, object]:
    """Freeze a JSON object with deterministic key order and string-only keys."""

    active = _enter_container(value, field_name, active_containers)
    try:
        keys: list[str] = []
        for key in value:
            if type(key) is not str:
                raise ValueError(f"{field_name} must use only string mapping keys")
            keys.append(key)
        items = {key: _freeze_json_value(value[key], field_name, active) for key in sorted(keys)}
        return MappingProxyType(items)
    finally:
        active.remove(id(value))


def _freeze_json_array(
    value: list[object] | tuple[object, ...],
    field_name: str,
    active_containers: set[int] | None,
) -> tuple[object, ...]:
    """Freeze a JSON array into a tuple after detecting recursive references."""

    active = _enter_container(value, field_name, active_containers)
    try:
        return _FrozenJSONArray(_freeze_json_value(item, field_name, active) for item in value)
    finally:
        active.remove(id(value))


def _enter_container(
    value: object,
    field_name: str,
    active_containers: set[int] | None,
) -> set[int]:
    """Reject a container already present in its own active ancestry."""

    active = active_containers if active_containers is not None else set()
    value_id = id(value)
    if value_id in active:
        raise ValueError(f"{field_name} must not contain a cycle")
    active.add(value_id)
    return active


__all__ = [
    "freeze_json_mapping",
    "freeze_json_value",
    "require_serialization_safe_integer",
    "thaw_json_mapping",
    "thaw_json_value",
]

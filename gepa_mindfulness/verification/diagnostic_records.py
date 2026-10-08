"""Exact serialization utilities for inert public verification diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, ClassVar, TypeVar

from ..core.evidence import EvidenceReference
from .epistemic_state import _array, _encode, _strings, _text, _unit
from .state import _require_exact_mapping, _restore_evidence_refs, _snapshot_evidence_refs

R = TypeVar("R", bound="DiagnosticRecord")


def public_refs(value: object, *, required: bool = False) -> tuple[EvidenceReference, ...]:
    """Snapshot public provenance; source labels do not authenticate evidence."""
    refs = _snapshot_evidence_refs(value)
    if required and not refs:
        raise ValueError("public provenance requires evidence_refs")
    if any(not ref.is_observable for ref in refs):
        raise ValueError("public provenance requires observable evidence")
    if len({ref.reference_id for ref in refs}) != len(refs):
        raise ValueError("evidence reference IDs must be unique")
    return refs


def strings(value: object, name: str, *, required: bool = False) -> tuple[str, ...]:
    """Snapshot unique identifiers or descriptions."""
    result = _strings(value, name, required=required)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique strings")
    return result


def choice(value: object, name: str, allowed: tuple[str, ...]) -> None:
    """Reject unknown or coerced categorical values."""
    if type(value) is not str or value not in allowed:
        raise ValueError(f"unsupported {name}: {value!r}")


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """Versioned exact JSON records, always excluded from optimization by default."""

    schema_version: ClassVar[str]
    restorers: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    def __post_init__(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Revalidate and detach the serialized record."""
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "training_eligibility": "DEVELOPMENT",
            **{field.name: encode(getattr(self, field.name)) for field in fields(self)},
        }

    @classmethod
    def from_dict(cls: type[R], data: object) -> R:
        """Restore exact version and fields without accepting authority extensions."""
        values = dict(
            _require_exact_mapping(
                data,
                {f.name for f in fields(cls)} | {"schema_version", "training_eligibility"},
                cls.__name__,
            )
        )
        if values.pop("schema_version") != cls.schema_version:
            raise ValueError("unsupported diagnostic schema_version")
        if values.pop("training_eligibility") != "DEVELOPMENT":
            raise ValueError("diagnostics require DEVELOPMENT eligibility")
        for name, restore in cls.restorers.items():
            values[name] = restore(values[name])
        return cls(**values)


def encode(value: Any) -> Any:
    """Serialize composed canonical records without altering their wire format."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    return _encode(value)


def restore_refs(value: object) -> tuple[EvidenceReference, ...]:
    """Restore canonical evidence references."""
    return _restore_evidence_refs(value, "diagnostic")


def records(value: object, cls: type[R]) -> tuple[R, ...]:
    """Detach exact typed records and reject duck-typed authority objects."""
    result = []
    for item in _array(value, cls.__name__):
        if type(item) is not cls:
            raise ValueError(f"expected exact {cls.__name__}")
        result.append(cls.from_dict(item.to_dict()))
    return tuple(result)


def restore_records(value: object, cls: type[R]) -> tuple[R, ...]:
    """Restore a sequence of one diagnostic record type."""
    return tuple(cls.from_dict(item) for item in _array(value, cls.__name__))


__all__ = ["DiagnosticRecord", "choice", "public_refs", "strings", "_text", "_unit"]

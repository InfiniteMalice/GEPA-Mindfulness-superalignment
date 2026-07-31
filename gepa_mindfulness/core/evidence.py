"""Typed provenance for observable and internal trajectory evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class EvidenceSourceKind(str, Enum):
    """The captured source category for one immutable evidence reference."""

    OBSERVABLE_OUTPUT = "observable_output"
    OBSERVABLE_ACTION = "observable_action"
    EXTERNAL_RECORD = "external_record"
    PRIVATE_REASONING = "private_reasoning"
    LATENT_STATE = "latent_state"
    ATTENTION_DATA = "attention_data"
    CACHE_DATA = "cache_data"


OBSERVABLE_EVIDENCE_SOURCE_KINDS = frozenset(
    {
        EvidenceSourceKind.OBSERVABLE_OUTPUT,
        EvidenceSourceKind.OBSERVABLE_ACTION,
        EvidenceSourceKind.EXTERNAL_RECORD,
    }
)


@dataclass(frozen=True)
class EvidenceReference:
    """An immutable reference whose source kind is declared when evidence is captured."""

    reference_id: str
    source_kind: EvidenceSourceKind

    def __post_init__(self) -> None:
        """Require a non-empty identifier and an enum value without string coercion."""
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("reference_id must be a non-empty string.")
        if not isinstance(self.source_kind, EvidenceSourceKind):
            raise ValueError("source_kind must be an EvidenceSourceKind.")

    @property
    def is_observable(self) -> bool:
        """Return whether rewards may use this source kind as observable evidence."""
        return self.source_kind in OBSERVABLE_EVIDENCE_SOURCE_KINDS

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""
        return {
            "reference_id": self.reference_id,
            "source_kind": self.source_kind.value,
        }

    @classmethod
    def from_dict(cls, data: object) -> "EvidenceReference":
        """Restore an exact evidence-reference object from JSON-compatible data."""
        if not isinstance(data, Mapping):
            raise ValueError("Evidence reference must be an object.")
        expected_fields = {"reference_id", "source_kind"}
        if set(data) != expected_fields:
            raise ValueError("Evidence reference requires exactly reference_id and source_kind.")
        reference_id = data["reference_id"]
        source_kind = data["source_kind"]
        if not isinstance(reference_id, str):
            raise ValueError("Evidence reference reference_id must be a string.")
        if not isinstance(source_kind, str):
            raise ValueError("Evidence reference source_kind must be a string.")
        try:
            parsed_kind = EvidenceSourceKind(source_kind)
        except ValueError as error:
            raise ValueError(f"Unknown evidence source kind {source_kind!r}.") from error
        return cls(reference_id=reference_id, source_kind=parsed_kind)


__all__ = [
    "EvidenceReference",
    "EvidenceSourceKind",
    "OBSERVABLE_EVIDENCE_SOURCE_KINDS",
]

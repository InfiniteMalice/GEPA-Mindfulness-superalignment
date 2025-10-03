"""Typed signatures used by the declarative DSPy-like pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping


@dataclass(frozen=True)
class ModuleSignature:
    """Lightweight descriptor for a module boundary.

    The project does not depend on the real DSPy package, but we mimic its
    declarative style by capturing the required inputs and the produced output
    field name.  These descriptors are consumed by :class:`GEPAChain` to ensure
    that each module publishes trace checkpoints with consistent metadata.
    """

    name: str
    input_fields: tuple[str, ...]
    output_field: str

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "inputs": self.input_fields,
            "output": self.output_field,
        }


Framing = ModuleSignature("Framing", ("inquiry",), "framing_text")
Evidence = ModuleSignature("Evidence", ("inquiry", "context"), "evidence_text")
Tensions = ModuleSignature("Tensions", ("evidence",), "tensions_text")
OptionsForecast = ModuleSignature("OptionsForecast", ("tensions",), "options_text")
Decision = ModuleSignature("Decision", ("options",), "decision_text")
Safeguards = ModuleSignature("Safeguards", ("decision",), "safeguards_text")
Reflection = ModuleSignature("Reflection", ("history",), "reflection_text")

ALL_SIGNATURES: tuple[ModuleSignature, ...] = (
    Framing,
    Evidence,
    Tensions,
    OptionsForecast,
    Decision,
    Safeguards,
    Reflection,
)


SignatureCallable = Callable[[Dict[str, Any]], str]

__all__ = [
    "ModuleSignature",
    "SignatureCallable",
    "Framing",
    "Evidence",
    "Tensions",
    "OptionsForecast",
    "Decision",
    "Safeguards",
    "Reflection",
    "ALL_SIGNATURES",
]

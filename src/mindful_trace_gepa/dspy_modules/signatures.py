"""Typed signatures used by the declarative DSPy-like pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

try:  # pragma: no cover - optional dependency
    import dspy  # type: ignore
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore


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


if dspy is not None:

    class DualPathFraming(dspy.Signature):
        """Frame the inquiry with careful and confident perspectives."""

        inquiry = dspy.InputField(desc="The user's question or problem")
        dual_path_framing = dspy.OutputField(desc="Dual-path framing following template")


    class DualPathEvidence(dspy.Signature):
        """Gather evidence using both careful and confident lenses."""

        dual_path_framing = dspy.InputField(desc="Framing from previous step")
        context = dspy.InputField(desc="Additional context")
        dual_path_evidence = dspy.OutputField(desc="Evidence gathered via both paths")


    class DualPathDecision(dspy.Signature):
        """Compare approaches and recommend an appropriate path."""

        dual_path_evidence = dspy.InputField(desc="Evidence from both paths")
        dual_path_decision = dspy.OutputField(desc="Comparison, recommendation, and rationale")


else:  # pragma: no cover - executed when dspy missing

    class DualPathFraming:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ImportError("dspy is required to use DualPathFraming")


    class DualPathEvidence:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ImportError("dspy is required to use DualPathEvidence")


    class DualPathDecision:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ImportError("dspy is required to use DualPathDecision")


__all__.extend(["DualPathFraming", "DualPathEvidence", "DualPathDecision"])

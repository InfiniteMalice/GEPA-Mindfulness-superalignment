"""Typed signatures used by the declarative DSPy-like pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

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
ClassifyReasoningUnits = ModuleSignature(
    "ClassifyReasoningUnits",
    ("task", "response"),
    "reasoning_units",
)
SelectReasoningUnits = ModuleSignature("SelectReasoningUnits", ("task",), "selected_units")
FrameTask = ModuleSignature("FrameTask", ("task",), "task_frame")
DetermineGrounding = ModuleSignature("DetermineGrounding", ("claim", "evidence"), "grounding")
SelectMethod = ModuleSignature("SelectMethod", ("task_frame",), "method")
TrackAssumptions = ModuleSignature("TrackAssumptions", ("task", "context"), "assumptions")
EstimateUncertainty = ModuleSignature("EstimateUncertainty", ("claim", "evidence"), "uncertainty")
CalibrationDecision = ModuleSignature("CalibrationDecision", ("confidence",), "calibration")
ScientificMethodCheck = ModuleSignature("ScientificMethodCheck", ("claim",), "scientific_check")
MDLControlGate = ModuleSignature("MDLControlGate", ("default", "controlled"), "mdl_gate")
UnitTraceEvaluator = ModuleSignature("UnitTraceEvaluator", ("trace",), "unit_trace_score")
CaseV3Classifier = ModuleSignature("CaseV3Classifier", ("response", "metadata"), "case_v3")
DetectInvariantsUnderTransformationSig = ModuleSignature(
    "DetectInvariantsUnderTransformation",
    ("original", "transformed", "target_property"),
    "invariant_properties",
)
ClassifyEquivalenceClassSig = ModuleSignature(
    "ClassifyEquivalenceClass",
    ("variants", "equivalence_criterion"),
    "classes",
)
CanonicalizeRepresentationSig = ModuleSignature(
    "CanonicalizeRepresentation",
    ("input_text", "canonical_schema"),
    "canonical_form",
)
DetectSymmetryBreakSig = ModuleSignature(
    "DetectSymmetryBreak",
    ("case_a", "case_b", "proposed_symmetry"),
    "symmetry_breaking_factors",
)
GenerateTransformationOrbitSig = ModuleSignature(
    "GenerateTransformationOrbit",
    ("seed_case", "allowed_transformations"),
    "generated_variants",
)

ALL_SIGNATURES: tuple[ModuleSignature, ...] = (
    Framing,
    Evidence,
    Tensions,
    OptionsForecast,
    Decision,
    Safeguards,
    Reflection,
    ClassifyReasoningUnits,
    SelectReasoningUnits,
    FrameTask,
    DetermineGrounding,
    SelectMethod,
    TrackAssumptions,
    EstimateUncertainty,
    CalibrationDecision,
    ScientificMethodCheck,
    MDLControlGate,
    UnitTraceEvaluator,
    CaseV3Classifier,
    DetectInvariantsUnderTransformationSig,
    ClassifyEquivalenceClassSig,
    CanonicalizeRepresentationSig,
    DetectSymmetryBreakSig,
    GenerateTransformationOrbitSig,
)


SignatureCallable = Callable[[dict[str, Any]], str]

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
    "ClassifyReasoningUnits",
    "SelectReasoningUnits",
    "FrameTask",
    "DetermineGrounding",
    "SelectMethod",
    "TrackAssumptions",
    "EstimateUncertainty",
    "CalibrationDecision",
    "ScientificMethodCheck",
    "MDLControlGate",
    "UnitTraceEvaluator",
    "CaseV3Classifier",
    "DetectInvariantsUnderTransformationSig",
    "ClassifyEquivalenceClassSig",
    "CanonicalizeRepresentationSig",
    "DetectSymmetryBreakSig",
    "GenerateTransformationOrbitSig",
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

    class DetectInvariantsUnderTransformation(dspy.Signature):
        """Identify what properties remain invariant under a transformation."""

        original = dspy.InputField()
        transformed = dspy.InputField()
        target_property = dspy.InputField()
        invariant_properties = dspy.OutputField()
        changed_properties = dspy.OutputField()

    class ClassifyEquivalenceClass(dspy.Signature):
        """Group variants into equivalence classes based on preserved structure."""

        variants = dspy.InputField()
        equivalence_criterion = dspy.InputField()
        classes = dspy.OutputField()
        symmetry_breaks = dspy.OutputField()

    class CanonicalizeRepresentation(dspy.Signature):
        """Convert surface-different variants into a canonical representation."""

        input_text = dspy.InputField()
        canonical_schema = dspy.InputField()
        canonical_form = dspy.OutputField()
        lost_information = dspy.OutputField()

    class DetectSymmetryBreak(dspy.Signature):
        """Detect when two apparently similar cases differ in a relevant way."""

        case_a = dspy.InputField()
        case_b = dspy.InputField()
        proposed_symmetry = dspy.InputField()
        symmetry_holds = dspy.OutputField()
        symmetry_breaking_factors = dspy.OutputField()

    class GenerateTransformationOrbit(dspy.Signature):
        """Generate or analyze variants reachable under allowed transformations."""

        seed_case = dspy.InputField()
        allowed_transformations = dspy.InputField()
        generated_variants = dspy.OutputField()
        invariant_to_test = dspy.OutputField()

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

    class DetectInvariantsUnderTransformation:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("dspy is required to use this signature")

    class ClassifyEquivalenceClass:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("dspy is required to use this signature")

    class CanonicalizeRepresentation:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("dspy is required to use this signature")

    class DetectSymmetryBreak:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("dspy is required to use this signature")

    class GenerateTransformationOrbit:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("dspy is required to use this signature")


__all__.extend(
    [
        "DualPathFraming",
        "DualPathEvidence",
        "DualPathDecision",
        "DetectInvariantsUnderTransformation",
        "ClassifyEquivalenceClass",
        "CanonicalizeRepresentation",
        "DetectSymmetryBreak",
        "GenerateTransformationOrbit",
    ]
)

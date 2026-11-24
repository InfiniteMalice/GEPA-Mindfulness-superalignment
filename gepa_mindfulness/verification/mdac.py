"""Diagnostic-only MDAC records for observed behavior and competing explanations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ..core.clarifying_abstention import FRAMEWORK_CASE_IDS
from ..core.evidence import EvidenceReference
from .diagnostic_records import (
    DiagnosticRecord,
    _text,
    _unit,
    choice,
    public_refs,
    records,
    restore_records,
    restore_refs,
    strings,
)

# Stable identifiers are independent of the canonical behavioral case manifest.
MECHANISMS = MappingProxyType(
    {
        "M01": ("Information loss", "Critical information retention"),
        "M02": ("Retrieval failure", "Relevant evidence recall"),
        "M03": ("Inappropriate memory reuse", "Retrieval-induced regression rate"),
        "M04": ("Contextual misunderstanding", "Semantic intent consistency"),
        "M05": ("Unrecognized uncertainty", "Decision coverage recall"),
        "M06": ("Miscalibrated confidence", "Calibration error"),
        "M07": ("Reasoning failure", "Reasoning transition validity"),
        "M08": ("Temporal information loss", "Temporal dependency retention"),
        "M09": ("Constitutional forgetting", "Ethical principle retention"),
        "M10": ("Value conflict mishandling", "Value decomposition accuracy"),
        "M11": ("Prediction failure", "Prediction–observation discrepancy"),
        "M12": ("Ethical generalization failure", "Constitutional OOD generalization"),
        "M13": ("Objective misgeneralization", "Objective invariance and sensitivity"),
        "M14": ("Reward hacking hypothesis", "Reward–behavior divergence"),
        "M15": ("Suspected strategic deception", "Convergent deception indicators"),
        "M16": ("Correction resistance", "Corrigibility effectiveness"),
    }
)
STATES = (
    "untested",
    "suspected",
    "supported",
    "strongly_supported",
    "contradicted",
    "inconclusive",
    "resolved",
    "regressed",
)


@dataclass(frozen=True, slots=True)
class MechanismAssessment(DiagnosticRecord):
    """One evaluator-declared explanation, with evidence for and against it."""

    schema_version = "mdac-mechanism-v1"
    restorers = {"supporting": restore_refs, "contradictory": restore_refs}

    mechanism: str
    state: str = "untested"
    confidence: float | None = None
    supporting: tuple[EvidenceReference, ...] = ()
    contradictory: tuple[EvidenceReference, ...] = ()
    alternatives: tuple[str, ...] = ()
    intervention_results: tuple[str, ...] = ()
    statistical_uncertainty: str | None = None
    recommended_curriculum: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        choice(self.mechanism, "mechanism", tuple(MECHANISMS))
        choice(self.state, "state", STATES)
        if self.confidence is not None:
            _unit(self.confidence, "confidence")
        if self.statistical_uncertainty is not None:
            _text(self.statistical_uncertainty, "statistical_uncertainty")
        for name in ("alternatives", "intervention_results", "recommended_curriculum"):
            object.__setattr__(self, name, strings(getattr(self, name), name))
        for name in ("supporting", "contradictory"):
            object.__setattr__(self, name, public_refs(getattr(self, name)))
        if self.state in {"supported", "strongly_supported"} and not self.supporting:
            raise ValueError("supported hypotheses require supporting evidence")
        if self.state == "contradicted" and not self.contradictory:
            raise ValueError("contradicted hypotheses require contradictory evidence")
        if self.mechanism == "M15" and self.state in {"supported", "strongly_supported"}:
            if len(self.supporting) < 2 or not self.alternatives:
                raise ValueError("M15 support requires multiple references and alternatives")


@dataclass(frozen=True, slots=True)
class MDACRecord(DiagnosticRecord):
    """An observation can be recorded before any mechanism has been identified."""

    schema_version = "mdac-observation-v1"
    restorers = {
        "assessments": lambda value: restore_records(value, MechanismAssessment),
        "provenance": restore_refs,
    }

    diagnostic_id: str
    observed_behavior: str
    expected_behavior: str | None = None
    task_id: str | None = None
    case_id: int | None = None
    run_id: str | None = None
    model_id: str | None = None
    checkpoint_id: str | None = None
    harness_version: str | None = None
    configuration_hash: str | None = None
    rl_method: str | None = None
    evaluator_version: str | None = None
    dataset_version: str | None = None
    seed: int | None = None
    constitutional_principles: tuple[str, ...] = ()
    assessments: tuple[MechanismAssessment, ...] = ()
    severity: float | None = None
    occurrences: int | None = None
    opportunities: int | None = None
    resolution_status: str = "untested"
    provenance: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        _text(self.diagnostic_id, "diagnostic_id")
        _text(self.observed_behavior, "observed_behavior")
        for name in (
            "expected_behavior",
            "task_id",
            "run_id",
            "model_id",
            "checkpoint_id",
            "harness_version",
            "configuration_hash",
            "rl_method",
            "evaluator_version",
            "dataset_version",
        ):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if self.case_id is not None and (
            type(self.case_id) is not int or self.case_id not in FRAMEWORK_CASE_IDS
        ):
            raise ValueError("case_id must identify a canonical case")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        choice(self.resolution_status, "resolution_status", STATES)
        if self.severity is not None:
            _unit(self.severity, "severity")
        if self.occurrences is not None or self.opportunities is not None:
            if (
                type(self.occurrences) is not int
                or type(self.opportunities) is not int
                or not 0 <= self.occurrences <= self.opportunities
            ):
                raise ValueError("counts must satisfy 0 <= occurrences <= opportunities")
        assessments = records(self.assessments, MechanismAssessment)
        if len({item.mechanism for item in assessments}) != len(assessments):
            raise ValueError("mechanisms must be unique within an observation")
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(self, "provenance", public_refs(self.provenance))
        object.__setattr__(
            self,
            "constitutional_principles",
            strings(self.constitutional_principles, "constitutional_principles"),
        )

    @property
    def frequency(self) -> float | None:
        if not self.opportunities or self.occurrences is None:
            return None
        return self.occurrences / self.opportunities


def load_diagnostic(payload: object, *, enabled: bool = False) -> MDACRecord | None:
    """Opt-in reader for an MDAC observation attached to an evaluation record."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    return MDACRecord.from_dict(payload) if enabled else None

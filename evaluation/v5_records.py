"""Immutable, registry-validated records for V5 evaluation observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from mindful_trace_gepa._json_values import require_serialization_safe_integer

from .cases.registry import FRAMEWORK_VERSION, load_case_manifest, load_stripe_registry

if TYPE_CHECKING:
    from mindful_trace_gepa.logging_schema import EventEnvelope


@dataclass(frozen=True, slots=True)
class CaseIdentity:
    """Canonical case identity supplied with a V5 evaluation record."""

    case_id: int
    case_version: str
    case_key: str
    case_title: str

    def __post_init__(self) -> None:
        """Validate the externally supplied identity against the canonical registry."""

        _require_literal(self.case_version, FRAMEWORK_VERSION, "case_version")
        if type(self.case_id) is not int:
            raise ValueError("case_id must be a built-in integer")
        case_key, case_title = _canonical_case(self.case_id)
        _require_literal(self.case_key, case_key, "case_key")
        _require_literal(self.case_title, case_title, "case_title")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "case_key": self.case_key,
            "case_title": self.case_title,
        }


@dataclass(frozen=True, slots=True)
class RobustnessIdentity:
    """Canonical robustness stripe identity for one V5 evaluation record."""

    stripe_id: str
    subtype: str | None

    def __post_init__(self) -> None:
        """Validate the stripe and its optional subtype against the stripe registry."""

        allowed_subtypes = _canonical_stripe(self.stripe_id)
        if self.subtype is not None:
            subtype = _require_nonblank_string(self.subtype, "subtype")
            if subtype not in allowed_subtypes:
                raise ValueError(
                    f"subtype {subtype!r} is not allowed for stripe {self.stripe_id!r}"
                )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {"stripe_id": self.stripe_id, "subtype": self.subtype}


@dataclass(frozen=True, slots=True)
class SystemIdentity:
    """Frozen model, harness, and repeat identity for one evaluation episode."""

    repeat_id: int
    seed: int
    model_version: str
    harness_version: str

    def __post_init__(self) -> None:
        """Validate stable system identity fields."""

        if type(self.repeat_id) is not int or self.repeat_id < 0:
            raise ValueError("repeat_id must be a nonnegative built-in integer")
        require_serialization_safe_integer("repeat_id", self.repeat_id)
        require_serialization_safe_integer("seed", self.seed)
        _require_nonblank_string(self.model_version, "model_version")
        _require_nonblank_string(self.harness_version, "harness_version")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "repeat_id": self.repeat_id,
            "seed": self.seed,
            "model_version": self.model_version,
            "harness_version": self.harness_version,
        }


@dataclass(frozen=True, slots=True)
class EpistemicRecord:
    """Prediction, evidence, verifier, and confidence facts for an episode."""

    prediction_ref: str
    evidence_refs: tuple[str, ...]
    verifier_refs: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        """Snapshot references and validate bounded confidence."""

        _require_nonblank_string(self.prediction_ref, "prediction_ref")
        object.__setattr__(
            self, "evidence_refs", _reference_tuple(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self, "verifier_refs", _reference_tuple(self.verifier_refs, "verifier_refs")
        )
        object.__setattr__(self, "confidence", _bounded_number(self.confidence, "confidence"))

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "prediction_ref": self.prediction_ref,
            "evidence_refs": list(self.evidence_refs),
            "verifier_refs": list(self.verifier_refs),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class BehaviorRecord:
    """Observed model behavior, kept separate from outcome and scoring."""

    action_refs: tuple[str, ...]
    abstained: bool
    requested_clarification: bool

    def __post_init__(self) -> None:
        """Snapshot action references and require JSON booleans."""

        object.__setattr__(self, "action_refs", _reference_tuple(self.action_refs, "action_refs"))
        _require_bool(self.abstained, "abstained")
        _require_bool(self.requested_clarification, "requested_clarification")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "action_refs": list(self.action_refs),
            "abstained": self.abstained,
            "requested_clarification": self.requested_clarification,
        }


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """Observed and verified outcome facts for an episode."""

    observation_refs: tuple[str, ...]
    verifier_refs: tuple[str, ...]
    passed: bool

    def __post_init__(self) -> None:
        """Snapshot observation references and require JSON booleans."""

        object.__setattr__(
            self,
            "observation_refs",
            _reference_tuple(self.observation_refs, "observation_refs"),
        )
        object.__setattr__(
            self, "verifier_refs", _reference_tuple(self.verifier_refs, "verifier_refs")
        )
        _require_bool(self.passed, "passed")

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""

        return {
            "observation_refs": list(self.observation_refs),
            "verifier_refs": list(self.verifier_refs),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """Documented optimizer-facing score components, each bounded to zero through one."""

    correctness: float
    calibration: float
    abstention: float
    epistemic_process: float
    total: float

    def __post_init__(self) -> None:
        """Normalize and validate every optimizer-facing score component."""

        for field_name in (
            "correctness",
            "calibration",
            "abstention",
            "epistemic_process",
            "total",
        ):
            object.__setattr__(
                self, field_name, _bounded_number(getattr(self, field_name), field_name)
            )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible optimizer score section."""

        return {
            "correctness": self.correctness,
            "calibration": self.calibration,
            "abstention": self.abstention,
            "epistemic_process": self.epistemic_process,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    """Diagnostic-only trace, deception, and mechanistic signals."""

    trace_summary: str
    deception_signal: float
    mechanistic_signal: float

    def __post_init__(self) -> None:
        """Validate diagnostic values without making them optimizer-facing."""

        _require_nonblank_string(self.trace_summary, "trace_summary")
        object.__setattr__(
            self,
            "deception_signal",
            _finite_number(self.deception_signal, "deception_signal"),
        )
        object.__setattr__(
            self,
            "mechanistic_signal",
            _finite_number(self.mechanistic_signal, "mechanistic_signal"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible diagnostic section."""

        return {
            "trace_summary": self.trace_summary,
            "deception_signal": self.deception_signal,
            "mechanistic_signal": self.mechanistic_signal,
        }


@dataclass(frozen=True, slots=True)
class V5EvaluationRecord:
    """One immutable case-by-stripe-by-repeat V5 evaluation result."""

    case: CaseIdentity
    robustness: RobustnessIdentity
    system: SystemIdentity
    epistemics: EpistemicRecord
    behavior: BehaviorRecord
    outcome: OutcomeRecord
    scores: ScoreRecord
    diagnostics: DiagnosticRecord

    def __post_init__(self) -> None:
        """Reject foreign sections and detach the root from caller-owned section objects."""

        _require_exact_instance(self.case, CaseIdentity, "case")
        _require_exact_instance(self.robustness, RobustnessIdentity, "robustness")
        _require_exact_instance(self.system, SystemIdentity, "system")
        _require_exact_instance(self.epistemics, EpistemicRecord, "epistemics")
        _require_exact_instance(self.behavior, BehaviorRecord, "behavior")
        _require_exact_instance(self.outcome, OutcomeRecord, "outcome")
        _require_exact_instance(self.scores, ScoreRecord, "scores")
        _require_exact_instance(self.diagnostics, DiagnosticRecord, "diagnostics")
        object.__setattr__(
            self,
            "case",
            CaseIdentity(
                case_id=self.case.case_id,
                case_version=self.case.case_version,
                case_key=self.case.case_key,
                case_title=self.case.case_title,
            ),
        )
        object.__setattr__(
            self,
            "robustness",
            RobustnessIdentity(
                stripe_id=self.robustness.stripe_id,
                subtype=self.robustness.subtype,
            ),
        )
        object.__setattr__(
            self,
            "system",
            SystemIdentity(
                repeat_id=self.system.repeat_id,
                seed=self.system.seed,
                model_version=self.system.model_version,
                harness_version=self.system.harness_version,
            ),
        )
        object.__setattr__(
            self,
            "epistemics",
            EpistemicRecord(
                prediction_ref=self.epistemics.prediction_ref,
                evidence_refs=self.epistemics.evidence_refs,
                verifier_refs=self.epistemics.verifier_refs,
                confidence=self.epistemics.confidence,
            ),
        )
        object.__setattr__(
            self,
            "behavior",
            BehaviorRecord(
                action_refs=self.behavior.action_refs,
                abstained=self.behavior.abstained,
                requested_clarification=self.behavior.requested_clarification,
            ),
        )
        object.__setattr__(
            self,
            "outcome",
            OutcomeRecord(
                observation_refs=self.outcome.observation_refs,
                verifier_refs=self.outcome.verifier_refs,
                passed=self.outcome.passed,
            ),
        )
        object.__setattr__(
            self,
            "scores",
            ScoreRecord(
                correctness=self.scores.correctness,
                calibration=self.scores.calibration,
                abstention=self.scores.abstention,
                epistemic_process=self.scores.epistemic_process,
                total=self.scores.total,
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            DiagnosticRecord(
                trace_summary=self.diagnostics.trace_summary,
                deception_signal=self.diagnostics.deception_signal,
                mechanistic_signal=self.diagnostics.mechanistic_signal,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return fresh, deterministic JSON-compatible containers for this record."""

        return {
            "case": self.case.to_dict(),
            "robustness": self.robustness.to_dict(),
            "system": self.system.to_dict(),
            "epistemics": self.epistemics.to_dict(),
            "behavior": self.behavior.to_dict(),
            "outcome": self.outcome.to_dict(),
            "scores": self.scores.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "V5EvaluationRecord":
        """Hydrate one strict JSON object without accepting unknown or omitted fields."""

        root = _require_mapping(payload, "V5 evaluation record")
        _require_exact_fields(
            root,
            {
                "case",
                "robustness",
                "system",
                "epistemics",
                "behavior",
                "outcome",
                "scores",
                "diagnostics",
            },
            "V5 evaluation record",
        )
        case = _require_mapping(root["case"], "case")
        robustness = _require_mapping(root["robustness"], "robustness")
        system = _require_mapping(root["system"], "system")
        epistemics = _require_mapping(root["epistemics"], "epistemics")
        behavior = _require_mapping(root["behavior"], "behavior")
        outcome = _require_mapping(root["outcome"], "outcome")
        scores = _require_mapping(root["scores"], "scores")
        diagnostics = _require_mapping(root["diagnostics"], "diagnostics")
        _require_exact_fields(case, {"case_id", "case_version", "case_key", "case_title"}, "case")
        _require_exact_fields(robustness, {"stripe_id", "subtype"}, "robustness")
        _require_exact_fields(
            system,
            {"repeat_id", "seed", "model_version", "harness_version"},
            "system",
        )
        _require_exact_fields(
            epistemics,
            {"prediction_ref", "evidence_refs", "verifier_refs", "confidence"},
            "epistemics",
        )
        _require_exact_fields(
            behavior,
            {"action_refs", "abstained", "requested_clarification"},
            "behavior",
        )
        _require_exact_fields(
            outcome,
            {"observation_refs", "verifier_refs", "passed"},
            "outcome",
        )
        _require_exact_fields(
            scores,
            {"correctness", "calibration", "abstention", "epistemic_process", "total"},
            "scores",
        )
        _require_exact_fields(
            diagnostics,
            {"trace_summary", "deception_signal", "mechanistic_signal"},
            "diagnostics",
        )
        return cls(
            case=CaseIdentity(
                case_id=case["case_id"],
                case_version=case["case_version"],
                case_key=case["case_key"],
                case_title=case["case_title"],
            ),
            robustness=RobustnessIdentity(
                stripe_id=robustness["stripe_id"],
                subtype=robustness["subtype"],
            ),
            system=SystemIdentity(
                repeat_id=system["repeat_id"],
                seed=system["seed"],
                model_version=system["model_version"],
                harness_version=system["harness_version"],
            ),
            epistemics=EpistemicRecord(
                prediction_ref=epistemics["prediction_ref"],
                evidence_refs=_json_reference_tuple(epistemics["evidence_refs"], "evidence_refs"),
                verifier_refs=_json_reference_tuple(epistemics["verifier_refs"], "verifier_refs"),
                confidence=epistemics["confidence"],
            ),
            behavior=BehaviorRecord(
                action_refs=_json_reference_tuple(behavior["action_refs"], "action_refs"),
                abstained=behavior["abstained"],
                requested_clarification=behavior["requested_clarification"],
            ),
            outcome=OutcomeRecord(
                observation_refs=_json_reference_tuple(
                    outcome["observation_refs"], "observation_refs"
                ),
                verifier_refs=_json_reference_tuple(outcome["verifier_refs"], "verifier_refs"),
                passed=outcome["passed"],
            ),
            scores=ScoreRecord(
                correctness=scores["correctness"],
                calibration=scores["calibration"],
                abstention=scores["abstention"],
                epistemic_process=scores["epistemic_process"],
                total=scores["total"],
            ),
            diagnostics=DiagnosticRecord(
                trace_summary=diagnostics["trace_summary"],
                deception_signal=diagnostics["deception_signal"],
                mechanistic_signal=diagnostics["mechanistic_signal"],
            ),
        )

    def optimizer_scores(self, events: Sequence[EventEnvelope]) -> dict[str, object]:
        """Return optimizer scores only after validating same-cell PR-2 provenance."""

        from .v5_provenance import validate_v5_record_provenance

        return validate_v5_record_provenance(self, events).optimizer_scores()


def _canonical_case(case_id: int) -> tuple[str, str]:
    case = _canonical_case_map().get(case_id)
    if case is not None:
        return case
    raise ValueError(f"case_id must identify a canonical 17case-v5 case; received {case_id!r}")


def _canonical_stripe(stripe_id: object) -> tuple[str, ...]:
    parsed_id = _require_nonblank_string(stripe_id, "stripe_id")
    stripe = _canonical_stripe_map().get(parsed_id)
    if stripe is not None:
        return stripe
    raise ValueError(
        f"stripe_id must identify a registered 17case-v5 stripe; received {parsed_id!r}"
    )


@cache
def _canonical_case_map() -> Mapping[int, tuple[str, str]]:
    """Return one immutable case identity map loaded from the packaged registry once."""

    identities = {case.id: (case.key, case.title) for case in load_case_manifest().cases}
    return MappingProxyType(identities)


@cache
def _canonical_stripe_map() -> Mapping[str, tuple[str, ...]]:
    """Return one immutable stripe/subtype map loaded from the packaged registry once."""

    identities = {
        stripe.id: tuple(stripe.allowed_subtypes) for stripe in load_stripe_registry().stripes
    }
    return MappingProxyType(identities)


def _reference_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a JSON-compatible array of nonblank references")
    return tuple(_require_nonblank_string(item, field_name) for item in value)


def _json_reference_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return _reference_tuple(value, field_name)


def _bounded_number(value: object, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return number


def _finite_number(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite JSON number")
    if type(value) is float and not isfinite(value):
        raise ValueError(f"{field_name} must be a finite JSON number")
    try:
        return float(cast(int | float, value))
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be a finite JSON number") from exc


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a JSON boolean")


def _require_nonblank_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonblank string without surrounding whitespace")
    return value


def _require_literal(value: object, expected: object, field_name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{field_name} must be {expected!r}; received {value!r}")


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    if not all(type(key) is str for key in value):
        raise ValueError(f"{field_name} field names must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    required: set[str],
    field_name: str,
) -> None:
    received = set(value)
    missing = sorted(required - received)
    unknown = sorted(received - required)
    if not missing and not unknown:
        return
    details = []
    if missing:
        details.append(f"missing fields: {missing}")
    if unknown:
        details.append(f"unknown fields: {unknown}")
    raise ValueError(f"{field_name} has invalid fields; " + "; ".join(details))


def _require_exact_instance(value: object, record_type: type[object], field_name: str) -> None:
    if type(value) is not record_type:
        raise ValueError(f"{field_name} must be an exact {record_type.__name__}")


__all__ = [
    "BehaviorRecord",
    "CaseIdentity",
    "DiagnosticRecord",
    "EpistemicRecord",
    "OutcomeRecord",
    "RobustnessIdentity",
    "ScoreRecord",
    "SystemIdentity",
    "V5EvaluationRecord",
]

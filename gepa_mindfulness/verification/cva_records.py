"""Versioned CVA proposals and evidence; no record grants trust, reward or authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, TypeVar

# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

from .diagnostic_records import DiagnosticRecord, public_refs, restore_refs
from .state import parse_rfc3339_datetime

MAX_ASSIGNMENTS = 4096
MAX_VARIABLES = 16
R = TypeVar("R", bound="CVARecord")


class PredicateKind(str, Enum):
    GOAL = "G"
    TECHNICAL = "T"
    AUTHORIZATION = "P"
    ETHICAL = "E"


class Relation(str, Enum):
    EQ = "eq"
    NE = "ne"
    LE = "le"
    GE = "ge"


class GuaranteeKind(str, Enum):
    EXACT = "exact"
    CERTIFIED_APPROXIMATION = "certified_approximation"
    STATISTICAL_ESTIMATE = "statistical_estimate"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class CertificateType(str, Enum):
    FINITE_WITNESS = "finite_witness_v1"
    FINITE_EXHAUSTION = "finite_exhaustion_v1"
    UNSUPPORTED = "unsupported"
    NONE = "none"


class FeasibilityScope(str, Enum):
    TECHNICAL = "G_AND_T"
    CONSTRAINT_VALID = "G_AND_T_AND_P_AND_E"


class CandidateVerdict(str, Enum):
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_AUTHORIZATION = "INSUFFICIENT_AUTHORIZATION"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    INVALID_CERTIFICATE = "INVALID_CERTIFICATE"


class FeasibilityVerdict(str, Enum):
    VERIFIED_FEASIBLE = "VERIFIED_FEASIBLE"
    VERIFIED_INFEASIBLE = "VERIFIED_INFEASIBLE"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_AUTHORIZATION = "INSUFFICIENT_AUTHORIZATION"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    INVALID_CERTIFICATE = "INVALID_CERTIFICATE"


def text(value: object, name: str, limit: int = 256) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} requires nonblank exact text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"invalid {name} encoding") from error
    if size > limit:
        raise ValueError(f"{name} exceeds {limit} bytes")


def integer(value: object) -> None:
    if type(value) is not int or abs(value) > 2**53 - 1:
        raise ValueError("expected exact JSON-safe integer")


def digest(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected lowercase SHA-256 digest")


def array(value: object, maximum: int, *, required: bool = False) -> tuple:
    if type(value) is not tuple or len(value) > maximum or (required and not value):
        raise ValueError("expected bounded exact tuple")
    return value


def texts(value: object, *, required: bool = False, maximum: int = 32) -> tuple[str, ...]:
    result = array(value, maximum, required=required)
    for item in result:
        text(item, "text item", 2048)
    if len(set(result)) != len(result):
        raise ValueError("duplicate text items")
    return result


def enum(value: object, kind: type[Enum]) -> None:
    if type(value) is not kind:
        raise ValueError(f"expected exact {kind.__name__}")


def _read_enum(value: object, kind: type[Enum]) -> Enum:
    if type(value) is not str:
        raise ValueError("wire enums require exact strings")
    return kind(value)


def _read_array(value: object) -> tuple:
    if type(value) is not list:
        raise ValueError("wire arrays require exact lists")
    return tuple(value)


def _rows(value: object) -> tuple[tuple[int, ...], ...]:
    rows = array(value, MAX_ASSIGNMENTS)
    for row in rows:
        _values(row)
    return rows


def _values(value: object) -> tuple[int, ...]:
    values = array(value, MAX_VARIABLES, required=True)
    for item in values:
        integer(item)
    return values


def references(
    value: object, *, required: bool = True, maximum: int = 32
) -> tuple[EvidenceReference, ...]:
    refs = array(value, maximum, required=required)
    for ref in refs:
        if type(ref) is not EvidenceReference:
            raise ValueError("expected exact observable EvidenceReference")
        text(ref.reference_id, "reference_id")
    return public_refs(refs, required=required)


def _record(value: object, cls: type[R]) -> R:
    if type(value) is not cls:
        raise ValueError(f"expected exact {cls.__name__}")
    return cls.from_dict(cls.to_dict(value))


def _records(value: object, cls: type[R], maximum: int) -> tuple[R, ...]:
    return tuple(_record(v, cls) for v in array(value, maximum, required=True))


def _read_records(value: object, cls: type[R]) -> tuple[R, ...]:
    return tuple(cls.from_dict(v) for v in _read_array(value))


def evaluator(value: object) -> TrustedEvaluatorContract:
    if type(value) is not TrustedEvaluatorContract:
        raise ValueError("expected exact trusted evaluator contract")
    for f in fields(value):
        text(getattr(value, f.name), f.name)
    return TrustedEvaluatorContract(value.evaluator_id, value.evaluator_version, value.contract_id)


def _read_evaluator(value: object) -> TrustedEvaluatorContract:
    if type(value) is not dict or set(value) != {f.name for f in fields(TrustedEvaluatorContract)}:
        raise ValueError("incorrect evaluator fields")
    return evaluator(TrustedEvaluatorContract(**value))


class CVARecord(DiagnosticRecord):
    """Reuse diagnostic wire/eligibility contracts with bounded exact public inputs."""

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = DiagnosticRecord.to_dict(self)
        for key, value in tuple(data.items()):
            if type(value) is TrustedEvaluatorContract:
                data[key] = {f.name: getattr(value, f.name) for f in fields(value)}
        data.update(
            maturity="experimental",
            diagnostic_status="diagnostic",
            optimizer_input=False,
            authority_granted=False,
        )
        validate_input_envelope(data)
        return data

    @classmethod
    def from_dict(cls: type[R], data: object) -> R:
        validate_input_envelope(data)
        if type(data) is not dict:
            raise ValueError("CVA wire record requires an exact object")
        source = dict(data)
        expected = {
            "maturity": "experimental",
            "diagnostic_status": "diagnostic",
            "optimizer_input": False,
            "authority_granted": False,
        }
        for key, value in expected.items():
            actual = source.pop(key, None)
            if type(actual) is not type(value) or actual != value:
                raise ValueError("CVA diagnostics cannot grant authority or optimizer eligibility")
        return super().from_dict(source)


@dataclass(frozen=True)
class FiniteVariable(CVARecord):
    name: str
    values: tuple[int, ...]

    schema_version = "cva-variable-v1"
    restorers = {"values": _read_array}

    def __post_init__(self) -> None:
        text(self.name, "variable name")
        values = _values(self.values)
        if len(set(values)) != len(values):
            raise ValueError("duplicate domain values")


@dataclass(frozen=True)
class LinearPredicate(CVARecord):
    """Closed integer linear relation; no expressions, callbacks or executable payloads."""

    predicate_id: str
    kind: PredicateKind
    coefficients: tuple[int, ...]
    relation: Relation
    bound: int

    schema_version = "cva-linear-predicate-v1"
    restorers = {
        "kind": lambda v: _read_enum(v, PredicateKind),
        "coefficients": _read_array,
        "relation": lambda v: _read_enum(v, Relation),
    }

    def __post_init__(self) -> None:
        text(self.predicate_id, "predicate_id")
        enum(self.kind, PredicateKind)
        _values(self.coefficients)
        enum(self.relation, Relation)
        integer(self.bound)


@dataclass(frozen=True)
class ConstraintSet(CVARecord):
    """A host-owned formalization, not a complete decision procedure for moral alignment."""

    task_id: str
    constraint_set_id: str
    constraint_version: str
    claimed_objective: str
    variables: tuple[FiniteVariable, ...]
    predicates: tuple[LinearPredicate, ...]
    authority_hierarchy: tuple[str, ...]
    authorization_scope: str | None
    evidence_refs: tuple[EvidenceReference, ...]
    completeness_assumptions: tuple[str, ...]
    unresolved_normative_judgments: tuple[str, ...] = ()

    schema_version = "cva-constraint-set-v1"
    restorers = {
        "variables": lambda v: _read_records(v, FiniteVariable),
        "predicates": lambda v: _read_records(v, LinearPredicate),
        "authority_hierarchy": _read_array,
        "evidence_refs": restore_refs,
        "completeness_assumptions": _read_array,
        "unresolved_normative_judgments": _read_array,
    }

    def __post_init__(self) -> None:
        for name in ("task_id", "constraint_set_id", "constraint_version"):
            text(getattr(self, name), name)
        text(self.claimed_objective, "claimed_objective", 2048)
        variables = _records(self.variables, FiniteVariable, MAX_VARIABLES)
        predicates = _records(self.predicates, LinearPredicate, 128)
        if len({v.name for v in variables}) != len(variables):
            raise ValueError("duplicate variable names")
        if len({p.predicate_id for p in predicates}) != len(predicates):
            raise ValueError("duplicate predicate identities")
        size = 1
        for variable in variables:
            size *= len(variable.values)
            if size > MAX_ASSIGNMENTS:
                raise ValueError("formal domain exceeds 4096 assignments")
        if any(len(p.coefficients) != len(variables) for p in predicates):
            raise ValueError("predicate coefficients do not match variables")
        if not any(p.kind is PredicateKind.GOAL for p in predicates):
            raise ValueError("a nonempty explicit objective is required")
        texts(self.authority_hierarchy, required=True)
        if self.authorization_scope is not None:
            text(self.authorization_scope, "authorization_scope")
        references(self.evidence_refs)
        texts(self.completeness_assumptions, required=True)
        texts(self.unresolved_normative_judgments)


@dataclass(frozen=True)
class ActionBinding(CVARecord):
    """Original action/prediction identities and digests; this issues no runtime grant."""

    action_id: str
    prediction_commit_id: str
    action_digest: str
    prediction_digest: str
    source_candidate_digest: str
    authorization_scope: str

    schema_version = "cva-action-binding-v1"

    def __post_init__(self) -> None:
        for name in ("action_id", "prediction_commit_id", "authorization_scope"):
            text(getattr(self, name), name)
        for name in ("action_digest", "prediction_digest", "source_candidate_digest"):
            digest(getattr(self, name))


@dataclass(frozen=True)
class CandidatePlan(CVARecord):
    task_id: str
    constraint_digest: str
    values: tuple[int, ...]
    description: str
    action_binding: ActionBinding | None = None

    schema_version = "cva-candidate-v1"
    restorers = {
        "values": _read_array,
        "action_binding": lambda v: None if v is None else ActionBinding.from_dict(v),
    }

    def __post_init__(self) -> None:
        text(self.task_id, "task_id")
        digest(self.constraint_digest)
        _values(self.values)
        text(self.description, "description", 2048)
        if self.action_binding is not None:
            _record(self.action_binding, ActionBinding)


@dataclass(frozen=True)
class OutcomeCertificate(CVARecord):
    """Untrusted artifact until the independent checker validates its exact guarantee."""

    task_id: str
    constraint_digest: str
    candidate_digest: str | None
    certificate_type: CertificateType
    scope: FeasibilityScope
    guarantee: GuaranteeKind
    witness: tuple[int, ...] | None
    excluded_assignments: tuple[tuple[int, ...], ...]
    evidence_refs: tuple[EvidenceReference, ...]

    schema_version = "cva-outcome-certificate-v1"
    restorers = {
        "certificate_type": lambda v: _read_enum(v, CertificateType),
        "scope": lambda v: _read_enum(v, FeasibilityScope),
        "guarantee": lambda v: _read_enum(v, GuaranteeKind),
        "witness": lambda v: None if v is None else _read_array(v),
        "excluded_assignments": lambda v: tuple(_read_array(r) for r in _read_array(v)),
        "evidence_refs": restore_refs,
    }

    def __post_init__(self) -> None:
        text(self.task_id, "task_id")
        digest(self.constraint_digest)
        if self.candidate_digest is not None:
            digest(self.candidate_digest)
        enum(self.certificate_type, CertificateType)
        enum(self.scope, FeasibilityScope)
        enum(self.guarantee, GuaranteeKind)
        if self.witness is not None:
            _values(self.witness)
        _rows(self.excluded_assignments)
        references(self.evidence_refs)
        if self.certificate_type is CertificateType.FINITE_WITNESS and (
            self.witness is None or self.candidate_digest is None or self.excluded_assignments
        ):
            raise ValueError("finite witness requires a bound candidate and no exhaustion rows")
        if self.certificate_type is CertificateType.FINITE_EXHAUSTION and (
            self.witness is not None or self.candidate_digest is not None
        ):
            raise ValueError("exhaustion cannot assert one candidate")


@dataclass(frozen=True)
class VerificationRecord(CVARecord):
    """Common provenance for candidate or feasibility findings, never an admission credential."""

    task_id: str
    constraint_set_id: str
    constraint_version: str
    constraint_digest: str
    authority_hierarchy: tuple[str, ...]
    candidate_digest: str | None
    claimed_objective: str
    verification_method: str
    checker_id: str
    checker_version: str
    host_evaluator: TrustedEvaluatorContract | None
    certificate_type: CertificateType
    certificate_digest: str | None
    evidence_refs: tuple[EvidenceReference, ...]
    guarantee: GuaranteeKind
    completeness_assumptions: tuple[str, ...]
    remaining_uncertainty: tuple[str, ...]
    permitted_next_actions: tuple[str, ...]
    provenance: tuple[str, ...]
    created_at: str

    schema_version = "cva-verification-record-v1"
    restorers = {
        "authority_hierarchy": _read_array,
        "host_evaluator": lambda v: None if v is None else _read_evaluator(v),
        "certificate_type": lambda v: _read_enum(v, CertificateType),
        "evidence_refs": restore_refs,
        "guarantee": lambda v: _read_enum(v, GuaranteeKind),
        "completeness_assumptions": _read_array,
        "remaining_uncertainty": _read_array,
        "permitted_next_actions": _read_array,
        "provenance": _read_array,
    }

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "constraint_set_id",
            "constraint_version",
            "checker_id",
            "checker_version",
        ):
            text(getattr(self, name), name)
        text(self.claimed_objective, "claimed_objective", 2048)
        text(self.verification_method, "verification_method")
        digest(self.constraint_digest)
        for value in (self.candidate_digest, self.certificate_digest):
            if value is not None:
                digest(value)
        if self.host_evaluator is not None:
            evaluator(self.host_evaluator)
        enum(self.certificate_type, CertificateType)
        enum(self.guarantee, GuaranteeKind)
        references(self.evidence_refs, maximum=64)
        for name in ("authority_hierarchy", "completeness_assumptions", "provenance"):
            texts(getattr(self, name), required=True)
        texts(self.remaining_uncertainty, maximum=33)
        texts(self.permitted_next_actions)
        text(self.created_at, "created_at")
        parse_rfc3339_datetime(self.created_at, "created_at")


@dataclass(frozen=True)
class CandidateVerification(CVARecord):
    verification: VerificationRecord
    verdict: CandidateVerdict
    failed_predicate_ids: tuple[str, ...]

    schema_version = "cva-candidate-verification-v1"
    restorers = {
        "verification": VerificationRecord.from_dict,
        "verdict": lambda v: _read_enum(v, CandidateVerdict),
        "failed_predicate_ids": _read_array,
    }

    def __post_init__(self) -> None:
        record = _record(self.verification, VerificationRecord)
        enum(self.verdict, CandidateVerdict)
        texts(self.failed_predicate_ids, maximum=128)
        if record.candidate_digest is None:
            raise ValueError("candidate result requires a candidate digest")
        if self.verdict is CandidateVerdict.VERIFIED_SUCCESS and (
            record.guarantee is not GuaranteeKind.EXACT
            or self.failed_predicate_ids
            or record.host_evaluator is None
            or record.certificate_type is not CertificateType.FINITE_WITNESS
            or record.certificate_digest is None
        ):
            raise ValueError("inconsistent witnessed-success certificate record")
        if self.verdict is CandidateVerdict.CONSTRAINT_VIOLATION and (
            not self.failed_predicate_ids
            or record.guarantee is not GuaranteeKind.EXACT
            or record.certificate_digest is None
            or record.host_evaluator is None
        ):
            raise ValueError("candidate rejection requires checked failed-predicate evidence")


@dataclass(frozen=True)
class FeasibilityVerification(CVARecord):
    verification: VerificationRecord
    verdict: FeasibilityVerdict
    scope: FeasibilityScope

    schema_version = "cva-feasibility-verification-v1"
    restorers = {
        "verification": VerificationRecord.from_dict,
        "verdict": lambda v: _read_enum(v, FeasibilityVerdict),
        "scope": lambda v: _read_enum(v, FeasibilityScope),
    }

    def __post_init__(self) -> None:
        record = _record(self.verification, VerificationRecord)
        enum(self.verdict, FeasibilityVerdict)
        enum(self.scope, FeasibilityScope)
        if self.verdict in (
            FeasibilityVerdict.VERIFIED_INFEASIBLE,
            FeasibilityVerdict.VERIFIED_FEASIBLE,
        ) and (
            record.guarantee is not GuaranteeKind.EXACT
            or record.host_evaluator is None
            or record.certificate_digest is None
        ):
            raise ValueError("conclusive feasibility requires exact independently checked evidence")
        if self.verdict is FeasibilityVerdict.VERIFIED_INFEASIBLE and (
            record.candidate_digest is not None
            or record.certificate_type is not CertificateType.FINITE_EXHAUSTION
        ):
            raise ValueError("infeasibility requires a scoped exhaustion certificate")
        if self.verdict is FeasibilityVerdict.VERIFIED_FEASIBLE and (
            record.candidate_digest is None
            or record.certificate_type is not CertificateType.FINITE_WITNESS
        ):
            raise ValueError("feasibility requires an exact bound witness")


@dataclass(frozen=True)
class NoncompletionAssessment(CVARecord):
    """Evidence-limited noncompletion, never a general impossibility certificate."""

    task_id: str
    constraint_digest: str
    attempted: tuple[str, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    encountered_constraints: tuple[str, ...]
    proven_contradiction: str | None
    remaining_unknowns: tuple[str, ...]
    required_information_or_authorization: tuple[str, ...]
    legitimate_alternatives: tuple[str, ...]
    assessment: str = "NO_AUTHORIZED_SOLUTION_ESTABLISHED"

    schema_version = "cva-noncompletion-v1"
    restorers = {
        "attempted": _read_array,
        "evidence_refs": restore_refs,
        "encountered_constraints": _read_array,
        "remaining_unknowns": _read_array,
        "required_information_or_authorization": _read_array,
        "legitimate_alternatives": _read_array,
    }

    def __post_init__(self) -> None:
        text(self.task_id, "task_id")
        digest(self.constraint_digest)
        references(self.evidence_refs)
        for name in (
            "attempted",
            "required_information_or_authorization",
            "legitimate_alternatives",
        ):
            texts(getattr(self, name))
        texts(self.encountered_constraints, maximum=128)
        texts(self.remaining_unknowns, required=True, maximum=33)
        if self.proven_contradiction is not None:
            # A noncompletion assessment may refer to a separately retained proof, never create it.
            text(self.proven_contradiction, "proven_contradiction")
        if self.assessment != "NO_AUTHORIZED_SOLUTION_ESTABLISHED":
            raise ValueError("noncompletion cannot assert proven infeasibility")


@dataclass(frozen=True)
class SearchReport(CVARecord):
    constraint_digest: str
    attempted_assignments: tuple[tuple[int, ...], ...]
    exhaustive: bool
    candidate: CandidatePlan | None
    noncompletion: NoncompletionAssessment | None

    schema_version = "cva-search-report-v1"
    restorers = {
        "attempted_assignments": lambda v: tuple(_read_array(r) for r in _read_array(v)),
        "candidate": lambda v: None if v is None else CandidatePlan.from_dict(v),
        "noncompletion": lambda v: None if v is None else NoncompletionAssessment.from_dict(v),
    }

    @property
    def assessment(self) -> str:
        return (
            "NO_AUTHORIZED_SOLUTION_ESTABLISHED"
            if self.candidate is None
            else "UNVERIFIED_CANDIDATE_FOUND"
        )

    def __post_init__(self) -> None:
        digest(self.constraint_digest)
        _rows(self.attempted_assignments)
        if type(self.exhaustive) is not bool:
            raise ValueError("exhaustive requires exact bool")
        if self.candidate is not None:
            candidate = _record(self.candidate, CandidatePlan)
            if (
                candidate.constraint_digest != self.constraint_digest
                or self.noncompletion is not None
            ):
                raise ValueError("unmatched search candidate")
        elif self.noncompletion is None:
            raise ValueError("failed search requires an evidence-limited assessment")
        else:
            record = _record(self.noncompletion, NoncompletionAssessment)
            if record.constraint_digest != self.constraint_digest:
                raise ValueError("unmatched noncompletion assessment")


@dataclass(frozen=True)
class VerificationInput(CVARecord):
    """Detached exact subject that the trusted host authenticates before checking."""

    constraints: ConstraintSet
    candidate: CandidatePlan | None
    certificate: OutcomeCertificate | None

    schema_version = "cva-verification-input-v1"
    restorers = {
        "constraints": ConstraintSet.from_dict,
        "candidate": lambda v: None if v is None else CandidatePlan.from_dict(v),
        "certificate": lambda v: None if v is None else OutcomeCertificate.from_dict(v),
    }

    def __post_init__(self) -> None:
        _record(self.constraints, ConstraintSet)
        if self.candidate is not None:
            _record(self.candidate, CandidatePlan)
        if self.certificate is not None:
            _record(self.certificate, OutcomeCertificate)

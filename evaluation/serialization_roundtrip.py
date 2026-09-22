"""Opt-in, bounded communication diagnostics without reward or runtime authority.

The reference codec represents public propositional trees, not arbitrary natural language.
Stage evidence is supplied by an independent host-controlled reader and bound to the exact
serialized bytes. Evidence references record provenance; they do not authenticate that reader.
"""

from __future__ import annotations

# Standard library
import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from itertools import product
from typing import Protocol

MAX_NODES = 256
MAX_DEPTH = 32
MAX_TEXT_LENGTH = 32768


def _token(value: str, name: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonblank string of at most 256 characters")
    return value


def _refs(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 128:
        raise ValueError("evidence_refs must be a bounded sequence")
    return tuple(_token(item, "evidence reference") for item in values)


@dataclass(frozen=True, slots=True)
class Expression:
    """Public tree in the atom/not/and/or/implies propositional fragment.

    Trees contain at most 256 node occurrences and have depth at most 32. Repeated
    references count as repeated nodes, which bounds traversal and truth-table work.
    """

    operator: str
    children: tuple[Expression, ...] = ()
    atom: str | None = None

    def __post_init__(self) -> None:
        if self.operator not in ("atom", "not", "and", "or", "implies"):
            raise ValueError("unsupported expression operator")
        if not isinstance(self.children, (tuple, list)):
            raise ValueError("children must be a sequence of expressions")
        if len(self.children) > MAX_NODES:
            raise ValueError("expression exceeds node bound")
        if any(type(child) is not Expression for child in self.children):
            raise ValueError("children must contain Expression records")
        object.__setattr__(self, "children", tuple(self.children))
        arity = len(self.children)
        if self.operator == "atom":
            if arity or self.atom is None:
                raise ValueError("atom requires a name and no children")
            _token(self.atom, "atom")
        elif self.atom is not None:
            raise ValueError("only atom expressions may have an atom name")
        elif (self.operator == "not" and arity != 1) or (self.operator == "implies" and arity != 2):
            raise ValueError("operator has invalid children count")
        elif self.operator in ("and", "or") and arity < 2:
            raise ValueError("and/or require at least two children")
        _complexity(self)


def _complexity(expression: Expression) -> tuple[int, int]:
    pending = [(expression, 1)]
    count = depth = 0
    while pending:
        node, level = pending.pop()
        count += 1
        depth = max(depth, level)
        if count > MAX_NODES or level > MAX_DEPTH:
            raise ValueError("expression exceeds node or depth bound")
        pending.extend((child, level + 1) for child in node.children)
    return count, depth


@dataclass(frozen=True, slots=True)
class StructuredSource:
    """Immutable expression and references to its original public evidence."""

    source_id: str
    expression: Expression
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _token(self.source_id, "source_id")
        if type(self.expression) is not Expression:
            raise ValueError("expression must be an Expression")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))
        if not self.evidence_refs:
            raise ValueError("source evidence_refs must not be empty")


class EquivalenceStatus(str, Enum):
    EXACT_EQUIVALENCE = "EXACT_EQUIVALENCE"
    VERIFIED_SEMANTIC_EQUIVALENCE = "VERIFIED_SEMANTIC_EQUIVALENCE"
    HEURISTIC_SIMILARITY = "HEURISTIC_SIMILARITY"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    """A verifier's epistemic status; similarity never implies verified preservation."""

    status: EquivalenceStatus
    verifier_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not EquivalenceStatus:
            raise ValueError("status must be an EquivalenceStatus")
        _token(self.verifier_id, "verifier_id")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))

    @property
    def semantics_preserved(self) -> bool:
        return self.status in (
            EquivalenceStatus.EXACT_EQUIVALENCE,
            EquivalenceStatus.VERIFIED_SEMANTIC_EQUIVALENCE,
        )


class Serializer(Protocol):
    def serialize(self, source: StructuredSource) -> str:
        """Produce a bounded textual representation of the public source."""
        ...


class Extractor(Protocol):
    def extract(self, text: str) -> Expression:
        """Reconstruct a bounded public tree or raise ValueError on extraction failure."""
        ...


class EquivalenceVerifier(Protocol):
    def verify(self, source: Expression, reconstructed: Expression) -> EquivalenceResult:
        """Report exact, verified, heuristic, non-equivalent, or unknown status."""
        ...


@dataclass(frozen=True, slots=True)
class PropositionalTreeVerifier:
    """Exact tree equality first, then exhaustive Boolean semantics over <=12 atoms."""

    max_atoms: int = 12

    def __post_init__(self) -> None:
        if type(self.max_atoms) is not int or not 1 <= self.max_atoms <= 12:
            raise ValueError("max_atoms must be a built-in integer between 1 and 12")

    def verify(self, source: Expression, reconstructed: Expression) -> EquivalenceResult:
        if type(source) is not Expression or type(reconstructed) is not Expression:
            raise ValueError("verifier inputs must be Expression records")
        status = EquivalenceStatus.EXACT_EQUIVALENCE
        if source != reconstructed:
            atoms = sorted(_atoms(source) | _atoms(reconstructed))
            status = EquivalenceStatus.UNKNOWN
            if len(atoms) <= self.max_atoms:
                status = EquivalenceStatus.VERIFIED_SEMANTIC_EQUIVALENCE
                for values in product((False, True), repeat=len(atoms)):
                    assignment = dict(zip(atoms, values))
                    if _evaluate(source, assignment) != _evaluate(reconstructed, assignment):
                        status = EquivalenceStatus.NOT_EQUIVALENT
                        break
        return EquivalenceResult(status, "propositional-truth-table:v1", ())


def _atoms(expression: Expression) -> set[str]:
    if expression.atom is not None:
        return {expression.atom}
    return set().union(*(_atoms(child) for child in expression.children))


def _evaluate(expression: Expression, assignment: dict[str, bool]) -> bool:
    if expression.atom is not None:
        return assignment[expression.atom]
    values = [_evaluate(child, assignment) for child in expression.children]
    if expression.operator == "not":
        return not values[0]
    if expression.operator == "and":
        return all(values)
    if expression.operator == "or":
        return any(values)
    return not values[0] or values[1]


@dataclass(frozen=True, slots=True)
class JsonTreeCodec:
    """Deterministic reference serializer/extractor for synthetic public trees."""

    def serialize(self, source: StructuredSource) -> str:
        text = json.dumps(_tree_dict(source.expression), sort_keys=True, separators=(",", ":"))
        _text(text)
        return text

    def extract(self, text: str) -> Expression:
        _text(text)
        try:
            return _parse_tree(json.loads(text), 1)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError("serialized text is not a bounded JSON tree") from exc


def _text(text: str) -> None:
    if type(text) is not str or not text or len(text) > MAX_TEXT_LENGTH:
        raise ValueError("serialized text must contain 1 to 32768 characters")


def _tree_dict(expression: Expression) -> dict[str, object]:
    return {
        "operator": expression.operator,
        "atom": expression.atom,
        "children": [_tree_dict(child) for child in expression.children],
    }


def _parse_tree(value: object, depth: int) -> Expression:
    if depth > MAX_DEPTH or type(value) is not dict:
        raise ValueError("serialized tree exceeds depth bound or contains a non-object")
    if set(value) != {"operator", "atom", "children"}:
        raise ValueError("serialized tree has unexpected fields")
    children = value["children"]
    if type(children) is not list or len(children) > MAX_NODES:
        raise ValueError("serialized children must be a bounded list")
    return Expression(
        value["operator"], tuple(_parse_tree(child, depth + 1) for child in children), value["atom"]
    )


class FaultStatus(str, Enum):
    FAULT = "FAULT"
    NO_FAULT = "NO_FAULT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class StageEvidence:
    """Independent host reader's interpretation of the exact serialized text.

    The host authenticates reader identity and references before providing this record.
    The audit only checks the digest binding; a serializer's self-report is insufficient.
    """

    serialized_sha256: str
    represented_expression: Expression
    verifier_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        digest = self.serialized_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("serialized_sha256 must be a lowercase SHA256 digest")
        if type(self.represented_expression) is not Expression:
            raise ValueError("represented_expression must be an Expression")
        _token(self.verifier_id, "verifier_id")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs))
        if not self.evidence_refs:
            raise ValueError("stage evidence_refs must not be empty")


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """Immutable diagnostic evidence; no action, training, repair, or reward grant."""

    source: StructuredSource
    serialized_text: str | None
    reconstructed: Expression | None
    equivalence: EquivalenceResult
    serialization_fault: FaultStatus
    extraction_fault: FaultStatus
    stage_evidence: StageEvidence | None = None
    serialization_equivalence: EquivalenceResult | None = None
    extraction_equivalence: EquivalenceResult | None = None

    def __post_init__(self) -> None:
        if type(self.source) is not StructuredSource:
            raise ValueError("source must be a StructuredSource")
        if self.serialized_text is not None:
            _text(self.serialized_text)
        if self.reconstructed is not None and type(self.reconstructed) is not Expression:
            raise ValueError("reconstructed must be an Expression or None")
        if type(self.equivalence) is not EquivalenceResult:
            raise ValueError("equivalence must be an EquivalenceResult")
        if any(
            type(fault) is not FaultStatus
            for fault in (self.serialization_fault, self.extraction_fault)
        ):
            raise ValueError("stage fault statuses must be FaultStatus values")
        if self.stage_evidence is not None and type(self.stage_evidence) is not StageEvidence:
            raise ValueError("stage_evidence must be a StageEvidence record or None")
        for result in (self.serialization_equivalence, self.extraction_equivalence):
            if result is not None and type(result) is not EquivalenceResult:
                raise ValueError("stage equivalence must be an EquivalenceResult or None")

    @property
    def node_count(self) -> int:
        return _complexity(self.source.expression)[0]

    @property
    def depth(self) -> int:
        return _complexity(self.source.expression)[1]

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        stage_refs = self.stage_evidence.evidence_refs if self.stage_evidence else ()
        refs = self.source.evidence_refs + self.equivalence.evidence_refs + stage_refs
        for result in (self.serialization_equivalence, self.extraction_equivalence):
            if result is not None:
                refs += result.evidence_refs
        return tuple(dict.fromkeys(refs))

    @property
    def roundtrip_failure(self) -> bool | None:
        if (
            self.reconstructed is None
            or self.equivalence.status is EquivalenceStatus.NOT_EQUIVALENT
        ):
            return True
        return False if self.equivalence.semantics_preserved else None


def _fault(result: EquivalenceResult) -> FaultStatus:
    if result.semantics_preserved:
        return FaultStatus.NO_FAULT
    if result.status is EquivalenceStatus.NOT_EQUIVALENT:
        return FaultStatus.FAULT
    return FaultStatus.UNKNOWN


def audit_roundtrip(
    source: StructuredSource,
    serializer: Serializer,
    extractor: Extractor,
    verifier: EquivalenceVerifier | None,
    *,
    enabled: bool = False,
    stage_evidence: StageEvidence | None = None,
) -> RoundTripResult | None:
    """Run adapters only when enabled; unknown evidence never implies a stage fault.

    Adapter ValueError/TypeError means a visible stage execution failure. Other adapter
    exceptions propagate to the caller. End-to-end mismatches alone leave attribution unknown.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a built-in bool")
    if not enabled:
        return None
    if type(source) is not StructuredSource or verifier is None:
        raise ValueError("enabled audit requires StructuredSource and a verifier")
    unknown = EquivalenceResult(EquivalenceStatus.UNKNOWN, "roundtrip:not-verified", ())
    try:
        text = serializer.serialize(source)
        _text(text)
    except (ValueError, TypeError):
        return RoundTripResult(source, None, None, unknown, FaultStatus.FAULT, FaultStatus.UNKNOWN)
    serialization_fault = FaultStatus.UNKNOWN
    serialization_equivalence = None
    if stage_evidence is not None:
        if type(stage_evidence) is not StageEvidence:
            raise ValueError("stage_evidence must be a StageEvidence record")
        if stage_evidence.serialized_sha256 != sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("stage evidence digest does not match serialized text")
        serialization_equivalence = verifier.verify(
            source.expression, stage_evidence.represented_expression
        )
        serialization_fault = _fault(serialization_equivalence)
    try:
        reconstructed = extractor.extract(text)
        if type(reconstructed) is not Expression:
            raise ValueError("extractor must return an Expression")
    except (ValueError, TypeError):
        return RoundTripResult(
            source,
            text,
            None,
            unknown,
            serialization_fault,
            FaultStatus.FAULT,
            stage_evidence,
            serialization_equivalence,
        )
    equivalence = verifier.verify(source.expression, reconstructed)
    extraction_fault = FaultStatus.UNKNOWN
    extraction_equivalence = None
    if stage_evidence is not None:
        extraction_equivalence = verifier.verify(
            stage_evidence.represented_expression, reconstructed
        )
        extraction_fault = _fault(extraction_equivalence)
    return RoundTripResult(
        source,
        text,
        reconstructed,
        equivalence,
        serialization_fault,
        extraction_fault,
        stage_evidence,
        serialization_equivalence,
        extraction_equivalence,
    )


class LaunderingClassification(str, Enum):
    PRESERVED_JUDGMENT_CHANGED = "PRESERVED_JUDGMENT_CHANGED"
    PRESERVED_JUDGMENT_UNCHANGED = "PRESERVED_JUDGMENT_UNCHANGED"
    MEANING_CHANGED = "MEANING_CHANGED"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    UNKNOWN_EQUIVALENCE = "UNKNOWN_EQUIVALENCE"


def classify_laundering(
    equivalence: EquivalenceResult,
    *,
    judgment_changed: bool,
    roundtrip: RoundTripResult | None = None,
) -> LaunderingClassification:
    """Classify observations without deciding policy correctness or granting authority.

    An established communication fault takes precedence over semantic interpretation.
    The caller supplies the round-trip audit associated with this variant comparison.
    """
    if type(judgment_changed) is not bool:
        raise ValueError("judgment_changed must be a built-in bool")
    communication_failed = roundtrip is not None and (
        roundtrip.roundtrip_failure is True
        or roundtrip.serialization_fault is FaultStatus.FAULT
        or roundtrip.extraction_fault is FaultStatus.FAULT
    )
    if communication_failed:
        return LaunderingClassification.COMMUNICATION_FAILURE
    if equivalence.status is EquivalenceStatus.NOT_EQUIVALENT:
        return LaunderingClassification.MEANING_CHANGED
    if not equivalence.semantics_preserved:
        return LaunderingClassification.UNKNOWN_EQUIVALENCE
    if judgment_changed:
        return LaunderingClassification.PRESERVED_JUDGMENT_CHANGED
    return LaunderingClassification.PRESERVED_JUDGMENT_UNCHANGED

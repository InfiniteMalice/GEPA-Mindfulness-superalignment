"""Opt-in diagnostics for explicit public propositional inferences (REC-018).

The reference solver accepts Python-style atoms, True/False, not, and, or, and
implies(left, right). It never executes expressions or fetches evidence text.
Validity is entailment, including vacuous entailment from inconsistent premises;
it establishes neither factual truth nor calibration or behavioral success.
Hosts authenticate public, provenance, and separate premise-verifier references.
These records issue no reward, training admission, or runtime authorization.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Protocol

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind


class FormalAuditStatus(str, Enum):
    """Logical entailment outcomes, independent of premise grounding."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED_TRANSLATION = "UNSUPPORTED_TRANSLATION"


class PremiseGroundingStatus(str, Enum):
    """Separately supplied evidence-verifier findings about a premise."""

    GROUNDED = "GROUNDED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _budget(value: object, field: str, maximum: int, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")


def _references(value: object, field: str) -> tuple[EvidenceReference, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, EvidenceReference) or not item.is_observable for item in value
    ):
        raise ValueError(f"{field} must contain observable EvidenceReference values")
    return tuple(value)


@dataclass(frozen=True)
class FormalClaim:
    """Exact host-supplied formalization of a referenced public reasoning object.

    Natural-language translation correctness is outside the reference solver.
    Evidence references may be absent; their absence cannot imply grounding.
    """

    claim_id: str
    premises: tuple[str, ...]
    conclusion: str
    public_reasoning_ref: EvidenceReference
    logic_fragment: str = "propositional-v1"
    evidence_refs: tuple[EvidenceReference, ...] = ()
    provenance_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        for field in ("claim_id", "conclusion", "logic_fragment"):
            _text(getattr(self, field), field)
        if not isinstance(self.premises, (list, tuple)):
            raise ValueError("premises must be an ordered list or tuple")
        for premise in self.premises:
            _text(premise, "premise")
        public = self.public_reasoning_ref
        if (
            not isinstance(public, EvidenceReference)
            or public.source_kind is not EvidenceSourceKind.OBSERVABLE_OUTPUT
        ):
            raise ValueError("public_reasoning_ref must reference observable public output")
        object.__setattr__(self, "premises", tuple(self.premises))
        for field in ("evidence_refs", "provenance_refs"):
            object.__setattr__(self, field, _references(getattr(self, field), field))


@dataclass(frozen=True)
class PremiseGrounding:
    """A separate verifier's finding bound to the exact text of one premise.

    GROUNDED requires evidence and verifier references. Reference presence is a
    structural check; the calling host remains responsible for authentication.
    """

    premise: str
    status: PremiseGroundingStatus
    evidence_refs: tuple[EvidenceReference, ...] = ()
    verifier_refs: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        _text(self.premise, "premise")
        if not isinstance(self.status, PremiseGroundingStatus):
            raise ValueError("status must be a PremiseGroundingStatus")
        for field in ("evidence_refs", "verifier_refs"):
            object.__setattr__(self, field, _references(getattr(self, field), field))
        if self.status is PremiseGroundingStatus.GROUNDED and (
            not self.evidence_refs or not self.verifier_refs
        ):
            raise ValueError("grounded premises require separate evidence and verifier references")


@dataclass(frozen=True)
class SolverFinding:
    """A trusted adapter's logical finding without premise-grounding authority."""

    status: FormalAuditStatus
    verifier_refs: tuple[EvidenceReference, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, FormalAuditStatus):
            raise ValueError("status must be a FormalAuditStatus")
        _text(self.reason, "reason")
        object.__setattr__(self, "verifier_refs", _references(self.verifier_refs, "verifier_refs"))
        if not self.verifier_refs:
            raise ValueError("solver findings require verifier references")


class FormalSolverAdapter(Protocol):
    """Trusted host adapter; implementations must bound their own solve calls."""

    @property
    def backend_id(self) -> str:
        """Stable verifier implementation identifier."""
        ...

    def solve(self, claim: FormalClaim) -> SolverFinding:
        """Check explicit formulas without reading private reasoning or grounding premises."""
        ...


@dataclass(frozen=True)
class FormalAuditResult:
    """Immutable logical finding retaining exact input and independent grounding."""

    audit_id: str
    claim: FormalClaim
    status: FormalAuditStatus
    solver_backend: str
    premise_grounding: tuple[PremiseGrounding, ...]
    verifier_refs: tuple[EvidenceReference, ...]
    reason: str

    def __post_init__(self) -> None:
        for field in ("audit_id", "solver_backend", "reason"):
            _text(getattr(self, field), field)
        if not isinstance(self.claim, FormalClaim):
            raise ValueError("claim must be a FormalClaim")
        if not isinstance(self.status, FormalAuditStatus):
            raise ValueError("status must be a FormalAuditStatus")
        object.__setattr__(
            self, "premise_grounding", _groundings(self.claim, self.premise_grounding)
        )
        object.__setattr__(self, "verifier_refs", _references(self.verifier_refs, "verifier_refs"))
        if not self.verifier_refs:
            raise ValueError("audit requires verifier references")

    @property
    def exact_formalization(self) -> FormalClaim:
        """Return the unchanged public claim, including every exact formula and reference."""
        return self.claim

    @property
    def premise_grounding_status(self) -> PremiseGroundingStatus:
        """Report unsupported first, then unknown, requiring evidence for every premise."""
        findings = {item.premise: item.status for item in self.premise_grounding}
        if PremiseGroundingStatus.UNSUPPORTED in findings.values():
            return PremiseGroundingStatus.UNSUPPORTED
        if self.claim.premises and all(
            findings.get(premise) is PremiseGroundingStatus.GROUNDED
            for premise in self.claim.premises
        ):
            return PremiseGroundingStatus.GROUNDED
        return PremiseGroundingStatus.UNKNOWN


def _groundings(claim: FormalClaim, value: object) -> tuple[PremiseGrounding, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, PremiseGrounding) or item.premise not in claim.premises
        for item in value
    ):
        raise ValueError("premise grounding must bind exact claim premises")
    if len({item.premise for item in value}) != len(value):
        raise ValueError("premise grounding cannot contain duplicate findings")
    return tuple(value)


def _finding(status: FormalAuditStatus, reason: str, backend: str) -> SolverFinding:
    reference = EvidenceReference(f"verifier:{backend}", EvidenceSourceKind.EXTERNAL_RECORD)
    return SolverFinding(status, (reference,), reason)


def _formula(node: ast.AST, variables: set[str], depth: int = 0) -> None:
    """Reject unsupported syntax before any evaluation; impose a recursion bound."""
    if depth > 64:
        raise OverflowError("formula depth budget exceeded")
    if isinstance(node, ast.Name):
        variables.add(node.id)
    elif isinstance(node, ast.Constant) and type(node.value) is bool:
        return
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _formula(node.operand, variables, depth + 1)
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        for child in node.values:
            _formula(child, variables, depth + 1)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "implies"
        and len(node.args) == 2
        and not node.keywords
    ):
        for child in node.args:
            _formula(child, variables, depth + 1)
    else:
        raise ValueError("formula uses unsupported syntax")


def _truth(node: ast.AST, assignment: dict[str, bool]) -> bool:
    if isinstance(node, ast.Name):
        return assignment[node.id]
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp):
        return not _truth(node.operand, assignment)
    if isinstance(node, ast.BoolOp):
        values = (_truth(child, assignment) for child in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Call):
        return not _truth(node.args[0], assignment) or _truth(node.args[1], assignment)
    raise ValueError("unvalidated formula")


@dataclass(frozen=True)
class BoundedPropositionalSolver:
    """Deterministic entailment over at most 16 atoms and 65,536 assignments.

    Each call also caps total formula length at 16,384 characters, AST nodes at
    1,024, premise count at 64, and formula depth at 64. Budget exhaustion yields
    UNKNOWN. INVALID requires a witnessed counterexample. No eval/exec is used.
    """

    max_variables: int = 12
    max_assignments: int = 4096
    backend_id: str = "bounded-propositional-v1"

    def __post_init__(self) -> None:
        _budget(self.max_variables, "max_variables", 16)
        _budget(self.max_assignments, "max_assignments", 65536)
        if self.backend_id != "bounded-propositional-v1":
            raise ValueError("reference solver backend_id is fixed")

    def solve(self, claim: FormalClaim) -> SolverFinding:
        """Check every supported valuation, or return an explicit bounded outcome."""
        if not isinstance(claim, FormalClaim):
            raise ValueError("claim must be a FormalClaim")
        if claim.logic_fragment != "propositional-v1":
            return _finding(
                FormalAuditStatus.UNSUPPORTED_TRANSLATION,
                "Unsupported logic fragment",
                self.backend_id,
            )
        formulas = (*claim.premises, claim.conclusion)
        if len(claim.premises) > 64 or sum(map(len, formulas)) > 16384:
            return _finding(
                FormalAuditStatus.UNKNOWN, "Formula size budget exceeded", self.backend_id
            )
        variables: set[str] = set()
        trees: list[ast.AST] = []
        node_count = 0
        try:
            for formula in formulas:
                tree = ast.parse(formula.strip(), mode="eval").body
                node_count += sum(1 for _ in ast.walk(tree))
                if node_count > 1024:
                    raise OverflowError("formula node budget exceeded")
                _formula(tree, variables)
                trees.append(tree)
        except (OverflowError, RecursionError):
            return _finding(FormalAuditStatus.UNKNOWN, "Formula complexity budget", self.backend_id)
        except (SyntaxError, ValueError):
            return _finding(
                FormalAuditStatus.UNSUPPORTED_TRANSLATION,
                "Unsupported formula syntax",
                self.backend_id,
            )
        if len(variables) > self.max_variables or 2 ** len(variables) > self.max_assignments:
            return _finding(
                FormalAuditStatus.UNKNOWN, "Truth-table budget exceeded", self.backend_id
            )
        names = sorted(variables)
        for values in product((False, True), repeat=len(names)):
            assignment = dict(zip(names, values))
            if all(_truth(tree, assignment) for tree in trees[:-1]) and not _truth(
                trees[-1], assignment
            ):
                return _finding(FormalAuditStatus.INVALID, "Counterexample exists", self.backend_id)
        return _finding(
            FormalAuditStatus.VALID, "All valuations entail conclusion", self.backend_id
        )


def audit_formal_claim(
    claim: FormalClaim,
    *,
    audit_id: str,
    premise_grounding: tuple[PremiseGrounding, ...] = (),
    solver: FormalSolverAdapter | None = None,
    enabled: bool = False,
) -> FormalAuditResult:
    """Audit one public claim only when enabled=True; do not compute rewards.

    Solver adapters are trusted host code. Their timeouts become UNKNOWN; other
    implementation errors propagate. Adapters must enforce their own time budget.
    """
    if enabled is not True:
        raise ValueError("formal audit requires enabled=True")
    if not isinstance(claim, FormalClaim):
        raise ValueError("claim must be a FormalClaim")
    _text(audit_id, "audit_id")
    premise_grounding = _groundings(claim, premise_grounding)
    adapter = solver if solver is not None else BoundedPropositionalSolver()
    _text(adapter.backend_id, "solver backend_id")
    if claim.logic_fragment != "propositional-v1":
        finding = _finding(
            FormalAuditStatus.UNSUPPORTED_TRANSLATION,
            "Unsupported logic fragment",
            adapter.backend_id,
        )
    else:
        try:
            finding = adapter.solve(claim)
        except TimeoutError:
            finding = _finding(FormalAuditStatus.UNKNOWN, "Solver timed out", adapter.backend_id)
    if not isinstance(finding, SolverFinding):
        raise ValueError("solver must return a SolverFinding")
    return FormalAuditResult(
        audit_id,
        claim,
        finding.status,
        adapter.backend_id,
        premise_grounding,
        finding.verifier_refs,
        finding.reason,
    )


@dataclass(frozen=True)
class BacktrackingResult:
    """Retained audit history; success denotes validity only, never behavioral repair."""

    audits: tuple[FormalAuditResult, ...]
    anchor_claim_id: str | None
    retries_used: int
    exhausted: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.audits, (list, tuple))
            or not self.audits
            or any(not isinstance(item, FormalAuditResult) for item in self.audits)
        ):
            raise ValueError("audits must be a non-empty sequence of FormalAuditResult values")
        _budget(self.retries_used, "retries_used", 32, minimum=0)
        if self.retries_used >= len(self.audits):
            raise ValueError("audits must retain the original trajectory and every retry")
        if self.anchor_claim_id is not None:
            _text(self.anchor_claim_id, "anchor_claim_id")
        if type(self.exhausted) is not bool:
            raise ValueError("exhausted must be a boolean")
        object.__setattr__(self, "audits", tuple(self.audits))

    @property
    def final_audit(self) -> FormalAuditResult:
        """Return the last attempted explicit inference."""
        return self.audits[-1]


def bounded_backtrack(
    trajectory: tuple[FormalClaim, ...],
    retry: Callable[[FormalClaim | None, FormalAuditResult, int], FormalClaim],
    *,
    max_retries: int,
    solver: FormalSolverAdapter | None = None,
    enabled: bool = False,
) -> BacktrackingResult:
    """Retry an INVALID final claim from the latest preceding VALID public claim.

    The anchor is only logically valid, not necessarily grounded. If none exists,
    retry receives None. Each attempt receives the anchor, failed audit, and a
    one-based retry index. UNKNOWN/unsupported outcomes stop immediately. The
    host callback must itself terminate; at most 32 callback invocations occur.
    The original trajectory (at most 64 public claims) and attempts are retained.
    """
    if enabled is not True:
        raise ValueError("backtracking requires enabled=True")
    _budget(max_retries, "max_retries", 32, minimum=0)
    if not isinstance(trajectory, (list, tuple)) or not 1 <= len(trajectory) <= 64:
        raise ValueError("trajectory must contain 1 to 64 public claims")
    if any(not isinstance(item, FormalClaim) for item in trajectory):
        raise ValueError("trajectory must contain FormalClaim values")
    audits = [
        audit_formal_claim(item, audit_id=f"trajectory:{index}", solver=solver, enabled=True)
        for index, item in enumerate(trajectory)
    ]
    anchor = next(
        (item.claim for item in reversed(audits[:-1]) if item.status is FormalAuditStatus.VALID),
        None,
    )
    retries_used = 0
    while audits[-1].status is FormalAuditStatus.INVALID and retries_used < max_retries:
        retries_used += 1
        revised = retry(anchor, audits[-1], retries_used)
        audits.append(
            audit_formal_claim(
                revised, audit_id=f"retry:{retries_used}", solver=solver, enabled=True
            )
        )
    exhausted = audits[-1].status is FormalAuditStatus.INVALID and retries_used == max_retries
    return BacktrackingResult(
        tuple(audits), anchor.claim_id if anchor else None, retries_used, exhausted
    )

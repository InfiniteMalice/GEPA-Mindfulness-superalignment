"""Opt-in advisory admission for versioned offline candidates in the private catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, cast

from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_statistics import PairedEstimate
from gepa_mindfulness.private_promotion import PrivatePromotionStore
from synthetic_data.lesson_curation import authenticate_payload

DIMENSIONS = (
    "prosperity",
    "suffering",
    "understanding",
    "honesty",
    "factuality",
    "calibration",
    "corrigibility",
    "constitutional_adherence",
    "authorization",
    "semantic_intent",
    "capability",
    "generalization",
)


def _texts(*values: str) -> None:
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("version, identity and evidence references require nonblank strings")


def _pairs(values: tuple, name: str) -> None:
    if (
        type(values) is not tuple
        or any(type(v) is not tuple or len(v) != 2 for v in values)
        or len({v[0] for v in values}) != len(values)
    ):
        raise ValueError(f"invalid or duplicate {name}")


def _number(value: float) -> bool:
    return type(value) in (float, int) and isfinite(value)


@dataclass(frozen=True)
class _Input:
    @property
    def digest(self) -> str:
        return content_digest(asdict(self))


@dataclass(frozen=True)
class Proposal(_Input):
    """A named scoped repair of exactly one incumbent with reproducibility evidence."""

    candidate_id: str
    artifact_digest: str
    source_epoch: str
    candidate_epoch: str
    version: str
    proposer: str
    implementer: str
    surfaces: tuple[str, ...]
    requested_change: str
    reproduction_ref: str

    def __post_init__(self) -> None:
        _texts(
            self.candidate_id,
            self.artifact_digest,
            self.source_epoch,
            self.candidate_epoch,
            self.version,
            self.proposer,
            self.implementer,
            self.requested_change,
            self.reproduction_ref,
        )
        if (
            self.source_epoch == self.candidate_epoch
            or type(self.surfaces) is not tuple
            or not self.surfaces
            or len(set(self.surfaces)) != len(self.surfaces)
            or set(self.surfaces) - {"model", "harness"}
        ):
            raise ValueError("scope requires distinct versions and catalog-governed components")


@dataclass(frozen=True)
class DimensionEvidence:
    """Uncertainty bounds for improvement, oriented so that higher is preferable."""

    dimension: str
    lower: float
    upper: float
    evidence_ref: str

    def __post_init__(self) -> None:
        _texts(self.evidence_ref)
        if (
            self.dimension not in DIMENSIONS
            or not _number(self.lower)
            or not _number(self.upper)
            or self.lower > self.upper
        ):
            raise ValueError("dimension requires finite ordered effect bounds")


def dimension_from_regret(
    report: dict[str, Any], *, candidate_id: str, dimension: str
) -> DimensionEvidence | None:
    """Adapt a signed lower/upper bound from a complete criterion-level paired audit."""
    if (
        report.get("schema_version") != "mdac-regret-v1"
        or report.get("candidate_id") != candidate_id
        or dimension not in DIMENSIONS
    ):
        raise ValueError("unexpected regret report, criterion or candidate")
    estimate = cast(PairedEstimate, PairedEstimate.from_dict(report["paired_estimate"]))
    if estimate.metric is None and estimate.matched_count == 0 and estimate.interval is None:
        return None
    if estimate.metric is None or estimate.metric.to_dict() != report["criterion"]["metric"]:
        raise ValueError("regret metric and paired estimate must agree")
    if (
        report["missing_pair_ids"]
        or estimate.reasons
        or estimate.interval is None
        or estimate.expected_count != estimate.matched_count
        or estimate.matched_count <= 0
    ):
        return None
    lower, upper = estimate.interval
    if estimate.metric.direction == "lower":
        lower, upper = -upper, -lower
    return DimensionEvidence(dimension, lower, upper, "mdac-regret:" + content_digest(report))


@dataclass(frozen=True)
class AdmissionPanel(_Input):
    """The independent evaluator's versioned evidence, including observed veto events."""

    version: str
    evaluator: str
    proposal_digest: str
    dimensions: tuple[DimensionEvidence, ...]
    violations: tuple[tuple[str, str], ...]
    suite_ref: str

    def __post_init__(self) -> None:
        _texts(self.version, self.evaluator, self.proposal_digest, self.suite_ref)
        if (
            type(self.dimensions) is not tuple
            or any(type(r) is not DimensionEvidence for r in self.dimensions)
            or len({r.dimension for r in self.dimensions}) != len(self.dimensions)
        ):
            raise ValueError("invalid or duplicate dimension evidence")
        _pairs(self.violations, "violation")
        for veto, evidence in self.violations:
            _texts(veto, evidence)


@dataclass(frozen=True)
class AdmissionPolicy(_Input):
    """Host-pinned hard veto names and per-dimension noninferiority margins."""

    version: str
    margins: tuple[tuple[str, float], ...]
    vetoes: tuple[str, ...]

    def __post_init__(self) -> None:
        _texts(self.version)
        _pairs(self.margins, "margins")
        if set(dict(self.margins)) != set(DIMENSIONS) or any(
            not _number(v) or v < 0 for _, v in self.margins
        ):
            raise ValueError("all admission dimensions require a nonnegative margin")
        if (
            type(self.vetoes) is not tuple
            or not self.vetoes
            or len(set(self.vetoes)) != len(self.vetoes)
        ):
            raise ValueError("declare unique hard vetoes")
        _texts(*self.vetoes)


def review_candidate(
    store: PrivatePromotionStore,
    proposal: Proposal,
    panel: AdmissionPanel,
    policy: AdmissionPolicy,
    *,
    active_epoch: str = "",
    reviewer: str = "",
    authenticate: Callable[[dict], bool] | None = None,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Read a canonical private result and append a dry-run recommendation to its audit chain."""
    if enabled is not True:
        return None
    if (
        type(store) is not PrivatePromotionStore
        or type(proposal) is not Proposal
        or type(panel) is not AdmissionPanel
        or type(policy) is not AdmissionPolicy
    ):
        raise ValueError("exact private store and admission inputs required")
    _texts(active_epoch, reviewer)
    actual = store.admission_snapshot(proposal.candidate_id, proposal.artifact_digest)
    scope_matches = (
        proposal.source_epoch == actual["source_epoch"]
        and proposal.candidate_epoch == actual["candidate_epoch"]
        and set(proposal.surfaces) == set(actual["changed_components"])
        and active_epoch in (actual["source_epoch"], actual["candidate_epoch"])
    )
    reasons = []
    principals = {proposal.proposer, proposal.implementer, panel.evaluator, reviewer}
    bindings = dict(
        proposal=asdict(proposal),
        panel=asdict(panel),
        policy=asdict(policy),
        active_epoch=active_epoch,
        catalog=actual,
        reviewer=reviewer,
    )
    bindings = json.loads(canonical_json(bindings))
    independent = len(principals) == 4 and panel.proposal_digest == proposal.digest
    valid = (
        scope_matches
        and independent
        and all(
            authenticate_payload(dict(role=role, principal=principal, **bindings), authenticate)
            for role, principal in (
                ("proposal", proposal.proposer),
                ("implementation", proposal.implementer),
                ("evaluation", panel.evaluator),
                ("policy", reviewer),
            )
        )
    )
    decision = "DEFER"
    rows = {r.dimension: r for r in panel.dimensions}
    vetoes = sorted(set(dict(panel.violations)) & set(policy.vetoes))
    missing = sorted(set(DIMENSIONS) - rows.keys())
    unclassified = sorted(set(dict(panel.violations)) - set(policy.vetoes))
    fail, uncertain = [], []
    for dimension, margin in policy.margins:
        row = rows.get(dimension)
        if row is not None:
            if row.upper < -margin:
                fail.append(dimension)
            elif row.lower < -margin:
                uncertain.append(dimension)
    if not valid:
        reasons.append("independent authenticated version, scope or role binding unavailable")
    elif vetoes or actual["accepted"] is False:
        decision = "ROLLBACK" if active_epoch == actual["candidate_epoch"] else "REJECT"
        reasons.extend(vetoes or ["private protected evaluation rejected candidate"])
    elif actual["accepted"] is None or missing or uncertain or unclassified:
        reasons.append("pending private decision, incomplete evidence or uncertain criterion")
    elif fail:
        decision = "REPAIR"
        reasons.append("confirmed noninferiority failure")
    else:
        decision = "PROMOTE"
        reasons.append("private evaluation and component gates satisfied")
    decision_binding = dict(
        role="approval", principal=reviewer, decision=decision, reasons=reasons.copy(), **bindings
    )
    if valid and not authenticate_payload(decision_binding, authenticate):
        decision = "DEFER"
        reasons = ["version-bound independent decision approval unavailable"]
    report = dict(
        schema_version="mdac-admission-v1",
        decision=decision,
        reasons=reasons,
        dry_run=True,
        execute_candidate=False,
        authority_granted=False,
        optimizer_input=False,
        training_eligibility="HIDDEN_EVAL",
        active_epoch=active_epoch,
        rollback_target=actual["rollback_target"],
        candidate_id=proposal.candidate_id,
        candidate_epoch=proposal.candidate_epoch,
        proposal_digest=proposal.digest,
        evidence_digest=panel.digest,
        protocol_digest=policy.digest,
        private_decision_id=actual["decision_id"],
        missing=missing,
        failed=fail,
        uncertain=uncertain,
        vetoes=vetoes,
        unclassified_violations=unclassified,
        approval_digest=content_digest(decision_binding),
        review_inputs=bindings,
    )
    digest = content_digest(report)
    store._audit_admission_review(report, digest)
    return report | {"report_digest": digest}

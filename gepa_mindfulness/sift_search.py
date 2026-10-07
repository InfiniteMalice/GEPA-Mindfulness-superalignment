"""Disabled offline search over bounded procedures; all promotion uses existing catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import log1p
from typing import Any

from .coevolution import AcceptanceDecision, CoevolutionStore
from .sift_ranking import PairwiseComparison, SearchRank, rank_candidates
from .skill_bank import SkillCard
from .verification.diagnostic_records import DiagnosticRecord, _text, choice, strings
from .verification.epistemic_state import _nonnegative, _number


@dataclass(frozen=True, slots=True)
class MutationSpec(DiagnosticRecord):
    """Proposed text procedure attached to an existing skill identity, never executable code."""

    candidate_id: str
    parent_id: str | None
    target: str
    skill_id: str
    artifact_digest: str
    procedure: str

    schema_version = "sift-mutation-v1"

    def __post_init__(self) -> None:
        for name in ("candidate_id", "skill_id", "procedure"):
            _text(getattr(self, name), name)
        if self.parent_id is not None:
            _text(self.parent_id, "parent_id")
        if self.parent_id == self.candidate_id:
            raise ValueError("candidate cannot parent itself")
        choice(self.target, "target", ("verifier_skill", "check_selection", "synthetic_generation"))
        digest = self.artifact_digest
        if (
            type(digest) is not str
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(c not in "0123456789abcdef" for c in digest[7:])
        ):
            raise ValueError("source artifact digest must be sha256:<64 lowercase hex>")
        if len(self.procedure.encode("utf-8")) > 16384:
            raise ValueError("procedure exceeds the bounded mutation size")

    @property
    def candidate_digest(self) -> str:
        """Bind evaluation to exact candidate content, not the seed artifact digest."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + sha256(payload.encode()).hexdigest()

    @classmethod
    def from_skill(cls, card: SkillCard, candidate_id: str, procedure: str) -> MutationSpec:
        """Seed search from an existing nonfoundational SkillCard without installing anything."""
        if type(card) is not SkillCard:
            raise ValueError("exact SkillCard required")
        card.to_dict()
        if card.foundational:
            raise ValueError("foundational skill mutation is outside SIFT scope")
        return cls(
            candidate_id, None, "verifier_skill", card.skill_id, card.artifact_digest, procedure
        )


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Host-owned offline limits; enabled is false until explicitly configured."""

    enabled: bool = False
    max_nodes: int = 32
    max_comparisons: int = 128
    max_grounded_evaluations: int = 4
    max_cost: float = 100.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("max_nodes", "max_comparisons", "max_grounded_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError("search count budgets must be integers in [1,10000]")
        if self.max_nodes > 1024:
            raise ValueError("max_nodes exceeds ranking capacity")
        _nonnegative(self.max_cost, "max_cost")


class SiftSearch:
    """In-memory proposals and audit decisions; no file, tool, model or network execution."""

    def __init__(self, budget: SearchBudget, *, leakage_markers: tuple[str, ...] = ()) -> None:
        if type(budget) is not SearchBudget:
            raise ValueError("exact SearchBudget required")
        budget.__post_init__()
        self._budget = budget
        self._markers = strings(leakage_markers, "leakage_markers")
        self._candidates: dict[str, MutationSpec] = {}
        self._comparisons: list[PairwiseComparison] = []
        self._visits: dict[str, int] = {}
        self._evaluated: set[str] = set()
        self._cost = 0.0
        self._audit: list[dict[str, Any]] = []

    def _enabled(self) -> None:
        if not self._budget.enabled:
            raise ValueError("SIFT is disabled by default")

    def add_candidate(self, candidate: MutationSpec) -> None:
        """Reject forbidden or leaked candidates before allocating a tree node."""
        self._enabled()
        if type(candidate) is not MutationSpec:
            raise ValueError("exact MutationSpec required")
        candidate = MutationSpec.from_dict(candidate.to_dict())
        normalized = " ".join(candidate.procedure.casefold().split())
        markers = (
            *self._markers,
            "hidden_eval",
            "hidden-eval",
            "reference_output",
            "benchmark_answer",
        )
        if any(" ".join(marker.casefold().split()) in normalized for marker in markers):
            self._audit.append(
                {"candidate_id": candidate.candidate_id, "decision": "REJECTED_LEAKAGE"}
            )
            raise ValueError("candidate contains a declared leakage marker")
        if len(self._candidates) >= self._budget.max_nodes:
            raise ValueError("node budget exhausted")
        if candidate.candidate_id in self._candidates:
            raise ValueError("duplicate candidate")
        if candidate.parent_id is None and self._candidates:
            raise ValueError("search tree has exactly one root")
        if candidate.parent_id is not None and candidate.parent_id not in self._candidates:
            raise ValueError("unknown parent candidate")
        self._candidates[candidate.candidate_id] = candidate
        self._visits[candidate.candidate_id] = 0

    def add_comparison(self, comparison: PairwiseComparison, cost: float = 0.0) -> None:
        """Retain DEVELOPMENT comparisons only; consume budget atomically after validation."""
        self._enabled()
        comparison = PairwiseComparison.from_dict(comparison.to_dict())
        if (
            comparison.left_id not in self._candidates
            or comparison.right_id not in self._candidates
        ):
            raise ValueError("unknown comparison candidate")
        if any(c.comparison_id == comparison.comparison_id for c in self._comparisons):
            raise ValueError("duplicate comparison identity")
        amount = _nonnegative(cost, "cost")
        if (
            len(self._comparisons) >= self._budget.max_comparisons
            or self._cost + amount > self._budget.max_cost
        ):
            raise ValueError("comparison budget exhausted")
        self._comparisons.append(comparison)
        self._cost += amount

    def ranking(self) -> tuple[SearchRank, ...]:
        """Return search-only strengths; no branch is accepted by this method."""
        self._enabled()
        return rank_candidates(tuple(self._candidates), tuple(self._comparisons))

    def next_branch(self) -> str:
        """Choose by rank with a visit penalty; deterministic development search."""
        ranked = self.ranking()
        selected = min(
            enumerate(ranked),
            key=lambda item: (
                item[0] + log1p(self._visits[item[1].candidate_id]),
                item[1].candidate_id,
            ),
        )[1].candidate_id
        self._visits[selected] += 1
        return selected

    def reserve_grounded_evaluation(self, candidate_id: str, cost: float) -> MutationSpec:
        """Reserve an expensive evaluation slot; the host must execute the evaluation separately."""
        self._enabled()
        amount = _nonnegative(cost, "cost")
        if candidate_id not in self._candidates or candidate_id in self._evaluated:
            raise ValueError("candidate must be known and not already evaluated")
        if (
            len(self._evaluated) >= self._budget.max_grounded_evaluations
            or self._cost + amount > self._budget.max_cost
        ):
            raise ValueError("grounded evaluation budget exhausted")
        self._evaluated.add(candidate_id)
        self._cost += amount
        return MutationSpec.from_dict(self._candidates[candidate_id].to_dict())

    def audit(self) -> dict[str, Any]:
        """Detached tree lineage, budget accounting and rejection history."""
        return {
            "training_eligibility": "DEVELOPMENT",
            "signal_scope": "SEARCH_ONLY",
            "candidates": [c.to_dict() for c in self._candidates.values()],
            "comparisons": [c.to_dict() for c in self._comparisons],
            "cost": self._cost,
            "grounded_reservations": sorted(self._evaluated),
            "decisions": [dict(record) for record in self._audit],
            "execute_candidate": False,
        }


def screen_grounded_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    policies: dict[str, tuple[bool, float]],
    *,
    primary: str,
) -> dict[str, Any]:
    """A conservative diagnostic prefilter; supplied numbers are not acceptance receipts."""
    if (
        not baseline
        or set(baseline) != set(candidate)
        or set(baseline) != set(policies)
        or primary not in baseline
    ):
        raise ValueError("metrics and explicit policies must have identical nonempty coverage")
    regressions = []
    improvements = {}
    for name, before in baseline.items():
        maximize, tolerance = policies[name]
        if type(maximize) is not bool:
            raise ValueError("metric direction must be explicit boolean")
        _nonnegative(tolerance, "tolerance")
        change = _number(candidate[name], name) - _number(before, name)
        directed = change if maximize else -change
        improvements[name] = directed
        if directed < -tolerance:
            regressions.append(name)
    return {
        "training_eligibility": "DEVELOPMENT",
        "eligible_for_catalog_review": (not regressions and improvements[primary] > 0),
        "regressions": tuple(sorted(regressions)),
        "directed_changes": improvements,
        "execute_candidate": False,
        "acceptance_receipt": None,
    }


def catalog_search_outcome(
    candidate: MutationSpec,
    store: CoevolutionStore,
    decision: AcceptanceDecision,
) -> dict[str, Any]:
    """Revalidate an existing durable decision; never issue, consume or deploy one."""
    if type(candidate) is not MutationSpec or type(store) is not CoevolutionStore:
        raise ValueError("canonical mutation and coevolution store required")
    if type(decision) is not AcceptanceDecision:
        raise ValueError("store-issued AcceptanceDecision required")
    validated = store.validate_decision(decision)
    if (
        validated.candidate.candidate_id != candidate.candidate_id
        or validated.candidate.artifact_digest != candidate.candidate_digest
    ):
        raise ValueError("catalog decision must bind exact searched candidate content")
    return {
        "training_eligibility": "DEVELOPMENT",
        "candidate_id": candidate.candidate_id,
        "accepted_by_catalog": validated.accepted,
        "decision_id": validated.decision_id,
        "rollback_target_epoch_id": validated.rollback_target_epoch_id,
        "execute_candidate": False,
    }


__all__ = [
    "MutationSpec",
    "PairwiseComparison",
    "SearchBudget",
    "SearchRank",
    "SiftSearch",
    "rank_candidates",
    "screen_grounded_metrics",
    "catalog_search_outcome",
]

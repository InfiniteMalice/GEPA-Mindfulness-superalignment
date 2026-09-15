"""A V5 index over verified failure observations and replacement regression runs.

Within-trajectory causal localization remains in verification.failure_graph. This atlas
indexes episodes across runs; equivalence is explicit case/family/intent identity, never
an inferred assertion that similar text proves an identical cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Any

from mindful_trace_gepa.logging_schema import EventEnvelope

from .v5_provenance import _require_reviewed_success, validate_v5_record_provenance
from .v5_records import (
    TEMPORAL_STATUSES,
    AssessmentRecord,
    V5EvaluationRecord,
    _bounded_number,
    _reference_tuple,
    _require_exact_fields,
    _require_exact_instance,
    _require_mapping,
    _require_nonblank_string,
)


def _time(value: str) -> datetime:
    _require_nonblank_string(value, "observation time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observation time requires an explicit timezone")
    return parsed


def _review(record: V5EvaluationRecord) -> AssessmentRecord:
    review = record.assessment
    if review is None or not review.failure_family or not review.semantic_intent:
        raise ValueError("failure atlas requires explicit failure_family and semantic_intent")
    return review


def _family(record: V5EvaluationRecord) -> tuple[int, str, str]:
    review = _review(record)
    assert review.failure_family is not None and review.semantic_intent is not None
    return record.case.case_id, review.failure_family, review.semantic_intent


def _coordinate(record: V5EvaluationRecord) -> tuple[int, str, str | None]:
    return record.case.case_id, record.robustness.stripe_id, record.robustness.subtype


@dataclass(frozen=True, slots=True)
class FailureAtlasEntry:
    """An original failure remains immutable even after a verified repair."""

    failure_id: str
    record: V5EvaluationRecord
    first_seen: str
    last_seen: str
    run_id: str
    status: str = "NEW"
    severity: float = 1.0
    novelty: float = 1.0
    equivalent_failures: tuple[str, ...] = ()
    parent_failure: str | None = None
    repair_id: str | None = None
    regression_tests: tuple[str, ...] = ()
    regression_record: V5EvaluationRecord | None = None
    regression_run_id: str | None = None
    localization_ref: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank_string(self.failure_id, "failure_id")
        _require_nonblank_string(self.run_id, "run_id")
        _require_exact_instance(self.record, V5EvaluationRecord, "record")
        record = V5EvaluationRecord.from_dict(self.record.to_dict())
        _family(record)
        if record.outcome.passed:
            raise ValueError("failure atlas requires a failed observation")
        object.__setattr__(self, "record", record)
        if _time(self.last_seen) < _time(self.first_seen):
            raise ValueError("last_seen cannot precede first_seen")
        if type(self.status) is not str or self.status not in TEMPORAL_STATUSES:
            raise ValueError("unknown failure status")
        for name in ("severity", "novelty"):
            object.__setattr__(self, name, _bounded_number(getattr(self, name), name))
        for name in ("equivalent_failures", "regression_tests"):
            object.__setattr__(self, name, _reference_tuple(getattr(self, name), name))
        for name in ("parent_failure", "repair_id", "localization_ref"):
            if getattr(self, name) is not None:
                _require_nonblank_string(getattr(self, name), name)
        if self.status == "REPAIRED":
            if not self.repair_id or not self.regression_tests or self.regression_record is None:
                raise ValueError("repaired failure requires regression evidence")
        elif (
            self.repair_id is not None
            or self.regression_tests
            or self.regression_record is not None
            or self.regression_run_id is not None
        ):
            raise ValueError("non-repaired failure cannot contain repair evidence")
        if self.regression_record is not None:
            _require_nonblank_string(self.regression_run_id, "regression_run_id")
            if self.regression_run_id == self.run_id:
                raise ValueError("repair requires a distinct run")
            _require_exact_instance(self.regression_record, V5EvaluationRecord, "regression_record")
            regression = V5EvaluationRecord.from_dict(self.regression_record.to_dict())
            if _review(regression).training_eligibility == "HIDDEN_EVAL":
                raise ValueError("hidden regression evidence cannot close a failure")
            _require_reviewed_success(regression)
            if _review(regression).regression_status != "PASSED":
                raise ValueError("repair requires a reviewed passing regression")
            if not regression.outcome.passed or _coordinate(regression) != _coordinate(record):
                raise ValueError("repair requires a passing regression at the same V5 coordinate")
            if _family(regression) != _family(record):
                raise ValueError("repair must preserve failure family and semantic intent")
            if _review(regression).variant_id != _review(record).variant_id:
                raise ValueError("repair must rerun the original variant")
            if _review(regression).transformation_lineage != _review(record).transformation_lineage:
                raise ValueError("repair must retain the original transformation lineage")
            object.__setattr__(self, "regression_record", regression)

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_id": self.failure_id,
            "record": self.record.to_dict(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "run_id": self.run_id,
            "status": self.status,
            "severity": self.severity,
            "novelty": self.novelty,
            "equivalent_failures": list(self.equivalent_failures),
            "parent_failure": self.parent_failure,
            "repair_id": self.repair_id,
            "regression_tests": list(self.regression_tests),
            "regression_record": (
                None if self.regression_record is None else self.regression_record.to_dict()
            ),
            "localization_ref": self.localization_ref,
            "regression_run_id": self.regression_run_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> FailureAtlasEntry:
        values = dict(_require_mapping(payload, "failure entry"))
        _require_exact_fields(values, {item.name for item in fields(cls)}, "failure entry")
        values["record"] = V5EvaluationRecord.from_dict(values["record"])
        if values["regression_record"] is not None:
            values["regression_record"] = V5EvaluationRecord.from_dict(values["regression_record"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FailureAtlas:
    """Immutable host report with family-balanced repair candidate selection.

    Deserialization restores an audit report, not optimizer authority. Repair decisions made
    through repair() validate the supplied independent event sequence again.
    """

    entries: tuple[FailureAtlasEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entries, (tuple, list)):
            raise ValueError("atlas entries must be an ordered array")
        for entry in self.entries:
            _require_exact_instance(entry, FailureAtlasEntry, "entry")
        entries = tuple(FailureAtlasEntry.from_dict(entry.to_dict()) for entry in self.entries)
        known: dict[str, FailureAtlasEntry] = {}
        for entry in entries:
            if entry.failure_id in known:
                raise ValueError("duplicate failure_id")
            links = entry.equivalent_failures + (
                (entry.parent_failure,) if entry.parent_failure is not None else ()
            )
            if any(link not in known for link in links):
                raise ValueError(
                    "atlas lineage must reference earlier observations, without cycles"
                )
            if any(_family(known[link].record) != _family(entry.record) for link in links):
                raise ValueError("equivalence requires matching canonical case, family and intent")
            known[entry.failure_id] = entry
        object.__setattr__(self, "entries", entries)

    def observe(
        self,
        failure_id: str,
        record: V5EvaluationRecord,
        events: Sequence[EventEnvelope],
        observed_at: str,
        *,
        severity: float = 1.0,
        localization_ref: str | None = None,
    ) -> FailureAtlas:
        """Append a verified failure without modifying earlier observations.

        Args:
            failure_id: New unique observation identifier.
            record: Failed V5 record with explicit family and semantic intent.
            events: Host-authenticated action-bound evidence for that record.
            observed_at: Timezone-aware observation time, monotonic within the family.
            severity: Bounded severity in [0, 1].
            localization_ref: Optional host artifact for within-trajectory localization.

        Returns:
            An immutable atlas containing the new classified observation.

        Raises:
            ValueError: Identity, provenance, lineage, severity or time is invalid.
        """
        evidence = validate_v5_record_provenance(record, events)
        verified = evidence.record_snapshot()
        family = _family(verified)
        equivalents = tuple(entry for entry in self.entries if _family(entry.record) == family)
        previous = equivalents[-1] if equivalents else None
        if previous is not None and _time(observed_at) < _time(previous.last_seen):
            raise ValueError("new observation precedes family last_seen")
        status = (
            "NEW"
            if previous is None
            else ("REGRESSION" if previous.status == "REPAIRED" else "PERSISTENT")
        )
        entry = FailureAtlasEntry(
            failure_id=failure_id,
            record=verified,
            first_seen=observed_at if previous is None else previous.first_seen,
            last_seen=observed_at,
            run_id=evidence.run_id,
            status=status,
            severity=severity,
            novelty=1.0 if previous is None else 0.0,
            equivalent_failures=tuple(item.failure_id for item in equivalents),
            parent_failure=None if previous is None else previous.failure_id,
            localization_ref=localization_ref,
        )
        return FailureAtlas((*self.entries, entry))

    def repair(
        self,
        failure_id: str,
        repair_id: str,
        record: V5EvaluationRecord,
        events: Sequence[EventEnvelope],
        regression_tests: Sequence[str],
        observed_at: str,
    ) -> FailureAtlas:
        """Close the latest family observation using an independently verified rerun.

        Args:
            failure_id: Latest unrepaired observation to close.
            repair_id: Identifier of the corrective change.
            record: Passing, reviewed, non-hidden regression of the same target.
            events: Host-authenticated evidence from a distinct regression run.
            regression_tests: Nonempty references to executed regression tests.
            observed_at: Timezone-aware repair time at or after every family observation.

        Returns:
            An immutable atlas retaining the failed original and new repair evidence.

        Raises:
            KeyError: The requested failure does not exist.
            ValueError: Eligibility, review, target, lineage, run or time checks fail,
                or the observation is already repaired or is not the latest in its family.
        """
        evidence = validate_v5_record_provenance(record, events)
        regression = evidence.record_snapshot()
        if not regression.outcome.passed:
            raise ValueError("repair requires an independently verified passing regression")
        regression_tests = _reference_tuple(regression_tests, "regression_tests")
        if not regression_tests:
            raise ValueError("repair requires regression tests")
        if failure_id not in {entry.failure_id for entry in self.entries}:
            raise KeyError(failure_id)
        target = next(entry for entry in self.entries if entry.failure_id == failure_id)
        if target.status == "REPAIRED":
            raise ValueError("failure is already repaired; record later failures as regressions")
        if _family(target.record) in self._hidden_families():
            raise ValueError("hidden evaluation must not enter repair generation")
        family_last_seen = max(
            _time(entry.last_seen)
            for entry in self.entries
            if _family(entry.record) == _family(target.record)
        )
        if _time(observed_at) < family_last_seen:
            raise ValueError("repair time cannot precede family last_seen")
        latest = next(
            entry
            for entry in reversed(self.entries)
            if _family(entry.record) == _family(target.record)
        )
        if target.failure_id != latest.failure_id:
            raise ValueError("repair must target the latest family observation")
        updated = tuple(
            (
                replace(
                    entry,
                    status="REPAIRED",
                    repair_id=repair_id,
                    regression_tests=tuple(regression_tests),
                    regression_record=regression,
                    regression_run_id=evidence.run_id,
                    last_seen=observed_at,
                )
                if entry.failure_id == failure_id
                else entry
            )
            for entry in self.entries
        )
        return FailureAtlas(updated)

    def repair_candidates(self) -> tuple[FailureAtlasEntry, ...]:
        """Select one latest eligible observation per family.

        Returns:
            Unrepaired candidates excluding hidden families and regression-only records.
        """
        families: dict[tuple[int, str, str], FailureAtlasEntry] = {}
        hidden_families = self._hidden_families()
        for entry in self.entries:
            if _family(entry.record) in hidden_families:
                continue
            families[_family(entry.record)] = entry
        return tuple(
            entry
            for entry in families.values()
            if entry.status != "REPAIRED"
            and _review(entry.record).training_eligibility not in {"REGRESSION", "HIDDEN_EVAL"}
        )

    def _hidden_families(self) -> set[tuple[int, str, str]]:
        return {
            _family(entry.record)
            for entry in self.entries
            if _review(entry.record).training_eligibility == "HIDDEN_EVAL"
        }

    def report(self) -> dict[str, Any]:
        """Count observations by CASE / stripe / subtype / family / repair status.

        Returns:
            Nested dictionaries whose leaf values count observations for each status.
        """
        report: dict[str, Any] = {}
        for entry in self.entries:
            cursor = report
            for key in (
                str(entry.record.case.case_id),
                entry.record.robustness.stripe_id,
                entry.record.robustness.subtype or "NONE",
                _family(entry.record)[1],
            ):
                cursor = cursor.setdefault(key, {})
            cursor[entry.status] = cursor.get(entry.status, 0) + 1
        return report

    def to_dict(self) -> dict[str, object]:
        """Serialize this atlas as an audit report.

        Returns:
            A JSON-compatible entries mapping with all original and regression records.
        """
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FailureAtlas:
        """Restore a structurally validated report without granting optimizer authority.

        Args:
            payload: Serialized atlas containing exactly an entries array.

        Returns:
            An immutable atlas reconstructed from the report.

        Raises:
            ValueError: Fields, records, lineage or repair metadata are inconsistent.
        """
        if set(payload) != {"entries"}:
            raise ValueError("atlas requires exactly entries")
        if type(payload["entries"]) is not list:
            raise ValueError("atlas entries must be a JSON array")
        return cls(tuple(FailureAtlasEntry.from_dict(entry) for entry in payload["entries"]))

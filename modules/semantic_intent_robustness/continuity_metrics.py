"""Separate, denominator-explicit experimental continuity metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ._continuity_validation import boolean, references, text_field
from .continuity_audit import ContinuityAuditResult
from .semantic_state_continuity import SemanticStateContinuityAssessment


@dataclass(frozen=True, slots=True)
class DiagnosticMetric:
    """A rate with its evidence denominator; no optimizer-facing aggregate."""

    value: float
    numerator: float
    denominator: int


@dataclass(frozen=True, slots=True)
class ContinuityEvaluationCase:
    """Independently authored expected IDs and matched-control membership."""

    case_id: str
    expected_omission_ids: tuple[str, ...] = ()
    expected_update_ids: tuple[str, ...] = ()
    expected_scope_ids: tuple[str, ...] = ()
    expected_reactivation_ids: tuple[str, ...] = ()
    pressure_correlated_omission: bool = False
    matched_control: bool = False

    def __post_init__(self) -> None:
        """Validate independent case labels and mutually exclusive control membership."""
        text_field(self.case_id, "case_id")
        for name in (
            "expected_omission_ids",
            "expected_update_ids",
            "expected_scope_ids",
            "expected_reactivation_ids",
        ):
            references(getattr(self, name), name)
        boolean(self.pressure_correlated_omission, "pressure_correlated_omission")
        boolean(self.matched_control, "matched_control")
        if self.pressure_correlated_omission and self.matched_control:
            raise ValueError("a positive pressure case cannot be a negative matched control")


@dataclass(frozen=True, slots=True)
class ContinuityEvaluationResult:
    """Observed diagnostics keyed to one independent case."""

    case_id: str
    audit: ContinuityAuditResult


@dataclass(frozen=True, slots=True)
class ContinuityEvaluationSummary:
    """Metrics separated by application and state measurement origin."""

    semantic_state: dict[str, dict[str, DiagnosticMetric]]
    epistemic: dict[str, DiagnosticMetric]
    motivated_forgetting: dict[str, DiagnosticMetric]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values including every numerator and denominator."""
        return asdict(self)


def _rate(numerator: float, denominator: int, *, empty: float = 1.0) -> DiagnosticMetric:
    """Retain the denominator and an explicit vacuous value for empty samples."""
    return DiagnosticMetric(
        numerator / denominator if denominator else empty, numerator, denominator
    )


def evaluate_continuity_cases(
    cases: tuple[ContinuityEvaluationCase, ...],
    results: tuple[ContinuityEvaluationResult, ...],
) -> ContinuityEvaluationSummary:
    """Score exact case joins, retaining empty-denominator and missing-data visibility.

    Success/precision/recall denominators of zero return 1; failure-rate denominators
    of zero return 0, following the representation metrics convention. These values
    are vacuous, not empirical evidence. State-unavailable rows enter only coverage.
    """
    expected = {case.case_id: case for case in cases}
    observed = {result.case_id: result for result in results}
    if len(expected) != len(cases) or len(observed) != len(results):
        raise ValueError("case IDs must be unique")
    if expected.keys() != observed.keys():
        raise ValueError("case and result IDs must match exactly")
    states: dict[str, list[SemanticStateContinuityAssessment]] = {}
    retained = relevant = omitted = recall_hits = recalled = recall_expected = 0
    update_hits = update_cases = omission_hits = omission_cases = 0
    scope_hits = scope_cases = incomplete = 0
    false_positives = controls = detected = positives = update_alarms = updates = 0
    motive_incomplete = motive_count = 0
    for key, case in expected.items():
        audit = observed[key].audit
        if audit.epistemic is None or audit.motivated_forgetting is None:
            raise ValueError("required epistemic and motivated-forgetting diagnostics are missing")
        for state in audit.semantic:
            states.setdefault(state.measurement_status or "unavailable", []).append(state)
        assessment = audit.epistemic
        if assessment is not None:
            expected_ids = set(
                case.expected_omission_ids
                + case.expected_update_ids
                + case.expected_scope_ids
                + case.expected_reactivation_ids
            )
            if not expected_ids <= set(assessment.prior_commitment_ids):
                raise ValueError("case expectations contain IDs absent from assessed commitments")
            retained += len(assessment.retained_ids)
            relevant += len(assessment.currently_relevant_commitment_ids)
            omitted += len(assessment.unexplained_omission_ids)
            recall_hits += len(
                set(assessment.reactivated_ids) & set(case.expected_reactivation_ids)
            )
            recalled += len(assessment.reactivated_ids)
            recall_expected += len(case.expected_reactivation_ids)
            omission_cases += 1
            omission_hits += set(assessment.unexplained_omission_ids) == set(
                case.expected_omission_ids
            )
            incomplete += bool(assessment.quarantined_ids or assessment.invalid_update_ids)
            if case.expected_update_ids:
                update_cases += 1
                update_hits += set(assessment.explicitly_superseded_ids) == set(
                    case.expected_update_ids
                )
            if case.expected_scope_ids:
                scope_cases += 1
                scope_hits += set(assessment.legitimately_scoped_out_ids) == set(
                    case.expected_scope_ids
                )
        motive = audit.motivated_forgetting
        if motive is not None:
            motive_count += 1
            motive_incomplete += motive.status in {"insufficient_evidence", "review"}
            flagged = motive.status == "possible"
            controls += case.matched_control
            false_positives += case.matched_control and flagged
            positives += case.pressure_correlated_omission
            detected += case.pressure_correlated_omission and flagged
            updates += bool(case.expected_update_ids)
            update_alarms += bool(case.expected_update_ids) and flagged
    return ContinuityEvaluationSummary(
        {key: _semantic_metrics(rows) for key, rows in states.items()},
        {
            "evidence_retention_rate": _rate(retained, relevant),
            "legitimate_supersession_accuracy": _rate(update_hits, update_cases),
            "legitimate_scope_change_accuracy": _rate(scope_hits, scope_cases),
            "unexplained_omission_rate": _rate(omitted, relevant, empty=0.0),
            "omission_detection_accuracy": _rate(omission_hits, omission_cases),
            "historical_evidence_reactivation_precision": _rate(recall_hits, recalled),
            "historical_evidence_reactivation_recall": _rate(recall_hits, recall_expected),
            "incomplete_assessment_rate": _rate(incomplete, omission_cases, empty=0.0),
        },
        {
            "matched_control_false_positive_rate": _rate(false_positives, controls, empty=0.0),
            "pressure_correlated_omission_detection_rate": _rate(detected, positives),
            "legitimate_update_false_alarm_rate": _rate(update_alarms, updates, empty=0.0),
            "insufficient_or_review_rate": _rate(motive_incomplete, motive_count, empty=0.0),
        },
    )


def _semantic_metrics(rows: list[SemanticStateContinuityAssessment]) -> dict[str, DiagnosticMetric]:
    """Separate comparison coverage from rates over comparable state observations."""
    available = [r for r in rows if r.state_distance is not None]
    same = [r for r in available if r.same_intent_expected]
    different = [r for r in available if not r.same_intent_expected]
    transitions = [r for r in same if r.transition_distance is not None]

    def state_stable(row: SemanticStateContinuityAssessment) -> bool:
        """Apply the recorded endpoint continuity threshold to available state."""
        return row.state_distance is not None and row.state_distance <= row.continuity_threshold

    return {
        "comparison_coverage": _rate(len(available), len(rows), empty=0.0),
        "same_intent_continuity": _rate(sum(state_stable(r) for r in same), len(same)),
        "same_intent_transition_continuity": _rate(
            sum(
                r.transition_distance is not None
                and r.transition_distance <= r.continuity_threshold
                for r in transitions
            ),
            len(transitions),
        ),
        "different_intent_separation": _rate(
            sum(
                r.state_distance is not None and r.state_distance >= r.separation_threshold
                for r in different
            ),
            len(different),
        ),
        "state_decomposition_agreement": _rate(
            sum(state_stable(r) == (r.semantic_decomposition_agreement == 1) for r in available),
            len(available),
        ),
        "state_policy_agreement": _rate(
            sum(state_stable(r) == r.policy_agreement for r in available),
            len(available),
        ),
        "unexplained_state_reset_rate": _rate(
            sum(bool(r.unexplained_state_reset) for r in same),
            len(same),
            empty=0.0,
        ),
        "policy_flip_without_intent_change_rate": _rate(
            sum(not r.policy_agreement for r in same),
            len(same),
            empty=0.0,
        ),
    }

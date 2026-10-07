"""Opt-in discriminating inquiry feeding the existing evidence-bound temporal estimator."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import log2
from typing import Any

from mindful_trace_gepa.logging_schema import EventEnvelope

from .check_records import CheckRequest, CheckResult
from .claim_verification import prioritize_checks
from .diagnostic_records import _text, records, strings
from .epistemic_state import EpistemicMeasurement
from .temporal_estimator import EstimatorResult, ScalarTemporalEstimator


def plan_inquiry(
    checks: tuple[CheckRequest, ...],
    hypothesis_predictions: dict[str, dict[str, str]],
    *,
    budget: float,
    unresolved_claims: tuple[str, ...] = (),
    requested_stop: bool = False,
    enabled: bool = False,
) -> dict[str, Any]:
    """Use normalized outcome entropy under a declared uniform-hypothesis heuristic.

    Distinguishable predictions measure experimental discrimination, not causal understanding
    or posterior confidence. Missing hypothesis coverage is an error, not information gain.
    """
    if type(enabled) is not bool or type(requested_stop) is not bool:
        raise ValueError("enabled and requested_stop must be booleans")
    unresolved = strings(unresolved_claims, "unresolved_claims")
    if not enabled:
        return {
            "selected_check_ids": (),
            "unresolved_claims": unresolved,
            "training_eligibility": "DEVELOPMENT",
            "disabled": True,
        }
    checks = records(checks, CheckRequest)
    if set(hypothesis_predictions) != {c.check_id for c in checks}:
        raise ValueError("each candidate check requires hypothesis predictions")
    hypothesis_ids: set[str] | None = None
    gain = {}
    for check in checks:
        predictions = hypothesis_predictions[check.check_id]
        if not isinstance(predictions, dict) or len(predictions) < 2:
            raise ValueError("at least two competing hypotheses are required")
        if hypothesis_ids is None:
            hypothesis_ids = set(predictions)
        elif hypothesis_ids != set(predictions):
            raise ValueError("every check must cover the same hypotheses")
        for key, value in predictions.items():
            _text(key, "hypothesis_id")
            _text(value, "predicted observation")
        count = len(predictions)
        entropy = -sum(
            (n / count) * log2(n / count) for n in Counter(predictions.values()).values()
        )
        gain[check.check_id] = entropy / log2(count)
    calibrated = tuple(replace(c, expected_information_gain=gain[c.check_id]) for c in checks)
    selected = prioritize_checks(calibrated, budget=budget, enabled=True)
    return {
        "training_eligibility": "DEVELOPMENT",
        "selected_check_ids": tuple(c.check_id for c in selected),
        "information_gain_proxy": gain,
        "unresolved_claims": unresolved,
        "premature_stop": requested_stop and bool(selected) and bool(unresolved),
        "budget_exhausted": bool(unresolved) and not selected,
        "predicted_cost": sum(c.verification_cost for c in selected),
        "mechanistic_understanding": "unassessed",
        "hypothesis_prior": "uniform_heuristic",
    }


def reconcile_inquiry(
    estimator: ScalarTemporalEstimator,
    events: tuple[EventEnvelope, ...],
    request: CheckRequest,
    result: CheckResult,
    *,
    enabled: bool = False,
    **reconciliation: Any,
) -> EstimatorResult | None:
    """Bind a public check to the canonical observation and invoke existing reconciliation.

    No generic confidence increase is inferred from check success. The existing estimator's
    source contract, measurement scale, independent-noise declaration and causal validation apply.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return None
    if type(estimator) is not ScalarTemporalEstimator:
        raise ValueError("canonical temporal estimator required")
    request = CheckRequest.from_dict(request.to_dict())
    result = CheckResult.from_dict(result.to_dict())
    if (request.check_id, request.claim_id, request.action_id) != (
        result.check_id,
        result.claim_id,
        result.action_id,
    ) or result.verdict == "unresolved":
        raise ValueError("inquiry requires a resolved matching check")
    measurement = reconciliation.get("measurement")
    if type(measurement) is not EpistemicMeasurement:
        raise ValueError("canonical measurement required")
    measurement = EpistemicMeasurement.from_dict(measurement.to_dict())
    if not measurement.evidence_refs or not set(measurement.evidence_refs).issubset(
        result.evidence_refs
    ):
        raise ValueError("measurement evidence must belong to the resolved check")
    observation = next(
        (event for event in events if event.event_id == reconciliation.get("observation_event_id")),
        None,
    )
    if observation is None or observation.payload.get("action_id") != request.action_id:
        raise ValueError("check must bind the observed action")
    return estimator.reconcile(events, **reconciliation)

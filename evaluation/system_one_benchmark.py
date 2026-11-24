"""Offline matched routing comparisons, with raw quality separate from guard correction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean, median
from typing import Any

from evaluation.schema import EvalResult
from evaluation.suites.calibration.brier_score import brier_score, expected_calibration_error
from gepa_mindfulness.factuality_observability.epistemic_routing import (
    RoutingBackend,
    RoutingPolicy,
    RoutingRequest,
    evaluate_routing_backend,
    prepare_routing,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction
from gepa_mindfulness.factuality_observability.typed_routing import (
    CalibrationLabel,
    RoutingCalibration,
)


@dataclass(frozen=True, slots=True)
class RoutingCase:
    """Host-authored public test unit with acceptable routes, unrelated to the 17 case IDs."""

    case_id: str
    request: RoutingRequest
    expected_actions: tuple[RecommendedAction, ...]

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id.strip():
            raise ValueError("case_id must be a nonblank string")
        if (
            type(self.expected_actions) is not tuple
            or not self.expected_actions
            or any(type(a) is not RecommendedAction for a in self.expected_actions)
            or len(set(self.expected_actions)) != len(self.expected_actions)
        ):
            raise ValueError("expected_actions must be a nonempty tuple of distinct actions")


def benchmark_routing(
    cases: tuple[RoutingCase, ...],
    backends: tuple[RoutingBackend, ...],
    *,
    policy: RoutingPolicy,
) -> dict[str, Any]:
    """Run each named backend once per frozen input, retaining every output and failure.

    Labels and features are prepared before any callback. Latency is callback wall time,
    includes cold starts, and excludes event validation. This is a local diagnostic, with
    no model download, network access, score aggregation, backend selection or promotion.
    Host callbacks may have side effects and must enforce their own service timeout.
    """
    if type(cases) is not tuple or not 1 <= len(cases) <= 10000:
        raise ValueError("cases must be a tuple of 1..10000 cases")
    if type(backends) is not tuple or not 1 <= len(backends) <= 32:
        raise ValueError("backends must be a tuple of 1..32 backends")
    cases = tuple(replace(case) for case in cases)
    backends = tuple(replace(backend) for backend in backends)
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate case_id")
    if len({(backend.name, backend.version) for backend in backends}) != len(backends):
        raise ValueError("duplicate backend name/version")
    prepared = tuple(
        (case.case_id, case.expected_actions, prepare_routing(case.request, policy))
        for case in cases
    )
    summaries = []
    for backend in backends:
        rows: list[dict[str, Any]] = []
        latencies = []
        for case_id, expected, features in prepared:
            result = evaluate_routing_backend(features, backend)
            latencies.append(result.latency_ms)
            rows.append(
                {
                    "case_id": case_id,
                    "expected_actions": [a.value for a in expected],
                    "raw_correct": result.proposed_action in expected,
                    "guarded_correct": result.action in expected,
                    "disallowed_proposal": (
                        result.proposed_action is not None
                        and result.proposed_action not in features.allowed_actions
                    ),
                    "assessment": result.to_dict(),
                }
            )
        summaries.append(
            {
                "name": backend.name,
                "version": backend.version,
                "cases": len(rows),
                "raw_correct": sum(row["raw_correct"] for row in rows),
                "guarded_correct": sum(row["guarded_correct"] for row in rows),
                "disallowed_proposals": sum(row["disallowed_proposal"] for row in rows),
                "backend_failures": sum(
                    row["assessment"]["backend_error"] is not None for row in rows
                ),
                "mean_latency_ms": mean(latencies),
                "median_latency_ms": median(latencies),
                "max_latency_ms": max(latencies),
                "results": rows,
            }
        )
    return {
        "schema_version": "system-one-benchmark-v1",
        "automatic_promotion": False,
        "backends": summaries,
    }


def benchmark_typed_calibration(
    calibration: RoutingCalibration,
    held_out: tuple[CalibrationLabel, ...],
    *,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Evaluate enumerable predictions on distinct, independently adjudicated opportunities."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not enabled:
        return None
    if type(calibration) is not RoutingCalibration or type(held_out) is not tuple:
        raise ValueError("expected routing calibration and held-out label tuple")
    calibration = replace(calibration)
    ids = set(calibration.sample_ids)
    first = calibration.labels[0]
    threshold = calibration.threshold
    accepted = correct = unknown = 0
    scored = []
    for row in held_out:
        if type(row) is not CalibrationLabel:
            raise ValueError("expected independent calibration label")
        row = replace(row)
        if row.case_id in ids or (row.target, row.model_version) != (
            first.target,
            first.model_version,
        ):
            raise ValueError("held-out cases need unique IDs and matching target and checkpoint")
        ids.add(row.case_id)
        if row.confidence is None:
            unknown += 1
            continue
        ok = row.choice == row.expected
        if threshold is not None and row.confidence >= threshold:
            accepted += 1
            correct += ok
        scored.append(
            EvalResult(
                eval_id=row.case_id,
                suite="mdac-typed-routing",
                category=row.target,
                prompt="",
                model_answer=row.choice.value if row.choice else "",
                gold_answer=row.expected.value,
                outcome="correct" if ok else "incorrect",
                confidence=row.confidence,
            )
        )
    return {
        "schema_version": "mdac-held-out-routing-v1",
        "target": first.target,
        "model_version": first.model_version,
        "threshold": threshold,
        "eligible": len(held_out),
        "accepted": accepted,
        "correct_among_accepted": correct,
        "unknown": unknown,
        "calibration_opportunities": len(scored),
        "coverage": accepted / len(held_out) if held_out else None,
        "selective_accuracy": correct / accepted if accepted else None,
        "brier_score": brier_score(scored),
        "expected_calibration_error": expected_calibration_error(scored),
    }

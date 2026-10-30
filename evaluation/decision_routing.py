"""Matched structured-routing diagnostics, using existing host labels and route guards."""

# Standard library
from __future__ import annotations

from dataclasses import asdict, replace
from statistics import mean

# Third-party
# Local
from evaluation.causal_records import content_digest
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.factuality_observability.decision_routing import (
    DecisionBackend,
    PublicDecisionContext,
    _probability,
    evaluate_decision_backend,
    prepare_decision_packet,
)
from gepa_mindfulness.factuality_observability.epistemic_routing import (
    RoutingPolicy,
    prepare_routing,
)
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

from .schema import EvalResult
from .suites.calibration.brier_score import brier_score, expected_calibration_error
from .system_one_benchmark import RoutingCase

_REVIEW = frozenset((Action.ROUTE_EXTERNAL, Action.DECOMPOSE_AND_VERIFY, Action.ESCALATE))


def _calibration(samples: list[EvalResult], bins: int) -> dict:
    return dict(
        calibration_samples=len(samples),
        top_label_brier=brier_score(samples),
        expected_calibration_error=expected_calibration_error(samples, bins=bins),
    )


def benchmark_decisions(
    cases: tuple[RoutingCase, ...],
    public_contexts: tuple[PublicDecisionContext, ...],
    backends: tuple[DecisionBackend, ...],
    *,
    label_evaluator: TrustedEvaluatorContract,
    confidence_threshold: float,
    calibration_bins: int,
    policy: RoutingPolicy = RoutingPolicy(),
) -> dict:
    """Freeze all state/labels first; model probabilities and costs remain captured diagnostics.

    This operation runs externally configured callbacks only when the existing routing policy
    is explicitly enabled. The host owns service timeouts, source authentication and labels.
    No expected label is given to a backend, and no candidate is promoted by this report.
    """
    if type(policy) is not RoutingPolicy:
        raise ValueError("expected existing routing policy")
    policy = replace(policy)
    if not policy.enabled:
        raise ValueError("decision benchmark requires enabled routing policy")
    if type(label_evaluator) is not TrustedEvaluatorContract:
        raise ValueError("existing label evaluator identity/version contract required")
    label_evaluator = replace(label_evaluator)
    if any(len(value) > 256 for value in asdict(label_evaluator).values()):
        raise ValueError("label evaluator identity exceeds limit")
    _probability(confidence_threshold)
    if type(calibration_bins) is not int or not 1 <= calibration_bins <= 100:
        raise ValueError("calibration_bins requires an integer in 1..100")
    if type(cases) is not tuple or not 1 <= len(cases) <= 128:
        raise ValueError("cases require a bounded tuple of 1..128 records")
    if type(public_contexts) is not tuple or len(public_contexts) != len(cases):
        raise ValueError("one public context per matched case required")
    if type(backends) is not tuple or not 1 <= len(backends) <= 8:
        raise ValueError("backends require a bounded tuple of 1..8 records")
    if any(type(c) is not RoutingCase for c in cases):
        raise ValueError("expected existing exact RoutingCase records")
    if any(type(b) is not DecisionBackend for b in backends):
        raise ValueError("expected exact decision backends")
    cases = tuple(replace(c) for c in cases)
    backends = tuple(replace(b) for b in backends)
    if len({c.case_id for c in cases}) != len(cases) or len(
        {(b.name, b.version) for b in backends}
    ) != len(backends):
        raise ValueError("duplicate case or backend identity")
    prepared = []
    for case, public in zip(cases, public_contexts):
        features = prepare_routing(case.request, policy)
        packet = prepare_decision_packet(features, public)
        prepared.append((case.case_id, tuple(case.expected_actions), features, packet))
    # Every row retains the complete packet plus bounded feature/readout/metric metadata.
    # Reserve 8 KiB for the latter (identities <=128 chars, 8 probabilities, exact safe counts,
    # finite latency, fixed error labels). This over-reserves; no unbounded callback text is kept.
    reservation = dict(
        label_evaluator=asdict(label_evaluator),
        packets=[packet.to_dict() for _ in backends for _, _, _, packet in prepared],
        routing_features=[f.to_dict() for _ in backends for _, _, f, _ in prepared],
        labels=[
            dict(case_id=key, expected_actions=[a.value for a in expected])
            for _ in backends
            for key, expected, _, _ in prepared
        ],
        metadata_capacity="x" * (8192 * len(cases) * len(backends)),
    )
    validate_input_envelope(reservation)
    results = []
    for backend in backends:
        rows, samples, latencies = [], [], []
        counts = dict(
            raw_correct=0,
            guarded_correct=0,
            raw_false_passes=0,
            guarded_false_passes=0,
            raw_false_escalations=0,
            guarded_false_escalations=0,
            review_required_count=0,
            raw_review_hits=0,
            guarded_review_hits=0,
            backend_failures=0,
            input_tokens=0,
            output_tokens=0,
        )
        for case_id, expected, features, packet in prepared:
            assessment = evaluate_decision_backend(
                features, packet.context, backend, confidence_threshold=confidence_threshold
            )
            raw, guarded = assessment.raw_action, assessment.routing.action
            raw_correct, guarded_correct = raw in expected, guarded in expected
            required_review = all(action in _REVIEW for action in expected)
            review_not_required = not any(action in _REVIEW for action in expected)
            counts["raw_correct"] += raw_correct
            counts["guarded_correct"] += guarded_correct
            counts["review_required_count"] += required_review
            counts["raw_review_hits"] += required_review and raw in _REVIEW
            counts["guarded_review_hits"] += required_review and guarded in _REVIEW
            for name, action in (("raw", raw), ("guarded", guarded)):
                counts[name + "_false_passes"] += action is Action.ACCEPT and action not in expected
                counts[name + "_false_escalations"] += review_not_required and action in _REVIEW
            counts["backend_failures"] += assessment.error is not None
            if assessment.confidence is not None:
                samples.append(
                    EvalResult(
                        eval_id=case_id,
                        suite="bounded-routing-diagnostic",
                        category=backend.kind,
                        prompt=packet.context.task,
                        model_answer=raw.value,
                        gold_answer=None,
                        outcome="correct" if raw_correct else "incorrect",
                        confidence=assessment.confidence,
                        metadata={"training_eligibility": "DEVELOPMENT"},
                    )
                )
                counts["input_tokens"] += assessment.readout.input_tokens
                counts["output_tokens"] += assessment.readout.output_tokens
            latencies.append(assessment.routing.latency_ms)
            rows.append(
                dict(
                    case_id=case_id,
                    expected_actions=[a.value for a in expected],
                    raw_correct=raw_correct,
                    guarded_correct=guarded_correct,
                    assessment=assessment.to_dict(),
                )
            )
        denominator = counts["review_required_count"]
        results.append(
            dict(
                name=backend.name,
                version=backend.version,
                model_version=backend.model_version,
                kind=backend.kind,
                cases=len(rows),
                token_counts_complete=counts["backend_failures"] == 0,
                **counts,
                **_calibration(samples, calibration_bins),
                raw_review_recall=counts["raw_review_hits"] / denominator if denominator else None,
                guarded_review_recall=(
                    counts["guarded_review_hits"] / denominator if denominator else None
                ),
                mean_latency_ms=mean(latencies),
                max_latency_ms=max(latencies),
                results=rows,
            )
        )
    report = dict(
        schema_version="bounded-decision-benchmark-v1",
        training_eligibility="DEVELOPMENT",
        optimizer_input=False,
        confers_authority=False,
        automatic_promotion=False,
        confidence_threshold=confidence_threshold,
        calibration_bins=calibration_bins,
        label_evaluator=asdict(label_evaluator),
        policy=dict(
            enabled=policy.enabled,
            uncertainty_threshold=policy.uncertainty_threshold,
            high_risk_threshold=policy.high_risk_threshold,
            max_state_age_seconds=policy.max_state_age_seconds,
        ),
        backends=results,
        cost_counts_independently_verified=False,
        calibration_qualified=False,
    )
    validate_input_envelope(report)
    report["result_digest"] = content_digest(report)
    return report

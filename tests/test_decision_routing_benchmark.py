"""Matched decision diagnostics must expose raw errors and preserve hidden labels."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.decision_routing import benchmark_decisions
from evaluation.system_one_benchmark import RoutingCase
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.factuality_observability.schemas import RecommendedAction as Action
from tests.test_decision_routing import backend, context, readout
from tests.test_epistemic_routing import ENABLED, request

LABEL_EVALUATOR = TrustedEvaluatorContract("host-labels", "fixture-v1", "routing-fixtures")


def inputs():
    low = request()
    high = replace(low, context=replace(low.context, domain_risk=0.95))
    return (
        RoutingCase("low", low, (Action.ACCEPT,)),
        RoutingCase("high", high, (Action.ROUTE_EXTERNAL,)),
    ), (context(low), context(high))


def run(cases=None, public=None, candidates=None):
    baseline, contexts = inputs()
    return benchmark_decisions(
        cases or baseline,
        public or contexts,
        candidates or (backend(),),
        label_evaluator=LABEL_EVALUATOR,
        policy=ENABLED,
        confidence_threshold=0.8,
        calibration_bins=5,
    )


def test_raw_errors_are_not_hidden_by_guard_correction():
    result = run()["backends"][0]
    assert result["raw_correct"] == 1 and result["guarded_correct"] == 2
    assert result["raw_false_passes"] == 1 and result["guarded_false_passes"] == 0
    assert result["raw_review_recall"] == 0.0 and result["guarded_review_recall"] == 1.0
    assert result["review_required_count"] == 1
    assert result["top_label_brier"] == pytest.approx(0.41)
    assert result["expected_calibration_error"] == pytest.approx(0.4)
    assert result["input_tokens"] == 200 and result["output_tokens"] == 0
    assert run()["optimizer_input"] is False


def test_label_evaluator_version_is_retained_separately_from_candidate_model():
    assert run()["label_evaluator"] == {
        "evaluator_id": "host-labels",
        "evaluator_version": "fixture-v1",
        "contract_id": "routing-fixtures",
    }


def test_unnecessary_escalation_and_empty_recall_denominator_are_explicit():
    cases, public = inputs()
    result = run(
        cases=cases[:1],
        public=public[:1],
        candidates=(backend(lambda p: readout(p, Action.ESCALATE)),),
    )["backends"][0]
    assert result["raw_false_escalations"] == 1
    assert result["review_required_count"] == 0 and result["raw_review_recall"] is None


def test_callbacks_do_not_receive_expected_labels():
    seen = []

    def predict(packet):
        seen.append(packet.to_dict())
        return readout(packet)

    run(candidates=(backend(predict),))
    assert len(seen) == 2
    assert all("expected_actions" not in str(p) for p in seen)


def test_callback_cannot_rewrite_recorded_host_policy():
    cases, public = inputs()
    policy = replace(ENABLED)

    def predict(packet):
        object.__setattr__(policy, "uncertainty_threshold", 0.25)
        return readout(packet)

    report = benchmark_decisions(
        cases,
        public,
        (backend(predict),),
        label_evaluator=LABEL_EVALUATOR,
        policy=policy,
        confidence_threshold=0.8,
        calibration_bins=5,
    )
    assert report["policy"]["uncertainty_threshold"] == 0.6


def test_all_inputs_validated_before_any_callback():
    cases, public = inputs()
    public = public[:1] + (replace(public[1], state_digest="0" * 64),)
    with pytest.raises(ValueError):
        run(public=public, candidates=(backend(lambda p: pytest.fail("callback ran early")),))


def test_bad_backend_has_no_valid_calibration_samples():
    result = run(candidates=(backend(lambda p: True),))["backends"][0]
    assert result["backend_failures"] == 2
    assert result["calibration_samples"] == 0
    assert result["top_label_brier"] is None and result["expected_calibration_error"] is None
    assert result["token_counts_complete"] is False


def test_report_capacity_is_reserved_before_prediction():
    cases, public = inputs()
    cases = tuple(replace(cases[0], case_id=f"case-{i}") for i in range(80))
    contexts = tuple(replace(public[0], artifact="x" * 65000) for _ in cases)
    with pytest.raises(ValueError, match="MiB|capacity"):
        run(cases, contexts, (backend(lambda p: pytest.fail("oversized report reached backend")),))


def test_report_capacity_includes_case_identifiers_and_all_backend_rows():
    cases, public = inputs()
    large = replace(cases[0], case_id="x" * 2_200_000)
    first = backend(lambda p: pytest.fail("oversized identifier report reached backend"))
    second = replace(first, name="other-backend")
    with pytest.raises(ValueError, match="MiB|capacity"):
        run((large,), public[:1], (first, second))

"""Analytical and adversarial tests for explicit scalar source fusion."""

import json
from dataclasses import replace

import pytest

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.epistemic_state import (
    Availability,
    CorrelationTreatment,
    EpistemicContext,
    EpistemicMeasurement,
)
from gepa_mindfulness.verification.scalar_fusion import fuse_scalar_measurements
from mindful_trace_gepa.confidence import ConfidenceSource
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion

CONTEXT = EpistemicContext("run", 0, EvaluatedSystemVersion("model", "harness"))
KNOWN = CorrelationTreatment.KNOWN_COVARIANCE
CI = CorrelationTreatment.COVARIANCE_INTERSECTION


def measurement(index: int, value: float = 2, variance: float = 1, **changes):
    return EpistemicMeasurement(
        **(
            dict(
                measurement_id=f"m{index}",
                context=CONTEXT,
                source=ConfidenceSource.TOOL_RESULT,
                representation_id="sensor-v1",
                target_dimension="value",
                value=value,
                variance=variance,
                uncertainty=None,
                evidence_refs=(EvidenceReference(f"e{index}", EvidenceSourceKind.EXTERNAL_RECORD),),
                provenance=(f"sensor-{index}",),
            )
            | changes
        )
    )


def fuse(items, **kwargs):
    return fuse_scalar_measurements(items, estimate_id="fused", **kwargs)


def known(items, matrix, **kwargs):
    return fuse(
        items, mode=KNOWN, covariance=matrix, covariance_provenance=("calibration-v1",), **kwargs
    )


def test_independent_correlated_and_unknown_confirmations_have_different_certainty():
    items = tuple(measurement(i) for i in range(5))
    independent = known(items, tuple(tuple(float(i == j) for j in range(5)) for i in range(5)))
    correlated = known(items, ((1,) * 5,) * 5)
    unknown = fuse(items)
    assert independent.estimate.state.variances == pytest.approx((0.2,))
    assert correlated.estimate.state.variances == (1,)
    assert unknown.estimate.state.variances == (1,)
    assert all(r.estimate.state.values == (2,) for r in (independent, correlated, unknown))
    assert unknown.correlation_treatment is CI
    assert independent.estimate.world_uncertainty < unknown.estimate.world_uncertainty


def test_ci_information_weights_and_known_convex_weights_are_explicit():
    items = (measurement(0, 0, 1), measurement(1, 10, 4))
    result = fuse(items)
    assert result.estimate.state.values == pytest.approx((2,))
    assert result.estimate.state.variances == pytest.approx((1.6,))
    assert result.weights == (0.5, 0.5)
    assert result.mean_weights == pytest.approx((0.8, 0.2))
    result = known(items, ((1, 1), (1, 4)), weights=(3, 1))
    assert result.estimate.state.values == (2.5,)
    assert result.estimate.state.variances == (1.1875,)


def test_conservative_bound_and_unresolved_modes_do_not_invent_independence():
    items = (measurement(0, 0, 1), measurement(1, 10, 4))
    bounded = fuse(items, mode=CorrelationTreatment.CONSERVATIVE_BOUND)
    assert bounded.estimate.state.values == (5,)
    assert bounded.estimate.state.variances == (4,)
    unresolved = fuse(
        (replace(items[0], variance=None), items[1]),
        mode=CorrelationTreatment.UNRESOLVED_CORRELATION,
    )
    assert unresolved.estimate.status is Availability.UNAVAILABLE
    assert unresolved.estimate.state is None
    assert unresolved.estimate.world_uncertainty is None


@pytest.mark.parametrize(
    "matrix",
    [
        ((1, 2), (2, 1)),
        ((1, 0.2), (0.3, 1)),
        ((2, 0), (0, 1)),
        ((1,), (0, 1)),
        ((1, True), (True, 1)),
        ((1, float("nan")), (0, 1)),
        ((1, "0"), (0, 1)),
        ((1, 1, 0), (1, 1, 0.1), (0, 0.1, 1)),
    ],
)
def test_invalid_covariance_is_rejected(matrix):
    with pytest.raises(ValueError):
        known(tuple(measurement(i) for i in range(len(matrix))), matrix)


def test_singular_negative_correlation_can_cancel_error_under_declared_model():
    result = known((measurement(0), measurement(1)), ((1, -1), (-1, 1)))
    assert result.estimate.state.variances == (0,)


@pytest.mark.parametrize(
    "changes",
    [
        {"context": replace(CONTEXT, run_id="other")},
        {"target_dimension": "other"},
        {"representation_id": "other"},
        {"measurement_id": "m0"},
    ],
)
def test_incompatible_measurements_fail(changes):
    with pytest.raises(ValueError):
        fuse((measurement(0), measurement(1, **changes)))


@pytest.mark.parametrize(
    "weights", [(0, 0), (-1, 1), (True, 1), ("1", 1), (float("inf"), 1), (1,), (1e-300, 1e300)]
)
def test_invalid_or_unreportable_weights_fail(weights):
    with pytest.raises(ValueError):
        fuse((measurement(0), measurement(1)), weights=weights)


@pytest.mark.parametrize("variance", [None, 0])
def test_numeric_modes_require_positive_available_variance(variance):
    with pytest.raises(ValueError):
        fuse((measurement(0, variance=variance),))


def test_covariance_requires_explicit_mode_and_provenance():
    items = (measurement(0), measurement(1))
    with pytest.raises(ValueError):
        fuse(items, covariance=((1, 0), (0, 1)))
    with pytest.raises(ValueError):
        fuse(items, mode=KNOWN, covariance=((1, 0), (0, 1)))
    with pytest.raises(ValueError):
        fuse(items, mode=KNOWN, covariance_provenance=("calibration",))


@pytest.mark.parametrize("shared", ["evidence", "group", "peers"])
def test_known_zero_covariance_cannot_erase_reported_dependence(shared):
    items = (measurement(0), measurement(1))
    if shared == "evidence":
        items = (items[0], replace(items[1], evidence_refs=items[0].evidence_refs))
    if shared == "group":
        items = tuple(replace(m, correlation_group="panel") for m in items)
    with pytest.raises(ValueError, match="dependence"):
        known(items, ((1, 0), (0, 1)), peer_exposed=shared == "peers")
    result = fuse(items, peer_exposed=shared == "peers")
    assert result.estimate.state.variances == (1,)
    assert known(items, ((1, 0.5), (0.5, 1)), peer_exposed=shared == "peers")


def test_extreme_finite_values_do_not_overflow_information_or_mean():
    result = fuse((measurement(0, 1e308, 1e-300), measurement(1, -1e308, 1e-300)))
    assert result.estimate.state.values == (0,)
    assert result.estimate.state.variances == (1e-300,)
    tiny = float.fromhex("0x0.0000000000001p-1022")
    with pytest.raises(ValueError, match="underflow"):
        known(
            (measurement(0, variance=tiny), measurement(1, variance=tiny)), ((tiny, 0), (0, tiny))
        )


def test_permutation_and_json_preserve_sources_evidence_and_provenance():
    items = (
        measurement(0, 0, 1),
        measurement(1, 10, 4, source=ConfidenceSource.HUMAN_ADJUDICATION),
    )
    left = fuse(items, weights=(3, 1))
    right = fuse(tuple(reversed(items)), weights=(1, 3))
    assert left.estimate.state == right.estimate.state
    payload = json.loads(json.dumps(left.to_dict(), allow_nan=False))
    assert payload["measurements"][1]["source"] == "HUMAN_ADJUDICATION"
    assert set(left.estimate.evidence_refs) == set(items[0].evidence_refs + items[1].evidence_refs)
    assert all(p in left.estimate.provenance for m in items for p in m.provenance)
    payload["estimate"]["state"]["values"][0] = 999
    object.__setattr__(items[0], "value", 999)
    assert left.measurements[0].value == 0
    assert left.estimate.model_uncertainty is None


@pytest.mark.parametrize("count", [0, 17])
def test_batch_size_is_bounded(count):
    with pytest.raises(ValueError):
        fuse(tuple(measurement(i) for i in range(count)))


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "known_covariance"},
        {"peer_exposed": 1},
        {"uncertainty_scale": True},
        {"uncertainty_scale": 0},
        {"uncertainty_scale": float("nan")},
    ],
)
def test_invalid_policy_fields_are_rejected(options):
    with pytest.raises(ValueError):
        fuse((measurement(0),), **options)


def test_missing_values_cannot_be_fused_as_zero_even_in_unresolved_mode():
    unavailable = measurement(0, value=None, variance=None, status=Availability.UNAVAILABLE)
    with pytest.raises(ValueError):
        fuse((unavailable,), mode=CorrelationTreatment.UNRESOLVED_CORRELATION)


def test_fractional_known_covariance_is_rounded_up_to_preserve_bound():
    from fractions import Fraction

    items = tuple(measurement(i) for i in range(3))
    result = known(items, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert Fraction(result.estimate.state.variances[0]) >= Fraction(1, 3)
    assert result.estimate.state.variances[0] == pytest.approx(1 / 3)

"""Hand-calculated dependency-aware effects and deterministic uncertainty."""

# Standard library
from dataclasses import replace

# Third-party
import pytest
from test_improvement_records import DIGEST

# Local
from evaluation.improvement_records import MetricSpec, ResamplingPolicy
from evaluation.improvement_statistics import (
    PairedObservation,
    estimate_overstatement,
    estimate_paired,
)

METRIC = MetricSpec("accuracy", "evaluation-ladder-v1", "representation_accuracy", DIGEST)
POLICY = ResamplingPolicy(seed=7)


def observation(name, delta, cluster=None, **kwargs):
    return PairedObservation(
        name,
        kwargs.pop("case_id", name),
        "condition",
        0,
        cluster or name,
        DIGEST,
        METRIC,
        0.0,
        delta,
        **kwargs,
    )


def estimate(rows, **kwargs):
    return estimate_paired(
        tuple(rows),
        expected_pair_ids=tuple(r.pair_id for r in rows),
        policy=kwargs.pop("policy", POLICY),
        dependencies_known=kwargs.pop("dependencies_known", True),
        **kwargs,
    )


def test_cluster_weights_do_not_count_cases_or_repeats_as_independent():
    rows = (
        observation("a", 1, "cluster-a"),
        observation("b", 0, "cluster-a"),
        observation("c", 0, "cluster-b"),
    )
    result = estimate(rows)
    assert result.raw_delta == 0.25
    assert result.cluster_count == 2
    assert result.interval is None
    duplicated = rows + (replace(rows[0], pair_id="repeat", repeat_id=1),)
    assert estimate(duplicated).raw_delta == 0.25


def test_twenty_cluster_boundary_reproducibility_and_fixed_system_scope():
    rows = tuple(observation(str(i), i % 2) for i in range(20))
    assert estimate(rows[:-1]).interval is None
    result = estimate(rows)
    assert result.interval is not None
    assert result == estimate(tuple(reversed(rows)))
    assert result.raw_delta == 0.5
    assert estimate(rows, dependencies_known=False).interval is None
    with pytest.raises(ValueError):
        estimate(rows + (rows[0],))
    with pytest.raises(ValueError):
        estimate(rows[:-1] + (replace(rows[-1], stratum_digest="b" * 64),))


def test_incomplete_roster_has_no_interval_or_silent_zero_fill():
    rows = tuple(observation(str(i), 1) for i in range(20))
    result = estimate_paired(
        rows,
        expected_pair_ids=tuple(r.pair_id for r in rows) + ("missing",),
        policy=POLICY,
        dependencies_known=True,
    )
    assert result.raw_delta == 1
    assert result.interval is None
    assert "incomplete_pairs" in result.reasons


@pytest.mark.parametrize("direction,sign", [("higher", 1), ("lower", -1)])
def test_overstatement_is_improvement_not_raw_failure_rate(direction, sign):
    metric = replace(METRIC, direction=direction)
    left = estimate((replace(observation("s", sign * 0.20), metric=metric),))
    right = estimate((replace(observation("a", sign * 0.05), metric=metric),))
    report = estimate_overstatement(left, right, policy=POLICY)
    assert report["overstatement"] == pytest.approx(0.15)
    assert report["interval"] is None


def test_overstatement_requires_disjoint_clusters_and_matching_contracts():
    left = estimate((observation("shared", 0.2),))
    with pytest.raises(ValueError):
        estimate_overstatement(left, left, policy=POLICY)
    right = estimate((replace(observation("audit", 0.05), metric=replace(METRIC, unit="other")),))
    with pytest.raises(ValueError):
        estimate_overstatement(left, right, policy=POLICY)


def test_independent_partition_resampling_has_known_constant_difference():
    selection = estimate(tuple(observation(f"s{i}", 0.2) for i in range(20)))
    audit = estimate(tuple(observation(f"a{i}", 0.05) for i in range(20)))
    result = estimate_overstatement(selection, audit, policy=POLICY)
    assert result["interval"] == pytest.approx((0.15, 0.15))


def test_unknown_estimate_is_not_zero():
    missing = estimate(())
    assert missing.raw_delta is None
    result = estimate_overstatement(estimate((observation("s", 0.1),)), missing, policy=POLICY)
    assert result["overstatement"] is None

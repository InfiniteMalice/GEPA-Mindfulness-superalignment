"""Conditional paired effects; repeats never add independent sampling units."""

# Standard library
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any

# Third-party
# Local
from .improvement_records import ImprovementRecord, MetricSpec, ResamplingPolicy


@dataclass(frozen=True)
class PairedObservation(ImprovementRecord):
    pair_id: str
    case_id: str
    condition_id: str
    repeat_id: int
    cluster_id: str
    stratum_digest: str
    metric: MetricSpec
    baseline_value: float
    candidate_value: float

    def _validate(self) -> None:
        if self.repeat_id < 0:
            raise ValueError("negative repeat")


@dataclass(frozen=True)
class PairedEstimate(ImprovementRecord):
    stratum_digest: str | None
    metric: MetricSpec | None
    policy: ResamplingPolicy
    raw_delta: float | None
    improvement: float | None
    cluster_ids: tuple[str, ...]
    cluster_means: tuple[float, ...]
    expected_count: int
    matched_count: int
    interval: tuple[float, ...] | None
    reasons: tuple[str, ...]

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_ids)


def _interval(replicates: list[float], policy: ResamplingPolicy) -> tuple[float, float]:
    ordered = sorted(replicates)

    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    tail = (1 - policy.confidence_level) / 2
    return quantile(tail), quantile(1 - tail)


def _draw(rng: random.Random, values: tuple[float, ...]) -> float:
    return fmean(rng.choice(values) for _ in values)


def estimate_paired(
    observations: tuple[PairedObservation, ...],
    *,
    expected_pair_ids: tuple[str, ...],
    policy: ResamplingPolicy,
    dependencies_known: bool,
) -> PairedEstimate:
    if type(observations) is not tuple or type(expected_pair_ids) is not tuple:
        raise ValueError("observations and roster require tuples")
    if type(dependencies_known) is not bool:
        raise ValueError("dependencies_known requires bool")
    if type(policy) is not ResamplingPolicy:
        raise ValueError("expected resampling policy")
    policy.__post_init__()
    if any(type(p) is not str or not p.strip() for p in expected_pair_ids):
        raise ValueError("invalid expected pair ID")
    if len(set(expected_pair_ids)) != len(expected_pair_ids):
        raise ValueError("duplicate expected pair IDs")
    rows = tuple(PairedObservation.from_dict(r.to_dict()) for r in observations)
    ids = {r.pair_id for r in rows}
    if len(ids) != len(rows) or ids - set(expected_pair_ids):
        raise ValueError("duplicate or unexpected pair ID")
    metric = rows[0].metric if rows else None
    stratum = rows[0].stratum_digest if rows else None
    if any(r.metric != metric or r.stratum_digest != stratum for r in rows):
        raise ValueError("mixed metric or configuration strata")
    groups = defaultdict(list)
    case_clusters, coordinates = {}, set()
    for row in sorted(rows, key=lambda r: r.pair_id):
        coordinate = (row.case_id, row.condition_id, row.repeat_id)
        if coordinate in coordinates:
            raise ValueError("duplicate paired repeat coordinates")
        coordinates.add(coordinate)
        if row.case_id in case_clusters and case_clusters[row.case_id] != row.cluster_id:
            raise ValueError("same case split across clusters")
        case_clusters[row.case_id] = row.cluster_id
        groups[(row.cluster_id, row.case_id, row.condition_id)].append(
            row.candidate_value - row.baseline_value
        )
    cases = defaultdict(list)
    for (cluster, case, _), deltas in sorted(groups.items()):
        cases[(cluster, case)].append(fmean(deltas))
    clusters = defaultdict(list)
    for (cluster, _), conditions in sorted(cases.items()):
        clusters[cluster].append(fmean(conditions))
    cluster_ids = tuple(sorted(clusters))
    means = tuple(fmean(clusters[c]) for c in cluster_ids)
    delta = fmean(means) if means else None
    sign = -1 if metric and metric.direction == "lower" else 1
    reasons = []
    if len(means) < policy.minimum_clusters:
        reasons.append("insufficient_clusters")
    if ids != set(expected_pair_ids):
        reasons.append("incomplete_pairs")
    if not dependencies_known:
        reasons.append("unknown_dependencies")
    interval = None
    if not reasons:
        rng = random.Random(policy.seed)
        interval = _interval([_draw(rng, means) for _ in range(policy.resamples)], policy)
    return PairedEstimate(
        stratum,
        metric,
        policy,
        delta,
        delta * sign if delta is not None else None,
        cluster_ids,
        means,
        len(expected_pair_ids),
        len(rows),
        interval,
        tuple(reasons),
    )


def estimate_overstatement(
    selection: PairedEstimate,
    independent: PairedEstimate,
    *,
    policy: ResamplingPolicy,
) -> dict[str, Any]:
    selection.__post_init__()
    independent.__post_init__()
    policy.__post_init__()
    if selection.policy != policy or independent.policy != policy:
        raise ValueError("resampling policy changed")
    if selection.improvement is None or independent.improvement is None:
        return dict(overstatement=None, interval=None, reason="missing_estimate")
    if (
        selection.metric != independent.metric
        or selection.stratum_digest != independent.stratum_digest
    ):
        raise ValueError("overstatement requires matching metric and configuration")
    if set(selection.cluster_ids) & set(independent.cluster_ids):
        raise ValueError("overstatement partitions share clusters")
    interval = None
    if selection.interval is not None and independent.interval is not None:
        rng = random.Random(policy.seed)
        sign = -1 if selection.metric.direction == "lower" else 1
        replicates = [
            sign * (_draw(rng, selection.cluster_means) - _draw(rng, independent.cluster_means))
            for _ in range(policy.resamples)
        ]
        interval = _interval(replicates, policy)
    return dict(
        overstatement=selection.improvement - independent.improvement,
        interval=interval,
        reason=None if interval is not None else "source_interval_unavailable",
        method="independent_partition_cluster_percentile",
        selection_reasons=list(selection.reasons),
        independent_reasons=list(independent.reasons),
        policy=policy.to_dict(),
    )

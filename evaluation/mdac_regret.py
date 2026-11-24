"""GUARD-inspired criterion-level retention analysis over existing paired audit records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.improvement_audit import summarize_journal
from evaluation.improvement_records import (
    AttemptJournal,
    ImprovementProtocol,
    ImprovementRecord,
    MetricSpec,
    ResamplingPolicy,
)
from evaluation.improvement_statistics import PairedObservation, estimate_paired


@dataclass(frozen=True)
class CriterionGate(ImprovementRecord):
    metric: MetricSpec
    dimension: str
    pass_threshold: float
    target: float
    severity: float
    hurt_weight: float
    gap_weight: float

    def _validate(self) -> None:
        if self.dimension not in (
            "user_facing_behavior",
            "factuality",
            "calibration",
            "ethical_reasoning",
            "agent_capability",
            "runtime_enforcement",
        ):
            raise ValueError("unknown retention dimension")
        if min(self.severity, self.hurt_weight, self.gap_weight) < 0:
            raise ValueError("priority weights must be nonnegative")


_TRANSITIONS = {
    (False, False): "unresolved_gap",
    (False, True): "improvement",
    (True, True): "retained",
    (True, False): "introduced_regression",
}


def evaluate_regret(
    candidate_id: str,
    observations: tuple[PairedObservation, ...],
    *,
    expected_pair_ids: tuple[str, ...],
    criterion: CriterionGate,
    policy: ResamplingPolicy,
    dependencies_known: bool,
    enabled: bool = False,
) -> dict[str, Any]:
    """Compute per-pair hurt/gap in criterion units and the paired uncertainty estimate."""
    if enabled is not True:
        raise ValueError("regret analysis requires enabled=True")
    if type(candidate_id) is not str or not candidate_id.strip():
        raise ValueError("candidate identity required")
    estimate = estimate_paired(
        observations,
        expected_pair_ids=expected_pair_ids,
        policy=policy,
        dependencies_known=dependencies_known,
    )
    if estimate.metric is not None and estimate.metric != criterion.metric:
        raise ValueError("criterion differs from the paired evidence rubric")
    sign = 1 if criterion.metric.direction == "higher" else -1
    transitions = dict.fromkeys(_TRANSITIONS.values(), 0)
    rows = []
    for row in sorted(observations, key=lambda r: r.pair_id):
        old, new = row.baseline_value, row.candidate_value
        transition = _TRANSITIONS[
            sign * (old - criterion.pass_threshold) >= 0,
            sign * (new - criterion.pass_threshold) >= 0,
        ]
        hurt = max(0.0, sign * (old - new))
        gap = max(0.0, sign * (criterion.target - new))
        priority = criterion.severity * (criterion.hurt_weight * hurt + criterion.gap_weight * gap)
        transitions[transition] += 1
        rows.append(
            dict(
                **row.to_dict(),
                transition=transition,
                score_delta=new - old,
                improvement=sign * (new - old),
                hurt=hurt,
                gap=gap,
                priority=priority,
            )
        )
    return dict(
        schema_version="mdac-regret-v1",
        candidate_id=candidate_id,
        criterion=criterion.to_dict(),
        expected_pair_ids=sorted(expected_pair_ids),
        transitions=transitions,
        rows=rows,
        missing_pair_ids=sorted(set(expected_pair_ids) - {r.pair_id for r in observations}),
        paired_estimate=estimate.to_dict(),
    )


def compare_candidates(reports: tuple[dict, ...], *, enabled: bool = False) -> dict[str, Any]:
    """Compare complete, matched candidate panels for one criterion."""
    if enabled is not True or len(reports) < 2:
        raise ValueError("enable comparison and supply at least two candidate reports")

    def panel(report):
        return (
            report["criterion"],
            report["expected_pair_ids"],
            [
                (
                    row["pair_id"],
                    row["case_id"],
                    row["condition_id"],
                    row["repeat_id"],
                    row["cluster_id"],
                    row["stratum_digest"],
                    row["baseline_value"],
                )
                for row in report["rows"]
            ],
        )

    first = panel(reports[0])
    scores = {}
    for report in reports:
        identity = report["candidate_id"]
        if (
            identity in scores
            or report["schema_version"] != "mdac-regret-v1"
            or report["missing_pair_ids"]
            or panel(report) != first
            or not report["rows"]
        ):
            raise ValueError("comparison requires distinct candidates with a common complete panel")
        scores[identity] = report["paired_estimate"]["improvement"]
    maximum, minimum = max(scores.values()), min(scores.values())
    best = sorted(cid for cid, score in scores.items() if score == maximum)
    worst = sorted(cid for cid, score in scores.items() if score == minimum)
    return dict(
        best=best[0] if len(best) == 1 else None,
        worst=worst[0] if len(worst) == 1 else None,
        best_candidates=best,
        worst_candidates=worst,
        criterion_contrast=maximum - minimum,
        scores=scores,
        transitions={r["candidate_id"]: dict(r["transitions"]) for r in reports},
    )


def rejection_history(
    protocol: ImprovementProtocol, journal: AttemptJournal, *, enabled: bool = False
) -> dict[str, Any]:
    """Identify exact repeated proposal digests in the existing version-bound attempt journal."""
    if enabled is not True:
        raise ValueError("enable rejection history explicitly")
    audit = summarize_journal(protocol, journal)
    rejected = [row for row in audit["candidates"] if row["status"] in ("failed", "rejected")]
    proposals: dict[str, list[str]] = {}
    for row in rejected:
        proposals.setdefault(row["spec"]["proposal_digest"], []).append(row["candidate_id"])
    return {
        "journal_id": journal.journal_id,
        "protocol_digest": journal.protocol_digest,
        "rejected_or_failed": rejected,
        "recurring_proposals": {key: ids for key, ids in proposals.items() if len(ids) > 1},
    }

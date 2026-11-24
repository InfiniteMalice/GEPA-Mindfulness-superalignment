"""Project sealed comparison and lifecycle receipts into evaluator-only partitioned telemetry."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import groupby
from math import fsum
from typing import Any

from gepa_mindfulness.verification.mdac import MECHANISMS

from .causal_records import content_digest
from .mdac_reporting import SplitManifest, _count, _validated_run, intervention_estimates

Row = dict[str, Any]
_PRIVATE = ("final_held_out", "ood")


class _Receipt:
    def __init__(self, report: Mapping[str, Any], manifest: SplitManifest, schema: str):
        source = dict(report)
        self.digest = source.pop("result_digest", None)
        if source.get("schema_version") != schema or self.digest != content_digest(source):
            raise ValueError("native experiment schema or digest mismatch")
        self.study = source["inputs"]["study"]
        if manifest.digest != source["inputs"]["split_digest"] or (
            manifest.dataset_version,
            manifest.evaluator_version,
        ) != (self.study["dataset_version"], self.study["evaluator_version"]):
            raise ValueError("native experiment split or versions mismatch")
        kind = source["evidence_kind"]
        if kind not in ("cpu_fixture", "model_backed") or kind != self.study["evidence_kind"]:
            raise ValueError("source evidence class mismatch")
        self.kind = "illustrative" if kind == "cpu_fixture" else "measured"
        self.manifest = manifest
        self.runs: list[Row] = []

    def emit(
        self,
        *,
        arm: str,
        method: str,
        seed: int,
        stage: str,
        step: int,
        checkpoint: str,
        partition: str,
        families: Sequence[str],
        points: list[Row],
        costs: Row,
        decisions: list[Row] | None = None,
    ) -> None:
        baseline = stage == "before" and method == "shared_baseline"
        if (not baseline and method not in self.study["methods"]) or seed not in self.study[
            "seeds"
        ]:
            raise ValueError("unknown source run seed or method")
        run = dict(
            schema_version="mdac-telemetry-v1",
            kind=self.kind,
            training_eligibility="HIDDEN_EVAL",
            run_id=f"{self.digest}:{arm}:{method}:{seed}:{step}:{stage}:{partition}",
            arm=arm,
            rl_method=method,
            model=self.study["checkpoint"],
            checkpoint=checkpoint,
            round=step,
            config_hash=self.study["model_config_digest"],
            dataset_version=self.manifest.dataset_version,
            evaluator_version=self.manifest.evaluator_version,
            seed=seed,
            split_digest=self.manifest.digest,
            partition=partition,
            family_ids=sorted(set(families)),
            cost=dict(
                training_tokens=None,
                evaluation_tokens=None,
                training_seconds=None,
                evaluation_seconds=None,
            )
            | costs,
            estimates=points,
            decisions=decisions or [],
        )
        self.runs.append(_validated_run(run, self.manifest))

    def point(
        self, name: str, group: str, values: Sequence[float | None], *, x: Any = None, y: Any = None
    ) -> Row:
        known = [v for v in values if v is not None]
        return dict(
            metric=name,
            group=group,
            value=fsum(known) / len(known) if known else None,
            n=len(known),
            missing=len(values) - len(known),
            interval=None,
            interval_method="descriptive complete-case mean; independent trials unasserted",
            source_metric=f"{name}/{group}",
            source_digest=self.digest,
            x=x,
            y=y,
        )

    def allocation(self, counts: Mapping[str, int]) -> list[Row]:
        total = sum(_count(c, "channel count") for c in counts.values())
        if total == 0:
            return []
        return [
            dict(self.point("allocation", channel, [count / total]), n=total)
            for channel, count in counts.items()
        ]


def _diagnostic_curves(src: _Receipt, rows: list[Row], side: str) -> list[Row]:
    # Confidence estimates refer to the evaluator's label-set prediction. A missing
    # reference cannot establish calibration, and the private labels stay in custody.
    samples = [
        (r[side]["diagnostic_confidence"], set(r[side]["mechanisms"]) == set(r["truth"]))
        for r in rows
        if r["truth"] is not None and r[side]["diagnostic_confidence"] is not None
    ]
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for confidence, accurate in samples:
        buckets[min(int(confidence * 10), 9)].append((confidence, accurate))
    missing = len(rows) - len(samples)
    group = "diagnostic label-set correctness"
    result = []
    for pairs in buckets.values():
        avg = fsum(c for c, _ in pairs) / len(pairs)
        result.append(
            dict(src.point("calibration", group, [ok for _, ok in pairs], x=avg), missing=missing)
        )
    answered = errors = 0
    for _, bucket in groupby(sorted(samples, reverse=True), key=lambda pair: pair[0]):
        for confidence, correct in bucket:
            answered += 1
            errors += int(not correct)
        coverage = answered / len(samples)
        point = src.point("selective_risk", group, [errors / answered], x=coverage)
        result.append(dict(point, n=answered, missing=missing))
    return result


def _comparison_points(src: _Receipt, rows: list[Row], side: str) -> list[Row]:
    values = [r[side] for r in rows]
    points = [
        src.point(metric, "all", [v[metric] for v in values])
        for metric in ("alignment", "capability", "diagnostic_confidence")
    ]
    points.extend(_diagnostic_curves(src, rows, side))
    for mechanism in MECHANISMS:
        points.append(
            src.point("mechanism_rate", mechanism, [mechanism in v["mechanisms"] for v in values])
        )
    if side == "after":
        for metric in ("alignment", "capability"):
            before = [r["before"][metric] for r in rows]
            after = [v[metric] for v in values]
            points.append(
                src.point(
                    "regression",
                    metric,
                    [b > a for b, a in zip(before, after)],
                    x="before",
                    y="declined",
                )
            )
        divergence: list[float | None] = []
        for r in rows:
            b, a = r["before"], r["after"]
            if None in (a["reward"], b["reward"], a["verified_behavior"], b["verified_behavior"]):
                divergence.append(None)
            else:
                divergence.append(
                    (a["reward"] - b["reward"]) - (a["verified_behavior"] - b["verified_behavior"])
                )
        points.append(src.point("reward_divergence", "verified behavior", divergence))
    return points


def comparison_runs(report: Mapping[str, Any], manifest: SplitManifest) -> list[Row]:
    """Export A–F cells, keeping training costs separate and baseline evaluations deduplicated."""
    src = _Receipt(report, manifest, "mdac-comparison-v1")
    baseline_digests: dict[tuple[int, str], str] = {}
    for cell in report["cells"]:
        arm, method, seed = (cell[k] for k in ("arm", "method", "seed"))
        families = cell["selected_families"]
        actual = cell["actual"]
        common = dict(method=method, seed=seed)
        src.emit(
            arm=arm,
            stage="training",
            step=1,
            checkpoint=cell["checkpoint_end"],
            partition="training",
            families=families,
            points=src.allocation(Counter(cell["schedule_channels"])),
            costs={k: actual[k] for k in ("training_tokens", "samples", "compute_units")},
            **common,
        )
        cases: dict[str, list[Row]] = defaultdict(list)
        seen_cases: set[str] = set()
        for judgment in cell["judgments"]:
            if judgment["case_id"] in seen_cases:
                raise ValueError("duplicate case in comparison cell")
            seen_cases.add(judgment["case_id"])
            if not {"family", "before_tokens", "after_tokens"} <= judgment.keys():
                raise ValueError("rerun comparison to record per-case family and token lineage")
            cases[judgment["partition"]].append(judgment)
        if set(cases) != set(_PRIVATE):
            raise ValueError("comparison must include distinct final held-out and OOD cases")
        for side, cost_key in (
            ("before", "baseline_evaluation_tokens"),
            ("after", "evaluation_tokens"),
        ):
            if (
                sum(
                    _count(r[side + "_tokens"], "evaluation tokens")
                    for rs in cases.values()
                    for r in rs
                )
                != actual[cost_key]
            ):
                raise ValueError("per-case costs disagree with native comparison receipt")
            for part, rows in cases.items():
                if side == "before":
                    identity = (seed, part)
                    digest = content_digest(
                        sorted(
                            (
                                r["case_id"],
                                r["family"],
                                r[side],
                                r["before_tokens"],
                                None if r["truth"] is None else sorted(r["truth"]),
                            )
                            for r in rows
                        )
                    )
                    if identity in baseline_digests:
                        if baseline_digests[identity] != digest:
                            raise ValueError("shared baseline changed across comparison arms")
                        continue
                    baseline_digests[identity] = digest
                src.emit(
                    arm="shared_baseline" if side == "before" else arm,
                    method="shared_baseline" if side == "before" else method,
                    seed=seed,
                    stage=side,
                    step=0 if side == "before" else 1,
                    checkpoint=cell["checkpoint_start" if side == "before" else "checkpoint_end"],
                    partition=part,
                    families=[r["family"] for r in rows],
                    points=_comparison_points(src, rows, side),
                    costs=dict(evaluation_tokens=sum(r[side + "_tokens"] for r in rows)),
                )
    return src.runs


def lifecycle_runs(report: Mapping[str, Any], manifest: SplitManifest) -> list[Row]:
    """Export native stage profiles and final admission evidence without private answers."""
    src = _Receipt(report, manifest, "mdac-alignment-lifecycle-v1")
    common = dict(
        arm=report["inputs"]["order"], method=src.study["methods"][0], seed=src.study["seeds"][0]
    )

    def profile(
        profile: Row, step: int, part: str, checkpoint: str, decisions: list[Row] | None = None
    ) -> None:
        if (
            profile["checkpoint"],
            profile["partition"],
            profile["evaluator"],
            profile["split_digest"],
        ) != (checkpoint, part, manifest.evaluator_version, manifest.digest):
            raise ValueError("lifecycle profile checkpoint or partition mismatch")
        groups: dict[str, list[bool]] = defaultdict(list)
        rows = profile["rows"]
        for r in rows:
            if type(r["correct"]) is not bool:
                raise ValueError("native profile correctness requires boolean observations")
            groups[r["dimension"]].append(r["correct"])
        points = [
            src.point(
                (
                    dim
                    if dim in ("alignment", "capability", "retention", "ethical_generalization")
                    else "profile_score"
                ),
                dim,
                values,
            )
            for dim, values in groups.items()
        ]
        src.emit(
            stage="profile",
            step=step,
            checkpoint=checkpoint,
            partition=part,
            families=[r["family"] for r in rows],
            points=points,
            decisions=decisions,
            costs=dict(
                evaluation_tokens=sum(_count(r["tokens"], "evaluation tokens") for r in rows)
            ),
            **common,
        )

    checkpoint = src.study["checkpoint"]
    profile(report["initial"], 0, "curriculum_validation", checkpoint)
    rounds = report["rounds"]
    for step, stage in enumerate(rounds, 1):
        if stage["source"] != checkpoint:
            raise ValueError("lifecycle checkpoint continuity mismatch")
        checkpoint = stage["checkpoint"]
        samples = _count(stage["samples"], "stage samples")
        if step == len(rounds):
            counts = dict(report["allocation"]["channel_counts"])
            counts["constitutional_anchors"] = samples - sum(counts.values())
        else:
            anchors = _count(stage["retention_samples"], "retention samples")
            counts = dict(retention=anchors, other_training=samples - anchors)
        src.emit(
            stage=stage["stage"],
            step=step,
            checkpoint=checkpoint,
            partition="training",
            families=stage["training_families"],
            points=src.allocation(counts),
            costs=dict(
                training_tokens=stage["tokens"],
                samples=samples,
                compute_units=stage["compute_units"],
            ),
            **common,
        )
        profile(stage["validation"], step, "curriculum_validation", checkpoint)
    profile(
        report["diagnostic_profile"],
        len(rounds) - 1,
        "diagnostic_development",
        rounds[-1]["source"],
    )
    diagnostics = src.runs[-1]["estimates"]
    for criterion in report["allocation"].get("inputs", {}).get("criteria", []):
        record = criterion["record"]
        name = criterion["mechanism"]
        denominator, numerator = record["opportunities"], record["occurrences"]
        if name is None:
            continue
        if denominator:
            point = src.point("mechanism_rate", name, [numerator / denominator])
            point.update(
                n=denominator, source_digest=content_digest(record), x=record["diagnostic_id"]
            )
            diagnostics.append(point)
        for assessment in record["assessments"]:
            if assessment["mechanism"] == name:
                point = src.point("diagnostic_confidence", name, [assessment["confidence"]])
                point.update(source_digest=content_digest(record), x=record["diagnostic_id"])
                diagnostics.append(point)
    development = report["diagnostic_profile"]
    dev_families = {r["family"] for r in development["rows"]}
    for case_report in report.get("case_interventions", []):
        points = intervention_estimates(case_report)
        roster = [entry["identity"] for entry in case_report["roster"]]
        families = {entry["task_id"] for entry in roster}
        if (
            not families
            or not families <= dev_families
            or any(entry["checkpoint"] != development["checkpoint"] for entry in roster)
        ):
            raise ValueError("case intervention must match the actual development profile")
        src.emit(
            stage=f"case-intervention-{case_report['result_digest']}",
            step=len(rounds) - 1,
            checkpoint=development["checkpoint"],
            partition="diagnostic_development",
            families=sorted(families),
            points=points,
            costs=dict(replay_calls=case_report.get("calls_used")),
            **common,
        )
    if "intervention" in report:
        family = report.get("intervention_family")
        if not family:
            raise ValueError(
                "native lifecycle receipt must bind the intervention calibration family"
            )
        src.emit(
            stage="intervention",
            step=len(rounds) - 1,
            checkpoint=rounds[-1]["source"],
            partition="intervention_calibration",
            families=[family],
            points=intervention_estimates(report["intervention"]),
            costs=dict(replay_calls=report["intervention"].get("calls_used")),
            **common,
        )
    receipt = report["admission"]
    decision = dict(
        outcome=receipt["decision"],
        candidate=receipt["candidate_id"],
        reason="; ".join(receipt["reasons"]),
        evidence_refs=[receipt["report_digest"]],
        rollback_target=receipt.get("rollback_target"),
    )
    profile(
        report["admission_profile"], len(rounds), "governance_admission", checkpoint, [decision]
    )
    if set(report.get("final_profiles", {})) != set(_PRIVATE):
        raise ValueError("rerun lifecycle with final held-out and OOD profiles")
    for partition in _PRIVATE:
        profile(report["final_profiles"][partition], len(rounds), partition, checkpoint)
    return src.runs

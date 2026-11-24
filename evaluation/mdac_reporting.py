"""Evaluator-side MDAC partition, telemetry and offline reporting contracts."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from gepa_mindfulness.verification.epistemic_state import _array, _number, _strings, _text
from gepa_mindfulness.verification.mdac_measurement import MetricResult
from gepa_mindfulness.verification.state import _require_exact_mapping

from .causal_records import canonical_json, content_digest

PARTITIONS = (
    "training",
    "diagnostic_development",
    "intervention_calibration",
    "curriculum_validation",
    "governance_admission",
    "final_held_out",
    "ood",
)
KINDS = ("measured", "illustrative", "unexecuted")
PLOTS = {
    "failure_heatmap": ("mechanism_rate",),
    "mechanism_trends": ("mechanism_rate",),
    "curriculum_allocation": ("allocation",),
    "calibration": ("calibration",),
    "risk_coverage": ("selective_risk",),
    "reward_behavior_divergence": ("reward_divergence",),
    "generalization": ("generalization_gap", "familiar", "novel"),
    "retention_regression": ("retention", "regression"),
    "intervention_recovery": ("intervention_recovery",),
    "cross_rl": ("alignment", "capability"),
    "mechanistic": ("temporal_drift", "attribution_difference"),
}
_METRICS = {metric for names in PLOTS.values() for metric in names} | {"diagnostic_confidence"}
_UNIT = {
    "mechanism_rate",
    "allocation",
    "calibration",
    "selective_risk",
    "familiar",
    "novel",
    "retention",
    "regression",
    "alignment",
    "capability",
    "diagnostic_confidence",
}
_RUN_KEYS = {
    "schema_version",
    "run_id",
    "round",
    "arm",
    "rl_method",
    "checkpoint",
    "model",
    "config_hash",
    "dataset_version",
    "evaluator_version",
    "split_digest",
    "seed",
    "partition",
    "family_ids",
    "kind",
    "cost",
    "estimates",
    "decisions",
}
_COSTS = ("training_tokens", "evaluation_tokens", "training_seconds", "evaluation_seconds")


def _exact(data: object, keys: set[str], name: str) -> dict[str, Any]:
    return dict(_require_exact_mapping(data, keys, f"{name} fields"))


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} requires a nonnegative integer")
    return value


@dataclass(frozen=True)
class SplitManifest:
    dataset_version: str
    evaluator_version: str
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text(self.dataset_version, "dataset_version")
        _text(self.evaluator_version, "evaluator_version")
        pairs = []
        for pair in _array(self.assignments, "assignments"):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("assignments require family and partition")
            family, partition = pair
            if partition not in PARTITIONS:
                raise ValueError("unknown scenario partition")
            pairs.append((_text(family, "family_id"), partition))
        if len({family for family, _ in pairs}) != len(pairs):
            raise ValueError("a scenario family may occur only once across all partitions")
        object.__setattr__(self, "assignments", tuple(sorted(pairs)))

    @classmethod
    def allocate(
        cls,
        family_ids: Sequence[str],
        counts: Mapping[str, int],
        *,
        seed: int,
        dataset_version: str,
        evaluator_version: str,
    ) -> SplitManifest:
        """Assign exactly the requested number of whole source families to each partition."""
        _count(seed, "seed")
        families = _strings(family_ids, "family IDs", required=True)
        sizes = _exact(counts, set(PARTITIONS), "partition counts")
        if sum(_count(n, "counts") for n in sizes.values()) != len(families):
            raise ValueError("partition counts must sum to the number of distinct families")
        ordered = sorted(families, key=lambda family: (content_digest([seed, family]), family))
        assignments: list[tuple[str, str]] = []
        offset = 0
        for partition in PARTITIONS:
            end = offset + sizes[partition]
            assignments.extend((family, partition) for family in ordered[offset:end])
            offset = end
        return cls(dataset_version, evaluator_version, tuple(assignments))

    @classmethod
    def from_dict(cls, data: object) -> SplitManifest:
        row = _exact(
            data,
            {"schema_version", "dataset_version", "evaluator_version", "assignments"},
            "split manifest",
        )
        if row["schema_version"] != "mdac-split-v1":
            raise ValueError("unknown split schema version")
        assignments = [
            _exact(a, {"family_id", "partition"}, "assignment")
            for a in _array(row["assignments"], "assignments")
        ]
        return cls(
            row["dataset_version"],
            row["evaluator_version"],
            tuple((a["family_id"], a["partition"]) for a in assignments),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            schema_version="mdac-split-v1",
            dataset_version=self.dataset_version,
            evaluator_version=self.evaluator_version,
            assignments=[dict(family_id=f, partition=p) for f, p in self.assignments],
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def require(
        self, family_id: str, partition: str, *, dataset_version: str, evaluator_version: str
    ) -> None:
        if (dataset_version, evaluator_version) != (self.dataset_version, self.evaluator_version):
            raise ValueError("dataset/evaluator version mismatch")
        if (family_id, partition) not in self.assignments:
            raise ValueError("scenario family missing or assigned to another partition")

    def training_input(self, record: Mapping[str, Any]) -> dict[str, str]:
        """Project a host-approved training example onto its three public fields."""
        self.require(
            _text(record.get("family_id"), "family_id"),
            "training",
            dataset_version=_text(record.get("dataset_version"), "dataset_version"),
            evaluator_version=_text(record.get("evaluator_version"), "evaluator_version"),
        )
        return {k: _text(record.get(k), k) for k in ("task_id", "family_id", "public_prompt")}


def metric_estimate(
    metric: str, result: MetricResult, *, group: str, x: Any = None, y: Any = None
) -> dict[str, Any]:
    """Copy a raw M01–M16 measurement with its original count and uncertainty basis."""
    estimate = dict(
        metric=metric,
        group=group,
        value=result.raw,
        n=result.denominator,
        missing=result.missing,
        interval=None if result.interval is None else list(result.interval),
        interval_method=result.uncertainty,
        x=x,
        y=y,
    )
    _estimate(estimate)
    return estimate


def intervention_estimates(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read paired DRS effects from a content-bound PR3 result (custody remains external)."""
    payload = dict(receipt)
    digest = payload.pop("result_digest", None)
    if payload.get("schema_version") != "mdac-interventions-v1" or digest != content_digest(
        payload
    ):
        raise ValueError("intervention receipt schema or digest mismatch")
    planned = _count(payload.get("planned_blocks"), "planned blocks")
    method = _text(payload.get("interval_method"), "interval method")
    effects = payload.get("effects")
    if not isinstance(effects, Mapping):
        raise ValueError("intervention effects must be a treatment mapping")
    points = []
    for treatment, effect in effects.items():
        if not isinstance(effect, Mapping):
            raise ValueError("intervention effect requires a mapping")
        n = _count(effect.get("matched_trials"), "matched trials")
        controls = _count(effect.get("control_passes"), "control passes")
        successes = _count(effect.get("intervention_passes"), "intervention passes")
        expected = (successes - controls) / n if n else None
        if n > planned or max(controls, successes) > n or expected != effect.get("drs"):
            raise ValueError("inconsistent matched intervention opportunity counts")
        point = dict(
            metric="intervention_recovery",
            group=_text(treatment, "treatment"),
            value=expected,
            n=n,
            missing=planned - n,
            interval=effect.get("interval"),
            interval_method=method,
            x=None,
            y=None,
        )
        _estimate(point)
        points.append(point)
    return points


def _estimate(data: object) -> None:
    row = _exact(
        data,
        {"metric", "group", "value", "n", "missing", "interval", "interval_method", "x", "y"},
        "estimate",
    )
    name = row["metric"]
    if name not in _METRICS:
        raise ValueError("unknown telemetry metric")
    _text(row["group"], "estimate group")
    _text(row["interval_method"], "interval method or reason unavailable")
    n = _count(row["n"], "sample size")
    _count(row["missing"], "missing opportunities")
    value = row["value"]
    if value is not None:
        value = _number(value, "estimate")
        if not n or (name in _UNIT and not 0 <= value <= 1):
            raise ValueError("estimate requires measured opportunities and valid scale")
        if name in ("generalization_gap", "intervention_recovery") and not -1 <= value <= 1:
            raise ValueError("signed effect outside [-1, 1]")
        if name == "reward_divergence" and not -2 <= value <= 2:
            raise ValueError("reward divergence outside [-2, 2]")
    if row["interval"] is not None:
        interval = _array(row["interval"], "interval")
        if len(interval) != 2 or value is None:
            raise ValueError("interval needs two bounds and a measured value")
        if _number(interval[0], "interval low") > _number(interval[1], "interval high"):
            raise ValueError("reversed interval")
    for axis in ("x", "y"):
        coordinate = row[axis]
        if coordinate is not None and type(coordinate) is not str:
            _number(coordinate, axis)
        if type(coordinate) is str:
            _text(coordinate, axis)
    if name in ("calibration", "selective_risk") and not 0 <= _number(row["x"], "x") <= 1:
        raise ValueError("curve requires confidence or coverage coordinate in [0, 1]")


def _validated_run(data: object, manifest: SplitManifest) -> dict[str, Any]:
    row = _exact(data, _RUN_KEYS, "run")
    if row["schema_version"] != "mdac-telemetry-v1" or row["kind"] not in KINDS:
        raise ValueError("invalid schema or measurement class")
    for k in (
        "run_id",
        "arm",
        "rl_method",
        "checkpoint",
        "model",
        "dataset_version",
        "evaluator_version",
    ):
        _text(row[k], k)
    if row["split_digest"] != manifest.digest or not re.fullmatch(
        "[0-9a-f]{64}", _text(row["config_hash"], "config_hash")
    ):
        raise ValueError("manifest digest or config hash invalid")
    _count(row["round"], "round")
    _count(row["seed"], "seed")
    families = _strings(row["family_ids"], "family_ids", required=True)
    for family in families:
        manifest.require(
            family,
            row["partition"],
            dataset_version=row["dataset_version"],
            evaluator_version=row["evaluator_version"],
        )
    cost = row["cost"]
    if cost is not None:
        for k, v in _exact(cost, set(_COSTS), "cost").items():
            if v is not None:
                if k.endswith("_tokens"):
                    _count(v, k)
                elif _number(v, k) < 0:
                    raise ValueError("cost must be nonnegative")
    estimates, decisions = _array(row["estimates"], "estimates"), _array(
        row["decisions"], "decisions"
    )
    if row["kind"] == "unexecuted" and (cost is not None or estimates or decisions):
        raise ValueError("unexecuted runs cannot claim costs, outcomes or decisions")
    seen = set()
    for estimate in estimates:
        _estimate(estimate)
        cell = tuple(estimate[k] for k in ("metric", "group", "x", "y"))
        if cell in seen:
            raise ValueError("duplicate estimate cell within run")
        seen.add(cell)
    for decision in decisions:
        d = _exact(decision, {"outcome", "reason", "candidate", "evidence_refs"}, "decision")
        if d["outcome"] not in ("PROMOTE", "REJECT", "DEFER", "REPAIR", "ROLLBACK"):
            raise ValueError("unknown governance decision")
        _text(d["candidate"], "candidate")
        _text(d["reason"], "decision reason")
        _strings(d["evidence_refs"], "decision evidence_refs", required=True)
    return json.loads(canonical_json(row))


def _summary(manifest: SplitManifest, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohorts = {}
    for kind in KINDS:
        group = [r for r in rows if r["kind"] == kind]
        costs = {
            k: [r["cost"][k] for r in group if r["cost"] is not None and r["cost"][k] is not None]
            for k in _COSTS
        }
        metrics: dict[str, list[dict[str, Any]]] = {}
        for r in group:
            identity = {
                k: r[k]
                for k in (
                    "run_id",
                    "arm",
                    "rl_method",
                    "checkpoint",
                    "partition",
                    "round",
                    "seed",
                    "family_ids",
                    "dataset_version",
                    "evaluator_version",
                    "config_hash",
                )
            }
            for estimate in r["estimates"]:
                metrics.setdefault(estimate["metric"], []).append({**identity, **estimate})
        cohorts[kind] = dict(
            run_count=len(group),
            runs=[{k: v for k, v in r.items() if k != "estimates"} for r in group],
            metrics=metrics,
            observed_costs={k: sum(v) if v else None for k, v in costs.items()},
            runs_with_cost={k: len(v) for k, v in costs.items()},
        )
    return dict(
        schema_version="mdac-summary-v1",
        split_digest=manifest.digest,
        partitions={
            p: sum(partition == p for _, partition in manifest.assignments) for p in PARTITIONS
        },
        cohorts=cohorts,
        limitations=[
            "Families, labels and provenance are supplied by the experiment host.",
            "Repeated seeds and dependent opportunities are displayed separately.",
            "Costs total the runs with measured cost for each unit.",
        ],
    )


def export_report(
    manifest: SplitManifest,
    runs: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    enabled: bool = False,
    plots: bool = True,
) -> dict[str, str]:
    """Validate all runs and publish a fresh directory with JSONL, summary and optional PNGs."""
    if enabled is not True or type(plots) is not bool:
        raise ValueError("report export requires enabled=True and a boolean plots setting")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    validated = [_validated_run(r, manifest) for r in runs]
    if len({r["run_id"] for r in validated}) != len(validated):
        raise ValueError("duplicate run ID")
    summary = _summary(manifest, validated)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        (stage / "telemetry.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in validated), encoding="utf-8"
        )
        (stage / "split.json").write_text(
            canonical_json(manifest.to_dict()) + "\n", encoding="utf-8"
        )
        if plots:
            from .mdac_report_plots import render_plots

            summary["plots"] = render_plots(summary["cohorts"], stage / "plots")
        else:
            summary["plots"] = []
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(destination)
    return {
        k: str(destination / name)
        for k, name in (
            ("summary", "summary.json"),
            ("telemetry", "telemetry.jsonl"),
            ("split", "split.json"),
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path, help="JSONL telemetry records")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = SplitManifest.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
    runs = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line]
    export_report(manifest, runs, args.output, enabled=args.enabled, plots=not args.no_plots)


if __name__ == "__main__":
    main()

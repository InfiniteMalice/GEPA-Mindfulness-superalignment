"""A-K ablation specifications and family-level summaries; no hidden-eval optimization."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from random import Random
from statistics import mean
from typing import Any

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.check_records import CheckRequest
from gepa_mindfulness.verification.epistemic_state import _nonnegative, _number, _text
from gepa_mindfulness.verification.uncertainty_inquiry import plan_inquiry


def ablation_matrix() -> dict[str, tuple[str, ...]]:
    """Return the requested independent and cumulative configurations, without enabling them."""
    full = (
        "decomposition",
        "resolver",
        "challenger",
        "sensitivity",
        "peo",
        "perspective",
        "curriculum",
        "human_skills",
    )
    return {
        "A": (),
        "B": ("decomposition",),
        "C": ("resolver",),
        "D": ("challenger",),
        "E": ("resolver", "challenger"),
        "F": ("decomposition", "resolver", "challenger", "sensitivity"),
        "G": ("decomposition", "resolver", "challenger", "sensitivity", "peo"),
        "H": ("perspective",),
        "I": ("curriculum",),
        "J": full,
        "K": full + ("sift",),
    }


def family_split(family_id: str, *, seed: int = 0) -> str:
    """Deterministically assign an entire semantic family to dev or held-out evaluation."""
    _text(family_id, "family_id")
    if type(seed) is not int:
        raise ValueError("seed must be integer")
    bucket = int(sha256(f"{seed}:{family_id}".encode()).hexdigest()[:8], 16) % 5
    return "HELD_OUT" if bucket == 0 else "DEVELOPMENT"


def _interval(values: list[float]) -> dict[str, Any]:
    if len(values) < 2:
        return {"mean": mean(values), "family_count": len(values), "bootstrap_95": None}
    rng = Random(0)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return {
        "mean": mean(values),
        "family_count": len(values),
        "bootstrap_95": [samples[24], samples[974]],
    }


def summarize_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize caller-measured trials, preserving metric coverage and all cost dimensions.

    Confidence intervals resample family means, not correlated perspective variants. Hidden
    evaluation records are rejected; protected evaluation remains in its existing private path.
    """
    seen = set()
    splits: dict[str, str] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if set(row) != {"run_id", "ablation", "family_id", "split", "metrics", "cost"}:
            raise ValueError("trial has missing or unknown fields")
        for name in ("run_id", "family_id"):
            _text(row[name], name)
        if row["run_id"] in seen:
            raise ValueError("duplicate run identity")
        seen.add(row["run_id"])
        if type(row["ablation"]) is not str or row["ablation"] not in ablation_matrix():
            raise ValueError("unknown ablation")
        if type(row["split"]) is not str or row["split"] not in {"DEVELOPMENT", "HELD_OUT"}:
            raise ValueError("split cannot expose hidden evaluation")
        family = row["family_id"]
        if family in splits and splits[family] != row["split"]:
            raise ValueError("semantic family leaked across splits")
        splits[family] = row["split"]
        if not isinstance(row["metrics"], dict) or not row["metrics"]:
            raise ValueError("nonempty separate metrics required")
        for name, value in row["metrics"].items():
            _text(name, "metric name")
            if value is not None:
                _number(value, name)
        costs = row["cost"]
        if not isinstance(costs, dict) or set(costs) != {
            "tool_calls",
            "latency_seconds",
            "verification_cost",
        }:
            raise ValueError("all three cost dimensions required")
        if type(costs["tool_calls"]) is not int:
            raise ValueError("tool_calls must be integer")
        for name, value in costs.items():
            _nonnegative(value, name)
        grouped[row["ablation"]].append(row)
    result = {}
    for ablation, trials in grouped.items():
        if len({trial["split"] for trial in trials}) != 1:
            raise ValueError("summarize each split separately")
        coverage = {tuple(sorted(trial["metrics"])) for trial in trials}
        if len(coverage) != 1:
            raise ValueError(
                "trials require matching metric coverage; use null for unmeasured values"
            )
        metrics = {}
        for name in trials[0]["metrics"]:
            families: dict[str, list[float]] = defaultdict(list)
            for trial in trials:
                if trial["metrics"][name] is not None:
                    families[trial["family_id"]].append(trial["metrics"][name])
            metrics[name] = (
                _interval([mean(values) for values in families.values()]) if families else None
            )
        result[ablation] = {
            "training_eligibility": "DEVELOPMENT",
            "split": trials[0]["split"],
            "run_count": len(trials),
            "metrics": metrics,
            "cost_totals": {
                name: sum(trial["cost"][name] for trial in trials)
                for name in ("tool_calls", "latency_seconds", "verification_cost")
            },
        }
    return result


def fixture_smoke() -> dict[str, Any]:
    """Exercise the real selector on a known discriminating experiment, not a model benchmark."""
    ref = EvidenceReference("fixture:known-input", EvidenceSourceKind.EXTERNAL_RECORD)
    checks = tuple(
        CheckRequest(
            name, "mechanism", "discrimination", name, 1, 1, 1, 1, (ref,), f"action:{name}"
        )
        for name in ("cosmetic", "discriminate")
    )
    result = plan_inquiry(
        checks,
        {
            "cosmetic": {"scale": "green", "offset": "green"},
            "discriminate": {"scale": "double", "offset": "plus-one"},
        },
        budget=1,
        unresolved_claims=("mechanism",),
        enabled=True,
    )
    return result | {
        "measurement_kind": "DETERMINISTIC_CONTRACT_FIXTURE",
        "model_benchmark_run": False,
    }

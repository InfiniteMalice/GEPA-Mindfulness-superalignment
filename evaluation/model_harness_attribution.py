"""Host-private matched 2x2 attribution; no runner, reward, admission or deployment."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from math import sqrt
from statistics import stdev

# Third-party
# Local
from evaluation.improvement_records import ImprovementRecord, _encode
from gepa_mindfulness.core.evidence import EvidenceReference
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.private_promotion import ComputeBudget
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope

_EFFECTS = ("model_only", "harness_only", "combined", "interaction")


@dataclass(frozen=True, kw_only=True)
class _PrivateRecord(ImprovementRecord):
    training_eligibility: str = "HIDDEN_EVAL"
    optimizer_input: bool = False
    confers_authority: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.training_eligibility != "HIDDEN_EVAL"
            or self.optimizer_input is not False
            or self.confers_authority is not False
        ):
            raise ValueError("private attribution cannot grant authority or optimizer eligibility")
        validate_input_envelope(_encode(self))


def _identity(value: str) -> None:
    if len(value) > 256:
        raise ValueError("identity exceeds limit")


def _evaluator(value: TrustedEvaluatorContract) -> None:
    for f in fields(value):
        v = getattr(value, f.name)
        if type(v) is not str:
            raise ValueError("evaluator identity requires exact strings")
        _identity(v)


@dataclass(frozen=True)
class AttributionProtocol(_PrivateRecord):
    protocol_id: str
    task_manifest_digest: str
    original_model: str
    updated_model: str
    original_harness: str
    candidate_harness: str
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: ComputeBudget
    evaluator: TrustedEvaluatorContract
    schema_version: str = "model-harness-attribution-protocol-v1"

    def _validate(self) -> None:
        if self.schema_version != "model-harness-attribution-protocol-v1":
            raise ValueError("unsupported attribution protocol")
        for name in (
            "protocol_id",
            "original_model",
            "updated_model",
            "original_harness",
            "candidate_harness",
        ):
            _identity(getattr(self, name))
        if (
            self.original_model == self.updated_model
            or self.original_harness == self.candidate_harness
        ):
            raise ValueError("distinct model and harness versions required")
        if not 2 <= len(self.seeds) <= 32 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("requires 2..32 unique seeds")
        if any(v < 0 for v in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if not 1 <= len(self.task_ids) <= 64 or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("requires 1..64 unique task identities")
        if len(self.seeds) * len(self.task_ids) > 128:
            raise ValueError("paired sample capacity exceeded")
        for task in self.task_ids:
            _identity(task)
        _evaluator(self.evaluator)


@dataclass(frozen=True)
class AttributionObservation(_PrivateRecord):
    """One externally metered/evaluated cell; claims require host source authentication."""

    task_id: str
    seed: int
    model_version: str
    harness_version: str
    task_manifest_digest: str
    evaluator: TrustedEvaluatorContract
    score: float
    task_success: bool
    protected_pass: bool | None
    tokens: int
    tool_calls: int
    wall_time_ms: int
    evidence_refs: tuple[EvidenceReference, ...]
    execution_status: str = "completed"
    schema_version: str = "model-harness-observation-v1"

    def _validate(self) -> None:
        if self.schema_version != "model-harness-observation-v1":
            raise ValueError("unsupported attribution observation")
        for name in ("task_id", "model_version", "harness_version"):
            _identity(getattr(self, name))
        if min(self.seed, self.tokens, self.tool_calls, self.wall_time_ms) < 0:
            raise ValueError("usage and seeds must be nonnegative")
        if not 0 <= self.score <= 1:
            raise ValueError("score requires a declared unit-range host metric")
        if self.execution_status not in ("completed", "failed", "timeout"):
            raise ValueError("invalid execution status")
        if self.execution_status != "completed" and (self.score != 0 or self.task_success):
            raise ValueError("failed executions must retain zero score and unsuccessful outcome")
        if not 1 <= len(self.evidence_refs) <= 32 or len(set(self.evidence_refs)) != len(
            self.evidence_refs
        ):
            raise ValueError("bounded unique outcome evidence required")
        for ref in self.evidence_refs:
            _identity(ref.reference_id)
        _evaluator(self.evaluator)


def _effects(scores: dict[str, Fraction]) -> dict[str, Fraction]:
    b, m, h, c = (scores[k] for k in ("baseline", "model_only", "harness_only", "combined"))
    return {
        "model_only": m - b,
        "harness_only": h - b,
        "combined": c - b,
        "interaction": c - m - h + b,
    }


def report_model_harness_attribution(
    protocol: AttributionProtocol,
    observations: tuple[AttributionObservation, ...],
    *,
    enabled: bool = False,
) -> dict:
    """Analyze externally captured, complete matched trials; report no causal identification.

    The host isolates workers, protects the manifest and evaluator, authenticates captures and
    enforces timeouts/metering. Digests only bind declarations. Budgets include failed attempts
    and are summed per seed in each of the four conditions. Reports remain HIDDEN_EVAL; they do
    not feed workflow selection, issue validation receipts or consume acceptance decisions.
    """
    if enabled is not True:
        raise ValueError("attribution requires enabled=True")
    if type(protocol) is not AttributionProtocol:
        raise ValueError("exact attribution protocol required")
    p = AttributionProtocol.from_dict(AttributionProtocol.to_dict(protocol))
    expected_count = 4 * len(p.task_ids) * len(p.seeds)
    if type(observations) is not tuple or len(observations) != expected_count:
        raise ValueError("complete matched four-cell sample required")
    if any(type(v) is not AttributionObservation for v in observations):
        raise ValueError("exact observations required")
    rows = tuple(
        AttributionObservation.from_dict(AttributionObservation.to_dict(v)) for v in observations
    )
    validate_input_envelope({"protocol": p.to_dict(), "observations": [v.to_dict() for v in rows]})
    versions = {
        (p.original_model, p.original_harness): "baseline",
        (p.updated_model, p.original_harness): "model_only",
        (p.original_model, p.candidate_harness): "harness_only",
        (p.updated_model, p.candidate_harness): "combined",
    }
    cells = {name: [] for name in versions.values()}
    samples, evidence_ids = {}, set()
    for row in rows:
        name = versions.get((row.model_version, row.harness_version))
        key = row.task_id, row.seed, name
        refs = {r.reference_id for r in row.evidence_refs}
        if (
            name is None
            or row.task_id not in p.task_ids
            or row.seed not in p.seeds
            or row.task_manifest_digest != p.task_manifest_digest
            or row.evaluator != p.evaluator
            or key in samples
            or refs & evidence_ids
        ):
            raise ValueError("unmatched, duplicate or reused outcome evidence")
        evidence_ids.update(refs)
        samples[key] = row
        cells[name].append(row)
    usage = {}
    for name, values in cells.items():
        usage[name] = []
        for seed in p.seeds:
            measured = {
                f: sum(getattr(r, f) for r in values if r.seed == seed)
                for f in ("tokens", "tool_calls", "wall_time_ms")
            }
            if any(v > getattr(p.budget, f) for f, v in measured.items()):
                raise ValueError("per-seed measured usage exceeds matched budget")
            usage[name].append(dict(seed=seed, **measured))
    paired, exact = [], []
    regressions = 0
    for task in p.task_ids:
        for seed in p.seeds:
            values = {k: samples[task, seed, k] for k in cells}
            effect = _effects({k: Fraction(v.score) for k, v in values.items()})
            exact.append((seed, effect))
            baseline = values["baseline"].protected_pass
            regressions += sum(
                baseline is True and v.protected_pass is False
                for k, v in values.items()
                if k != "baseline"
            )
            paired.append(
                dict(task_id=task, seed=seed, effects={k: float(v) for k, v in effect.items()})
            )
    per_seed = [
        {
            "seed": seed,
            "effects": {
                key: float(sum(e[key] for s, e in exact if s == seed) / len(p.task_ids))
                for key in _EFFECTS
            },
        }
        for seed in p.seeds
    ]
    result = {
        "schema_version": "model-harness-attribution-v1",
        "training_eligibility": "HIDDEN_EVAL",
        "optimizer_input": False,
        "authority_granted": False,
        "promote_candidate": False,
        "protocol": p.to_dict(),
        "observations": [r.to_dict() for r in rows],
        "paired_samples": len(exact),
        "paired_rows": paired,
        "effects": {k: float(sum(e[k] for _, e in exact) / len(exact)) for k in _EFFECTS},
        "per_seed_effects": per_seed,
        "seed_standard_error": {
            k: stdev(s["effects"][k] for s in per_seed) / sqrt(len(per_seed)) for k in _EFFECTS
        },
        "protected_regressions": regressions,
        "protected_failures": sum(r.protected_pass is False for r in rows),
        "protected_unknown": sum(r.protected_pass is None for r in rows),
        "protected_non_regression": all(r.protected_pass is True for r in rows),
        "cells": {
            k: dict(
                mean_score=float(sum(Fraction(r.score) for r in v) / len(v)),
                task_successes=sum(r.task_success for r in v),
                execution_failures=sum(r.execution_status != "completed" for r in v),
                sample_count=len(v),
                usage=usage[k],
            )
            for k, v in cells.items()
        },
    }
    validate_input_envelope(result)
    return result

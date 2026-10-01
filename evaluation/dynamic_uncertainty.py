"""Matched behavioral evaluation of policies using verified epistemic trajectories."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from gepa_mindfulness.factuality_observability.schemas import RecommendedAction
from gepa_mindfulness.training.contrastive import _digest, _text
from gepa_mindfulness.training.dynamic_uncertainty import (
    ACTIONS,
    Behavior,
    DecisionInput,
    DecisionVerifier,
    TrajectoryExample,
    _configuration,
    _verified_tables,
    prepare_trajectories,
)
from gepa_mindfulness.training.eligibility import TrainingEligibility


@dataclass(frozen=True, slots=True)
class DecisionBackend:
    """A frozen host-owned policy configuration producing proposals only."""

    version: str
    decide: Callable[[DecisionInput], RecommendedAction]

    def __post_init__(self) -> None:
        """Reject unnamed versions or noncallable policy configurations."""
        _text(self.version, "backend version")
        if not callable(self.decide):
            raise ValueError("backend requires a decision callback")


def compare_decisions(
    examples: Iterable[TrajectoryExample],
    backends: Mapping[str, DecisionBackend | None],
    *,
    verifier: DecisionVerifier,
    training_examples: Iterable[TrajectoryExample] = (),
    seed: int = 0,
    enabled: bool = False,
) -> dict[str, Any]:
    """Evaluate frozen policies with the same verified tables and presentation order.

    The caller controls policy initialization, inference budgets and checkpoint history.
    Declared split checks do not establish what an external checkpoint previously saw.
    Numeric history is reported separately and never added to behavioral reward.

    Args:
        examples: Explicit non-TRAIN trajectories; partial behavior coverage is reported.
        backends: Named, versioned policies; None explicitly declares an unavailable arm.
        verifier: Host-trusted behavioral evaluator, shared by every arm.
        training_examples: Union of declared training catalogs for split-overlap checks.
        seed: Shared presentation seed in [0, 2**32).
        enabled: Literal True opts in to calling the evaluators and policies.

    Returns:
        Per-behavior scores, raw proposals, diagnostics, coverage and split-check status.

    Raises:
        ValueError: Catalogs, contracts, backend identities or policy outputs are invalid.
    """
    _configuration(enabled, seed)
    if not isinstance(backends, Mapping) or not backends:
        raise ValueError("declare at least one named backend")
    frozen_backends = dict(backends)
    for name, backend in frozen_backends.items():
        _text(name, "backend name")
        if backend is not None:
            if type(backend) is not DecisionBackend:
                raise ValueError("backend must be DecisionBackend or None")
            backend.__post_init__()
            frozen_backends[name] = replace(backend)
    prepared = prepare_trajectories(examples, for_training=False)
    training = tuple(training_examples)
    if training:
        train = prepare_trajectories(training, for_training=True)
        for field in ("example_id", "source_group", "fingerprint"):
            if {getattr(p, field) for p in prepared} & {getattr(p, field) for p in train}:
                raise ValueError(f"training/evaluation overlap in {field}")
    tables, assessment_digest, evaluator = _verified_tables(prepared, verifier)
    order = list(range(len(prepared)))
    random.Random(seed).shuffle(order)
    results: dict[str, Any] = {}
    for name, backend in frozen_backends.items():
        if backend is None:
            results[name] = {"available": False}
            continue
        rows: list[dict[str, Any]] = []
        for index in order:
            item = prepared[index]
            action = backend.decide(replace(item.input))
            if type(action) is not RecommendedAction:
                raise ValueError("policy must return a typed RecommendedAction")
            score = tables[index][ACTIONS.index(action)]
            rows.append(
                dict(
                    example_id=item.example_id,
                    behavior=item.behavior.value,
                    action=action.value,
                    verified_score=score,
                    best_action=(
                        score == max(tables[index])
                        if max(tables[index]) != min(tables[index])
                        else None
                    ),
                )
            )
        by_behavior = {}
        for behavior in Behavior:
            subset = [row for row in rows if row["behavior"] == behavior.value]
            by_behavior[behavior.value] = dict(
                count=len(subset),
                mean_verified_score=(
                    sum(row["verified_score"] for row in subset) / len(subset) if subset else None
                ),
            )
        informative = [row for row in rows if row["best_action"] is not None]
        results[name] = dict(
            available=True,
            version=backend.version,
            rows=rows,
            by_behavior=by_behavior,
            mean_verified_score=sum(row["verified_score"] for row in rows) / len(rows),
            best_action_rate=(
                sum(row["best_action"] for row in informative) / len(informative)
                if informative
                else None
            ),
        )
    restrictions = (
        TrainingEligibility.DEVELOPMENT,
        TrainingEligibility.REGRESSION,
        TrainingEligibility.HIDDEN_EVAL,
    )
    split = max((p.eligibility for p in prepared), key=restrictions.index)
    missing = [b.value for b in Behavior if b not in {p.behavior for p in prepared}]
    return dict(
        schema_version="dynamic-uncertainty-evaluation-v1",
        training_eligibility=split.value,
        dataset_digest=_digest([(p.example_id, p.source_digest) for p in prepared]),
        assessment_digest=assessment_digest,
        evaluator=evaluator,
        seed=seed,
        presentation_order=[prepared[index].example_id for index in order],
        split_check="passed" if training else "unavailable",
        complete_behavior_coverage=not missing,
        missing_behaviors=missing,
        backends=results,
        diagnostics=[
            dict(example_id=p.example_id, history=json.loads(p.input.history_json))
            for p in prepared
        ],
        confers_authority=False,
    )

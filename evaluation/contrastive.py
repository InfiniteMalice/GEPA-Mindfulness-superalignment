"""Matched four-arm contrastive experiment; backend score scales stay separate."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from cognitive_pairwise_training.schemas import PairwiseReasoningExample
from gepa_mindfulness.training.contrastive import NegativeFamily, _digest, prepare_pairs

COMPARISON_ARMS = ("classifier", "jev", "clm", "clm_curriculum")


@dataclass(frozen=True)
class RankingBackend:
    """Caller-supplied scorer; name/version identify a frozen evaluation configuration."""

    name: str
    version: str
    score: Callable[[str, tuple[str, str]], Sequence[float]]

    def __post_init__(self) -> None:
        if self.name not in COMPARISON_ARMS:
            raise ValueError("backend name must identify a comparison arm")
        if type(self.version) is not str or not self.version.strip() or not callable(self.score):
            raise ValueError("backend requires a version and callable scorer")


def _scores(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("backend scores must be a two-element list or tuple")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in value):
        raise ValueError("backend scores must be finite numbers")
    scores = float(value[0]), float(value[1])
    if not math.isfinite(scores[0] - scores[1]):
        raise ValueError("backend scores must have a finite margin")
    return scores


def compare_backends(
    examples: Iterable[PairwiseReasoningExample],
    backends: Mapping[str, RankingBackend | None],
    *,
    training_examples: Iterable[PairwiseReasoningExample] = (),
    enabled: bool = False,
) -> dict[str, Any]:
    """Measure all available arms on the same public pairs in both answer orders.

    Supply the union of the arms' training catalogs to check declared split overlap.
    Without that catalog the report explicitly marks split checking unavailable.
    Metadata cannot establish what an external checkpoint actually saw in training.
    Ties and a wrong preference in either presentation count as incorrect.
    """
    if enabled is not True:
        raise ValueError("contrastive experiments require enabled=True")
    if not isinstance(backends, Mapping) or set(backends) != set(COMPARISON_ARMS):
        raise ValueError("declare all four comparison arms, using None for unavailable arms")
    frozen_backends = dict(backends)
    for name, backend in frozen_backends.items():
        if backend is not None and (type(backend) is not RankingBackend or backend.name != name):
            raise ValueError("backend identity must match its arm")
    pairs = prepare_pairs(examples, for_training=False)
    train = tuple(training_examples)
    if train:
        train_pairs = prepare_pairs(train, for_training=True)
        for field in ("source_group", "problem_id", "pair_id", "fingerprint"):
            if {getattr(p, field) for p in pairs} & {getattr(p, field) for p in train_pairs}:
                raise ValueError(f"training/evaluation overlap in {field}")
        if {" ".join(p.prompt.split()) for p in pairs} & {
            " ".join(p.prompt.split()) for p in train_pairs
        }:
            raise ValueError("training/evaluation overlap in normalized prompt")
    results: dict[str, Any] = {}
    for name in COMPARISON_ARMS:
        backend = frozen_backends[name]
        if backend is None:
            results[name] = {"status": "unavailable"}
            continue
        rows: list[dict[str, Any]] = []
        for pair in pairs:
            forward = _scores(backend.score(pair.prompt, (pair.chosen, pair.rejected)))
            reverse = _scores(backend.score(pair.prompt, (pair.rejected, pair.chosen)))
            margins = forward[0] - forward[1], reverse[1] - reverse[0]
            rows.append(
                dict(
                    pair_id=pair.pair_id,
                    family=pair.family.value,
                    forward_scores=list(forward),
                    reverse_scores=list(reverse),
                    mean_margin=margins[0] / 2 + margins[1] / 2,
                    correct=all(m > 0 for m in margins),
                    tied=any(m == 0 for m in margins),
                    order_disagreement=(margins[0] > 0) != (margins[1] > 0),
                )
            )
        families = {}
        for family in NegativeFamily:
            members = [row for row in rows if row["family"] == family.value]
            if members:
                count = len(members)
                families[family.value] = dict(
                    count=count,
                    accuracy=sum(row["correct"] for row in members) / count,
                    mean_margin=math.fsum(row["mean_margin"] / count for row in members),
                    tie_rate=sum(row["tied"] for row in members) / count,
                    order_disagreement_rate=sum(row["order_disagreement"] for row in members)
                    / count,
                )
        results[name] = dict(
            status="measured",
            version=backend.version,
            accuracy=sum(row["correct"] for row in rows) / len(rows),
            families=families,
            records=rows,
        )
    return dict(
        schema_version="contrastive-comparison-v1",
        training_eligibility="REGRESSION",
        dataset_digest=_digest([(p.pair_id, p.source_digest) for p in pairs]),
        split_check=(
            "checked_declared_training_catalog" if train else "training_catalog_not_supplied"
        ),
        missing_families=[f.value for f in NegativeFamily if f not in {p.family for p in pairs}],
        arms=results,
    )

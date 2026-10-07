"""Regularized Bradley-Terry search signals; never optimizer-facing reward."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .verification.diagnostic_records import DiagnosticRecord, _text, _unit, records, strings
from .verification.epistemic_state import _number


@dataclass(frozen=True, slots=True)
class PairwiseComparison(DiagnosticRecord):
    """Cheap judge preference over public candidate procedures, including ties."""

    comparison_id: str
    left_id: str
    right_id: str
    left_win: float
    judge_id: str
    split: str

    schema_version = "sift-pairwise-v1"

    def __post_init__(self) -> None:
        for name in ("comparison_id", "left_id", "right_id", "judge_id"):
            _text(getattr(self, name), name)
        _unit(self.left_win, "left_win")
        if self.left_id == self.right_id:
            raise ValueError("pairwise comparison requires distinct candidates")
        if self.split != "DEVELOPMENT":
            raise ValueError(
                "search comparisons require DEVELOPMENT; hidden evaluation is forbidden"
            )


@dataclass(frozen=True, slots=True)
class SearchRank(DiagnosticRecord):
    """An uncalibrated log-strength used only to allocate offline search budget."""

    candidate_id: str
    log_strength: float
    signal_scope: str = "SEARCH_ONLY"

    schema_version = "sift-rank-v1"

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        _number(self.log_strength, "log_strength")
        if self.signal_scope != "SEARCH_ONLY":
            raise ValueError("SIFT ranking must remain SEARCH_ONLY")


def rank_candidates(
    candidate_ids: tuple[str, ...],
    comparisons: tuple[PairwiseComparison, ...],
    *,
    regularization: float = 1.0,
    iterations: int = 300,
) -> tuple[SearchRank, ...]:
    """Fit penalized log-likelihood by deterministic bounded gradient ascent.

    Positive L2 regularization keeps separated or disconnected comparison sets finite.
    Ties contribute half a win. Stable lexical tie-breaking makes fixture runs reproducible.
    """
    ids = strings(candidate_ids, "candidate_ids", required=True)
    if len(ids) > 1024:
        raise ValueError("ranking supports at most 1024 candidates")
    comparisons = records(comparisons, PairwiseComparison)
    penalty = _number(regularization, "regularization")
    if penalty <= 0 or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("positive regularization and bounded integer iterations required")
    if len(comparisons) > 10000 or len({c.comparison_id for c in comparisons}) != len(comparisons):
        raise ValueError("comparison IDs must be unique within a 10000-comparison bound")
    if any(c.left_id not in ids or c.right_id not in ids for c in comparisons):
        raise ValueError("comparison references unknown candidate")
    strength = dict.fromkeys(ids, 0.0)
    step = 1.0 / (len(comparisons) + penalty)
    for _ in range(iterations):
        gradient = {key: -penalty * value for key, value in strength.items()}
        for comparison in comparisons:
            delta = strength[comparison.left_id] - strength[comparison.right_id]
            probability = 1 / (1 + exp(-max(-700, min(700, delta))))
            residual = comparison.left_win - probability
            gradient[comparison.left_id] += residual
            gradient[comparison.right_id] -= residual
        for key in ids:
            strength[key] += step * gradient[key]
    return tuple(
        SearchRank(key, value)
        for key, value in sorted(strength.items(), key=lambda pair: (-pair[1], pair[0]))
    )

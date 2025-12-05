"""Logging helpers for EGGROLL + MDT experiments."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence


@dataclasses.dataclass
class EggrollLogRecord:
    """Structured record for a single EGGROLL candidate evaluation."""

    generation: int
    candidate_id: int
    perturbation_seed: int | None
    fitness_raw: float | None
    fitness_normalized: float | None
    task_reward: float | None
    ethics_score: float | None
    deception_penalty: float | None
    mdt_embedding: Sequence[float] | None
    grn_summary: Mapping[str, Any] | None
    attribution_metadata: Mapping[str, Any] | None


class EggrollMDTLogger:
    """Accumulated logger that preserves attribution-graph compatibility."""

    def __init__(self) -> None:
        self.records: list[EggrollLogRecord] = []

    def log(
        self,
        generation: int,
        candidate_id: int,
        perturbation_seed: int | None,
        fitness_raw: float | None,
        fitness_normalized: float | None,
        task_reward: float | None,
        ethics_score: float | None,
        deception_penalty: float | None,
        mdt_embedding: Sequence[float] | None,
        grn_summary: Mapping[str, Any] | None,
        attribution_metadata: Mapping[str, Any] | None,
    ) -> None:
        record = EggrollLogRecord(
            generation=generation,
            candidate_id=candidate_id,
            perturbation_seed=perturbation_seed,
            fitness_raw=fitness_raw,
            fitness_normalized=fitness_normalized,
            task_reward=task_reward,
            ethics_score=ethics_score,
            deception_penalty=deception_penalty,
            mdt_embedding=list(mdt_embedding) if mdt_embedding is not None else None,
            grn_summary=grn_summary,
            attribution_metadata=attribution_metadata,
        )
        self.records.append(record)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dataclasses.asdict(record) for record in self.records]


__all__ = ["EggrollLogRecord", "EggrollMDTLogger"]

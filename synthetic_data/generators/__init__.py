"""Import-safe synthetic case generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from evaluation.cases.registry import load_case_manifest
from evaluation.v5_runner import V5EvaluationCell
from gepa_mindfulness.training.eligibility import TrainingEligibility, require_training_eligible


@dataclass(frozen=True)
class GenerationMetadata:
    """Explicit reviewed coordinates for one seed; no inferred case classification."""

    cell: V5EvaluationCell | None = None
    training_eligibility: TrainingEligibility = TrainingEligibility.DEVELOPMENT
    transformations: tuple[str, ...] = ()
    verification_method: str = "human_review_required"

    def __post_init__(self) -> None:
        if self.cell is not None and not isinstance(self.cell, V5EvaluationCell):
            raise ValueError("cell must be a validated V5EvaluationCell")
        if not isinstance(self.training_eligibility, TrainingEligibility):
            raise ValueError("training_eligibility must be TrainingEligibility")
        if not isinstance(self.transformations, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.transformations
        ):
            raise ValueError("transformations must be a tuple of nonempty strings")
        if not isinstance(self.verification_method, str) or not self.verification_method.strip():
            raise ValueError("verification_method must be a nonempty string")


def attach_generation_metadata(
    cases: list[dict[str, Any]],
    *,
    generator: str,
    invariant_field: str,
    failure_field: str,
    cell_metadata: Mapping[str, GenerationMetadata] | None = None,
    for_training: bool = False,
) -> list[dict[str, Any]]:
    """Attach reconstructable metadata to seed records while retaining source labels."""
    if type(for_training) is not bool:
        raise ValueError("for_training must be a boolean")
    supplied = {} if cell_metadata is None else cell_metadata
    if not isinstance(supplied, Mapping):
        raise ValueError("cell_metadata must map authored IDs to GenerationMetadata")
    if set(supplied) - {case["case_id"] for case in cases}:
        raise ValueError("cell_metadata contains an unknown authored case ID")
    result: list[dict[str, Any]] = []
    for case in cases:
        selected = supplied.get(case["case_id"], GenerationMetadata())
        if not isinstance(selected, GenerationMetadata):
            raise ValueError("cell_metadata values must be GenerationMetadata")
        cell = selected.cell
        identity = (
            next(item for item in load_case_manifest().cases if item.id == cell.case_id)
            if cell is not None
            else None
        )
        metadata: dict[str, object] = {
            "source_template": case["case_id"],
            "canonical_case_id": None if identity is None else identity.id,
            "canonical_case_key": None if identity is None else identity.key,
            "case_version": None if cell is None else cell.case_version,
            "stripe": None if cell is None else cell.stripe_id,
            "stripe_subtype": None if cell is None else cell.subtype,
            "repeat_id": None if cell is None else cell.repeat_id,
            "model_version": None if cell is None else cell.model_version,
            "harness_version": None if cell is None else cell.harness_version,
            "generator": generator,
            "seed": 0 if cell is None else cell.seed,
            "transformations": list(selected.transformations),
            "transformation_lineage": list(selected.transformations),
            "expected_invariant": case[invariant_field],
            "expected_failure_signal": case[failure_field],
            "verification_method": selected.verification_method,
            "training_eligibility": selected.training_eligibility.value,
            "holdout_status": (
                "HIDDEN_EVAL"
                if selected.training_eligibility is TrainingEligibility.HIDDEN_EVAL
                else "not_held_out"
            ),
        }
        # Preserve pre-existing restrictions; a caller-supplied TRAIN cannot erase them.
        if "metadata" in case:
            metadata["source_record"] = case
        enriched = {**case, "metadata": metadata}
        if for_training or selected.training_eligibility is TrainingEligibility.TRAIN:
            require_training_eligible(enriched)
        result.append(enriched)
    return result

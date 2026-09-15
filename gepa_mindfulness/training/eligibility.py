"""Shared optimizer-input eligibility policy, independent of evaluation schemas."""

from collections.abc import Mapping
from enum import Enum


class TrainingEligibility(str, Enum):
    """The declared use of a record; only TRAIN is optimizer-visible."""

    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    REGRESSION = "REGRESSION"
    HIDDEN_EVAL = "HIDDEN_EVAL"


def require_training_eligible(metadata: Mapping[str, object]) -> None:
    """Reject non-training or malformed labels anywhere in retained provenance.

    Untagged legacy records remain accepted. An outer TRAIN label never overrides
    a nested restriction. This check cannot recognize a record with stripped provenance.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("training_eligibility metadata must be a mapping")
    pending: list[object] = [metadata]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if not isinstance(value, (Mapping, list, tuple)) or id(value) in visited:
            continue
        visited.add(id(value))
        if not isinstance(value, Mapping):
            pending.extend(value)
            continue
        case = value.get("case")
        if isinstance(case, Mapping) and case.get("case_version") == "17case-v5":
            _require_successful_v5_outcome(value)
        if "v5_record" in value:
            _require_successful_v5_outcome(value["v5_record"])
        if "training_eligibility" in value:
            label = value["training_eligibility"]
            if not isinstance(label, (str, TrainingEligibility)):
                raise ValueError("training_eligibility must be a canonical eligibility string")
            try:
                eligibility = TrainingEligibility(label)
            except ValueError as error:
                raise ValueError("training_eligibility must be a canonical eligibility") from error
            if eligibility is not TrainingEligibility.TRAIN:
                raise ValueError(f"training_eligibility {eligibility.value} forbids optimization")
        if "holdout_status" in value and value["holdout_status"] not in (
            "not_held_out",
            "TRAIN",
        ):
            raise ValueError("training_eligibility forbids held-out or malformed holdout_status")
        for field in ("metadata", "source_record"):
            if field in value and not isinstance(value[field], Mapping):
                raise ValueError(f"training_eligibility provenance {field} must be a mapping")
        pending.extend(value.values())


def _require_successful_v5_outcome(record: object) -> None:
    """Check explicitly identified V5 outcomes without classifying unrelated legacy objects."""
    if not isinstance(record, Mapping) or any(
        not isinstance(record.get(field), Mapping) for field in ("case", "system", "outcome")
    ):
        raise ValueError("V5 record must retain structured case, system, and outcome")
    if record["outcome"].get("passed") is not True:
        raise ValueError("V5 failed or malformed outcome is audit-only; repair before train")

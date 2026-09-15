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

    Args:
        metadata: Retained input record or metadata, including nested source records.

    Returns:
        None when the retained record is eligible for optimization.

    Raises:
        ValueError: A restriction, failed V5 outcome, malformed provenance, or missing
            generated-data review prevents admission.
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
        if "generator" in value and "source_template" in value:
            _require_reviewed_generation(value)
        if "holdout_status" in value and value["holdout_status"] not in (
            "not_held_out",
            "TRAIN",
        ):
            raise ValueError("training_eligibility forbids held-out or malformed holdout_status")
        for field in ("metadata", "source_record"):
            if field in value and not isinstance(value[field], Mapping):
                raise ValueError(f"training_eligibility provenance {field} must be a mapping")
        pending.extend(value.values())


def _require_reviewed_generation(value: Mapping[str, object]) -> None:
    """Validate generated TRAIN coordinates and a host-authenticated completed review."""
    # Import only at admission to keep the shared policy import-safe for V5 consumers.
    from evaluation.cases.registry import load_case_manifest
    from evaluation.v5_runner import V5EvaluationCell
    from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind

    if value.get("training_eligibility") != TrainingEligibility.TRAIN:
        raise ValueError("generated training_eligibility requires explicit TRAIN")
    if value.get("review_completed") is not True:
        raise ValueError("generated TRAIN requires completed human review authorization")
    reviewer = value.get("reviewed_by")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("generated TRAIN requires an identified human reviewer")
    reference = EvidenceReference.from_dict(value.get("review_authorization"))
    if reference.source_kind is not EvidenceSourceKind.EXTERNAL_RECORD:
        raise ValueError("human review authorization requires an external record")
    try:
        cell = V5EvaluationCell(
            value["canonical_case_id"],
            value["case_version"],
            value["stripe"],
            value["stripe_subtype"],
            value["repeat_id"],
            value["seed"],
            value["model_version"],
            value["harness_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("generated TRAIN requires a validated V5 cell") from error
    case = next(case for case in load_case_manifest().cases if case.id == cell.case_id)
    if value.get("canonical_case_key") != case.key:
        raise ValueError("generated TRAIN case key does not match its validated cell")


def _require_successful_v5_outcome(record: object) -> None:
    """Check explicitly identified V5 outcomes without classifying unrelated legacy objects."""
    if not isinstance(record, Mapping) or any(
        not isinstance(record.get(field), Mapping) for field in ("case", "system", "outcome")
    ):
        raise ValueError("V5 record must retain structured case, system, and outcome")
    if record["outcome"].get("passed") is not True:
        raise ValueError("V5 failed or malformed outcome is audit-only; repair before train")

"""Objective decomposition helpers for validator capture analysis."""

# Standard library
from __future__ import annotations

import re
from typing import Any

# Local
from .schema import ObjectiveStructure

VALIDATOR_CUES = (
    "validator",
    "rubric",
    "must pass",
    "unit test",
    "assert",
    "json schema",
    "pydantic",
    "required field",
    "make the checker pass",
    "checker pass",
    "pass schema",
    "return only valid json",
    "benchmark",
)

SENSITIVE_CUES = {
    "cyber": ("exploit", "malware", "evade detection", "payload", "credential"),
    "bio": ("pathogen", "toxin", "dosage"),
    "privacy": ("private data", "ssn", "password", "api key"),
    "toxicity": ("toxicity examples", "slur", "hate speech"),
}

OUTPUT_CUES = ("json", "csv", "yaml", "table", "schema", "pydantic")


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _extract_data_slots(text: str) -> list[str]:
    slots = [
        match.group(1).strip() for match in re.finditer(r'"([a-zA-Z0-9_\- ]{3,40})"\s*:', text)
    ]
    if slots:
        return sorted(set(slots))

    candidate_slots: list[str] = []
    for token in (
        "dangerous_payload",
        "exploit_example",
        "toxin_recipe",
        "dosage",
        "credential",
        "private_data",
        "api_key",
        "password",
        "ssn",
    ):
        if token in text:
            candidate_slots.append(token)
    return sorted(set(candidate_slots))


def _infer_domain(text: str) -> str | None:
    for domain, cues in SENSITIVE_CUES.items():
        if _contains_any(text, cues):
            return domain
    if "medical" in text or "health" in text:
        return "medical"
    if "classifier" in text or "dataset" in text:
        return "dataset"
    return None


def _imperative_conflicts(text: str) -> list[str]:
    conflicts: list[str] = []
    if _contains_any(
        text,
        (
            "toxin",
            "pathogen",
            "dosage",
            "exploit",
            "malware",
            "credential",
            "private data",
            "evade detection",
            "self-medication",
            "payload",
            "dangerous_payload",
        ),
    ):
        conflicts.append("Reduce Human Suffering")
    if _contains_any(text, ("private data", "credential", "password", "api key", "ssn")):
        conflicts.append("Increase Human Prosperity")
    if _contains_any(text, ("fabricate", "mislead benchmark", "poison dataset")):
        conflicts.append("Increase Scientific Knowledge")
    return sorted(set(conflicts))


def decompose_objective(
    input_text: str,
    metadata: dict[str, Any] | None = None,
) -> ObjectiveStructure:
    """Decompose a task into objective/validator structure with inspectable heuristics."""

    metadata = metadata or {}
    text = input_text.strip()
    lowered = text.lower()

    notes: list[str] = []
    if _contains_any(lowered, VALIDATOR_CUES):
        notes.append("Validator-like success criteria detected")
    if "fill placeholders" in lowered or "complete missing data" in lowered:
        notes.append("Task asks for structural completion of missing slots")

    explicit_success_condition = None
    for cue in (
        "must pass",
        "make the checker pass",
        "checker pass",
        "pass schema",
        "unit test",
        "return only valid json",
        "benchmark",
    ):
        if cue in lowered:
            explicit_success_condition = cue
            break

    implicit_success_condition = None
    if explicit_success_condition is None and _contains_any(lowered, VALIDATOR_CUES):
        implicit_success_condition = "optimize output for validator compliance"

    validator_or_rubric = None
    if _contains_any(lowered, VALIDATOR_CUES):
        validator_or_rubric = "validator/rubric/schema/test present"

    required_output_format = None
    for cue in OUTPUT_CUES:
        if cue in lowered:
            required_output_format = cue
            break

    data_slots = _extract_data_slots(text)
    requested_capability = metadata.get("requested_capability")
    if requested_capability is None:
        if "generate" in lowered and "dataset" in lowered:
            requested_capability = "dataset_generation"
        elif "fill" in lowered and data_slots:
            requested_capability = "slot_filling"
        else:
            requested_capability = "task_completion"

    domain = metadata.get("domain") or _infer_domain(lowered)
    tool_context = metadata.get("tool_context", [])

    conflicts = _imperative_conflicts(lowered)
    harm_if_completed = None
    if conflicts:
        harm_if_completed = "Local completion may require unsafe or privacy-violating content"

    safe_completion_available = bool(
        conflicts
        and _contains_any(
            lowered,
            (
                "placeholder",
                "dummy",
                "synthetic",
                "non-executable",
                "approved dataset",
                "abstract label",
            ),
        )
    )
    if conflicts and not safe_completion_available:
        safe_completion_available = True
        notes.append("Safe completion path inferred via redaction or non-operational placeholders")

    uncertainty = 0.2 if validator_or_rubric else 0.55
    if conflicts and not data_slots:
        uncertainty = 0.45

    return ObjectiveStructure(
        surface_task=text,
        explicit_success_condition=explicit_success_condition,
        implicit_success_condition=implicit_success_condition,
        validator_or_rubric=validator_or_rubric,
        required_output_format=required_output_format,
        data_slots=data_slots,
        requested_capability=requested_capability,
        domain=domain,
        tool_context=list(tool_context),
        harm_if_completed=harm_if_completed,
        conflict_with_imperatives=conflicts,
        safe_completion_available=safe_completion_available,
        uncertainty=uncertainty,
        notes=notes,
    )

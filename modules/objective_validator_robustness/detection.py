"""Detection logic for validator capture and overlay tier selection."""

# Standard library
from __future__ import annotations

# Local
from .schema import ObjectiveStructure, ValidatorCaptureSignal


def detect_validator_capture(structure: ObjectiveStructure) -> ValidatorCaptureSignal:
    """Detect whether local validator success likely conflicts with alignment imperatives."""

    reasons: list[str] = []
    has_validator_pressure = bool(
        structure.validator_or_rubric
        or structure.explicit_success_condition
        or structure.implicit_success_condition
    )
    if has_validator_pressure:
        reasons.append("Task contains validator/rubric/test success pressure")

    requires_sensitive_completion = bool(
        structure.conflict_with_imperatives
        or structure.harm_if_completed
        or any(
            slot
            for slot in structure.data_slots
            if slot.lower()
            in {"dangerous_payload", "toxin_recipe", "credential", "password", "api_key", "ssn"}
        )
    )
    if requires_sensitive_completion:
        reasons.append("Completion appears to require sensitive or harmful slot content")

    local_success_conflicts_with_safety = has_validator_pressure and requires_sensitive_completion
    if local_success_conflicts_with_safety:
        reasons.append("Local success target conflicts with global imperative safety")

    harmful_content_is_structurally_required = bool(
        has_validator_pressure and structure.data_slots
    ) and bool(requires_sensitive_completion)
    if harmful_content_is_structurally_required:
        reasons.append("Harmful fields appear embedded as required structure")

    task_appears_benign = has_validator_pressure and not any(
        token in structure.surface_task.lower()
        for token in ("attack", "harm", "bypass", "abuse", "exploit directly")
    )

    direct_request_would_likely_be_refused = bool(requires_sensitive_completion)
    safe_alternative_exists = structure.safe_completion_available

    confidence = 0.2
    if has_validator_pressure:
        confidence += 0.2
    if requires_sensitive_completion:
        confidence += 0.25
    if local_success_conflicts_with_safety:
        confidence += 0.2
    if harmful_content_is_structurally_required:
        confidence += 0.15
    confidence = min(confidence, 0.95)

    return ValidatorCaptureSignal(
        has_validator_pressure=has_validator_pressure,
        requires_sensitive_completion=requires_sensitive_completion,
        local_success_conflicts_with_safety=local_success_conflicts_with_safety,
        harmful_content_is_structurally_required=harmful_content_is_structurally_required,
        task_appears_benign=task_appears_benign,
        direct_request_would_likely_be_refused=direct_request_would_likely_be_refused,
        safe_alternative_exists=safe_alternative_exists,
        confidence=confidence,
        reasons=reasons,
    )


def validator_overlay_tier(signal: ValidatorCaptureSignal) -> str:
    """Map detection signals to V0..V5 overlay tiers."""

    if not signal.has_validator_pressure and not signal.requires_sensitive_completion:
        return "V0"
    if signal.has_validator_pressure and not signal.requires_sensitive_completion:
        return "V1"
    if signal.has_validator_pressure and signal.requires_sensitive_completion:
        if not signal.local_success_conflicts_with_safety:
            return "V2"
        if not signal.harmful_content_is_structurally_required:
            return "V3"
        if signal.safe_alternative_exists:
            return "V4"
        return "V5"
    return "V2"

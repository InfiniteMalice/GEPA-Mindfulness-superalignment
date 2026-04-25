"""Policy routing for validator capture safety decisions."""

# Standard library
from __future__ import annotations

# Local
from .schema import ObjectiveStructure, ValidatorCaptureSignal, ValidatorPolicyDecision

HIGH_RISK_DOMAINS = {"bio", "cyber", "medical", "weapons", "privacy"}


def decide_validator_policy(
    structure: ObjectiveStructure,
    signal: ValidatorCaptureSignal,
) -> ValidatorPolicyDecision:
    """Select an action that preserves benign value while blocking unsafe completion pressure."""

    domain = (structure.domain or "").lower()
    blocked = sorted(set(structure.data_slots))

    if not signal.has_validator_pressure and not signal.requires_sensitive_completion:
        return ValidatorPolicyDecision(
            action="allow",
            rationale="No validator capture pressure or safety conflict detected.",
            safe_response_strategy="Proceed with normal task completion.",
            blocked_elements=[],
            allowed_elements=["full_task"],
            recommended_trace_note="V0 baseline: no objective-validator conflict.",
        )

    if signal.has_validator_pressure and not signal.requires_sensitive_completion:
        return ValidatorPolicyDecision(
            action="allow",
            rationale="Validator pressure exists, but required outputs remain benign.",
            safe_response_strategy=(
                "Complete requested format while preserving standard quality checks."
            ),
            blocked_elements=[],
            allowed_elements=["schema", "logic", "formatting"],
            recommended_trace_note="V1 parsed objective with benign validator requirements.",
        )

    if (
        signal.local_success_conflicts_with_safety
        and signal.safe_alternative_exists
        and domain not in {"bio", "medical", "weapons"}
    ):
        return ValidatorPolicyDecision(
            action="transform",
            rationale=(
                "Unsafe slots are structurally required, "
                "but safe placeholders can preserve utility."
            ),
            safe_response_strategy=(
                "Keep schema shape and replace dangerous content with "
                "[REDACTED_SAFE_PLACEHOLDER] or non-executable dummy values."
            ),
            blocked_elements=blocked,
            allowed_elements=["schema shape", "validation guidance", "synthetic benign controls"],
            recommended_trace_note="V4: transformed completion to avoid validator capture.",
        )

    if signal.local_success_conflicts_with_safety and domain in HIGH_RISK_DOMAINS:
        if domain in {"bio", "medical", "weapons"}:
            return ValidatorPolicyDecision(
                action="escalate",
                rationale="High-risk domain with validator-required unsafe content.",
                safe_response_strategy=(
                    "Decline operational details, provide high-level safety framing, "
                    "and escalate review."
                ),
                blocked_elements=blocked,
                allowed_elements=["risk explanation", "safety resources"],
                recommended_trace_note="V5: high-risk validator capture; escalation triggered.",
            )
        return ValidatorPolicyDecision(
            action="refuse",
            rationale=(
                "Task completion would require unsafe or privacy-invasive " "operational content."
            ),
            safe_response_strategy=(
                "Refuse dangerous slots and offer non-operational alternatives "
                "or synthetic invalid data."
            ),
            blocked_elements=blocked,
            allowed_elements=["safe template", "placeholder fields", "benign controls"],
            recommended_trace_note="V5: validator capture required refusal of harmful content.",
        )

    if signal.requires_sensitive_completion and not signal.local_success_conflicts_with_safety:
        return ValidatorPolicyDecision(
            action="bound",
            rationale=(
                "Potential dual-use request detected without clear structural " "harm requirement."
            ),
            safe_response_strategy=(
                "Provide bounded, non-operational guidance and request " "safer constraints."
            ),
            blocked_elements=blocked,
            allowed_elements=["high-level explanation", "defensive best practices"],
            recommended_trace_note="V3: bounded response under uncertainty.",
        )

    return ValidatorPolicyDecision(
        action="ask_clarifying",
        rationale="Insufficient context to determine whether validator requirements are safe.",
        safe_response_strategy=(
            "Ask for legitimate use context, approved dataset source, and safety constraints."
        ),
        blocked_elements=blocked,
        allowed_elements=["clarification questions"],
        recommended_trace_note="V2/V3 ambiguity: clarification requested.",
    )

"""Over-refusal guard logic."""

from __future__ import annotations

from .abstention_policy import detect_refusal
from .types import AtomicClaim, ClaimSupport, ScopedAlternative


def find_scoped_alternative(
    prompt: str,
    answer: str,
    claims: list[AtomicClaim],
    supports: list[ClaimSupport],
    safety_context: dict[str, bool] | None = None,
) -> ScopedAlternative:
    safety_context = safety_context or {}
    if safety_context.get("must_refuse", False):
        return ScopedAlternative(
            False, "refuse", "Safety policy requires refusal.", refusal_required=True
        )

    refused = detect_refusal(answer)
    supported = [s for s in supports if s.support_label in {"supported", "partially_supported"}]
    if refused and supported:
        return ScopedAlternative(
            True, "answer_with_qualifications", "Refusal not required; scoped answer possible."
        )
    if any(s.support_label == "partially_supported" for s in supports):
        return ScopedAlternative(True, "answer_partially", "Return only supported portion.")
    if claims and all(s.support_label == "unsupported" for s in supports):
        return ScopedAlternative(
            True, "ask_clarifying_question", "Clarify scope or request evidence."
        )
    return ScopedAlternative(False, "abstain", "No useful supported scope found.")

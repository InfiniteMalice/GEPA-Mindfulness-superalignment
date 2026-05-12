"""MDL compression-control helpers for Schema V3."""

from __future__ import annotations

from .case_v3 import MDLControlOverlay


def mdl_control_gate(
    *,
    default_answer: str | None,
    controlled_answer: str | None,
    grounding_conflict: bool = False,
    causal_conflict: bool = False,
) -> MDLControlOverlay:
    """Escalate when a fast/default answer conflicts with controlled checks."""
    conflict = bool(grounding_conflict or causal_conflict)
    return MDLControlOverlay(
        default_answer=default_answer,
        controlled_answer=controlled_answer,
        default_control_conflict=conflict,
        escalation_required=conflict,
        escalation_taken=conflict and controlled_answer is not None,
        compression_candidate=not conflict,
        compression_guardrails=["preserve_grounding", "preserve_causal_checks"],
    )

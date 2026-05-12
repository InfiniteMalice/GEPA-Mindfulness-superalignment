"""Causal and scientific-method helpers for Schema V3."""

from __future__ import annotations

from .case_v3 import CausalScientificOverlay, ControlOverlay


def causal_confounding_overlay(confounder: str) -> CausalScientificOverlay:
    """Create metadata for a correlation-versus-causation confounding case."""
    return CausalScientificOverlay(
        causal_types=["common_cause_confounding", "interventionist"],
        scientific_controls=["scientific_method_check", "confounder_detection"],
        confounders_considered=[confounder],
        falsification_conditions=["effect disappears under randomized intervention"],
        alternative_hypotheses=["observed correlation is explained by common cause"],
        causal_claim_strength="weak",
    )


def scientific_method_required_control() -> ControlOverlay:
    """Return the control overlay required for causal confounding examples."""
    return ControlOverlay(
        required_controls=["scientific_method_check", "uncertainty_estimation"],
        observed_controls=["scientific_method_check", "uncertainty_estimation"],
        answer_mode_decision="scoped_answer",
        grounding_status="grounded",
        calibration_status="calibrated",
        method_selection_status="scientific_method_selected",
    )

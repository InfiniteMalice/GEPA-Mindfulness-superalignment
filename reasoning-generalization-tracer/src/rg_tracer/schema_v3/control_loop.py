"""Control-loop registry for the 13-case Schema V3 overlay."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ControlLoopEntry:
    """Machine-readable description of one metacognitive control operation."""

    name: str
    display_name: str
    definition: str
    inputs: list[str]
    outputs: list[str]
    failure_modes: list[str]
    example_domains: list[str]
    composition_partners: list[str]
    scoring_notes: str
    subchecks: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable registry entry."""
        return dataclasses.asdict(self)


CONTROL_OPERATIONS = (
    "task_framing",
    "epistemic_grounding",
    "method_selection",
    "reasoning_unit_selection",
    "assumption_tracking",
    "uncertainty_estimation",
    "calibration_decision",
    "consistency_checking",
    "error_localization",
    "revision_control",
    "scientific_method_check",
    "epistemic_boundary_abstention",
    "mdl_compression_control",
)

SCIENTIFIC_METHOD_SUBCHECKS = (
    "hypothesis_formation",
    "operationalization",
    "prediction",
    "falsification_condition",
    "control_group",
    "confounder_detection",
    "measurement_validity",
    "replication_check",
    "effect_size_reasoning",
    "alternative_hypothesis_comparison",
)

MDL_COMPRESSION_SUBCHECKS = (
    "fast_default_answer",
    "controlled_deliberative_answer",
    "disagreement_conflict_signal",
    "escalation_rule",
    "compression_candidate_rule",
    "guardrails_against_unsafe_overcompression",
)


def _control_entry(name: str) -> ControlLoopEntry:
    display = name.replace("_", " ").title()
    subchecks: list[str] = []
    if name == "scientific_method_check":
        subchecks = list(SCIENTIFIC_METHOD_SUBCHECKS)
    elif name == "mdl_compression_control":
        subchecks = list(MDL_COMPRESSION_SUBCHECKS)
    return ControlLoopEntry(
        name=name,
        display_name=display,
        definition=f"Invoke {display.lower()} as a public control-loop operation.",
        inputs=["task", "evidence", "candidate_answer", "confidence"],
        outputs=["control_decision", "status", "repair_or_route"],
        failure_modes=[f"missing_{name}", f"misapplied_{name}"],
        example_domains=["qa", "science", "safety", "code", "calibration"],
        composition_partners=["task_framing", "uncertainty_estimation"],
        scoring_notes=(
            "Reward observed public control use when required; do not penalize hidden "
            "thought directly. Observable overclaiming remains handled by answer and "
            "confidence components."
        ),
        subchecks=subchecks,
    )


def _duplicate_names(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


def build_control_registry() -> dict[str, ControlLoopEntry]:
    """Build the control-loop registry keyed by operation name."""
    duplicates = _duplicate_names(CONTROL_OPERATIONS)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"CONTROL_OPERATIONS contains duplicate names: {joined}")
    return {name: _control_entry(name) for name in CONTROL_OPERATIONS}


CONTROL_LOOP_REGISTRY = build_control_registry()

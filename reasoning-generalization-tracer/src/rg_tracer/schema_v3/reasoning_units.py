"""Reasoning-unit registry for the 13-case Schema V3 overlay."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ReasoningUnitEntry:
    """Machine-readable description of one compositional reasoning unit."""

    name: str
    display_name: str
    definition: str
    input_type: list[str]
    output_type: list[str]
    dependencies: list[str]
    minimal_tests: list[str]
    transfer_tests: list[str]
    failure_modes: list[str]
    composition_partners: list[str]
    example_domains: list[str]
    subtypes: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable registry entry."""
        return dataclasses.asdict(self)


CAUSAL_SUBTYPES = [
    "temporal_order",
    "mechanism",
    "counterfactual_dependence",
    "interventionist",
    "common_cause_confounding",
    "necessary_vs_sufficient",
    "enabling_vs_triggering",
    "overdetermination",
    "probabilistic_causality",
    "dose_response",
    "level_shifting",
    "material_formal_efficient_final",
    "agentic_causality",
    "normative_responsibility",
]

GROUP_THEORETIC_SUBTYPES = [
    "identity_transformation",
    "inverse_transformation",
    "closure_under_operation",
    "associativity_of_composition",
    "symmetry_detection",
    "invariance_under_transformation",
    "equivalence_class_reasoning",
    "orbit_reasoning",
    "stabilizer_reasoning",
    "quotient_structure_reasoning",
    "normal_form_reasoning",
    "isomorphism_detection",
    "symmetry_breaking",
    "conservation_law_analogy",
]

_REASONING_FAMILIES = [
    "recursive",
    "functional_composition",
    "type_constrained_composition",
    "abstraction",
    "instantiation",
    "variable_binding",
    "relational_composition",
    "constraint_composition",
    "decomposition",
    "proof_step_composition",
    "analogy",
    "invariant_preservation",
    "contextual_modulation",
    "causal_reasoning",
    "hierarchical_composition",
    "dialectical_composition",
    "compression_expansion",
    "group_theoretic_reasoning",
]


def _generic_entry(name: str) -> ReasoningUnitEntry:
    display = name.replace("_", " ").title()
    return ReasoningUnitEntry(
        name=name,
        display_name=display,
        definition=f"Apply {display.lower()} as a public reasoning operation.",
        input_type=["problem_state", "context"],
        output_type=["updated_reasoning_state"],
        dependencies=[],
        minimal_tests=[f"identify when {name} is required"],
        transfer_tests=[f"transfer {name} across a new domain"],
        failure_modes=[f"missing_{name}", f"misapplied_{name}"],
        composition_partners=["abstraction", "constraint_composition"],
        example_domains=["math", "natural_language", "safety", "code"],
    )


def _causal_entry() -> ReasoningUnitEntry:
    return dataclasses.replace(
        _generic_entry("causal_reasoning"),
        display_name="Causal Reasoning",
        definition=(
            "Reason about temporal order, mechanisms, interventions, confounders, "
            "counterfactuals, levels, and causal claim strength."
        ),
        input_type=["claim", "evidence", "candidate_mechanism"],
        output_type=["causal_type", "claim_strength", "confounders", "tests"],
        dependencies=["temporal_order", "abstraction", "scientific_method_check"],
        minimal_tests=[
            "distinguish correlation from causation",
            "identify common-cause confounding",
            "state an intervention that would change an outcome",
        ],
        transfer_tests=[
            "medical confounding",
            "social-science confounding",
            "software performance causal diagnosis",
        ],
        failure_modes=["post_hoc_fallacy", "confounder_neglect", "overclaimed_causality"],
        composition_partners=["scientific_method_check", "abstraction", "decomposition"],
        example_domains=["science", "medicine", "policy", "debugging"],
        subtypes=CAUSAL_SUBTYPES,
    )


def _group_entry() -> ReasoningUnitEntry:
    return ReasoningUnitEntry(
        name="group_theoretic_reasoning",
        display_name="Group-Theoretic Reasoning",
        definition=(
            "Reason over transformations, symmetries, invariants, equivalence classes, "
            "reversible operations, canonical forms, and structure-preserving mappings. "
            "This family helps detect when surface changes do or do not alter the "
            "relevant underlying structure."
        ),
        input_type=[
            "object_or_representation",
            "transformation_or_set_of_transformations",
            "target_property",
        ],
        output_type=[
            "invariant_properties",
            "changed_properties",
            "equivalence_class",
            "canonical_form",
            "symmetry_breaks",
            "reachable_variants",
        ],
        dependencies=[
            "invariant_preservation",
            "abstraction",
            "relational_composition",
            "type_constrained_composition",
            "analogy",
        ],
        minimal_tests=[
            "variable renaming preserves equation validity",
            "translation preserves factual claim",
            "paraphrase preserves user intent",
            "rotation preserves shape identity",
            "inverse operation restores original state",
        ],
        transfer_tests=[
            "math transformation invariance",
            "semantic laundering detection",
            "code refactor behavior preservation",
            "legal reformatting obligation preservation",
            "causal affordance under euphemism",
        ],
        failure_modes=[
            "surface_form_bias",
            "false_equivalence",
            "missed_symmetry_break",
            "treating_irreversible_operations_as_reversible",
            "failing_to_canonicalize_variants",
            "confusing_same_topic_with_same_intent",
        ],
        composition_partners=[
            "invariant_preservation",
            "abstraction",
            "instantiation",
            "analogy",
            "contextual_modulation",
            "causal_reasoning",
            "proof_step_composition",
            "constraint_composition",
        ],
        example_domains=[
            "arithmetic",
            "algebra",
            "natural_language",
            "safety",
            "legal_reasoning",
            "code_refactoring",
            "mechanistic_interpretability",
        ],
        subtypes=GROUP_THEORETIC_SUBTYPES,
    )


def build_reasoning_registry() -> dict[str, ReasoningUnitEntry]:
    """Build the reasoning-unit registry keyed by unit name."""
    registry = {name: _generic_entry(name) for name in _REASONING_FAMILIES}
    registry["causal_reasoning"] = _causal_entry()
    registry["group_theoretic_reasoning"] = _group_entry()
    return registry


REASONING_UNIT_REGISTRY = build_reasoning_registry()

"""PlurPO-inspired framing diagnostics, not simulated votes on truth or authorization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .claim_graph import ClaimGraph
from .diagnostic_records import DiagnosticRecord, _text, _unit, strings
from .perspective_records import PERSPECTIVES, Perspective


@dataclass(frozen=True, slots=True)
class Judgment(DiagnosticRecord):
    """Core public judgments; empathy and explanation are deliberately outside invariance."""

    factual_judgment: str
    evidence_standard: str
    hard_constraints: tuple[str, ...]
    response_mode: str
    permissible_action_class: str
    confidence: float
    normative_judgment: str = "unresolved"

    schema_version = "perspective-judgment-v1"

    def __post_init__(self) -> None:
        for name in (
            "factual_judgment",
            "evidence_standard",
            "response_mode",
            "permissible_action_class",
            "normative_judgment",
        ):
            _text(getattr(self, name), name)
        object.__setattr__(
            self, "hard_constraints", strings(self.hard_constraints, "hard_constraints")
        )
        _unit(self.confidence, "confidence")


def perspective_variants(seed: Perspective, *, enabled: bool = False) -> tuple[Perspective, ...]:
    """Generate transformation specifications; the host renders and audits semantic identity."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if not enabled:
        return ()
    seed = Perspective.from_dict(seed.to_dict())
    return tuple(
        replace(
            seed,
            variant_id=f"{seed.variant_id}:{role}",
            perspective=role,
            source_variant_id=seed.variant_id,
            material_facts_changed=False,
        )
        for role in PERSPECTIVES
    )


def compare_perspectives(
    original: Perspective,
    variant: Perspective,
    before: Judgment,
    after: Judgment,
    *,
    expected_stakeholders: tuple[str, ...] = (),
    observed_stakeholders: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Report core drift and coverage against a host-supplied stakeholder reference set.

    Material changes leave justification unresolved; they do not automatically excuse changes
    to authorization. Hallucination means absence from the reference set, not proven nonexistence.
    """
    original = Perspective.from_dict(original.to_dict())
    variant = Perspective.from_dict(variant.to_dict())
    before = Judgment.from_dict(before.to_dict())
    after = Judgment.from_dict(after.to_dict())
    if original.semantic_core_id != variant.semantic_core_id:
        raise ValueError("perspective comparisons require one semantic core")
    if (
        not variant.material_facts_changed
        and original.material_facts_id != variant.material_facts_id
    ):
        raise ValueError("unchanged facts require matching material_facts_id")
    expected = set(strings(expected_stakeholders, "expected_stakeholders"))
    observed = set(strings(observed_stakeholders, "observed_stakeholders"))
    fields = (
        "factual_judgment",
        "evidence_standard",
        "response_mode",
        "permissible_action_class",
        "normative_judgment",
    )
    changed = [name for name in fields if getattr(before, name) != getattr(after, name)]
    constraints_changed = set(before.hard_constraints) != set(after.hard_constraints)
    if constraints_changed:
        changed.append("hard_constraints")
    spurious = bool(changed) and not variant.material_facts_changed
    return {
        "training_eligibility": "DEVELOPMENT",
        "changed_fields": tuple(changed),
        "hard_constraint_instability": constraints_changed,
        "confidence_shift": after.confidence - before.confidence,
        "response_mode_shift": before.response_mode != after.response_mode,
        "spurious_framing_transition": spurious,
        "transition": (
            "SPURIOUS_FRAMING_TRANSITION"
            if spurious
            else ("UNRESOLVED" if changed else "NO_TRANSITION_EXPECTED")
        ),
        "stakeholder_omissions": tuple(sorted(expected - observed)),
        "stakeholder_hallucinations": tuple(sorted(observed - expected)),
        "semantic_laundering_scrutiny": spurious,
    }


def perspective_claim_divergence(
    before: ClaimGraph,
    after: ClaimGraph,
    *,
    material_facts_changed: bool,
) -> dict[str, Any]:
    """Locate changed public premises beneath conclusions, without inferring hidden reasoning.

    Earliest means a changed decision-relevant node without a changed dependency below it.
    This is a graph diagnostic, not a temporal claim or a causal attribution.
    """
    if type(material_facts_changed) is not bool:
        raise ValueError("material change must be explicit boolean")
    left = {node.claim.claim_id: node for node in ClaimGraph.from_dict(before.to_dict()).nodes}
    right = {node.claim.claim_id: node for node in ClaimGraph.from_dict(after.to_dict()).nodes}
    changed = {
        key
        for key in left.keys() | right.keys()
        if left.get(key) != right.get(key)
        and any(
            node is not None and node.decision_importance > 0
            for node in (left.get(key), right.get(key))
        )
    }
    parents = {
        edge.parent_claim_id
        for edge in before.dependencies + after.dependencies
        if edge.child_claim_id in changed
    }
    return {
        "training_eligibility": "DEVELOPMENT",
        "changed_claim_ids": tuple(sorted(changed)),
        "earliest_divergence_claim_ids": tuple(sorted(changed - parents)),
        "material_facts_changed": material_facts_changed,
        "semantic_laundering_scrutiny": bool(changed) and not material_facts_changed,
        "justification": "UNRESOLVED" if changed else "NO_TRANSITION_EXPECTED",
    }

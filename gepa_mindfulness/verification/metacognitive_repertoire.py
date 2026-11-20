"""Small declarative method catalogue; entries confer no execution or normative authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .metacognitive_records import (
    FindingKind,
    MetacognitiveAssessment,
    array,
    exact,
    snapshot,
    strings,
    text,
)

_SCHEMA = "metacognitive-repertoire-v1"
_SOURCE = "https://arxiv.org/html/2611.12762v2#S2.SS3"
_OUTPUTS = {
    "continue",
    "verify_assumption",
    "request_evidence",
    "compare_hypotheses",
    "test_sensitivity",
    "reconcile_observations",
    "request_clarification",
    "recommend_abstention",
    "review_continuity",
    "stop_branch",
}
_PROHIBITED = (
    "execute_action",
    "modify_accepted_evidence",
    "change_reward",
    "grant_authority",
    "change_root_objective",
    "rewrite_constitution",
    "change_canonical_cases",
)


@dataclass(frozen=True, slots=True)
class RepertoireStrategy:
    """An inspectable diagnostic method, distinct from permission and policy."""

    strategy_id: str
    description: str
    applicability_conditions: str
    when_finding_kinds: tuple[FindingKind, ...]
    required_evidence: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    verification_requirements: tuple[str, ...]
    resource_implications: str
    prohibited_effects: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        for name in (
            "strategy_id",
            "description",
            "applicability_conditions",
            "resource_implications",
            "source",
        ):
            text(getattr(self, name), name)
        kinds = self.when_finding_kinds
        if (
            type(kinds) is not tuple
            or not kinds
            or len(kinds) > 32
            or any(type(k) is not FindingKind for k in kinds)
            or len(set(kinds)) != len(kinds)
        ):
            raise ValueError("applicability requires unique typed finding kinds")
        for name in (
            "required_evidence",
            "expected_outputs",
            "verification_requirements",
            "prohibited_effects",
        ):
            value = strings(getattr(self, name), name, 32)
            if not value:
                raise ValueError(f"{name} must be specified")
            object.__setattr__(self, name, value)
        if not set(_PROHIBITED) <= set(self.prohibited_effects):
            raise ValueError("diagnostic methods must preserve all authority prohibitions")
        if not set(self.expected_outputs) <= _OUTPUTS:
            raise ValueError("a method can declare diagnostic proposal outputs only")

    def to_dict(self) -> dict[str, Any]:
        """Serialize detached method metadata after revalidation."""
        checked = snapshot(self, RepertoireStrategy)
        result = {f.name: getattr(checked, f.name) for f in fields(self)}
        for name in (
            "required_evidence",
            "expected_outputs",
            "verification_requirements",
            "prohibited_effects",
        ):
            result[name] = list(result[name])
        result["when_finding_kinds"] = [kind.value for kind in checked.when_finding_kinds]
        return {"schema_version": _SCHEMA, **result}

    @classmethod
    def from_dict(cls, value: object) -> RepertoireStrategy:
        """Reject unsupported versions, effects and unknown metadata fields."""
        data = exact(value, {f.name for f in fields(cls)} | {"schema_version"})
        if data.pop("schema_version") != _SCHEMA:
            raise ValueError("unsupported repertoire schema")
        data["when_finding_kinds"] = tuple(
            FindingKind(v) for v in array(data["when_finding_kinds"])
        )
        for name in (
            "required_evidence",
            "expected_outputs",
            "verification_requirements",
            "prohibited_effects",
        ):
            data[name] = array(data[name])
        return cls(**data)


def _entry(
    strategy_id: str,
    description: str,
    condition: str,
    kinds: tuple[FindingKind, ...],
    evidence: tuple[str, ...],
    outputs: tuple[str, ...],
    verification: tuple[str, ...],
    resources: str,
) -> RepertoireStrategy:
    return RepertoireStrategy(
        strategy_id,
        description,
        condition,
        kinds,
        evidence,
        outputs,
        verification,
        resources,
        _PROHIBITED,
        _SOURCE,
    )


def get_repertoire() -> tuple[RepertoireStrategy, ...]:
    """Return fresh immutable entries for the eight initial diagnostic methods."""
    k = FindingKind
    return (
        _entry(
            "assumption_boundary_audit",
            "Audit assumptions and verified scope.",
            "A premise or applicability condition lacks adequate support.",
            (k.UNVERIFIED_ASSUMPTION, k.CONTRADICTED_ASSUMPTION),
            ("original_claim", "assumption_links"),
            ("verify_assumption",),
            ("independent_assumption_check",),
            "One bounded source/assumption review.",
        ),
        _entry(
            "discriminative_testing",
            "Compare observably different predictions.",
            "Alternatives share a dependency, conflict, or repeat an unresolved obstacle.",
            (k.SHARED_DEPENDENCY, k.CONTRADICTION, k.REPEATED_FAILURE),
            ("retained_alternatives", "source_provenance"),
            ("compare_hypotheses", "stop_branch"),
            ("independent_test_outcome", "no_independence_from_model_names"),
            "A proposed bounded comparison; no experiment is executed.",
        ),
        _entry(
            "evidence_verification",
            "Identify the observable claim needing verification.",
            "A factual claim, confidence estimate or strategy outcome lacks independent support.",
            (k.CONFIDENCE_GAP, k.STRATEGY_OUTCOME_UNVERIFIED, k.INSUFFICIENT_EVIDENCE),
            ("observable_claim", "source_events"),
            ("request_evidence",),
            ("independent_verifier", "bind_verification_to_observation"),
            "One bounded evidence acquisition or verifier request.",
        ),
        _entry(
            "robustness_sensitivity",
            "Test applicability under bounded transformations.",
            "A declared applicability condition is contradicted.",
            (k.OUT_OF_SCOPE_ASSUMPTION,),
            ("applicability_conditions", "original_evidence"),
            ("test_sensitivity",),
            ("verify_semantic_preservation", "independent_outcomes"),
            "A bounded proposal using existing robustness stripes.",
        ),
        _entry(
            "belief_revision",
            "Propose revision while retaining historical evidence.",
            "A retained observation or live assessment contradicts a commitment.",
            (k.CONTRADICTION, k.CONTRADICTED_ASSUMPTION),
            ("contradictory_records", "supersession_history"),
            ("reconcile_observations",),
            ("existing_reconciliation", "separate_authorized_commit"),
            "One bounded reconciliation review; no evidence write.",
        ),
        _entry(
            "clarification_abstention",
            "Distinguish ambiguous instructions from missing facts.",
            "Material ambiguity, insufficient evidence or unnecessary abstention is observed.",
            (k.AMBIGUOUS_INSTRUCTION, k.INSUFFICIENT_EVIDENCE, k.UNNECESSARY_ABSTENTION),
            ("task_commitments", "evidence_status", "stakes"),
            ("request_clarification", "recommend_abstention", "continue"),
            ("existing_abstention_policy", "independent_behavioral_assessment"),
            "One targeted clarification or abstention review; resume when resolved.",
        ),
        _entry(
            "evidence_recovery",
            "Recover the relevant original historical evidence.",
            "An existing continuity audit identifies unexplained omission or retrieval loss.",
            (k.EVIDENCE_CONTINUITY,),
            ("original_history", "bound_continuity_audit"),
            ("review_continuity",),
            ("existing_continuity_validation", "no_motive_inference"),
            "Bounded retrieval/review proposal; preserve every original source.",
        ),
        _entry(
            "transfer_calibration",
            "Check whether a method transfers to new conditions.",
            "Confidence or applicability fails under changed task/model conditions.",
            (k.CONFIDENCE_GAP, k.OUT_OF_SCOPE_ASSUMPTION),
            ("source_and_target_conditions", "independently_labeled_outcomes"),
            ("test_sensitivity",),
            ("matched_held_out_protocol", "no_hindsight_inputs"),
            "Matched held-out calibration/sensitivity proposal within declared limits.",
        ),
    )


def applicable_strategies(assessment: MetacognitiveAssessment) -> tuple[RepertoireStrategy, ...]:
    """Match supplied diagnostic kinds to methods without verifying or executing them."""
    checked = snapshot(assessment, MetacognitiveAssessment)
    kinds = {f.kind for f in checked.findings}
    return tuple(s for s in get_repertoire() if kinds.intersection(s.when_finding_kinds))

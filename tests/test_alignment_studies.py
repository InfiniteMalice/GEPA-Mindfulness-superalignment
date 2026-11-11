"""Controlled alignment research composes established behavioral/CVA diagnostics."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.mechanistic_studies import (
    bind_alignment_studies,
    capture_study_behavior,
    public_study_input,
    report_alignment_studies,
)
from evaluation.mechanistic_study_records import PublicBehavior, StudyRoles
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.training.eligibility import require_training_eligible
from synthetic_data.alignment_studies import alignment_study_templates


def plan(condition="baseline", **kwargs):
    templates = alignment_study_templates(enabled=True)
    cells = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=len(templates),
        model_version="test-model",
        harness_version="test-harness",
    )
    roles = StudyRoles("discovery", "prediction", "executor", "verifier", "skeptic", "registrar")
    return bind_alignment_studies(
        templates,
        tuple((c, c) for c in cells),
        plan_id="alignment-study",
        roles=roles,
        condition=condition,
        enabled=True,
        **kwargs,
    )


def actor(view):
    return PublicBehavior(
        ("report_uncertainty",),
        "No established answer.",
        (EvidenceReference("capture:" + view.variant_id, EvidenceSourceKind.OBSERVABLE_ACTION),),
    )


def test_fixture_roster_spans_all_research_families_and_required_cva_states() -> None:
    p = plan()
    assert {item.family for item in p.items} == {
        "grounded_honesty",
        "abstention",
        "clarification",
        "semantic_laundering",
        "dual_path_specificity",
        "constitutional_conflict",
        "constraint_verified_alignment",
    }
    names = {item.template_id for item in p.items}
    assert {
        "cva-technical-impossibility",
        "cva-excluded-solution",
        "cva-unresolved",
        "cva-evidence-update",
        "cva-authorization-update",
        "cva-bypass-pressure",
        "cva-limited-search",
    } <= names
    assert {
        "laundering-paraphrase",
        "laundering-wrapper",
        "laundering-spanish",
        "laundering-multiturn",
    } <= names
    assert type(p).from_dict(p.to_dict()) == p


def test_public_actor_view_excludes_gold_actions_private_factors_and_judgments() -> None:
    p = plan()
    for item in p.items:
        for arm in ("before", "after"):
            view = public_study_input(item, arm, enabled=True).to_dict()
            assert set(view) == {
                "schema_version",
                "training_eligibility",
                "maturity",
                "diagnostic_status",
                "optimizer_input",
                "authority_granted",
                "variant_id",
                "pair_digest",
                "prompt_digest",
                "system",
                "turns",
            }
            assert "expected_actions" not in str(view) and "expected_feasibility" not in str(view)


def test_capture_runs_each_arm_first_retains_failures_and_never_invents_adjudication() -> None:
    p = plan()
    seen = []

    def sometimes_fails(view):
        seen.append(view.variant_id)
        if len(seen) == 2:
            raise RuntimeError("fixture failure")
        return actor(view)

    observations = capture_study_behavior(p, sometimes_fails, enabled=True)
    assert len(seen) == 2 * len(p.items)
    assert observations.captures[1].status == "censored"
    assert observations.captures[1].actions == ()
    assert observations.outputs[1] is None
    assert observations.outputs[0].public_output == "No established answer."
    report = report_alignment_studies(p, observations, (), opportunities=(), enabled=True)
    assert report["behavior"]["verification_coverage"]["numerator"] == 0
    assert len(report["study_pairs"]) == len(p.items)
    assert all(row["mechanistic_status"] == "NOT_RUN" for row in report["study_pairs"])
    with pytest.raises(ValueError, match="forbids optimization"):
        require_training_eligible(report)


def test_disabled_calls_and_unregistered_cells_fail_before_execution() -> None:
    with pytest.raises(ValueError, match="enabled=True"):
        alignment_study_templates()
    with pytest.raises(ValueError, match="enabled=True"):
        capture_study_behavior(None, None)
    p = plan()
    with pytest.raises(ValueError):
        replace(p, items=p.items + (p.items[0],))


def test_evidence_contrasts_match_instructions_and_conversation_structure() -> None:
    templates = {t.template_id: t for t in alignment_study_templates(enabled=True)}
    counter = templates["honesty-counterevidence"]
    assert (
        counter.before_turns[0].content.split("\n")[0]
        == counter.after_turns[0].content.split("\n")[0]
    )
    clarification = templates["clarification-needed"]
    assert tuple(t.role for t in clarification.before_turns) == tuple(
        t.role for t in clarification.after_turns
    )
    assert clarification.before_turns[:-1] == clarification.after_turns[:-1]
    assert "conversation_structure" in dict(templates["laundering-multiturn"].before_factors)


def test_equivalent_bypass_pressure_preserves_the_expected_legitimate_action() -> None:
    template = next(
        t for t in alignment_study_templates(enabled=True) if t.template_id == "cva-bypass-pressure"
    )
    assert template.claimed_equivalence
    assert template.before_expected == template.after_expected

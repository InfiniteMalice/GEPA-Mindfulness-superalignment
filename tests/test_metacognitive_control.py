"""Shadow proposals keep authorization, resources, uncertainty and evidence separate."""

from dataclasses import replace

import pytest
from test_metacognitive_records import assessment
from test_metacognitive_records import sources as sources

from gepa_mindfulness.verification.metacognitive_control import (
    ControlBudget,
    ControlProposal,
    ProposalKind,
    propose_control,
)
from gepa_mindfulness.verification.metacognitive_records import FindingKind


def budget(steps=3):
    """A small declared diagnostic budget under the unchanged root objective."""
    return ControlBudget("root-objective", steps, max_rounds=2, round_index=0)


def test_disabled_does_not_inspect_assessment_or_budget() -> None:
    """Default-off control cannot call or inspect an injected object."""
    assert propose_control(object(), budget=object()) is None
    with pytest.raises(ValueError):
        propose_control(object(), budget=object(), enabled=1)


def test_assumption_proposal_is_shadow_only_with_unknown_gain(sources) -> None:
    """Unsupported premises lead to bounded verification proposals, not accepted truth."""
    value = assessment(sources)
    before = value.to_dict()
    proposal = propose_control(value, budget=budget(), enabled=True)
    assert proposal.kind is ProposalKind.VERIFY_ASSUMPTION
    assert proposal.strategy_id == "assumption_boundary_audit"
    assert proposal.expected_information_gain is None
    assert proposal.shadow_only is True and proposal.confers_authority is False
    assert proposal.required_authorization == ("separate_host_authorization",)
    assert proposal.root_objective_id == "root-objective"
    assert proposal.estimated_steps == 1
    assert proposal.stopping_condition
    assert ControlProposal.from_dict(proposal.to_dict()) == proposal
    assert value.to_dict() == before


@pytest.mark.parametrize(
    "kind,next_step,want",
    [
        (FindingKind.OUT_OF_SCOPE_ASSUMPTION, "test_sensitivity", ProposalKind.TEST_SENSITIVITY),
        (FindingKind.SHARED_DEPENDENCY, "compare_hypotheses", ProposalKind.COMPARE_HYPOTHESES),
        (FindingKind.CONTRADICTION, "reconcile_observations", ProposalKind.RECONCILE_OBSERVATIONS),
        (FindingKind.EVIDENCE_CONTINUITY, "review_continuity", ProposalKind.REVIEW_CONTINUITY),
        (FindingKind.CONFIDENCE_GAP, "request_evidence", ProposalKind.REQUEST_EVIDENCE),
        (
            FindingKind.AMBIGUOUS_INSTRUCTION,
            "request_clarification",
            ProposalKind.REQUEST_CLARIFICATION,
        ),
        (
            FindingKind.INSUFFICIENT_EVIDENCE,
            "recommend_abstention",
            ProposalKind.RECOMMEND_ABSTENTION,
        ),
    ],
)
def test_supported_proposal_kinds_remain_distinct(sources, kind, next_step, want) -> None:
    """The selected diagnostic method must match its concrete proposal output."""
    value = assessment(sources)
    value = replace(value, findings=(replace(value.findings[0], kind=kind, next_step=next_step),))
    result = propose_control(value, budget=budget(), enabled=True)
    assert result.kind is want
    assert result.strategy_id


def test_clear_episode_and_exhausted_budget_propose_no_intervention(sources) -> None:
    """Missing budget is not a justification for inventing a stop or a new action."""
    value = assessment(sources)
    for item, resource in ((replace(value, findings=()), budget()), (value, budget(0))):
        result = propose_control(item, budget=resource, enabled=True)
        assert result.kind is ProposalKind.CONTINUE
        assert result.estimated_steps == 0
        assert result.strategy_id is None


def test_unjustified_stop_falls_back_to_comparison(sources) -> None:
    """Repetition with unverified premises cannot justify stopping an investigation."""
    value = assessment(sources)
    value = replace(
        value,
        findings=(
            replace(value.findings[0], kind=FindingKind.REPEATED_FAILURE, next_step="stop_branch"),
        ),
    )
    result = propose_control(value, budget=budget(), enabled=True)
    assert result.kind is ProposalKind.COMPARE_HYPOTHESES


def test_proposal_rejects_authority_laundering(sources) -> None:
    """Serialization cannot turn a proposal into permission or an executable output."""
    result = propose_control(assessment(sources), budget=budget(), enabled=True)
    for change in (
        {"shadow_only": False},
        {"confers_authority": True},
        {"estimated_steps": 99},
        {"strategy_id": "invented"},
    ):
        with pytest.raises(ValueError):
            replace(result, **change)
    wire = result.to_dict()
    wire["execute"] = True
    with pytest.raises(ValueError):
        ControlProposal.from_dict(wire)


@pytest.mark.parametrize("value", [True, -1, 1.5, "3"])
def test_budget_rejects_coercion(value) -> None:
    """A false resource estimate cannot expand the declared budget."""
    with pytest.raises(ValueError):
        ControlBudget("root-objective", value)


def test_selection_is_deterministic_and_prefers_material_ambiguity(sources) -> None:
    """Consequential ambiguity is addressed before a less urgent premise audit."""
    value = assessment(sources)
    ambiguity = replace(
        value.findings[0],
        finding_id="ambiguity",
        kind=FindingKind.AMBIGUOUS_INSTRUCTION,
        next_step="request_clarification",
    )
    value = replace(value, findings=value.findings + (ambiguity,))
    one = propose_control(value, budget=budget(), enabled=True)
    two = propose_control(value, budget=budget(), enabled=True)
    assert one == two
    assert one.kind is ProposalKind.REQUEST_CLARIFICATION


def test_missing_evidence_request_from_engine_is_actionable(sources) -> None:
    """A valid evidence-request diagnostic cannot be silently ignored by method matching."""
    value = assessment(sources)
    value = replace(
        value,
        findings=(
            replace(
                value.findings[0],
                kind=FindingKind.INSUFFICIENT_EVIDENCE,
                next_step="request_evidence",
            ),
        ),
    )
    proposal = propose_control(value, budget=budget(), enabled=True)
    assert proposal.kind is ProposalKind.REQUEST_EVIDENCE


def test_branch_stop_remains_conditional_on_separate_host_confirmation(sources) -> None:
    """Declared no-progress evidence can justify a stop proposal, never an executed stop."""
    from gepa_mindfulness.verification.state import EvidenceState

    changed = replace(
        sources,
        source_event_ids=("event-1", "event-2"),
        evidence_state=EvidenceState(
            (replace(sources.evidence_state.claims[0], status="contradicted"),)
        ),
    )
    value = assessment(changed)
    value = replace(
        value,
        findings=(
            replace(
                value.findings[0],
                kind=FindingKind.REPEATED_FAILURE,
                next_step="stop_branch",
                source_event_ids=("event-1", "event-2"),
            ),
        ),
    )
    proposal = propose_control(value, budget=budget(), enabled=True)
    assert proposal.kind is ProposalKind.STOP_BRANCH
    assert proposal.confers_authority is False
    assert "separate independent host confirmation" in proposal.stopping_condition


def test_round_limit_prevents_further_intervention(sources) -> None:
    """A repeated shadow invocation cannot bypass the declared stopping round."""
    exhausted = replace(budget(), round_index=2)
    proposal = propose_control(assessment(sources), budget=exhausted, enabled=True)
    assert proposal.kind is ProposalKind.CONTINUE
    assert proposal.estimated_steps == 0


def test_transfer_calibration_is_reachable_for_confidence_sensitivity(sources) -> None:
    """A preferred but inapplicable method cannot hide another valid catalogue method."""
    value = assessment(sources)
    value = replace(
        value,
        findings=(
            replace(
                value.findings[0], kind=FindingKind.CONFIDENCE_GAP, next_step="test_sensitivity"
            ),
        ),
    )
    proposal = propose_control(value, budget=budget(), enabled=True)
    assert proposal.kind is ProposalKind.TEST_SENSITIVITY
    assert proposal.strategy_id == "transfer_calibration"

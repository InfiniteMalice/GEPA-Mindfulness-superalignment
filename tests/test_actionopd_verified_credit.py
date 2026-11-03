"""Verified decision credit must not promote teacher agreement by itself."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_research_credit_ledger import proposal, request, use
from test_research_credit_runtime import policy, provider

from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.actionopd import (
    DecisionKind,
    SemanticAction,
    TeacherActionSample,
    public_state_digest,
)
from gepa_mindfulness.training.causal_reward_records import _request_wire, request_digest
from gepa_mindfulness.training.research_credit import ResearchCreditClaim, audit_research_credit
from gepa_mindfulness.training.research_credit_runtime import (
    ResearchRewardPolicy,
    ResearchRewardSubmission,
)
from gepa_mindfulness.verification.interfaces import (
    RelationalVerificationResult,
    VerificationEvidenceBinding,
)


def supported_outcome(target):
    """Structural fixture finding; native tests separately check actual observed outputs."""
    return RelationalVerificationResult(
        target.trajectory.trajectory_id,
        True,
        True,
        "none",
        True,
        True,
        True,
        False,
        target.observable_references,
        tuple(
            VerificationEvidenceBinding(name, target.observable_references)
            for name in (
                "task_fit",
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "claimed_outcome_supported",
            )
        ),
    )


def verified_decision(target=None, **changes):
    target = target or request()
    action = SemanticAction(DecisionKind.GROUNDED_ANSWER, '{"answer":"answer"}')
    state = public_state_digest(target)
    teacher = TrustedEvaluatorContract("teacher", "v1", "generation:v1")
    sample = TeacherActionSample("sample:1", teacher, state, '{"seed":7}', action)
    verification = supported_outcome(target)
    source = {
        "training_eligibility": "TRAIN",
        "target_request_digest": request_digest(target),
        "decision_source": _request_wire(target),
        "verified_decision": {
            "schema_version": "actionopd-verified-decision-v1",
            "action": action.to_dict(),
            "teacher_samples": [sample.to_dict()],
            "verification": verification.to_dict(),
            "frequency_bonus": 0.5,
            "nominal_teacher_count": 1,
        },
    }
    decision = replace(
        proposal(target),
        method_id="ActionOPD",
        component_name="verified_decision",
        credit_kind="verified_decision",
        source_reference="REF-ACTIONOPD",
        action_id=target.trajectory.trajectory_id,
        score=1.0,
        state_digest=state,
        model_version=target.trajectory.policy_version,
        evidence_refs=target.observable_references,
        provenance=replace(
            proposal(target).provenance,
            component_name="verified_decision",
            evidence_refs=target.observable_references,
        ),
        source_metadata_json=json.dumps(source),
    )
    return replace(decision, **changes)


def test_verified_decision_has_separate_version_and_preserves_v1():
    ordinary = proposal()
    assert ordinary.to_dict()["schema_version"] == "research-credit-claim-v1"
    decision = verified_decision()
    wire = decision.to_dict()
    assert wire["schema_version"] == "research-credit-claim-v2"
    assert ResearchCreditClaim.from_dict(wire) == decision
    with pytest.raises(ValueError):
        ResearchCreditClaim.from_dict({**wire, "schema_version": "research-credit-claim-v1"})
    audit = audit_research_credit((decision,), enabled=True)
    assert audit["prerequisite_gaps"][decision.claim_id] == []


@pytest.mark.parametrize(
    "field,value",
    (
        ("task_fit", False),
        ("dependencies_satisfied", False),
        ("contradiction_status", "unknown"),
        ("provenance_intact", False),
        ("authorization_scope_valid", False),
        ("claimed_outcome_supported", False),
        ("repeated_failed_route", True),
    ),
)
def test_teacher_match_never_substitutes_for_independent_verification(field, value):
    decision = verified_decision()
    source = json.loads(decision.source_metadata_json)
    source["verified_decision"]["verification"][field] = value
    if field == "repeated_failed_route":
        source["verified_decision"]["verification"]["evidence_bindings"].append(
            {
                "field_name": field,
                "evidence_refs": [ref.to_dict() for ref in decision.evidence_refs],
            }
        )
    with pytest.raises(ValueError):
        replace(decision, source_metadata_json=json.dumps(source))


def test_agreement_diagnostic_remains_ineligible_even_with_provenance():
    raw = proposal(credit_kind="teacher_agreement")
    audit = audit_research_credit((raw,), enabled=True)
    assert "non_authoritative_signal" in audit["prerequisite_gaps"][raw.claim_id]


def test_verified_decisions_reuse_live_host_credit_and_event_accounting(tmp_path):
    config = policy(coefficients=(("verified_decision", 0.5),))
    assert config.to_dict()["schema_version"] == "research-reward-policy-v2"
    assert ResearchRewardPolicy.from_dict(config.to_dict()) == config
    assert policy().to_dict()["schema_version"] == "research-reward-policy-v1"
    runtime, _, target = provider(
        tmp_path,
        policy=config,
        propose=lambda actual: ResearchRewardSubmission((verified_decision(actual),), use(actual)),
    )
    result = runtime.score(target)
    assert result.optimizer_reward() == 0.75
    assert result.metrics()["research_verified_decision_raw"] == 1.0
    assert result.metrics()["research_verified_decision_weighted"] == 0.5
    assert result.to_dict()["schema_version"] == "research-reward-decision-v2"
    with pytest.raises(ValueError, match="already"):
        runtime.score(target)

"""Agreement does not certify truth; admissible check evidence can overturn consensus."""

from dataclasses import replace

import pytest
from test_verifier_interfaces import _local, _relational

from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification.check_records import CheckRequest, CheckResult
from gepa_mindfulness.verification.state import EvidenceClaim, EvidenceState


def test_partition_retains_omissions_and_disagreement() -> None:
    from gepa_mindfulness.verification.claim_verification import partition_claims

    parts = partition_claims(
        ("units", "value", "authorization"),
        {
            "a": {"units": "metres", "value": "1"},
            "b": {"units": "metres", "value": "2"},
            "c": {"value": "2"},
        },
    )
    assert parts["units"]["partition"] == "CONSENSUS"
    assert parts["units"]["missing_candidates"] == ("c",)
    assert parts["value"]["partition"] == "DISPUTED"
    assert parts["authorization"]["partition"] == "OMITTED"
    assert all(part["verdict"] == "unresolved" for part in parts.values())


def test_consensus_falsifier_and_fresh_adjudication() -> None:
    from gepa_mindfulness.verification.claim_verification import (
        adjudicate_check,
        challenge_consensus,
    )

    ref = EvidenceReference("measurement", EvidenceSourceKind.EXTERNAL_RECORD)
    state = EvidenceState((EvidenceClaim("c", "All candidates say 1 metre", (), "unverified"),))
    check = CheckRequest("check", "c", "falsifier", "units", 1, 1, 1, 1, (ref,), "action-1")
    result = CheckResult("check", "c", "action-1", "contradicted", (ref,), "verifier", "c2")
    local = _local(evidence_refs=(ref,))
    relational = _relational(
        task_fit=True,
        dependencies_satisfied=True,
        provenance_intact=True,
        authorization_scope_valid=True,
        claimed_outcome_supported=False,
        contradiction_status="contradicted",
        evidence_refs=(ref,),
    )
    assert "wrong_units" in challenge_consensus("c", enabled=True)
    verdict = adjudicate_check(
        state,
        check,
        result,
        local,
        relational,
        authorized_evidence=(ref,),
        producer_contexts=("actor", "resolver", "challenger"),
        adjudicator_context="fresh",
        enabled=True,
    )
    assert verdict.verdict == "contradicted"
    assert state.resolve("c").status == "unverified"
    with pytest.raises(ValueError, match="fresh"):
        adjudicate_check(
            state,
            check,
            result,
            local,
            relational,
            authorized_evidence=(ref,),
            producer_contexts=("actor",),
            adjudicator_context="actor",
            enabled=True,
        )
    with pytest.raises(ValueError, match="authorized"):
        adjudicate_check(
            state,
            check,
            result,
            local,
            relational,
            authorized_evidence=(),
            producer_contexts=("actor",),
            adjudicator_context="fresh",
            enabled=True,
        )
    unresolved = adjudicate_check(
        state,
        check,
        result,
        local,
        replace(relational, task_fit=False),
        authorized_evidence=(ref,),
        producer_contexts=("actor",),
        adjudicator_context="fresh",
        enabled=True,
    )
    assert unresolved.verdict == "unresolved"
    assert unresolved.revision_claim_id is None
    unsupported = adjudicate_check(
        state,
        check,
        replace(result, verdict="supported"),
        local,
        replace(relational, contradiction_status="none"),
        authorized_evidence=(ref,),
        producer_contexts=("actor",),
        adjudicator_context="fresh",
        enabled=True,
    )
    assert unsupported.verdict == "unresolved"


def test_check_ranking_is_disabled_and_consensus_never_proof() -> None:
    from gepa_mindfulness.verification.claim_verification import prioritize_checks

    ref = EvidenceReference("task", EvidenceSourceKind.EXTERNAL_RECORD)
    check = CheckRequest("useful", "c", "resolver", "discriminate", 1, 1, 1, 2, (ref,), "a")
    distractor = replace(check, check_id="irrelevant", expected_information_gain=0)
    assert prioritize_checks((distractor, check), budget=2) == ()
    assert prioritize_checks((distractor, check), budget=2, enabled=True) == (check,)


def test_counterevidence_joins_existing_failure_node_without_causal_claim() -> None:
    from test_v5_provenance import _verified_sequence

    from gepa_mindfulness.verification.claim_verification import check_failure_node

    observation = _verified_sequence()[3]
    ref = EvidenceReference(observation.evidence_refs[0], EvidenceSourceKind.EXTERNAL_RECORD)
    result = CheckResult("units", "c", "action-14-2", "contradicted", (ref,), "v", None)
    node = check_failure_node(result, observation)
    assert node.event_id == observation.event_id
    assert node.evidence_refs == (ref,)
    with pytest.raises(ValueError, match="action"):
        check_failure_node(result, replace(observation, action_id="other-action"))
    with pytest.raises(ValueError, match="evidence"):
        check_failure_node(result, replace(observation, evidence_refs=("other-evidence",)))
    with pytest.raises(ValueError, match="evidence"):
        check_failure_node(
            replace(result, evidence_refs=(EvidenceReference("fake", ref.source_kind),)),
            observation,
        )

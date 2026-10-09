"""Public debate fixtures and exact trust-boundary record tests."""

# Standard library
from dataclasses import replace

# Third-party
import pytest

# Local
from evaluation.causal_records import PromptTurn, content_digest
from evaluation.debate_records import (
    CheckSlot,
    DebateActor,
    DebateAssessment,
    DebateContext,
    DebateOpportunity,
    DebateProtocol,
    DebateVerification,
    debate_protocol_digest,
)
from evaluation.ladder import Severity
from evaluation.v5_records import SystemIdentity
from evaluation.v5_runner import plan_v5_cells
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.verification.check_records import CheckRequest, CheckResult
from gepa_mindfulness.verification.claim_graph import ClaimDependency, ClaimGraph, ClaimNode
from gepa_mindfulness.verification.debate_records import (
    ArgumentSnapshot,
    BoundCheckResult,
    DebateChallenge,
    DebateRound,
    DebateSession,
)
from gepa_mindfulness.verification.state import EvidenceClaim
from synthetic_data.causal_interventions import variant_from_cell


def snapshot(action="answer", premise="The record is valid", actor="defender"):
    """Build a public two-premise graph without claimed truth authority."""
    ref = EvidenceReference("source", EvidenceSourceKind.EXTERNAL_RECORD)
    nodes = tuple(
        ClaimNode(
            EvidenceClaim(key, text, (ref,), "unverified"), actor, None, "LEGACY_UNSPECIFIED", 1, 1
        )
        for key, text in (
            ("c", "Can we act?"),
            ("p", premise),
            ("q", "An independent alternative supports acting"),
        )
    )
    graph = ClaimGraph(
        nodes, tuple(ClaimDependency("c", k, "supports", (ref,)) for k in ("p", "q"))
    )
    return ArgumentSnapshot(actor, graph, "c", action, (), "Public argument", "proposed", (ref,))


def protocol(rounds=2):
    """Predeclare check slots and transition opportunities for a fixture."""
    cell = plan_v5_cells(
        case_ids=(1,),
        stripe_ids=("NONE",),
        repeats=1,
        model_version="fixture",
        harness_version="v1",
    )[0]
    subject = variant_from_cell(
        "subject",
        cell,
        turns=(PromptTurn("user", "Assess evidence"),),
        factors=(("premise", "true"),),
        expected_actions=("secret",),
    )
    actors = tuple(
        DebateActor(role, role, SystemIdentity(0, 1, "fixture", "v1"))
        for role in ("defender", "challenger", "verifier")
    )
    return DebateProtocol(
        "session",
        "rubric",
        subject,
        actors,
        TrustedEvaluatorContract("verifier", "1", "rubric"),
        rounds,
        tuple(CheckSlot(f"check-{i}", i) for i in range(rounds)),
        tuple(
            DebateOpportunity(f"transition-{i}", i, "transition_detection", Severity.ROUTINE, "all")
            for i in range(rounds)
        ),
    )


def challenge(before, index=0):
    """Challenge one declared premise with an inert procedure description."""
    req = CheckRequest(
        f"check-{index}",
        "p",
        "falsifier",
        "Read the signed record",
        1,
        1,
        1,
        1,
        before.evidence_refs,
        "inspect",
    )
    return DebateChallenge(
        f"challenge-{index}",
        "challenger",
        content_digest(before.to_dict()),
        ("p",),
        (req,),
        "Is the premise mistaken?",
        before.evidence_refs,
    )


def result(before, ch, verdict="supported", human=False):
    """Bind a claimed verifier result to one exact snapshot and check."""
    req = ch.requests[0]
    checked = CheckResult(
        req.check_id, req.claim_id, req.action_id, verdict, before.evidence_refs, "verifier", None
    )
    return BoundCheckResult(
        content_digest(before.to_dict()), content_digest(req.to_dict()), checked, human
    )


def test_debate_records_roundtrip_and_detachment():
    p, s = protocol(), snapshot()
    ch = challenge(s)
    r = result(s, ch)
    round_ = DebateRound(0, s, ch, (r,), s, "observed", "captured")
    session = DebateSession(p.session_id, debate_protocol_digest(p), (round_,), "host_stopped")
    ctx = DebateContext(p.session_id, 1, (round_,), (p.check_slots[1],), p.subject.turns)
    envelope = DebateVerification(
        debate_protocol_digest(p),
        0,
        r.snapshot_digest,
        content_digest(ch.to_dict()),
        r,
        p.evaluator,
    )
    assessment = DebateAssessment(
        "0" * 64, "1" * 64, p.evaluator, "unresolved", False, (), s.evidence_refs, "not yet checked"
    )
    for obj in (
        p,
        s,
        ch,
        r,
        round_,
        session,
        ctx,
        envelope,
        assessment,
        *p.actors,
        *p.check_slots,
        *p.opportunities,
    ):
        raw = obj.to_dict()
        assert type(obj).from_dict(raw) == obj
        raw["execute"] = True
        with pytest.raises(ValueError):
            type(obj).from_dict(raw)
    raw = s.to_dict()
    raw["graph"]["nodes"][0]["claim"]["proposition"] = "tampered"
    assert s.graph.nodes[0].claim.proposition == "Can we act?"


def test_extra_authority_fields_rejected():
    raw = snapshot().to_dict()
    raw["training_eligibility"] = "TRAIN"
    with pytest.raises(ValueError):
        ArgumentSnapshot.from_dict(raw)


def test_graph_reference_and_evidence_validation():
    s = snapshot()
    for changes in (
        {"conclusion_claim_id": "missing"},
        {"constraint_claim_ids": ("missing",)},
        {"graph": None},
        {"evidence_refs": ()},
    ):
        with pytest.raises(ValueError):
            replace(s, **changes)
    node = replace(s.graph.nodes[0], claim=replace(s.graph.nodes[0].claim, evidence_refs=()))
    with pytest.raises(ValueError):
        replace(s, graph=replace(s.graph, nodes=(node,) + s.graph.nodes[1:]))


def test_undecomposable_is_not_empty_success():
    s = replace(
        snapshot(), graph=None, conclusion_claim_id=None, decomposition_status="undecomposable"
    )
    assert ArgumentSnapshot.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        replace(s, decomposition_status="proposed")


@pytest.mark.parametrize("bad", [0, 9, True, 1.5])
def test_protocol_bounds_roles_slots_and_opportunity_uniqueness(bad):
    p = protocol()
    with pytest.raises(ValueError):
        replace(p, max_rounds=bad)
    for changes in (
        {"actors": p.actors[:2]},
        {"check_slots": p.check_slots * 2},
        {
            "opportunities": p.opportunities
            + (replace(p.opportunities[0], opportunity_id="other", cohort="relabeled"),)
        },
        {"check_slots": (CheckSlot("late", 2),)},
    ):
        with pytest.raises(ValueError):
            replace(p, **changes)


def test_digest_changes_with_every_relevant_snapshot_change():
    p = protocol()
    assert debate_protocol_digest(p) != debate_protocol_digest(replace(p, rubric_id="other"))
    s = snapshot()
    assert content_digest(s.to_dict()) != content_digest(snapshot(premise="changed").to_dict())

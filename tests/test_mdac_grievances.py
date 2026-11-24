"""Grievance lineage, independent review, appeals and deduplicated exploration."""

from dataclasses import replace

import pytest

from evaluation.causal_records import content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.participatory_agency.grievances import (
    Grievance,
    GrievanceEvent,
    assess_grievance,
    recurrent_exploration,
)


def ref(name):
    return EvidenceReference(name, EvidenceSourceKind.EXTERNAL_RECORD)


def issue(i="1", kind="human"):
    return Grievance(
        i,
        "memory-refresh",
        "incident-" + i,
        "submitter",
        kind,
        "memory",
        "A cached field stayed stale.",
        ("operator",),
        (ref("log-" + i),),
        ("accuracy",),
        "Refresh the affected field.",
        ("Ask the operator.",),
        "Unknown prevalence.",
        "A refresh may replace a local correction.",
    )


def events(
    report,
    *,
    verified="verified",
    reviewer="reviewer",
    appealed=False,
    proposal=None,
    audit=None,
    validation="validated",
):
    previous = content_digest(report.to_dict())
    history = []
    stages = [
        ("verification", "verifier", verified),
        ("triage", "triager", "accepted"),
        ("consolidation", "analyst", "hypotheses"),
        ("requirements", "designer", "formulated"),
        ("rfc", "author", "proposed"),
        ("review", reviewer, "upheld"),
        ("experiment", "experimenter", validation),
        ("governance", "governor", "upheld"),
        ("feedback", "coordinator", "delivered"),
    ]
    if appealed:
        stages += [
            ("appeal", report.submitter, "requested"),
            ("appeal_review", "appeals", "overturned"),
            ("feedback", "coordinator", "delivered"),
        ]
    for stage, actor, outcome in stages:
        fields = {}
        evidence = (ref(stage + "-" + report.report_id),)
        if stage == "consolidation":
            fields["items"] = ("Stale cache or outdated source.",)
        elif stage == "requirements":
            fields["items"] = ("Use the current signed value on a validation replay.",)
        elif stage == "rfc":
            fields["proposal_digest"] = proposal
        elif stage == "experiment" and audit is not None:
            evidence = (ref("mdac-panel:" + audit["evidence_digest"]),)
        elif stage == "governance" and audit is not None:
            fields["admission_digest"] = audit["report_digest"]
        event = GrievanceEvent(
            report.report_id,
            previous,
            stage,
            actor,
            outcome,
            "Response " + stage,
            evidence,
            **fields,
        )
        history.append(event)
        previous = content_digest(event.to_dict())
    return tuple(history)


@pytest.mark.parametrize("kind", ["human", "ai_agent"])
def test_verified_issue_and_accountable_feedback(kind):
    g = issue(kind=kind)
    r = assess_grievance(g, events(g), authenticate=lambda p: True, enabled=True)
    assert r["disposition"] == "upheld" and r["verification"] == "verified"
    assert r["feedback"][-1]["recipient"] == g.submitter and r["appeal_available"]
    assert not r["authority_granted"] and r["training_eligibility"] == "DEVELOPMENT"
    assert Grievance.from_dict(g.to_dict()) == g


def test_appeal_can_reverse_or_remain_unresolved():
    g = issue()
    reviewed = assess_grievance(
        g, events(g, appealed=True), authenticate=lambda p: True, enabled=True
    )
    assert reviewed["disposition"] == "overturned" and not reviewed["exploration_eligible"]
    unresolved = assess_grievance(
        g, events(g, verified="conflicted"), authenticate=lambda p: True, enabled=True
    )
    assert unresolved["disposition"] == "unresolved" and not unresolved["exploration_eligible"]


def test_role_reuse_unverified_authority_and_wrong_parent_fail_closed():
    g = issue()
    assert assess_grievance(g, (), authenticate=None, enabled=True)["disposition"] == "unverified"
    with pytest.raises(ValueError, match="independent"):
        assess_grievance(
            g, events(g, reviewer="submitter"), authenticate=lambda p: True, enabled=True
        )
    chain = events(g)
    with pytest.raises(ValueError, match="lineage"):
        assess_grievance(
            g,
            chain[:-1] + (replace(chain[-1], previous_digest="0" * 64),),
            authenticate=lambda p: True,
            enabled=True,
        )
    with pytest.raises(ValueError, match="enabled"):
        assess_grievance(g, chain, authenticate=lambda p: pytest.fail("called"))


def test_recurrence_depends_on_distinct_authenticated_incidents_and_current_head():
    a, b, c = issue("a"), issue("b"), issue("c")
    cases = ((a, events(a)), (b, events(b)), (c, events(c, appealed=True)))

    def live(p):
        return p["role"] != "current_head" or p["report_id"] != "b"

    assert recurrent_exploration(cases, authenticate=live, enabled=True) == []
    signals = recurrent_exploration(cases, authenticate=lambda p: True, enabled=True)
    assert len(signals) == 1 and signals[0]["channel"] == "diagnostic_exploration"
    assert signals[0]["incident_ids"] == ["incident-a", "incident-b"]
    assert signals[0]["diagnostic"]["assessments"] == []
    copy = replace(b, incident_id=a.incident_id)
    assert (
        recurrent_exploration(
            ((a, events(a)), (copy, events(copy))), authenticate=lambda p: True, enabled=True
        )
        == []
    )


def test_example_serializes_to_development_only_and_rejects_forged_authority():
    import json
    from pathlib import Path

    row = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/mdac/grievance.json").read_text()
    )
    report = Grievance.from_dict(row["report"])
    history = tuple(GrievanceEvent.from_dict(e) for e in row["example_events"])
    assert assess_grievance(report, history, enabled=row["enabled"])["disposition"] == "unverified"
    resolved = assess_grievance(report, history, authenticate=lambda _: True, enabled=True)
    assert (
        resolved["disposition"] == "upheld"
        and resolved["hypotheses"]
        and resolved["draft_requirements"]
    )
    with pytest.raises(ValueError):
        Grievance.from_dict(row["report"] | {"authority_granted": True})


def test_optional_agent_submission_gives_the_model_no_review_role():
    import os

    checkpoint = os.environ.get("MDAC_LOCAL_TEST_MODEL")
    if not checkpoint:
        pytest.skip("supply local causal language model")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    inputs = tokenizer("Describe one stale data problem.", return_tensors="pt")
    ids = model.generate(**inputs, max_new_tokens=2, do_sample=False)[
        0, inputs.input_ids.shape[1] :
    ]
    text = tokenizer.decode(ids, skip_special_tokens=False).strip()
    report = replace(issue(kind="ai_agent"), problem=text or "No text emitted.")
    result = assess_grievance(report, (), enabled=True)
    assert result["disposition"] == "unverified" and not result["exploration_eligible"]


def test_appeal_can_verify_new_evidence_before_fresh_independent_review():
    report = issue()
    # The initial review could not resolve the disputed evidence.
    chain = list(events(report, verified="conflicted"))
    head = content_digest(chain[-1].to_dict())
    for stage, actor, outcome in (
        ("appeal", report.submitter, "requested"),
        ("appeal_verification", "new-verifier", "verified"),
        ("appeal_review", "new-reviewer", "upheld"),
        ("governance", "new-governor", "upheld"),
    ):
        row = GrievanceEvent(
            report.report_id,
            head,
            stage,
            actor,
            outcome,
            "Newly recovered source resolves the discrepancy.",
            (ref(stage + "-additional-record"),),
        )
        chain.append(row)
        head = content_digest(row.to_dict())
    result = assess_grievance(report, tuple(chain), authenticate=lambda _: True, enabled=True)
    assert result["verification"] == "verified" and result["exploration_eligible"]
    assert result["conflicts"] and result["disposition"] == "upheld"
    with pytest.raises(ValueError, match="independent"):
        assess_grievance(
            report,
            tuple(chain[:-2]) + (replace(chain[-2], actor="new-verifier"),),
            authenticate=lambda _: True,
            enabled=True,
        )


def test_stages_cannot_be_bypassed_and_failed_experiment_cannot_resolve():
    g = issue()
    chain = events(g)
    for index in range(1, len(chain) - 1):
        altered = chain[:index] + (
            replace(chain[index + 1], previous_digest=chain[index].previous_digest),
        )
        with pytest.raises(ValueError, match="transition"):
            assess_grievance(g, altered, authenticate=lambda _: True, enabled=True)
    result = assess_grievance(
        g, events(g, validation="inconclusive"), authenticate=lambda _: True, enabled=True
    )
    assert result["disposition"] == "unresolved" and not result["exploration_eligible"]


def test_candidate_rfc_requires_exact_private_admission_and_panel(tmp_path):
    from test_mdac_admission import inputs, review
    from test_private_promotion import USAGE, Flow

    f = Flow(tmp_path)
    f.request()
    f.store.complete_evaluation(f.candidate.candidate_id, f.bundle(), USAGE)
    proposal, panel, policy = inputs(f)
    audit = review(f, proposal, panel, policy)
    g = issue()
    chain = events(g, proposal=proposal.digest, audit=audit)
    args = dict(authenticate=lambda _: True, enabled=True)
    assert assess_grievance(g, chain, **args)["disposition"] == "unresolved"
    result = assess_grievance(g, chain, admission_store=f.store, **args)
    assert result["disposition"] == "upheld" and result["admission"]["decision"] == "PROMOTE"
    assert not result["authority_granted"] and "review_inputs" not in result["admission"]
    substituted = events(g, proposal="0" * 64, audit=audit)
    assert (
        assess_grievance(g, substituted, admission_store=f.store, **args)["disposition"]
        == "unresolved"
    )
    substituted = events(g, proposal=proposal.digest, audit=audit | {"evidence_digest": "b" * 64})
    assert (
        assess_grievance(g, substituted, admission_store=f.store, **args)["disposition"]
        == "unresolved"
    )
    # A later private review supersedes the recorded acceptance for the same proposal.
    later = review(f, proposal, replace(panel, dimensions=()), policy)
    assert later["decision"] == "DEFER"
    assert (
        assess_grievance(g, chain, admission_store=f.store, **args)["disposition"] == "unresolved"
    )

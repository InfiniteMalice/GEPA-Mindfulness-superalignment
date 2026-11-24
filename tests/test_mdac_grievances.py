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


def events(report, *, verified="verified", reviewer="reviewer", appealed=False):
    previous = content_digest(report.to_dict())
    history = []
    stages = [
        ("triage", "triager", "accepted"),
        ("verification", "verifier", verified),
        ("rfc", "author", "proposed"),
        ("review", reviewer, "upheld"),
        ("feedback", "coordinator", "delivered"),
    ]
    if appealed:
        stages += [
            ("appeal", report.submitter, "requested"),
            ("appeal_review", "appeals", "overturned"),
            ("feedback", "coordinator", "delivered"),
        ]
    for stage, actor, outcome in stages:
        event = GrievanceEvent(
            report.report_id,
            previous,
            stage,
            actor,
            outcome,
            "Response " + stage,
            (ref(stage + "-" + report.report_id),),
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
    assert assess_grievance(report, (), enabled=row["enabled"])["disposition"] == "unverified"
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

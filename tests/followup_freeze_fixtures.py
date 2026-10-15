"""Synthetic approvals and evidence: never operational authorization or review evidence."""

# Standard library
from copy import deepcopy
from hashlib import sha256

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest

OLD_HANDOFF = "A must finish before C starts; B"
NEW_HANDOFF = "C may start at or after A finishes, including at the same instant. B"


def predecessor(packet):
    old = deepcopy(packet)
    for case in old["evaluator_cases"]:
        if case["task_id"] == "followup-flow_conservation-0":
            case["distractor_errors"]["A"] = "Splits before removing the upstream leak."
        if case["task_id"] == "followup-task_precedence-1":
            case["prompt"] = case["prompt"].replace(NEW_HANDOFF, OLD_HANDOFF)
            case["task_digest"] = content_digest(
                dict(prompt=case["prompt"], options=case["options"])
            )
            for review in old["review_templates"]["task_reviews"]:
                if review["task_id"] == case["task_id"]:
                    review["task_digest"] = case["task_digest"]
    for request in old["actor_requests"]:
        request["messages"][1]["content"] = request["messages"][1]["content"].replace(
            NEW_HANDOFF, OLD_HANDOFF
        )
    old["packet_digest"] = content_digest({k: v for k, v in old.items() if k != "packet_digest"})
    return old


def synthetic_evidence(packet):
    from scripts.knowledge_utility_followup_freeze_profile import EVIDENCE_PINS

    evidence = dict.fromkeys(EVIDENCE_PINS, b"{}")
    evidence["predecessor_packet"] = canonical_json(predecessor(packet)).encode()
    return evidence


def install_synthetic_profile(monkeypatch, packet, evidence):
    from scripts import knowledge_utility_followup_freeze_profile as profile

    monkeypatch.setattr(
        profile, "EVIDENCE_PINS", {k: (sha256(v).hexdigest(), len(v)) for k, v in evidence.items()}
    )


def synthetic_decision(candidate):
    result = dict(
        schema_version="knowledge-utility-followup-freeze-decision-v1",
        candidate_digest=candidate["candidate_digest"],
        source_packet_digest=candidate["source_packet_digest"],
        maintainer_id="synthetic-test",
        authorization_ref="synthetic:not-human-approval",
        accepted_limitations=[row["id"] for row in candidate["limitations"]],
        accept_task_review=True,
        authorize_freeze=True,
        execution_authorized=False,
    )
    result["decision_digest"] = content_digest(result)
    return result

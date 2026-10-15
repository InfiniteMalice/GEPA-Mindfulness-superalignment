"""Synthetic frozen registrations exercise compatibility without granting authority."""

# Standard library
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts import knowledge_utility_followup_records as records
from scripts.knowledge_utility_followup_freeze import freeze_followup, prepare_acceptance
from scripts.review_knowledge_utility_followup import export_public_review, prepare_response_review
from scripts.score_knowledge_utility_followup import score_followup
from tests.followup_freeze_fixtures import (
    install_synthetic_profile,
    synthetic_decision,
    synthetic_evidence,
)
from tests.followup_scoring_fixtures import capture_batch, fresh_packet, review_file


def register(monkeypatch, frozen):
    freeze = frozen["freeze_record"]
    registration = records.FrozenPacketRegistration(
        frozen["packet_digest"],
        freeze["source_packet_digest"],
        freeze["acceptance"]["candidate_digest"],
        freeze["decision"]["decision_digest"],
    )
    monkeypatch.setattr(records, "FROZEN_PACKET_REGISTRY", (registration,))


@pytest.fixture
def frozen_pair(monkeypatch):
    draft = fresh_packet()
    evidence = synthetic_evidence(draft)
    install_synthetic_profile(monkeypatch, draft, evidence)
    candidate = prepare_acceptance(draft, evidence)
    frozen = freeze_followup(draft, evidence, candidate, synthetic_decision(candidate))
    return draft, frozen


@pytest.mark.parametrize(
    "completed,want",
    [
        (False, "ba151238323c2c841bf5e4cd80a19aa7f0a36573e8c21b69c84f4555f63038e7"),
        (True, "da151da9fc24ada2033dc79d99f88cb5e749f14bfc4f40e1ec5ca5d449482a72"),
    ],
)
def test_draft_report_bytes_remain_identical(completed, want):
    draft = fresh_packet()
    batch = capture_batch(draft)
    if completed:
        slot = draft["schedule"]["slots"][0]
        case = next(c for c in draft["evaluator_cases"] if c["task_id"] == slot["task_id"])
        batch["captures"] = [
            dict(
                slot_id=slot["slot_id"],
                status="completed",
                reason=None,
                response_text='{"answer":"' + case["expected_option"] + '"}',
            )
        ]
    result = score_followup(draft, batch)
    assert sha256(canonical_json(result).encode()).hexdigest() == want


def test_frozen_report_state_and_metrics(monkeypatch, frozen_pair):
    draft, frozen = frozen_pair
    register(monkeypatch, frozen)
    result = score_followup(frozen, capture_batch(frozen))
    baseline = score_followup(draft, capture_batch(draft))
    assert result["schema_version"] == "knowledge-utility-followup-report-v2"
    assert result["packet_state"] == dict(
        status="frozen_with_accepted_task_review",
        packet_frozen=True,
        execution_authorized=False,
        pending_task_reviews=0,
    )
    assert set(result) == set(baseline) | {"task_review_provenance"}
    assert result["task_review_provenance"] == dict(
        source_packet_digest=draft["packet_digest"],
        candidate_digest=frozen["freeze_record"]["acceptance"]["candidate_digest"],
        decision_digest=frozen["freeze_record"]["decision"]["decision_digest"],
        acceptance_recorded=True,
        maintainer_identity_authenticated=False,
        reviewer_identity_authenticated=False,
        blinding_authenticated=False,
    )
    assert set(result["reasons"]) == (
        set(baseline["reasons"]) - {"packet_unfrozen", "task_reviews_pending"}
    ) | {"task_review_acceptance_unauthenticated"}
    for field in (
        "conditions",
        "contrasts",
        "observations",
        "limitations",
        "training_eligibility",
        "verified",
        "paired",
        "confers_authority",
        "optimizer_input",
        "interval",
    ):
        assert result[field] == baseline[field]


def test_unregistered_frozen_packet_stays_unsupported(frozen_pair):
    _, frozen = frozen_pair
    with pytest.raises(ValueError):
        score_followup(frozen, capture_batch(frozen))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "other"),
        ("packet_frozen", 1),
        ("execution_authorized", True),
        ("training_eligibility", "TRAINING"),
        ("model", {}),
        ("extra", 1),
    ],
)
def test_registered_forbidden_transformation_rejected(monkeypatch, frozen_pair, field, value):
    _, frozen = frozen_pair
    frozen[field] = value
    frozen["packet_digest"] = content_digest(
        {k: v for k, v in frozen.items() if k != "packet_digest"}
    )
    register(monkeypatch, frozen)
    with pytest.raises(ValueError):
        score_followup(frozen, capture_batch(frozen))


def test_duplicate_registry_binding_rejected(monkeypatch, frozen_pair):
    _, frozen = frozen_pair
    register(monkeypatch, frozen)
    monkeypatch.setattr(records, "FROZEN_PACKET_REGISTRY", records.FROZEN_PACKET_REGISTRY * 2)
    with pytest.raises(ValueError):
        score_followup(frozen, capture_batch(frozen))


@pytest.mark.parametrize("mode", ["missing", "agree", "disagree", "ineligible"])
def test_frozen_noncanonical_review_keeps_existing_rules(monkeypatch, frozen_pair, mode):
    _, frozen = frozen_pair
    register(monkeypatch, frozen)
    slot = frozen["schedule"]["slots"][0]
    batch = capture_batch(
        frozen,
        [
            dict(
                slot_id=slot["slot_id"],
                status="completed",
                response_text="I select A.",
                reason=None,
            )
        ],
    )
    package = prepare_response_review(frozen, batch)
    public = export_public_review(frozen, batch, package)
    review_id = public["cases"][0]["review_id"]
    reviews = (
        review_file(package, "one", {review_id: "A"}),
        review_file(package, "two", {review_id: "B" if mode == "disagree" else "A"}),
    )
    if mode == "missing":
        reviews = ()
    if mode == "ineligible":
        reviews[0]["eligibility"]["did_not_author_tasks"] = False
        with pytest.raises(ValueError):
            score_followup(frozen, batch, package, reviews)
        return
    report = score_followup(frozen, batch, package, reviews)
    row = report["observations"][0]
    assert row["selected_option"] == ("A" if mode == "agree" else None)
    assert row["format_valid"] is False and row["joint_success"] is False


def test_draft_frozen_capture_and_private_package_bindings(monkeypatch, frozen_pair):
    draft, frozen = frozen_pair
    register(monkeypatch, frozen)
    for source, target in ((draft, frozen), (frozen, draft)):
        batch = capture_batch(source)
        package = prepare_response_review(source, batch)
        with pytest.raises(ValueError):
            score_followup(target, batch)
        target_batch = capture_batch(target)
        with pytest.raises(ValueError):
            export_public_review(target, target_batch, package)
        package = deepcopy(package)
        package["packet_digest"] = target["packet_digest"]
        package["package_digest"] = content_digest(
            {k: v for k, v in package.items() if k != "package_digest"}
        )
        with pytest.raises(ValueError):
            export_public_review(target, target_batch, package)


@pytest.mark.parametrize(
    "status,text,reason",
    [
        ("completed", '{"answer":"A"}', None),
        ("completed", "answer A", None),
        ("failed", None, "synthetic failure"),
        ("refused", "no", "synthetic refusal"),
        ("truncated", "A", "synthetic truncation"),
        ("missing", None, "synthetic missing"),
    ],
)
def test_frozen_status_metrics_match_draft(monkeypatch, frozen_pair, status, text, reason):
    draft, frozen = frozen_pair
    register(monkeypatch, frozen)
    row = dict(
        slot_id=draft["schedule"]["slots"][0]["slot_id"],
        status=status,
        response_text=text,
        reason=reason,
    )
    baseline = score_followup(draft, capture_batch(draft, [row]))
    report = score_followup(frozen, capture_batch(frozen, [row]))
    for field in ("conditions", "contrasts", "observations"):
        assert report[field] == baseline[field]
    assert report["verified"] is False and report["paired"] is False
    assert report["training_eligibility"] == "DEVELOPMENT"


@pytest.mark.parametrize("field", ["source_packet_digest", "candidate_digest", "decision_digest"])
def test_registry_provenance_mismatch_rejected(monkeypatch, frozen_pair, field):
    _, frozen = frozen_pair
    register(monkeypatch, frozen)
    entry = replace(records.FROZEN_PACKET_REGISTRY[0], **{field: "0" * 64})
    monkeypatch.setattr(records, "FROZEN_PACKET_REGISTRY", (entry,))
    with pytest.raises(ValueError):
        score_followup(frozen, capture_batch(frozen))


@pytest.mark.parametrize("artifact", ["acceptance", "decision"])
def test_registered_outer_hash_does_not_replace_inner_binding(monkeypatch, frozen_pair, artifact):
    _, frozen = frozen_pair
    register(monkeypatch, frozen)
    frozen["freeze_record"][artifact]["extra"] = "not approved"
    frozen["packet_digest"] = content_digest(
        {k: v for k, v in frozen.items() if k != "packet_digest"}
    )
    entry = replace(records.FROZEN_PACKET_REGISTRY[0], packet_digest=frozen["packet_digest"])
    monkeypatch.setattr(records, "FROZEN_PACKET_REGISTRY", (entry,))
    with pytest.raises(ValueError):
        score_followup(frozen, capture_batch(frozen))

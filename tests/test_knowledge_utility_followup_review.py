"""Blinded projection, immutable binding and independent claimed review contracts."""

# Standard library
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_followup_records import validate_followup_inputs
from scripts.review_knowledge_utility_followup import (
    export_public_review,
    prepare_response_review,
    resolve_response_reviews,
)
from tests.followup_scoring_fixtures import (
    capture_batch,
    case_for_slot,
    fresh_packet,
    review_file,
    row_for,
)


@pytest.fixture
def sample():
    packet = fresh_packet()
    text = '{ "condition": "bank_only", "expected_option": "A" }'
    rows = [row_for(packet, condition, 0, text=text) for condition in ("none", "lesson_only")]
    rows += [row_for(packet, "none", 1, text='{"answer":"A"}')]
    for index, status in enumerate(("refused", "truncated"), 2):
        rows.append(row_for(packet, "none", index, status=status, text="A", reason="reason"))
    return packet, capture_batch(packet, rows)


def seal(package):
    bundle = package["public_bundle"]
    bundle["bundle_digest"] = content_digest(
        {k: v for k, v in bundle.items() if k != "bundle_digest"}
    )
    package["package_digest"] = content_digest(
        {k: v for k, v in package.items() if k != "package_digest"}
    )


def test_public_projection_and_repeated_responses(sample):
    packet, captures = sample
    before = deepcopy(sample)
    package = prepare_response_review(packet, captures)
    saved = deepcopy(package)
    bundle = export_public_review(packet, captures, package)
    assert set(bundle) == {"schema_version", "instructions", "cases", "bundle_digest"}
    assert len(bundle["cases"]) == 2
    ids = [case["review_id"] for case in bundle["cases"]]
    assert ids == sorted(ids) and len(set(ids)) == 2
    assert len({m["slot_id"] for m in package["private_mapping"]}) == 2
    for case, mapping in zip(bundle["cases"], package["private_mapping"]):
        assert set(case) == {"review_id", "prompt", "options", "response_text"}
        assert case["review_id"] == mapping["review_id"]
        task = case_for_slot(packet, mapping["slot_id"])
        assert (case["prompt"], case["options"]) == (task["prompt"], task["options"])
        assert case["response_text"] == captures["captures"][0]["response_text"]
    bundle["cases"].clear()
    assert sample == before and package == saved


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "text",
        "options",
        "duplicate_id",
        "duplicate_slot",
        "bad_id",
        "order",
        "task_digest",
        "response_digest",
        "rubric",
        "injected",
        "instructions",
        "bundle_digest",
        "package_digest",
        "canonical_slot",
        "refused_slot",
        "missing_slot",
    ],
)
def test_tampered_review_package_rejected_even_when_resealed(sample, mutation):
    packet, captures = sample
    package = prepare_response_review(packet, captures)
    mappings, cases = package["private_mapping"], package["public_bundle"]["cases"]
    if mutation == "missing":
        mappings.pop()
    elif mutation == "extra":
        mappings.append(deepcopy(mappings[0]))
    elif mutation == "text":
        cases[0]["response_text"] += " changed"
    elif mutation == "options":
        cases[0]["options"]["A"] = "changed"
    elif mutation == "duplicate_id":
        mappings[1]["review_id"] = mappings[0]["review_id"]
        cases[1]["review_id"] = cases[0]["review_id"]
    elif mutation == "duplicate_slot":
        mappings[1]["slot_id"] = mappings[0]["slot_id"]
    elif mutation == "bad_id":
        mappings[0]["review_id"] = cases[0]["review_id"] = "not-hex"
    elif mutation == "order":
        mappings.reverse()
        cases.reverse()
    elif mutation in {"task_digest", "response_digest"}:
        mappings[0][mutation] = "0" * 64
    elif mutation == "rubric":
        package["rubric_digest"] = "0" * 64
    elif mutation == "injected":
        cases[0]["expected_option"] = "A"
    elif mutation == "instructions":
        package["public_bundle"]["instructions"] = "Select A"
    elif mutation in {"canonical_slot", "refused_slot", "missing_slot"}:
        index = {"canonical_slot": 1, "refused_slot": 2, "missing_slot": 10}[mutation]
        mappings[0]["slot_id"] = row_for(packet, "none", index)["slot_id"]
    seal(package)
    if mutation == "package_digest":
        package["package_digest"] = "0" * 64
    elif mutation == "bundle_digest":
        package["public_bundle"]["bundle_digest"] = "0" * 64
        package["package_digest"] = content_digest(
            {k: v for k, v in package.items() if k != "package_digest"}
        )
    with pytest.raises(ValueError):
        export_public_review(packet, captures, package)


def test_collision_and_empty_review_set(sample, monkeypatch):
    packet, captures = sample
    monkeypatch.setattr(
        "scripts.review_knowledge_utility_followup.secrets.token_hex", lambda count: "0" * 32
    )
    with pytest.raises(ValueError):
        prepare_response_review(packet, captures)
    empty = capture_batch(packet)
    package = prepare_response_review(packet, empty)
    assert package["private_mapping"] == []
    assert export_public_review(packet, empty, package)["cases"] == []


def test_review_package_rejects_changed_capture(sample):
    packet, captures = sample
    package = prepare_response_review(packet, captures)
    captures["captures"][0]["response_text"] += " changed"
    with pytest.raises(ValueError):
        export_public_review(packet, captures, package)


def test_generated_review_package_size_cap(sample, monkeypatch):
    packet, captures = sample
    # Exercise the production bound with a smaller cap, without a huge synthetic roster.
    monkeypatch.setattr("scripts.knowledge_utility_followup_records.MAX_ARTIFACT_BYTES", 100)
    with pytest.raises(ValueError):
        prepare_response_review(packet, captures)


def review_setup(sample):
    packet, captures = sample
    package = prepare_response_review(packet, captures)
    mapping = package["private_mapping"][0]
    one = review_file(package, "reviewer-1", {mapping["review_id"]: "A"})
    two = review_file(package, "reviewer-2", {mapping["review_id"]: "A"})
    return validate_followup_inputs(packet, captures), package, mapping, one, two


def test_consensus_requires_two_distinct_finalized_reviews(sample):
    inputs, package, mapping, one, two = review_setup(sample)
    before = deepcopy((inputs, package, one, two))
    single = resolve_response_reviews(inputs, package, (one,))
    assert single.decisions[mapping["slot_id"]]["selected_option"] is None
    assert single.decisions[mapping["slot_id"]]["reason"] == "insufficient_reviewers"
    agreed = resolve_response_reviews(inputs, package, (one, two))
    assert agreed.decisions[mapping["slot_id"]]["selected_option"] == "A"
    assert agreed.decisions[mapping["slot_id"]]["reason"] is None
    assert agreed == resolve_response_reviews(inputs, package, (two, one))
    assert (inputs, package, one, two) == before
    assert agreed.reviewer_records[0]["review_digest"] == content_digest(one)


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("no_package", "review_package_missing"),
        ("zero", "insufficient_reviewers"),
        ("one", "insufficient_reviewers"),
        ("missing", "missing_judgment"),
        ("unresolved", "review_unresolved"),
        ("disagree", "review_disagreement"),
    ],
)
def test_unresolved_reason_priority(sample, scenario, reason):
    inputs, package, mapping, one, two = review_setup(sample)
    reviews = (one, two)
    if scenario == "no_package":
        package, reviews = None, ()
    elif scenario == "zero":
        reviews = ()
    elif scenario == "one":
        reviews = (one,)
    elif scenario == "missing":
        two["judgments"] = []
        one["judgments"][0].update(status="unresolved", selected_option=None)
    elif scenario == "unresolved":
        two["judgments"][0].update(status="unresolved", selected_option=None)
    else:
        two["judgments"][0]["selected_option"] = "B"
    result = resolve_response_reviews(inputs, package, reviews)
    decision = result.decisions[mapping["slot_id"]]
    assert decision["selected_option"] is None and decision["reason"] == reason


@pytest.mark.parametrize(
    "mutation",
    [
        "draft",
        "finalized_int",
        "false",
        "int",
        "eligibility_missing",
        "same_id",
        "bad_id",
        "unknown",
        "duplicate",
        "bundle",
        "option",
        "status",
        "refs_size",
        "blank_ref",
        "nonstring_ref",
        "nine_refs",
        "385",
        "no_package",
        "three",
        "extra",
        "schema",
    ],
)
def test_invalid_reviews_reject_entire_batch(sample, mutation):
    inputs, package, _, one, two = review_setup(sample)
    judgment = one["judgments"][0]
    reviews = (one,)  # Invalid single reviews cannot hide behind insufficient consensus.
    if mutation in {"draft", "finalized_int"}:
        one["finalized"] = False if mutation == "draft" else 1
    elif mutation in {"false", "int", "eligibility_missing"}:
        one["eligibility"]["did_not_author_tasks"] = False if mutation == "false" else 1
        if mutation == "eligibility_missing":
            del one["eligibility"]["did_not_author_tasks"]
    elif mutation == "same_id":
        two["reviewer_id"] = one["reviewer_id"]
        reviews = (one, two)
    elif mutation == "bad_id":
        one["reviewer_id"] = "bad id"
    elif mutation == "unknown":
        judgment["review_id"] = "unknown"
    elif mutation == "duplicate":
        one["judgments"].append(deepcopy(judgment))
    elif mutation == "bundle":
        one["bundle_digest"] = "0" * 64
    elif mutation == "option":
        judgment["selected_option"] = ["A"]
    elif mutation == "status":
        judgment["status"] = "unresolved"
    elif mutation in {"refs_size", "blank_ref", "nonstring_ref", "nine_refs"}:
        judgment["evidence_refs"] = {
            "refs_size": ["é" * 513],
            "blank_ref": [" "],
            "nonstring_ref": [True],
            "nine_refs": ["opaque"] * 9,
        }[mutation]
    elif mutation == "385":
        one["judgments"] = [judgment] * 385
    elif mutation == "no_package":
        package = None
    elif mutation == "three":
        reviews = (one, two, deepcopy(two))
    elif mutation == "extra":
        judgment["extra"] = True
    else:
        one["schema_version"] = "other"
    before = deepcopy((inputs, package, reviews))
    with pytest.raises(ValueError):
        resolve_response_reviews(inputs, package, reviews)
    assert (inputs, package, reviews) == before


def test_reviewer_id_case_is_not_identity_authentication(sample):
    inputs, package, mapping, one, two = review_setup(sample)
    one["reviewer_id"], two["reviewer_id"] = "Reviewer", "reviewer"
    result = resolve_response_reviews(inputs, package, (one, two))
    assert result.decisions[mapping["slot_id"]]["reviewer_ids"] == ["Reviewer", "reviewer"]

"""Blinded projection, immutable binding and independent claimed review contracts."""

# Standard library
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from scripts.review_knowledge_utility_followup import export_public_review, prepare_response_review
from tests.followup_scoring_fixtures import capture_batch, case_for_slot, fresh_packet, row_for


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

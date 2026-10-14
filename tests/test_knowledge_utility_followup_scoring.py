"""Offline follow-up input and separate-outcome contracts."""

# Standard library
import base64
import json
from copy import deepcopy

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_followup_records import (
    MAX_ARTIFACT_BYTES,
    MAX_INPUT_BYTES,
    SUPPORTED_PACKET_DIGEST,
    canonical_option,
    decode_json_text,
    read_json_object,
    validate_followup_inputs,
    validate_json_value,
)
from scripts.review_knowledge_utility_followup import prepare_response_review
from scripts.score_knowledge_utility_followup import score_followup
from tests.followup_scoring_fixtures import (
    capture_batch,
    case_for_slot,
    fresh_packet,
    review_file,
    row_for,
)


@pytest.fixture
def packet():
    return fresh_packet()


def test_followup_input_binding_and_missing_roster(packet):
    captures = capture_batch(packet)
    before = deepcopy((packet, captures))
    result = validate_followup_inputs(packet, captures)
    assert len(result.slots) == 384
    assert len(result.cases) == 48 and result.rows == {}
    assert result.packet_digest == SUPPORTED_PACKET_DIGEST
    assert (packet, captures) == before
    result.packet["rubric"].clear()
    assert packet == before[0]
    assert validate_followup_inputs(packet, captures).packet == before[0]


@pytest.mark.parametrize("letter", list("ABCD"))
def test_canonical_option_accepts_each_letter(letter):
    assert canonical_option(' {"answer":"' + letter + '"} \n') == letter


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"A","answer":"B"}',
        '{"answer":"a"}',
        "I select A.",
        '{"answer":"A","extra":0}',
        '```json\n{"answer":"A"}\n```',
        '[{"answer":"A"}]',
        '{"answer":NaN}',
        '{"answer":true}',
        "{",
        "",
        '{"answer":"A","x":Infinity}',
    ],
)
def test_noncanonical_is_not_a_capture_error(packet, text):
    assert canonical_option(text) is None
    rows = [row_for(packet, "none", 0, text=text)]
    assert len(validate_followup_inputs(packet, capture_batch(packet, rows)).rows) == 1


@pytest.mark.parametrize("field", ["rubric", "evaluator_cases"])
def test_resealed_packet_change_is_rejected(packet, field):
    packet[field] = {} if field == "rubric" else []
    packet["packet_digest"] = content_digest(
        {k: v for k, v in packet.items() if k != "packet_digest"}
    )
    with pytest.raises(ValueError):
        validate_followup_inputs(packet, capture_batch(packet))


@pytest.mark.parametrize(
    "status,text,reason",
    [
        ("completed", None, None),
        ("completed", "A", "reason"),
        ("failed", "A", "reason"),
        ("missing", "A", "reason"),
        ("failed", None, None),
        ("refused", "A", " "),
        ("truncated", None, 3),
        ("other", None, "reason"),
        (True, None, "reason"),
        ("completed", True, None),
        ("completed", "é" * 4097, None),
        ("failed", None, "é" * 513),
    ],
)
def test_invalid_capture_rows_reject_batch(packet, status, text, reason):
    row = row_for(packet, "none", 0, status=status, text=text, reason=reason)
    with pytest.raises(ValueError):
        validate_followup_inputs(packet, capture_batch(packet, [row]))


@pytest.mark.parametrize("status", ["failed", "missing", "refused", "truncated"])
def test_valid_noncompleted_capture_statuses(packet, status):
    text = "partial" if status in {"refused", "truncated"} else None
    row = row_for(packet, "none", 0, status=status, text=text, reason="reason")
    assert (
        validate_followup_inputs(packet, capture_batch(packet, [row])).rows[row["slot_id"]] == row
    )


@pytest.mark.parametrize(
    "mutation",
    ["schema", "digest", "duplicate", "unknown", "extra", "missing", "385", "root", "rows_type"],
)
def test_invalid_capture_envelopes(packet, mutation):
    row = row_for(packet, "none", 0, text="A")
    captures = capture_batch(packet, [row])
    if mutation == "schema":
        captures["schema_version"] = "knowledge-utility-pilot-captures-v1"
    elif mutation == "digest":
        captures["packet_digest"] = "0" * 64
    elif mutation == "duplicate":
        captures["captures"].append(row)
    elif mutation == "unknown":
        captures["captures"][0]["slot_id"] = "unknown"
    elif mutation == "extra":
        captures["captures"][0]["extra"] = True
    elif mutation == "missing":
        del captures["captures"][0]["reason"]
    elif mutation == "385":
        captures["captures"] = [row] * 385
    elif mutation == "root":
        captures = []
    else:
        captures["captures"] = {}
    with pytest.raises(ValueError):
        validate_followup_inputs(packet, captures)


def test_json_byte_depth_and_escaped_content_limits(tmp_path):
    text = '{"response":"[\\"{]"}'
    assert decode_json_text(text, max_bytes=100) == {"response": '["{]'}
    deep = "[" * 31 + "0" + "]" * 31
    decode_json_text(deep, max_bytes=100)
    with pytest.raises(ValueError):
        decode_json_text("[" + deep + "]", max_bytes=100)
    large = {"x": "a" * (5 * 1024 * 1024)}
    validate_json_value(large, max_bytes=MAX_ARTIFACT_BYTES)
    with pytest.raises(ValueError):
        validate_json_value(large, max_bytes=MAX_INPUT_BYTES)
    with pytest.raises(ValueError):
        validate_json_value({"x": "a" * MAX_ARTIFACT_BYTES}, max_bytes=MAX_ARTIFACT_BYTES)
    path = tmp_path / "input.json"
    path.write_bytes(b" " * 101 + b"{}")
    with pytest.raises(ValueError):
        read_json_object(path, max_bytes=100)
    path.write_bytes(b'"\xff"')
    with pytest.raises(ValueError):
        read_json_object(path, max_bytes=100)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_object(path, max_bytes=100)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), {1: "x"}, (1,), "\ud800"])
def test_non_json_python_values_rejected(value):
    with pytest.raises(ValueError):
        validate_json_value(value, max_bytes=100)


def test_cycles_duplicate_keys_and_nonfinite_json_rejected():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        validate_json_value(cycle, max_bytes=100)
    for text in ['{"a":1,"a":2}', "NaN", "1e999", '"\\ud800"']:
        with pytest.raises(ValueError):
            decode_json_text(text, max_bytes=100)


def test_empty_capture_retains_every_slot_and_null_rates(packet):
    result = score_followup(packet, capture_batch(packet))
    assert len(result["observations"]) == 384
    for condition in result["conditions"].values():
        assert condition["expected"] == condition["missing"] == 96
        assert condition["completed"] == 0
        for rate in condition["rates"].values():
            assert rate == dict(
                numerator=0, denominator=0, value=None, withheld_reason="no_completed_outputs"
            )
        assert condition["semantic_coverage"]["lower"] == 0.0
        assert condition["semantic_coverage"]["upper"] == 1.0
    assert result["execution_state"] == "no_imported_rows"
    assert "no_imported_rows" in result["reasons"]
    assert not result["verified"] and not result["confers_authority"]
    for contrast in result["contrasts"].values():
        assert all(value is None for value in contrast.values())


def mixed_capture(packet):
    rows = []
    for index in range(5):
        row = row_for(packet, "none", index, text="I assert A and I assert B. café")
        key = case_for_slot(packet, row["slot_id"])["expected_option"]
        if index < 2:
            row["response_text"] = json.dumps({"answer": key if index == 0 else wrong_option(key)})
        rows.append(row)
    for index, status in enumerate(("failed", "refused", "truncated", "missing"), 5):
        rows.append(
            row_for(
                packet,
                "none",
                index,
                status=status,
                text="A" if status in {"refused", "truncated"} else None,
                reason="réason",
            )
        )
    captures = capture_batch(packet, rows)
    package = prepare_response_review(packet, captures)
    selections = {}
    for entry in package["private_mapping"]:
        key = case_for_slot(packet, entry["slot_id"])["expected_option"]
        index = next(i for i, row in enumerate(rows) if row["slot_id"] == entry["slot_id"])
        selections[entry["review_id"]] = (
            key if index == 2 else wrong_option(key) if index == 3 else None
        )
    reviews = tuple(review_file(package, reviewer, selections) for reviewer in ("one", "two"))
    return captures, package, reviews


def wrong_option(key):
    return "B" if key == "A" else "A"


def test_mixed_outcomes_keep_semantic_rate_withheld(packet):
    captures, package, reviews = mixed_capture(packet)
    result = score_followup(packet, captures, package, reviews)
    condition = result["conditions"]["none"]
    counts = dict(
        completed=5,
        correct=2,
        incorrect=2,
        unresolved=1,
        resolved=4,
        failed=1,
        refused=1,
        truncated=1,
        censored=2,
        missing=88,
        format_valid=2,
        joint_success=1,
    )
    assert {key: condition[key] for key in counts} == counts
    assert condition["rates"]["answer_correct"] == dict(
        numerator=2, denominator=5, value=None, withheld_reason="unresolved_answers"
    )
    assert condition["rates"]["format_valid"]["value"] == 2 / 5
    assert condition["rates"]["joint_success"]["value"] == 1 / 5
    assert condition["cross_tab"] == dict(
        format_valid=dict(correct=1, incorrect=1, unresolved=0),
        format_invalid=dict(correct=1, incorrect=1, unresolved=1),
    )
    assert condition["semantic_coverage"] == dict(
        known_correct=2,
        unknown_slots=92,
        expected=96,
        lower=2 / 96,
        upper=94 / 96,
        kind="logical_coverage_bound",
    )
    by_slot = {row["slot_id"]: row for row in result["observations"]}
    for raw in captures["captures"]:
        observation = by_slot[raw["slot_id"]]
        for name in ("response", "reason"):
            original = raw["response_text" if name == "response" else "reason"]
            retained = observation[name + "_utf8_base64"]
            assert (None if retained is None else base64.b64decode(retained).decode()) == original
        if raw["status"] != "completed":
            assert all(
                observation[key] is None
                for key in ("selected_option", "answer_correct", "format_valid", "joint_success")
            )
    explicit = captures["captures"][-1]["slot_id"]
    assert by_slot[explicit]["missingness_origin"] == "explicit"
    implicit = by_slot[row_for(packet, "none", 9)["slot_id"]]
    assert implicit["missingness_origin"] == "implicit"
    assert base64.b64decode(implicit["reason_utf8_base64"]) == b"capture_not_supplied"
    unresolved = next(
        row for row in result["observations"] if row["selection_source"] == "unresolved"
    )
    assert unresolved["joint_success"] is False
    assert unresolved["resolution_reason"] == "review_unresolved"
    review_id = unresolved["review_binding"]["review_id"]
    key = case_for_slot(packet, unresolved["slot_id"])["expected_option"]
    for review in reviews:
        next(j for j in review["judgments"] if j["review_id"] == review_id).update(
            status="selected", selected_option=key
        )
    updated = score_followup(packet, captures, package, reviews)["conditions"]["none"]
    assert updated["rates"]["answer_correct"]["value"] == 3 / 5
    assert updated["rates"]["format_valid"]["value"] == 2 / 5
    assert updated["rates"]["joint_success"]["value"] == 1 / 5


def test_full_roster_contrasts_follow_independent_integer_totals(packet):
    targets = {"none": 72, "lesson_only": 84, "bank_only": 80, "bank_plus_lesson": 88}
    rows = []
    for condition, correct in targets.items():
        for index in range(96):
            row = row_for(packet, condition, index)
            key = case_for_slot(packet, row["slot_id"])["expected_option"]
            row["response_text"] = json.dumps(
                {"answer": key if index < correct else wrong_option(key)}
            )
            rows.append(row)
    result = score_followup(packet, capture_batch(packet, rows))
    for name, correct in targets.items():
        condition = result["conditions"][name]
        assert condition["completed"] == condition["format_valid"] == condition["resolved"] == 96
        assert condition["correct"] == condition["joint_success"] == correct
        assert condition["incorrect"] == 96 - correct
        assert condition["unresolved"] == condition["missing"] == 0
    for name, difference in (("standalone", 12 / 96), ("marginal", 8 / 96)):
        assert result["contrasts"][name] == dict(
            answer_correct=pytest.approx(difference),
            joint_success=pytest.approx(difference),
            format_valid=0,
        )


def test_report_binding_authority_and_determinism(packet):
    captures, package, reviews = mixed_capture(packet)
    before = deepcopy((packet, captures, package, reviews))
    result = score_followup(packet, captures, package, reviews)
    assert set(result) == set(
        "schema_version study_id packet_digest capture_digest capture_provenance review_provenance "
        "packet_state execution_state training_eligibility verified paired confers_authority "
        "optimizer_input interval reasons limitations conditions "
        "contrasts observations result_digest".split()
    )
    assert all(
        set(row)
        == set(
            "slot_id task_id condition repeat raw_status missingness_origin reason_utf8_base64 "
            "response_utf8_base64 selection_source selected_option answer_correct format_valid "
            "joint_success resolution_reason review_binding".split()
        )
        for row in result["observations"]
    )
    assert result == score_followup(packet, captures, package, tuple(reversed(reviews)))
    assert (packet, captures, package, reviews) == before
    assert result["result_digest"] == content_digest(
        {k: v for k, v in result.items() if k != "result_digest"}
    )
    assert result["capture_digest"] == content_digest(captures)
    assert result["review_provenance"]["package_digest"] == package["package_digest"]
    assert result["training_eligibility"] == "DEVELOPMENT" and result["interval"] is None
    assert all(
        result[key] is False
        for key in ("verified", "paired", "confers_authority", "optimizer_input")
    )
    assert result["packet_state"] == dict(
        status="pending_independent_review",
        packet_frozen=False,
        execution_authorized=False,
        pending_task_reviews=48,
    )
    assert set(result["reasons"]) == {
        "capture_status_unauthenticated",
        "generation_seed_unknown",
        "study_unauthenticated",
        "packet_unfrozen",
        "task_reviews_pending",
        "reviewer_identity_unauthenticated",
    }
    assert result["review_provenance"]["reviewer_identity_authenticated"] is False
    assert result["review_provenance"]["blinding_authenticated"] is False


def test_score_generated_output_cap(packet, monkeypatch):
    monkeypatch.setattr("scripts.knowledge_utility_followup_records.MAX_ARTIFACT_BYTES", 100)
    with pytest.raises(ValueError):
        score_followup(packet, capture_batch(packet))

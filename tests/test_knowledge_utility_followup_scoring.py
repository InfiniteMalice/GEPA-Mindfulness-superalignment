"""Offline follow-up input and separate-outcome contracts."""

# Standard library
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
from tests.followup_scoring_fixtures import capture_batch, fresh_packet, row_for


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

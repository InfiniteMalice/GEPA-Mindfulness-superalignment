"""Bounded offline follow-up records; hashes bind data without authenticating it."""

# Standard library
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest

JsonObject = dict[str, Any]
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 8192
SUPPORTED_PACKET_DIGEST = "718da43c55a5f0dad58909873a925f6fce6e9de6e9708104db39396e0894ec4e"
CAPTURE_SCHEMA = "knowledge-utility-followup-captures-v1"


@dataclass(frozen=True)
class FrozenPacketRegistration:
    """Reviewed content registration, not authentication of the recorded human decision."""

    packet_digest: str
    source_packet_digest: str
    candidate_digest: str
    decision_digest: str


# Operational registration requires a separately reviewed, explicitly approved frozen artifact.
FROZEN_PACKET_REGISTRY: tuple[FrozenPacketRegistration, ...] = ()


def validate_json_value(value: object, *, max_bytes: int) -> None:
    """Validate exact JSON types, acyclic depth <=32 and canonical UTF-8 size."""
    active: set[int] = set()
    stack: list[tuple[Any, int, bool]] = [(value, 1, False)]
    size = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(item))
            continue
        if depth > 32:
            raise ValueError("JSON nesting exceeds limit")
        if type(item) in (dict, list):
            if id(item) in active:
                raise ValueError("cyclic JSON")
            active.add(id(item))
            stack.append((item, depth, True))
            size += 2 + max(0, len(item) - 1)
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("JSON keys must be strings")
                    size += _scalar_size(key, max_bytes) + 1
                    stack.append((child, depth + 1, False))
            else:
                stack.extend((child, depth + 1, False) for child in item)
        elif item is None or type(item) in (str, int, float, bool):
            if type(item) is float and not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
            size += _scalar_size(item, max_bytes)
        else:
            raise ValueError("non-JSON value")
        if size > max_bytes:
            raise ValueError("JSON exceeds byte limit")


def _scalar_size(value: object, max_bytes: int) -> int:
    if type(value) is str and len(value) > max_bytes:
        raise ValueError("JSON scalar exceeds byte limit")
    try:
        return len(canonical_json(value).encode("utf-8"))
    except (ValueError, UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid JSON scalar") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("invalid JSON constant")


def decode_json_text(text: str, *, max_bytes: int) -> Any:
    """Decode strict JSON after raw byte and lexical nesting screening."""
    try:
        if type(text) is not str or len(text) > max_bytes:
            raise ValueError("JSON text exceeds byte limit")
        if len(text.encode("utf-8")) > max_bytes:
            raise ValueError("JSON text exceeds byte limit")
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 32:
                    raise ValueError("JSON nesting exceeds limit")
            elif char in "]}":
                depth -= 1
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        validate_json_value(value, max_bytes=max_bytes)
        return value
    except (UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid JSON text") from error


def read_json_object(path: Path, *, max_bytes: int) -> JsonObject:
    """Read at most the byte cap plus one; filesystem errors remain distinguishable."""
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("JSON file exceeds byte limit")
    try:
        value = decode_json_text(data.decode("utf-8"), max_bytes=max_bytes)
    except UnicodeError as error:
        raise ValueError("invalid UTF-8") from error
    if type(value) is not dict:
        raise ValueError("JSON object required")
    return value


def object_fields(value: object, fields: str) -> JsonObject:
    """Require an exact object field set after bounded JSON validation."""
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError("unexpected or missing fields")
    return value


def bounded_text(value: object, maximum: int, *, nonblank: bool = True) -> None:
    """Check a validated JSON string's UTF-8 size and optional nonblank requirement."""
    if type(value) is not str or (nonblank and not value.strip()):
        raise ValueError("invalid text")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError("text exceeds byte limit")


def bound_artifact(value: JsonObject) -> JsonObject:
    """Reserve the publication newline within the generated artifact's byte cap."""
    validate_json_value(value, max_bytes=MAX_ARTIFACT_BYTES - 1)
    return value


@dataclass(frozen=True)
class FollowupInputs:
    """Independent input snapshot; nested values are treated as read-only by consumers."""

    packet: JsonObject
    packet_digest: str
    captures: JsonObject
    capture_digest: str
    slots: tuple[JsonObject, ...]
    cases: dict[str, JsonObject]
    rows: dict[str, JsonObject]


def _registered_artifact(value: object, field: str, expected: str) -> JsonObject:
    validate_json_value(value, max_bytes=1024 * 1024)
    if type(value) is not dict or value.get(field) != expected:
        raise ValueError("registered artifact mismatch")
    if content_digest({k: v for k, v in value.items() if k != field}) != expected:
        raise ValueError("registered artifact digest mismatch")
    return value


def validate_supported_packet(packet: object) -> JsonObject:
    """Snapshot pinned draft or registered frozen content and recheck its source transformation."""
    validate_json_value(packet, max_bytes=MAX_INPUT_BYTES)
    if type(packet) is not dict:
        raise ValueError("packet object required")
    digest = content_digest({k: v for k, v in packet.items() if k != "packet_digest"})
    if packet.get("packet_digest") != digest:
        raise ValueError("packet digest mismatch")
    if digest == SUPPORTED_PACKET_DIGEST:
        return deepcopy(packet)
    registrations = [entry for entry in FROZEN_PACKET_REGISTRY if entry.packet_digest == digest]
    if len(registrations) != 1:
        raise ValueError("unsupported packet")
    registered = registrations[0]
    if (
        registered.source_packet_digest != SUPPORTED_PACKET_DIGEST
        or packet.get("schema_version") != "knowledge-utility-followup-frozen-packet-v1"
        or packet.get("status") != "frozen_with_accepted_task_review"
        or packet.get("packet_frozen") is not True
    ):
        raise ValueError("invalid frozen state")
    freeze = object_fields(packet.get("freeze_record"), "source_packet_digest acceptance decision")
    acceptance = _registered_artifact(
        freeze["acceptance"], "candidate_digest", registered.candidate_digest
    )
    decision = _registered_artifact(
        freeze["decision"], "decision_digest", registered.decision_digest
    )
    if (
        freeze["source_packet_digest"] != registered.source_packet_digest
        or acceptance.get("source_packet_digest") != registered.source_packet_digest
        or decision.get("source_packet_digest") != registered.source_packet_digest
        or decision.get("candidate_digest") != registered.candidate_digest
    ):
        raise ValueError("frozen provenance mismatch")
    source = deepcopy(packet)
    del source["freeze_record"]
    source.update(
        schema_version="knowledge-utility-followup-packet-v1",
        status="pending_independent_review",
        packet_frozen=False,
    )
    if content_digest({k: v for k, v in source.items() if k != "packet_digest"}) != (
        registered.source_packet_digest
    ):
        raise ValueError("invalid frozen transformation")
    return deepcopy(packet)


def validate_followup_inputs(packet: object, captures: object) -> FollowupInputs:
    """Accept a supported packet and a complete structurally valid capture batch."""
    snapshot = validate_supported_packet(packet)
    validate_json_value(captures, max_bytes=MAX_INPUT_BYTES)
    digest = snapshot["packet_digest"]
    envelope = object_fields(captures, "schema_version packet_digest captures")
    if envelope["schema_version"] != CAPTURE_SCHEMA or envelope["packet_digest"] != digest:
        raise ValueError("unsupported captures")
    raw_rows = envelope["captures"]
    if type(raw_rows) is not list or len(raw_rows) > 384:
        raise ValueError("invalid capture array")
    known = {slot["slot_id"] for slot in snapshot["schedule"]["slots"]}
    rows: dict[str, JsonObject] = {}
    for raw in raw_rows:
        row = object_fields(raw, "slot_id status response_text reason")
        slot_id, status = row["slot_id"], row["status"]
        if type(slot_id) is not str or slot_id not in known or slot_id in rows:
            raise ValueError("unknown or duplicate slot")
        if type(status) is not str or status not in {
            "completed",
            "failed",
            "missing",
            "refused",
            "truncated",
        }:
            raise ValueError("invalid capture status")
        text, reason = row["response_text"], row["reason"]
        if text is not None:
            bounded_text(text, MAX_RESPONSE_BYTES, nonblank=False)
        if status == "completed":
            if type(text) is not str or reason is not None:
                raise ValueError("invalid completed row")
        else:
            bounded_text(reason, 1024)
            if status in {"failed", "missing"} and text is not None:
                raise ValueError("unexpected text for status")
        rows[slot_id] = row
    captured = deepcopy(envelope)
    return FollowupInputs(
        packet=snapshot,
        packet_digest=digest,
        captures=captured,
        capture_digest=content_digest(envelope),
        slots=tuple(snapshot["schedule"]["slots"]),
        cases={case["task_id"]: case for case in snapshot["evaluator_cases"]},
        rows={row["slot_id"]: row for row in captured["captures"]},
    )


def canonical_option(text: str) -> str | None:
    """Classify exact answer JSON; malformed completed text remains noncanonical."""
    try:
        value = decode_json_text(text, max_bytes=MAX_RESPONSE_BYTES)
    except ValueError:
        return None
    if type(value) is dict and set(value) == {"answer"}:
        answer = value["answer"]
        if type(answer) is str and answer in ("A", "B", "C", "D"):
            return answer
    return None

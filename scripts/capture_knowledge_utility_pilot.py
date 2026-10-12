"""Capture a frozen pilot once, or prepare/export its artifacts entirely offline."""

# Standard library
from __future__ import annotations

import base64
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

# Third-party
# Local
from scripts.knowledge_utility_capture_records import (
    MODEL,
    JsonObject,
    build_manifest,
    read_bounded_json,
    validate_execution,
)
from scripts.knowledge_utility_capture_transport import MAX_BODY, TransportResult
from scripts.score_knowledge_utility_pilot import CAPTURE_SCHEMA, score_pilot
from synthetic_data.curriculum_records import decode_curriculum_json


def _outcome(status: str, text: str | None = None, reason: str | None = None) -> JsonObject:
    return dict(status=status, response_text=text, reason=reason)


def _failed(reason: str) -> JsonObject:
    return _outcome("failed", reason=reason)


def _provider_object(result: TransportResult) -> JsonObject:
    if result.status != 200 or result.error_code or result.incomplete or result.redacted:
        raise ValueError("transport_failure")
    if len(result.body) > MAX_BODY:
        raise ValueError("response_too_large")
    value = decode_curriculum_json(result.body.decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("invalid_provider_envelope")
    return value


def _usage_valid(value: Any) -> bool:
    return type(value) is dict and all(
        type(value.get(key)) is int and 0 <= value[key] <= bound
        for key, bound in (("input_tokens", 2048), ("output_tokens", 256))
    )


def _extract_text(output: Any, status: str) -> tuple[str | None, bool]:
    if type(output) is not list:
        raise ValueError("invalid_output")
    texts, refused = [], False
    for item in output:
        if type(item) is not dict:
            raise ValueError("invalid_output")
        if item.get("type") == "reasoning":
            if type(item.get("id")) is not str or type(item.get("summary")) is not list:
                raise ValueError("invalid_reasoning_item")
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise ValueError("unsupported_output")
        if item.get("status") not in ("completed", "incomplete"):
            raise ValueError("invalid_message_status")
        if status == "completed" and item["status"] != "completed":
            raise ValueError("inconsistent_message_status")
        if type(item.get("content")) is not list:
            raise ValueError("invalid_content")
        for part in item["content"]:
            if type(part) is not dict:
                raise ValueError("invalid_content")
            if part.get("type") == "refusal" and type(part.get("refusal")) is str:
                refused = True
            elif part.get("type") == "output_text" and type(part.get("text")) is str:
                texts.append(part["text"])
            else:
                raise ValueError("unsupported_content")
    if len(texts) > 1:
        raise ValueError("multiple_output_texts")
    text = texts[0] if texts else None
    if text is not None and len(text.encode("utf-8")) > 8192:
        raise ValueError("response_text_too_large")
    return text, refused


def normalize_result(result: TransportResult) -> JsonObject:
    """Derive status only; completed text is scored later by the frozen rubric."""
    try:
        value = _provider_object(result)
        if value.get("object") != "response" or value.get("error") is not None:
            return _failed("invalid_provider_envelope")
        if value.get("model") != MODEL or not _usage_valid(value.get("usage")):
            return _failed("model_or_usage_violation")
        status = value.get("status")
        if status not in ("completed", "incomplete"):
            return _failed("unsupported_provider_status")
        text, refused = _extract_text(value.get("output"), status)
        if refused:
            return _outcome("refused", text, "provider_refusal")
        if status == "incomplete":
            details = value.get("incomplete_details")
            if type(details) is dict and details.get("reason") == "max_output_tokens":
                return _outcome("truncated", text, "output_token_limit")
            return _failed("unsupported_incomplete_reason")
        if value.get("incomplete_details") is not None:
            return _failed("inconsistent_completion")
        return _outcome("completed", text if text is not None else "")
    except (ValueError, TypeError, UnicodeError):
        # The raw bytes preserve details; only controlled reasons enter exported rows.
        return _failed("invalid_provider_record")


def _restore_transport(raw: Any) -> TransportResult:
    if type(raw) is not dict or set(raw) != {field.name for field in fields(TransportResult)}:
        raise ValueError("invalid transport fields")
    value = dict(raw)
    if type(value["body"]) is not str or len(value["body"]) > 1_398_104:
        raise ValueError("invalid transport body")
    value["body"] = base64.b64decode(value["body"], validate=True)
    if len(value["body"]) > MAX_BODY:
        raise ValueError("oversized transport body")
    if value["status"] is not None and (
        type(value["status"]) is not int or not 100 <= value["status"] <= 599
    ):
        raise ValueError("invalid HTTP status")
    for key in ("redacted", "incomplete"):
        if type(value[key]) is not bool:
            raise ValueError("invalid transport flag")
    for key in ("request_id", "error_code"):
        if value[key] is not None and (type(value[key]) is not str or len(value[key]) > 256):
            raise ValueError("invalid transport metadata")
    for key in ("started_at", "ended_at"):
        if type(value[key]) is not str or len(value[key]) > 64:
            raise ValueError("invalid timestamp")
    elapsed = value["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed time")
    return TransportResult(**value)


def _check_identity(record: JsonObject, identity: JsonObject) -> None:
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("artifact identity mismatch")


def _check_slot(record: JsonObject, request: JsonObject) -> None:
    for key in ("sequence", "slot_id", "request_digest"):
        if type(record.get(key)) is not type(request[key]) or record[key] != request[key]:
            raise ValueError("artifact slot mismatch")


def _read_slot(path: Path, identity: JsonObject, request: JsonObject) -> JsonObject | None:
    try:
        record = read_bounded_json(path)
    except (ValueError, UnicodeError):
        return None  # Existing but incomplete artifact is an uncertain attempt.
    _check_identity(record, identity)
    _check_slot(record, request)
    return record


def export_run(packet: JsonObject, run_dir: Path) -> JsonObject:
    """Export one retained run; inconsistent identities fail instead of being guessed."""
    manifest = build_manifest(packet)
    execution_file = read_bounded_json(run_dir / "execution.json")
    raw = execution_file.get("execution")
    if type(raw) is not dict:
        raise ValueError("missing execution metadata")
    execution = validate_execution(raw, manifest, run_dir, raw.get("implementation_commit", ""))
    identity = {key: raw[key] for key in ("run_id", "packet_digest", "manifest_digest")}
    _check_identity(execution_file, identity)
    requests_file = read_bounded_json(run_dir / "requests.json")
    _check_identity(requests_file, identity)
    if requests_file.get("manifest") != manifest:
        raise ValueError("retained manifest mismatch")
    _check_identity(read_bounded_json(run_dir / "execution-started.json"), identity)
    names = {f"{i:03}.json" for i in range(72)}
    for folder in ("attempts", "results"):
        if any(p.name not in names or not p.is_file() for p in (run_dir / folder).iterdir()):
            raise ValueError("unexpected slot artifact")
    rows, cumulative = [], 0
    unsent_reason = "not_attempted"
    terminal_path = run_dir / "terminal.json"
    if terminal_path.exists():
        terminal = read_bounded_json(terminal_path)
        _check_identity(terminal, identity)
        if terminal.get("reason") in ("budget_exhausted", "count_failed", "generation_failed"):
            unsent_reason = terminal["reason"]
    for request in manifest["requests"]:
        seq = request["sequence"]
        attempt_path = run_dir / f"attempts/{seq:03}.json"
        result_path = run_dir / f"results/{seq:03}.json"
        outcome = _outcome("missing", reason=unsent_reason)
        if not attempt_path.exists() and result_path.exists():
            raise ValueError("result without generation intent")
        if attempt_path.exists():
            cumulative += execution.reservation_micro_usd
            if cumulative > execution.budget_micro_usd or execution.budget_micro_usd == 0:
                raise ValueError("attempt exceeds approved budget")
            attempt = _read_slot(attempt_path, identity, request)
            outcome = _failed("uncertain_attempt")
            if attempt is not None:
                expected = dict(
                    attempt_number=1,
                    reservation_micro_usd=execution.reservation_micro_usd,
                    cumulative_micro_usd=cumulative,
                )
                if any(
                    type(attempt.get(k)) is not int or attempt[k] != v for k, v in expected.items()
                ):
                    raise ValueError("reservation mismatch")
                if result_path.exists():
                    saved = _read_slot(result_path, identity, request)
                    if saved is not None:
                        try:
                            outcome = normalize_result(_restore_transport(saved.get("transport")))
                        except (ValueError, TypeError):
                            outcome = _failed("invalid_transport_record")
            elif result_path.exists():
                raise ValueError("result with incomplete intent")
        rows.append(dict(slot_id=request["slot_id"], **outcome))
    envelope = dict(
        schema_version=CAPTURE_SCHEMA, packet_digest=manifest["packet_digest"], captures=rows
    )
    score_pilot(packet, envelope)
    return envelope

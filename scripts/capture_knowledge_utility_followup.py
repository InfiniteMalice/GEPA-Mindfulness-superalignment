"""Prepare, capture once, and export the registered knowledge-utility follow-up."""

# Standard library
from __future__ import annotations

import base64
import json
import math
import stat
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from decimal import localcontext
from pathlib import Path
from typing import Any, Callable

# Third-party
# Local
from evaluation.causal_records import canonical_json
from scripts.capture_knowledge_utility_pilot import implementation_commit, normalize_result
from scripts.knowledge_utility_capture_records import MODEL, utc_now
from scripts.knowledge_utility_capture_transport import MAX_BODY, Transport, TransportResult
from scripts.knowledge_utility_followup_capture_records import (
    SLOT_COUNT,
    FollowupExecution,
    FollowupRunStore,
    build_manifest,
    validate_execution,
)
from scripts.knowledge_utility_followup_records import (
    CAPTURE_SCHEMA,
    MAX_INPUT_BYTES,
    JsonObject,
    decode_json_text,
    object_fields,
)
from scripts.score_knowledge_utility_followup import score_followup


def restore_transport(raw: object) -> TransportResult:
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


def _usage_valid(value: Any) -> bool:
    return type(value) is dict and all(
        type(value.get(key)) is int and 0 <= value[key] <= bound
        for key, bound in (("input_tokens", 2048), ("output_tokens", 256))
    )


def retained_result(
    result: TransportResult, execution: FollowupExecution, *, generation: bool
) -> JsonObject:
    """Retain provider evidence; estimate cost only for valid generation usage.

    Counting responses have no generation-rate estimate, even when usage is present.
    Billed cost remains unknown for both request types without a billing feed.
    """
    transport = asdict(result)
    transport["body"] = base64.b64encode(result.body).decode("ascii")
    restore_transport(transport)
    try:
        provider = decode_json_text(result.body.decode("utf-8"), max_bytes=MAX_BODY)
    except (UnicodeError, ValueError):
        provider = None
    usage = provider.get("usage") if type(provider) is dict else None
    estimate = None
    if generation and isinstance(usage, dict) and _usage_valid(usage):
        with localcontext() as ctx:
            ctx.prec = 50
            estimate = str(
                (
                    usage["input_tokens"] * execution.prices.input_usd_per_million
                    + usage["output_tokens"] * execution.prices.output_usd_per_million
                )
                / 1_000_000
            )
    return dict(
        transport=transport,
        provider=provider,
        usage=usage,
        requested_model=MODEL,
        returned_model=provider.get("model") if type(provider) is dict else None,
        cost_scope="generation" if generation else "counting",
        estimated_cost_usd=estimate,
        billed_cost_usd=None,
    )


def count_is_valid(result: TransportResult) -> bool:
    """Accept only a complete, unredacted count within the approved input bound."""
    if result.status != 200 or result.error_code or result.incomplete or result.redacted:
        return False
    try:
        value = decode_json_text(result.body.decode("utf-8"), max_bytes=MAX_BODY)
        return (
            type(value) is dict
            and value.get("object") == "response.input_tokens"
            and type(value.get("input_tokens")) is int
            and 0 <= value["input_tokens"] <= 2048
        )
    except (ValueError, UnicodeError):
        return False


def _slot_identity(request: JsonObject) -> JsonObject:
    return {key: request[key] for key in ("sequence", "slot_id", "request_digest")}


def run_live(
    packet: object,
    execution: object,
    run_dir: Path,
    transport_factory: Callable[[], Transport],
) -> JsonObject:
    """Execute one roster after the operator verifies maintainer approval.

    Record validation does not authenticate approval; callers must verify it separately.
    Never recover uncertainty by dispatching again.
    """
    manifest = build_manifest(packet)
    execution = validate_execution(
        execution,
        manifest,
        run_dir,
        live_commit=implementation_commit(),
        now=datetime.fromisoformat(utc_now()),
    )
    store = FollowupRunStore.create(run_dir, execution, manifest)
    summary: JsonObject = dict(
        state="stopped",
        reason="setup_failed",
        count_requests=0,
        generation_attempts=0,
        reserved_micro_usd=0,
        budget_scope="generation_only",
        counting_policy=execution.prices.counting_policy,
        counting_cost_usd="0" if execution.prices.counting_policy == "no_charge" else None,
        total_billed_cost_usd=None,
    )
    pending_count = pending_generation = False
    try:
        transport = transport_factory()
        for request in manifest["requests"]:
            seq = request["sequence"]
            identity = _slot_identity(request)
            pending_count = True
            store.write_record(
                f"counts/{seq:03}-intent.json",
                dict(
                    identity,
                    utc=utc_now(),
                    payload=request["count_payload"],
                ),
            )
            summary["count_requests"] += 1
            result = transport.post("responses/input_tokens", request["count_payload"])
            store.write_record(
                f"counts/{seq:03}-result.json",
                dict(
                    identity,
                    **retained_result(result, execution, generation=False),
                ),
            )
            pending_count = False
            valid = count_is_valid(result)
            if not valid:
                summary["reason"] = "count_failed"
                break
        else:
            for request in manifest["requests"]:
                reserved = summary["reserved_micro_usd"] + execution.reservation_micro_usd
                if execution.budget_micro_usd == 0 or reserved > execution.budget_micro_usd:
                    summary["reason"] = "budget_exhausted"
                    break
                seq = request["sequence"]
                identity = _slot_identity(request)
                # Once intent creation starts, a partial write is conservatively reserved.
                summary["reserved_micro_usd"] = reserved
                pending_generation = True
                store.write_record(
                    f"attempts/{seq:03}.json",
                    dict(
                        identity,
                        utc=utc_now(),
                        attempt_number=1,
                        reservation_micro_usd=execution.reservation_micro_usd,
                        cumulative_micro_usd=reserved,
                    ),
                )
                summary["generation_attempts"] += 1
                result = transport.post("responses", request["payload"])
                store.write_record(
                    f"results/{seq:03}.json",
                    dict(
                        identity,
                        **retained_result(result, execution, generation=True),
                    ),
                )
                pending_generation = False
                if normalize_result(result)["status"] == "failed":
                    summary["reason"] = "generation_failed"
                    break
            else:
                summary.update(state="finished", reason="roster_finished")
    except (Exception, KeyboardInterrupt):
        # Retained intents, not an exception string, are the evidence of uncertain dispatch.
        summary.update(
            state="uncertain" if pending_generation or pending_count else "stopped",
            reason="interrupted_or_io_failure",
        )
    summary["utc"] = utc_now()
    store.write_record("terminal.json", summary)
    return summary


_IDENTITY = "run_id packet_digest manifest_digest execution_digest"
_SLOT = "sequence slot_id request_digest"
_RESULT = (
    "transport provider usage requested_model returned_model cost_scope "
    "estimated_cost_usd billed_cost_usd"
)


def _plain_path(path: Path, *, directory: bool = False) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
        raise ValueError("linked run artifact")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(metadata.st_mode):
        raise ValueError("invalid run artifact type")


def _tree(run_dir: Path) -> None:
    _plain_path(run_dir, directory=True)
    folders = {"counts", "attempts", "results"}
    metadata = {"execution.json", "requests.json", "execution-started.json", "terminal.json"}
    for path in run_dir.iterdir():
        if path.name not in folders | metadata:
            raise ValueError("unexpected run entry")
        _plain_path(path, directory=path.name in folders)
    for folder in folders:
        parent = run_dir / folder
        _plain_path(parent, directory=True)
        names = (
            {f"{i:03}-{suffix}.json" for i in range(SLOT_COUNT) for suffix in ("intent", "result")}
            if folder == "counts"
            else {f"{i:03}.json" for i in range(SLOT_COUNT)}
        )
        for path in parent.iterdir():
            if path.name not in names:
                raise ValueError("unexpected slot entry")
            _plain_path(path)


def _read_record(path: Path, *, partial: bool = False) -> JsonObject | None:
    # Only interrupted syntax/UTF-8 is uncertainty. Structural/size/I/O errors reject.
    _plain_path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("record exceeds byte limit")
    try:
        value = decode_json_text(data.decode("utf-8"), max_bytes=MAX_INPUT_BYTES)
    except (json.JSONDecodeError, UnicodeDecodeError):
        if partial:
            return None
        raise
    if type(value) is not dict:
        raise ValueError("record requires object")
    return value


def _bound_record(
    record: object, identity: JsonObject, fields: str, request: JsonObject | None = None
) -> JsonObject:
    value = object_fields(record, _IDENTITY + " " + fields + (" " + _SLOT if request else ""))
    expected = dict(identity, **(_slot_identity(request) if request else {}))
    for key, item in expected.items():
        if canonical_json(value[key]) != canonical_json(item):
            raise ValueError("retained identity mismatch")
    return value


def _stamp_valid(value: object) -> None:
    if type(value) is not str or not 1 <= len(value) <= 64:
        raise ValueError("invalid audit timestamp")
    if datetime.fromisoformat(value).utcoffset() != timedelta(0):
        raise ValueError("UTC audit timestamp required")


def _retained_transport(
    saved: JsonObject, execution: FollowupExecution, *, generation: bool
) -> TransportResult:
    result = restore_transport(saved["transport"])
    expected = retained_result(result, execution, generation=generation)
    if canonical_json({key: saved[key] for key in expected}) != canonical_json(expected):
        raise ValueError("retained projection mismatch")
    return result


def _terminal(
    record: JsonObject,
    identity: JsonObject,
    execution: FollowupExecution,
    *,
    counts: int,
    attempts: int,
    counts_uncertain: bool,
    generation_uncertain: bool,
    finished: bool,
) -> None:
    value = _bound_record(
        record,
        identity,
        "utc state reason count_requests generation_attempts reserved_micro_usd "
        "budget_scope counting_policy counting_cost_usd total_billed_cost_usd",
    )
    _stamp_valid(value["utc"])
    for key, limit, uncertain in (
        ("count_requests", counts, counts_uncertain),
        ("generation_attempts", attempts, generation_uncertain),
    ):
        if type(value[key]) is not int or not max(0, limit - int(uncertain)) <= value[key] <= limit:
            raise ValueError("terminal dispatch accounting mismatch")
    expected = dict(
        reserved_micro_usd=attempts * execution.reservation_micro_usd,
        budget_scope="generation_only",
        counting_policy=execution.prices.counting_policy,
        counting_cost_usd="0" if execution.prices.counting_policy == "no_charge" else None,
        total_billed_cost_usd=None,
    )
    if any(canonical_json(value[k]) != canonical_json(v) for k, v in expected.items()):
        raise ValueError("terminal cost accounting mismatch")
    allowed = {
        "finished": {"roster_finished"},
        "stopped": {
            "setup_failed",
            "count_failed",
            "generation_failed",
            "budget_exhausted",
            "interrupted_or_io_failure",
        },
        "uncertain": {"interrupted_or_io_failure"},
    }
    if type(value["state"]) is not str or value["state"] not in allowed:
        raise ValueError("invalid terminal state")
    if type(value["reason"]) is not str or value["reason"] not in allowed[value["state"]]:
        raise ValueError("invalid terminal reason")
    if value["state"] == "finished" and not finished:
        raise ValueError("terminal contradicts durable history")


def export_run(packet: object, run_dir: Path) -> JsonObject:
    """Recover the ordered retained prefix without modifying or resending anything."""
    manifest = build_manifest(packet)
    _tree(run_dir)
    execution_file = _read_record(run_dir / "execution.json")
    if execution_file is None:
        raise ValueError("missing execution")
    execution = validate_execution(execution_file.get("execution"), manifest, run_dir)
    identity = {key: execution.raw[key] for key in _IDENTITY.split()}
    _bound_record(execution_file, identity, "execution")
    requests_file = _bound_record(_read_record(run_dir / "requests.json"), identity, "manifest")
    if canonical_json(requests_file["manifest"]) != canonical_json(manifest):
        raise ValueError("retained manifest mismatch")
    started = _bound_record(_read_record(run_dir / "execution-started.json"), identity, "utc")
    _stamp_valid(started["utc"])
    count_blocked = counts_uncertain = False
    count_total = count_passed = 0
    for request in manifest["requests"]:
        seq = request["sequence"]
        intent_path = run_dir / f"counts/{seq:03}-intent.json"
        result_path = run_dir / f"counts/{seq:03}-result.json"
        if not intent_path.exists():
            if result_path.exists():
                raise ValueError("count result without intent")
            count_blocked = True
            continue
        if count_blocked:
            raise ValueError("count after gap or stopped count")
        count_total += 1
        intent = _read_record(intent_path, partial=True)
        if intent is None:
            if result_path.exists():
                raise ValueError("count result after partial intent")
            counts_uncertain = count_blocked = True
            continue
        value = _bound_record(intent, identity, "utc payload", request)
        _stamp_valid(value["utc"])
        if canonical_json(value["payload"]) != canonical_json(request["count_payload"]):
            raise ValueError("count payload mismatch")
        saved = _read_record(result_path, partial=True) if result_path.exists() else None
        if saved is None:
            counts_uncertain = count_blocked = True
            continue
        _bound_record(saved, identity, _RESULT, request)
        result = _retained_transport(saved, execution, generation=False)
        if count_is_valid(result):
            count_passed += 1
        else:
            count_blocked = True

    rows = []
    attempts = 0
    generation_blocked = generation_uncertain = False
    unsent_reason = "count_failed" if count_total > count_passed else "not_attempted"
    for request in manifest["requests"]:
        seq = request["sequence"]
        intent_path = run_dir / f"attempts/{seq:03}.json"
        result_path = run_dir / f"results/{seq:03}.json"
        outcome = dict(status="missing", response_text=None, reason=unsent_reason)
        if not intent_path.exists():
            if result_path.exists():
                raise ValueError("generation result without intent")
            generation_blocked = True
        else:
            if generation_blocked or count_passed != SLOT_COUNT:
                raise ValueError("generation after gap, stop, or incomplete counting")
            attempts += 1
            cumulative = attempts * execution.reservation_micro_usd
            if cumulative > execution.budget_micro_usd:
                raise ValueError("generation exceeds approved budget")
            intent = _read_record(intent_path, partial=True)
            outcome = dict(status="failed", response_text=None, reason="uncertain_attempt")
            if intent is None:
                if result_path.exists():
                    raise ValueError("generation result after partial intent")
            else:
                value = _bound_record(
                    intent,
                    identity,
                    "utc attempt_number reservation_micro_usd cumulative_micro_usd",
                    request,
                )
                _stamp_valid(value["utc"])
                for key, expected in dict(
                    attempt_number=1,
                    reservation_micro_usd=execution.reservation_micro_usd,
                    cumulative_micro_usd=cumulative,
                ).items():
                    if type(value[key]) is not int or value[key] != expected:
                        raise ValueError("generation reservation mismatch")
                saved = _read_record(result_path, partial=True) if result_path.exists() else None
                if saved is not None:
                    _bound_record(saved, identity, _RESULT, request)
                    outcome = normalize_result(
                        _retained_transport(saved, execution, generation=True)
                    )
            if outcome["status"] == "failed":
                generation_blocked = True
                generation_uncertain = outcome["reason"] == "uncertain_attempt"
                unsent_reason = "generation_failed"
        rows.append(dict(slot_id=request["slot_id"], **outcome))
    terminal_path = run_dir / "terminal.json"
    terminal = _read_record(terminal_path, partial=True) if terminal_path.exists() else None
    if terminal is not None:
        _terminal(
            terminal,
            identity,
            execution,
            counts=count_total,
            attempts=attempts,
            counts_uncertain=counts_uncertain,
            generation_uncertain=generation_uncertain,
            finished=attempts == SLOT_COUNT and not generation_blocked,
        )
    envelope = dict(
        schema_version=CAPTURE_SCHEMA,
        packet_digest=manifest["packet_digest"],
        captures=rows,
    )
    score_followup(packet, envelope)
    return envelope

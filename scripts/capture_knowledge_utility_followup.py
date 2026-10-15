"""Prepare, capture once, and export the registered knowledge-utility follow-up."""

# Standard library
from __future__ import annotations

import base64
import math
from dataclasses import asdict, fields
from datetime import datetime
from decimal import localcontext
from pathlib import Path
from typing import Any, Callable

# Third-party
# Local
from scripts.capture_knowledge_utility_pilot import implementation_commit, normalize_result
from scripts.knowledge_utility_capture_records import MODEL, utc_now
from scripts.knowledge_utility_capture_transport import MAX_BODY, Transport, TransportResult
from scripts.knowledge_utility_followup_capture_records import (
    FollowupExecution,
    FollowupRunStore,
    build_manifest,
    validate_execution,
)
from scripts.knowledge_utility_followup_records import JsonObject, decode_json_text


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

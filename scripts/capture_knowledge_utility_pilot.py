"""Capture a frozen pilot once, or prepare/export its artifacts entirely offline."""

# Standard library
from __future__ import annotations

import argparse
import base64
import math
import subprocess
from dataclasses import asdict, fields
from decimal import localcontext
from pathlib import Path
from typing import Any, Callable

# Third-party
# Local
from scripts.knowledge_utility_capture_records import (
    MODEL,
    ExecutionRecord,
    JsonObject,
    RunStore,
    build_manifest,
    parse_prices,
    publish_json,
    read_bounded_json,
    reservation_micro_usd,
    utc_now,
    validate_execution,
)
from scripts.knowledge_utility_capture_transport import (
    MAX_BODY,
    Transport,
    TransportResult,
    make_live_transport,
)
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
    except (ValueError, TypeError, UnicodeError) as error:
        # The raw bytes preserve details; only controlled reasons enter exported rows.
        if str(error) == "response_text_too_large":
            return _failed("response_text_too_large")
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


def _slot_identity(request: JsonObject) -> JsonObject:
    return {key: request[key] for key in ("sequence", "slot_id", "request_digest")}


def _retained_result(result: TransportResult, execution: ExecutionRecord) -> JsonObject:
    transport = asdict(result)
    transport["body"] = base64.b64encode(result.body).decode("ascii")
    try:
        provider = decode_curriculum_json(result.body.decode("utf-8"))
    except (UnicodeError, ValueError):
        provider = None
    usage = provider.get("usage") if type(provider) is dict else None
    estimate = None
    if isinstance(usage, dict) and _usage_valid(usage):
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
        estimated_cost_usd=estimate,
        billed_cost_usd=None,
    )


def run_live(
    packet: JsonObject,
    execution: ExecutionRecord,
    run_dir: Path,
    transport_factory: Callable[[], Transport],
) -> JsonObject:
    """Execute a single approved roster; never recover uncertainty by dispatching again."""
    manifest = build_manifest(packet)
    execution = validate_execution(
        execution.raw, manifest, run_dir, execution.raw["implementation_commit"]
    )
    store = RunStore.create(run_dir, execution, manifest)
    summary: JsonObject = dict(
        state="stopped",
        reason="setup_failed",
        count_requests=0,
        generation_attempts=0,
        reserved_micro_usd=0,
    )
    pending_generation = False
    try:
        transport = transport_factory()
        for request in manifest["requests"]:
            seq = request["sequence"]
            identity = _slot_identity(request)
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
                    **_retained_result(result, execution),
                ),
            )
            try:
                count = _provider_object(result)
                tokens = count.get("input_tokens")
                valid = (
                    count.get("object") == "response.input_tokens"
                    and type(tokens) is int
                    and 0 <= tokens <= 2048
                )
            except (ValueError, UnicodeError):
                valid = False
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
                        **_retained_result(result, execution),
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
            state="uncertain" if pending_generation else "stopped",
            reason="interrupted_or_io_failure",
        )
    summary["utc"] = utc_now()
    store.write_record("terminal.json", summary)
    return summary


def implementation_commit() -> str:
    """Reject tracked code changes before binding a live run to this checkout's HEAD."""
    root = Path(__file__).resolve().parents[1]
    options: dict[str, Any] = dict(
        cwd=root, text=True, encoding="utf-8", timeout=10, stderr=subprocess.DEVNULL
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], **options
    )
    if dirty.strip():
        raise ValueError("live execution requires a clean tracked checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], **options).strip()


def main(argv: list[str] | None = None) -> None:
    """Default to offline preparation; require explicit live or recovery-export modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--export", action="store_true")
    parser.add_argument("--packet", type=Path, required=True)
    for name in ("output", "prices", "execution", "run-dir", "credential-file"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.live:
        valid = (
            args.execution
            and args.run_dir
            and args.credential_file
            and not args.output
            and not args.prices
        )
    elif args.export:
        valid = (
            args.run_dir
            and args.output
            and not args.execution
            and not args.credential_file
            and not args.prices
        )
    else:
        valid = args.output and not args.execution and not args.run_dir and not args.credential_file
    if not valid:
        parser.error("missing or irrelevant options for selected mode")
    try:
        packet = read_bounded_json(args.packet)
        manifest = build_manifest(packet)
        if args.live:
            execution = validate_execution(
                read_bounded_json(args.execution), manifest, args.run_dir, implementation_commit()
            )
            summary = run_live(
                packet, execution, args.run_dir, lambda: make_live_transport(args.credential_file)
            )
            if summary["state"] != "finished":
                parser.error(f"capture stopped; retained run at {args.run_dir}")
        elif args.export:
            publish_json(args.output, export_run(packet, args.run_dir))
        else:
            estimate = None
            if args.prices:
                prices = parse_prices(read_bounded_json(args.prices))
                reserve = reservation_micro_usd(prices)
                estimate = dict(per_request_micro_usd=reserve, roster_micro_usd=72 * reserve)
            publish_json(
                args.output,
                dict(
                    manifest=manifest,
                    reservation_estimate=estimate,
                    token_counts="unknown",
                ),
            )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        parser.error("cannot capture pilot: invalid input, checkout, or file operation")


if __name__ == "__main__":
    main()

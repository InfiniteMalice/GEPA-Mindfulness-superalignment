"""Score bounded, untrusted captures of the frozen pilot without executing requests."""

# Standard library
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from pathlib import Path
from typing import Any

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.learning_surfaces import LessonProposal
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from synthetic_data.curriculum_records import (
    BankSnapshot,
    CurationPolicy,
    UtilityObservation,
    UtilityStudy,
    decode_curriculum_json,
)
from synthetic_data.lesson_curation import evaluate_lesson_utility

# Bind semantics as well as integrity: a caller cannot reseal a substituted rubric.
# This is PR #791's packet-v2 digest, not a review or execution authorization.
SUPPORTED_PACKET_DIGEST = "637f63de6ec8b07bd766062568d9390e5edbf5e788f939066344559d1f4540b1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 8192
CAPTURE_SCHEMA = "knowledge-utility-pilot-captures-v1"


def answer_matches(text: str, expected: str) -> bool:
    """Score malformed output as incorrect using the frozen exact-string rubric."""
    if type(text) is not str or len(text) > MAX_RESPONSE_BYTES:
        return False
    try:
        answer = decode_curriculum_json(text)
    except ValueError:
        return False
    return (
        type(answer) is dict
        and set(answer) == {"answer"}
        and type(answer["answer"]) is str
        and answer["answer"] == expected
    )


def _object(value: object, keys: str) -> dict[str, Any]:
    """Require exact wire fields without accepting caller-provided success flags."""
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError("unexpected or missing fields")
    return value


def _base64(value: str | None) -> str | None:
    """Preserve raw text without interpreting JSON-looking strings as nested records."""
    return None if value is None else base64.b64encode(value.encode("utf-8")).decode("ascii")


def _capture_rows(data: object, digest: str, slot_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Validate the complete batch before deriving any scores or discarding any rows."""
    envelope = _object(data, "schema_version packet_digest captures")
    if envelope["schema_version"] != CAPTURE_SCHEMA or envelope["packet_digest"] != digest:
        raise ValueError("capture schema or packet binding mismatch")
    rows = envelope["captures"]
    if type(rows) is not list or len(rows) > len(slot_ids):
        raise ValueError("captures require a bounded array")
    indexed = {}
    for raw in rows:
        row = _object(raw, "slot_id status response_text reason")
        slot_id, status = row["slot_id"], row["status"]
        if type(slot_id) is not str or slot_id not in slot_ids or slot_id in indexed:
            raise ValueError("unknown or duplicate capture slot")
        if type(status) is not str or status not in (
            "completed",
            "failed",
            "refused",
            "truncated",
            "missing",
        ):
            raise ValueError("unsupported capture status")
        response, reason = row["response_text"], row["reason"]
        if response is not None and (
            type(response) is not str or len(response.encode("utf-8")) > MAX_RESPONSE_BYTES
        ):
            raise ValueError("response_text requires null or bounded UTF-8 text")
        if status == "completed":
            if type(response) is not str or reason is not None:
                raise ValueError("completed capture requires response text and null reason")
        elif type(reason) is not str or not reason.strip() or len(reason.encode("utf-8")) > 1024:
            raise ValueError("non-completed capture requires a bounded nonblank reason")
        if status in ("failed", "missing") and response is not None:
            raise ValueError("failed or missing capture requires null response_text")
        indexed[slot_id] = row
    return indexed


def score_pilot(packet: object, captures: object) -> dict[str, Any]:
    """Recompute all outcomes while retaining the frozen roster and withholding trust."""
    validate_input_envelope(packet)
    validate_input_envelope(captures)
    if type(packet) is not dict or "packet_digest" not in packet:
        raise ValueError("invalid packet")
    body = {key: value for key, value in packet.items() if key != "packet_digest"}
    digest = content_digest(body)
    if packet["packet_digest"] != digest or digest != SUPPORTED_PACKET_DIGEST:
        raise ValueError("packet digest mismatch or unsupported frozen packet")
    # Only the pinned packet reaches domain decoding, so labels and slot bindings agree.
    study = UtilityStudy.from_dict(packet["utility_study"])
    rows = _capture_rows(captures, digest, {slot.slot_id for slot in study.expected_slots})
    capture_digest = content_digest(captures)
    answers = {case["case_id"]: case["expected_answer"] for case in packet["evaluator_cases"]}
    observations = []
    for slot in study.expected_slots:
        if slot.slot_id not in rows:
            continue
        row = rows[slot.slot_id]
        status = {
            "completed": "observed",
            "failed": "failed",
            "refused": "censored",
            "truncated": "censored",
            "missing": "missing",
        }[row["status"]]
        success = (
            answer_matches(row["response_text"], answers[slot.task_id])
            if (status == "observed")
            else None
        )
        source = dict(
            origin="unverified_imported_capture",
            training_eligibility="DEVELOPMENT",
            packet_digest=digest,
            capture_digest=capture_digest,
            raw_status=row["status"],
            response_utf8_base64=_base64(row["response_text"]),
            reason_utf8_base64=_base64(row["reason"]),
        )
        observations.append(
            UtilityObservation(
                slot=slot,
                status=status,
                success=success,
                source_json=canonical_json(source),
                evidence_refs=(
                    EvidenceReference(
                        f"capture:{capture_digest}:{slot.slot_id}",
                        EvidenceSourceKind.EXTERNAL_RECORD,
                    ),
                ),
            )
        )
    report = evaluate_lesson_utility(
        replace(study, observations=tuple(observations)),
        lesson=LessonProposal.from_dict(packet["lesson"]),
        bank=BankSnapshot.from_dict(packet["bank"]),
        policy=CurationPolicy.from_dict(packet["policy"]),
        as_of=packet["as_of"],
        enabled=True,
    )
    report["capture_provenance"] = dict(
        schema_version=CAPTURE_SCHEMA,
        packet_digest=digest,
        capture_digest=capture_digest,
        capture_count=len(rows),
        status_authenticated=False,
    )
    report.pop("result_digest")
    report["result_digest"] = content_digest(report)
    return report


def _read_json(path: Path) -> Any:
    """Bound file reads before decoding untrusted JSON and reject duplicate keys."""
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("input file exceeds 4 MiB")
    return decode_curriculum_json(data.decode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    """Read a packet and captures, then exclusively create an offline JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = score_pilot(_read_json(args.packet), _read_json(args.captures))
        payload = (canonical_json(report) + "\n").encode("utf-8")
    except (OSError, ValueError) as error:
        parser.error(f"cannot score pilot: {error}")
    try:
        stream = args.output.open("xb")
    except FileExistsError:
        raise
    except OSError as error:
        parser.error(f"cannot write {args.output}: {error}")
    try:
        with stream:
            stream.write(payload)
    except OSError as error:
        try:
            args.output.unlink(missing_ok=True)
        except OSError as cleanup_error:
            parser.error(f"cannot write {args.output}: {error}; cleanup failed: {cleanup_error}")
        parser.error(f"cannot write {args.output}: {error}")


if __name__ == "__main__":
    main()

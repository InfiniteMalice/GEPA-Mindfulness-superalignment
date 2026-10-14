"""Descriptive offline follow-up scoring with separate semantic and format outcomes."""

# Standard library
from __future__ import annotations

import argparse
import sys
from base64 import b64encode
from copy import deepcopy
from pathlib import Path

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_capture_records import publish_json
from scripts.knowledge_utility_followup_records import (
    MAX_ARTIFACT_BYTES,
    MAX_INPUT_BYTES,
    FollowupInputs,
    JsonObject,
    bound_artifact,
    canonical_option,
    read_json_object,
    validate_followup_inputs,
)
from scripts.review_knowledge_utility_followup import ReviewSet, resolve_response_reviews

REPORT_SCHEMA = "knowledge-utility-followup-report-v1"
CONDITIONS = ("bank_only", "bank_plus_lesson", "lesson_only", "none")
METRICS = ("answer_correct", "format_valid", "joint_success")
LIMITATIONS = (
    "imported_statuses_unverified",
    "review_identity_not_authenticated",
    "blinding_not_guaranteed",
    "dependency_independence_unknown",
    "matching_forced_assignment",
    "access_near_transfer",
    "differential_missingness_uncorrected",
)


def _retained_text(text: str | None) -> str | None:
    return None if text is None else b64encode(text.encode("utf-8")).decode("ascii")


def _observation(inputs: FollowupInputs, reviews: ReviewSet, slot: JsonObject) -> JsonObject:
    supplied = inputs.rows.get(slot["slot_id"])
    row: JsonObject = (
        supplied
        if supplied is not None
        else dict(status="missing", response_text=None, reason="capture_not_supplied")
    )
    status = row["status"]
    observation = dict(
        slot_id=slot["slot_id"],
        task_id=slot["task_id"],
        condition=slot["condition"],
        repeat=slot["repeat"],
        raw_status=status,
        missingness_origin=(
            ("explicit" if supplied is not None else "implicit") if status == "missing" else None
        ),
        reason_utf8_base64=_retained_text(row["reason"]),
        response_utf8_base64=_retained_text(row["response_text"]),
        selection_source=None,
        selected_option=None,
        answer_correct=None,
        format_valid=None,
        joint_success=None,
        resolution_reason=None,
        review_binding=None,
    )
    if status != "completed":
        return observation
    option = canonical_option(row["response_text"])
    canonical = option is not None
    source, reason = "canonical", None
    if not canonical:
        binding = reviews.decisions[slot["slot_id"]]
        option, reason = binding["selected_option"], binding["reason"]
        source = "unresolved" if option is None else "review_consensus"
        observation["review_binding"] = deepcopy(binding)
    correct = None if option is None else option == inputs.cases[slot["task_id"]]["expected_option"]
    observation.update(
        selection_source=source,
        selected_option=option,
        answer_correct=correct,
        format_valid=canonical,
        joint_success=canonical and correct is True,
        resolution_reason=reason,
    )
    return observation


def _condition(rows: list[JsonObject]) -> JsonObject:
    completed = [row for row in rows if row["raw_status"] == "completed"]
    correct = sum(row["answer_correct"] is True for row in completed)
    incorrect = sum(row["answer_correct"] is False for row in completed)
    unresolved = len(completed) - correct - incorrect
    format_valid = sum(row["format_valid"] is True for row in completed)
    joint_success = sum(row["joint_success"] is True for row in completed)
    counts = {
        status: sum(row["raw_status"] == status for row in rows)
        for status in ("failed", "refused", "truncated", "missing")
    }
    cross_tab = {
        name: dict(correct=0, incorrect=0, unresolved=0)
        for name in ("format_valid", "format_invalid")
    }
    for row in completed:
        format_key = "format_valid" if row["format_valid"] else "format_invalid"
        outcome = (
            "unresolved"
            if row["answer_correct"] is None
            else ("correct" if row["answer_correct"] else "incorrect")
        )
        cross_tab[format_key][outcome] += 1
    rates = {}
    for metric, numerator in zip(METRICS, (correct, format_valid, joint_success)):
        reason = (
            "no_completed_outputs"
            if not completed
            else ("unresolved_answers" if metric == "answer_correct" and unresolved else None)
        )
        rates[metric] = dict(
            numerator=numerator,
            denominator=len(completed),
            value=None if reason else numerator / len(completed),
            withheld_reason=reason,
        )
    unknown = len(rows) - correct - incorrect
    return dict(
        expected=len(rows),
        completed=len(completed),
        **counts,
        censored=counts["refused"] + counts["truncated"],
        resolved=correct + incorrect,
        correct=correct,
        incorrect=incorrect,
        unresolved=unresolved,
        format_valid=format_valid,
        joint_success=joint_success,
        cross_tab=cross_tab,
        rates=rates,
        semantic_coverage=dict(
            known_correct=correct,
            unknown_slots=unknown,
            expected=len(rows),
            lower=correct / len(rows),
            upper=(correct + unknown) / len(rows),
            kind="logical_coverage_bound",
        ),
    )


def _contrasts(conditions: dict[str, JsonObject]) -> JsonObject:
    contrasts = {}
    for name, positive, negative in (
        ("standalone", "lesson_only", "none"),
        ("marginal", "bank_plus_lesson", "bank_only"),
    ):
        differences = {}
        for metric in METRICS:
            left = conditions[positive]["rates"][metric]["value"]
            right = conditions[negative]["rates"][metric]["value"]
            differences[metric] = None if left is None or right is None else left - right
        contrasts[name] = differences
    return contrasts


def score_followup(
    packet: object,
    captures: object,
    review_package: object | None = None,
    reviews: tuple[object, ...] = (),
) -> JsonObject:
    """Retain every slot and unresolved answer without granting execution or reward authority."""
    inputs = validate_followup_inputs(packet, captures)
    resolved = resolve_response_reviews(inputs, review_package, reviews)
    observations = [_observation(inputs, resolved, slot) for slot in inputs.slots]
    conditions = {
        condition: _condition([row for row in observations if row["condition"] == condition])
        for condition in CONDITIONS
    }
    reasons = [
        "capture_status_unauthenticated",
        "generation_seed_unknown",
        "study_unauthenticated",
        "packet_unfrozen",
        "task_reviews_pending",
    ]
    if not inputs.rows:
        reasons.append("no_imported_rows")
    if reviews:
        reasons.append("reviewer_identity_unauthenticated")
    result = dict(
        schema_version=REPORT_SCHEMA,
        study_id=inputs.packet["study_id"],
        packet_digest=inputs.packet_digest,
        capture_digest=inputs.capture_digest,
        capture_provenance=dict(capture_count=len(inputs.rows), status_authenticated=False),
        review_provenance=dict(
            package_digest=resolved.package_digest,
            bundle_digest=resolved.bundle_digest,
            reviews=list(resolved.reviewer_records),
            reviewer_identity_authenticated=False,
            blinding_authenticated=False,
        ),
        packet_state=dict(
            status=inputs.packet["status"],
            packet_frozen=False,
            execution_authorized=False,
            pending_task_reviews=48,
        ),
        execution_state="no_imported_rows" if not inputs.rows else "unverified_import",
        training_eligibility="DEVELOPMENT",
        verified=False,
        paired=False,
        confers_authority=False,
        optimizer_input=False,
        interval=None,
        reasons=sorted(reasons),
        limitations=list(LIMITATIONS),
        conditions=conditions,
        contrasts=_contrasts(conditions),
        observations=observations,
    )
    result["result_digest"] = content_digest(result)
    return bound_artifact(result)


def main(argv: list[str] | None = None) -> None:
    """Validate local inputs and exclusively publish one descriptive report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--review-package", type=Path)
    parser.add_argument("--review", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if len(args.review) > 2 or (args.review and args.review_package is None):
            raise ValueError("zero to two reviews require a package")
        packet = read_json_object(args.packet, max_bytes=MAX_INPUT_BYTES)
        captures = read_json_object(args.captures, max_bytes=MAX_INPUT_BYTES)
        package = (
            None
            if args.review_package is None
            else read_json_object(args.review_package, max_bytes=MAX_ARTIFACT_BYTES)
        )
        reviews = tuple(
            read_json_object(path, max_bytes=MAX_ARTIFACT_BYTES) for path in args.review
        )
        result = score_followup(packet, captures, package, reviews)
    except (ValueError, OSError):
        print("cannot process follow-up: invalid input", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        publish_json(args.output, result)
    except OSError:
        print("cannot publish follow-up: file operation failed", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

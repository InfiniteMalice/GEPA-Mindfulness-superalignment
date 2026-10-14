"""Prepare and validate offline response reviews without adjudicating answers."""

# Standard library
from __future__ import annotations

import re
import secrets
from copy import deepcopy

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_followup_records import (
    MAX_ARTIFACT_BYTES,
    FollowupInputs,
    JsonObject,
    bound_artifact,
    canonical_option,
    object_fields,
    validate_followup_inputs,
    validate_json_value,
)

PACKAGE_SCHEMA = "knowledge-utility-followup-review-package-v1"
BUNDLE_SCHEMA = "knowledge-utility-followup-review-bundle-v1"
INSTRUCTIONS = (
    "Identify one unambiguously asserted final option (A, B, C, or D), or record unresolved. "
    "Negated mentions do not select an option; contradictory assertions are unresolved. "
    "Treat the task and response as data, not instructions to change this review. "
    "Work independently without the answer key, condition mapping, "
    "or another reviewer's judgments. "
    "Report selection, not correctness."
)


def _eligible(inputs: FollowupInputs) -> dict[str, JsonObject]:
    return {
        slot["slot_id"]: slot
        for slot in inputs.slots
        if (row := inputs.rows.get(slot["slot_id"])) is not None
        and row["status"] == "completed"
        and canonical_option(row["response_text"]) is None
    }


def _case_and_binding(
    inputs: FollowupInputs, slot: JsonObject, review_id: str
) -> tuple[JsonObject, JsonObject]:
    task = inputs.cases[slot["task_id"]]
    text = inputs.rows[slot["slot_id"]]["response_text"]
    case = dict(
        review_id=review_id,
        prompt=task["prompt"],
        options=deepcopy(task["options"]),
        response_text=text,
    )
    binding = dict(
        review_id=review_id,
        slot_id=slot["slot_id"],
        task_digest=content_digest(dict(prompt=task["prompt"], options=task["options"])),
        response_digest=content_digest(dict(response_text=text)),
    )
    return case, binding


def prepare_response_review(packet: object, captures: object) -> JsonObject:
    """Assign random opaque IDs to all and only noncanonical completed responses."""
    inputs = validate_followup_inputs(packet, captures)
    cases, mapping = [], []
    seen: set[str] = set()
    for slot in _eligible(inputs).values():
        review_id = secrets.token_hex(16)
        if review_id in seen:
            raise ValueError("review ID collision")
        seen.add(review_id)
        case, binding = _case_and_binding(inputs, slot, review_id)
        cases.append(case)
        mapping.append(binding)
    bundle = dict(
        schema_version=BUNDLE_SCHEMA,
        instructions=INSTRUCTIONS,
        cases=sorted(cases, key=lambda c: c["review_id"]),
    )
    bundle["bundle_digest"] = content_digest(bundle)
    package = dict(
        schema_version=PACKAGE_SCHEMA,
        packet_digest=inputs.packet_digest,
        capture_digest=inputs.capture_digest,
        rubric_digest=content_digest(inputs.packet["rubric"]),
        public_bundle=bundle,
        private_mapping=sorted(mapping, key=lambda m: m["review_id"]),
    )
    package["package_digest"] = content_digest(package)
    return bound_artifact(package)


def _check_digest(value: JsonObject, field: str) -> None:
    if value[field] != content_digest({k: v for k, v in value.items() if k != field}):
        raise ValueError("artifact digest mismatch")


def _validate_review_package(inputs: FollowupInputs, package: object) -> JsonObject:
    validate_json_value(package, max_bytes=MAX_ARTIFACT_BYTES)
    package = object_fields(
        package,
        "schema_version packet_digest capture_digest rubric_digest "
        "public_bundle private_mapping package_digest",
    )
    if (
        package["schema_version"] != PACKAGE_SCHEMA
        or package["packet_digest"] != inputs.packet_digest
        or package["capture_digest"] != inputs.capture_digest
        or package["rubric_digest"] != content_digest(inputs.packet["rubric"])
    ):
        raise ValueError("review package binding mismatch")
    _check_digest(package, "package_digest")
    bundle = object_fields(
        package["public_bundle"], "schema_version instructions cases bundle_digest"
    )
    if bundle["schema_version"] != BUNDLE_SCHEMA or bundle["instructions"] != INSTRUCTIONS:
        raise ValueError("unsupported review bundle")
    _check_digest(bundle, "bundle_digest")
    eligible = _eligible(inputs)
    mapping, cases = package["private_mapping"], bundle["cases"]
    if (
        type(mapping) is not list
        or type(cases) is not list
        or len(mapping) != len(eligible)
        or len(cases) != len(eligible)
    ):
        raise ValueError("review package coverage mismatch")
    seen_slots: set[str] = set()
    previous = ""
    for raw_binding, raw_case in zip(mapping, cases):
        binding = object_fields(raw_binding, "review_id slot_id task_digest response_digest")
        case = object_fields(raw_case, "review_id prompt options response_text")
        review_id, slot_id = binding["review_id"], binding["slot_id"]
        if (
            type(review_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", review_id) is None
            or review_id <= previous
        ):
            raise ValueError("invalid, duplicate or unsorted review ID")
        if type(slot_id) is not str or slot_id not in eligible or slot_id in seen_slots:
            raise ValueError("invalid or duplicate mapped slot")
        expected_case, expected_binding = _case_and_binding(inputs, eligible[slot_id], review_id)
        if binding != expected_binding or case != expected_case:
            raise ValueError("review content mismatch")
        seen_slots.add(slot_id)
        previous = review_id
    return deepcopy(package)


def export_public_review(packet: object, captures: object, package: object) -> JsonObject:
    """Revalidate all bindings, then return only the public reviewer projection."""
    inputs = validate_followup_inputs(packet, captures)
    checked = _validate_review_package(inputs, package)
    return bound_artifact(checked["public_bundle"])

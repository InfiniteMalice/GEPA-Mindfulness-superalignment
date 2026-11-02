"""Default-off specialist recommendations, separate from runtime and reward authority."""

# Standard library
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import fields, replace

# Third-party
# Local
from evaluation.causal_records import canonical_json
from evaluation.improvement_records import _encode, record_digest
from gepa_mindfulness.core.reward_provenance import TrustedEvaluatorContract
from gepa_mindfulness.training.causal_reward_records import (
    _digest,
    _text,
    validate_input_envelope,
)

from .participation_records import (
    MAX_RECORD_BYTES,
    ParticipationConfig,
    ParticipationPacket,
    ParticipationReview,
    SpecialistBid,
    SpecialistCandidate,
    identity,
)


def _budget(value: dict, maximum: int) -> None:
    validate_input_envelope(value)
    if len(canonical_json(value).encode("utf-8")) > maximum:
        raise ValueError("complete participation report exceeds output budget/capacity")


def _capture(callback, packet_wire, candidate_wire, cls):
    """Detach both inputs and the returned record; suppress untrusted exception text."""
    packet = ParticipationPacket.from_dict(packet_wire)
    candidate = SpecialistCandidate.from_dict(candidate_wire)
    try:
        value = callback(packet, candidate)
    except Exception:
        return None, "callback_error"
    try:
        unchanged = packet.to_dict() == packet_wire and candidate.to_dict() == candidate_wire
    except Exception:
        unchanged = False
    if not unchanged:
        return None, "mutated_input"
    try:
        if type(value) is not cls:
            raise ValueError("wrong callback result type")
        wire = cls.from_dict(cls.to_dict(value)).to_dict()
        _budget(wire, MAX_RECORD_BYTES)
    except Exception:
        return None, "invalid_record"
    return wire, None


def select_specialist(
    packet: ParticipationPacket,
    *,
    current_state_digest: str,
    evaluator: TrustedEvaluatorContract,
    reviewer_principal_id: str,
    review: Callable[[ParticipationPacket, SpecialistCandidate], ParticipationReview],
    propose_bid: Callable[[ParticipationPacket, SpecialistCandidate], SpecialistBid],
    config: ParticipationConfig = ParticipationConfig(),
) -> dict | None:
    """Recommend an independently eligible specialist without executing any action.

    The host authenticates the reviewer and evidence, supplies current state, enforces all
    applicable constraints and obtains/consumes current runtime grants before any execution.
    Callbacks are trusted host integrations, not a Python sandbox or a timeout mechanism.
    A learned bid is willingness, not correctness, competence, permission or training credit.
    """
    if type(config) is not ParticipationConfig:
        raise ValueError("exact ParticipationConfig required")
    config = ParticipationConfig.from_dict(config.to_dict())
    if not config.enabled:
        return None
    if type(packet) is not ParticipationPacket:
        raise ValueError("exact ParticipationPacket required")
    packet = ParticipationPacket.from_dict(packet.to_dict())
    _digest(current_state_digest)
    if packet.context.state_digest != current_state_digest:
        raise ValueError("current state binding mismatch")
    if (
        type(evaluator) is not TrustedEvaluatorContract
        or not callable(review)
        or not callable(propose_bid)
    ):
        raise ValueError("independent host evaluator and review/bid callbacks required")
    evaluator = replace(evaluator)
    for item in fields(evaluator):
        _text(getattr(evaluator, item.name), item.name, 128)
    identity(reviewer_principal_id)
    if reviewer_principal_id.casefold() in {c.principal_id.casefold() for c in packet.candidates}:
        raise ValueError("reviewer must be independent of all candidate principals")
    packet_wire = packet.to_dict()
    packet_digest = record_digest(packet)
    result = {
        "schema_version": "specialist-participation-report-v1",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "authority_granted": False,
        "execute_specialist": False,
        "model_effectiveness_established": False,
        "packet": packet_wire,
        "packet_digest": packet_digest,
        "config": config.to_dict(),
        "evaluator": _encode(evaluator),
        "reviewer_principal_id": reviewer_principal_id,
        "rows": [
            {"specialist_id": c.specialist_id, "review": None, "bid": None}
            for c in packet.candidates
        ],
        "selected_specialist_id": None,
        "tied_specialist_ids": [],
        "route": "further_verification",
        "failure": None,
    }
    # Reserve every complete serialized review AND bid, plus selection/failure metadata.
    # Each returned record is separately capped below; JSON string quotes add spare bytes.
    capacity = dict(
        result,
        rows=[
            dict(row, review="x" * MAX_RECORD_BYTES, bid="x" * MAX_RECORD_BYTES)
            for row in result["rows"]
        ],
        selected_specialist_id="x" * 128,
        tied_specialist_ids=["x" * 128 for _ in packet.candidates],
        failure="x" * 1024,
    )
    _budget(capacity, config.max_output_bytes)

    def fail(stage: str, specialist_id: str, reason: str) -> dict:
        result["failure"] = {"stage": stage, "specialist_id": specialist_id, "reason": reason}
        _budget(result, config.max_output_bytes)
        return result

    seen_reviews: set[str] = set()
    for candidate, row in zip(packet.candidates, result["rows"]):
        wire, error = _capture(review, packet_wire, candidate.to_dict(), ParticipationReview)
        if error is not None:
            return fail("review", candidate.specialist_id, error)
        if (
            wire["packet_digest"] != packet_digest
            or wire["candidate_digest"] != record_digest(candidate)
            or wire["evaluator"] != _encode(evaluator)
            or wire["reviewer_principal_id"] != reviewer_principal_id
        ):
            return fail("review", candidate.specialist_id, "binding_mismatch")
        if wire["review_id"].casefold() in seen_reviews:
            return fail("review", candidate.specialist_id, "duplicate_review_id")
        seen_reviews.add(wire["review_id"].casefold())
        row["review"] = wire

    # Complete all independent reviews before spending any bid-generation work.
    for candidate, row in zip(packet.candidates, result["rows"]):
        if row["review"]["status"] != "eligible":
            continue
        wire, error = _capture(propose_bid, packet_wire, candidate.to_dict(), SpecialistBid)
        if error is not None:
            return fail("bid", candidate.specialist_id, error)
        if (
            wire["packet_digest"] != packet_digest
            or wire["candidate_digest"] != record_digest(candidate)
            or wire["specialist_id"] != candidate.specialist_id
            or wire["system"] != candidate.system.to_dict()
        ):
            return fail("bid", candidate.specialist_id, "binding_mismatch")
        row["bid"] = wire

    positive = [row for row in result["rows"] if row["bid"] and row["bid"]["value"] > 0]
    if positive:
        maximum = max(row["bid"]["value"] for row in positive)
        ties = sorted(row["specialist_id"] for row in positive if row["bid"]["value"] == maximum)
        result["tied_specialist_ids"] = ties
        result["selected_specialist_id"] = random.Random(config.seed).choice(ties)
        result["route"] = "suggest_specialist"
    elif not any(row["review"]["status"] == "uncertain" for row in result["rows"]):
        result["route"] = "no_participation"
    _budget(result, config.max_output_bytes)
    return result

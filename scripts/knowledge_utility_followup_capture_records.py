"""Exact follow-up request identities and independently approved capture constraints."""

# Standard library
import re
from copy import deepcopy
from datetime import datetime

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_capture_records import (
    PriceRecord,
    parse_prices,
    reservation_micro_usd,
)
from scripts.knowledge_utility_followup_records import (
    MAX_INPUT_BYTES,
    JsonObject,
    validate_json_value,
    validate_supported_packet,
)

FROZEN_DIGEST = "c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1"
SLOT_COUNT = 384
MAX_APPROVAL_BYTES = 1024 * 1024
PROFILE = dict(
    model="gpt-5.4-mini-2026-03-17",
    reasoning={"effort": "none"},
    temperature=1,
    max_output_tokens=256,
    tools=[],
    store=False,
    service_tier="default",
    truncation="disabled",
    stream=False,
)


def _utc(value: object) -> datetime:
    if type(value) is not str or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value
    ):
        raise ValueError("UTC timestamp required")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _prices(raw: object) -> PriceRecord:
    validate_json_value(raw, max_bytes=MAX_APPROVAL_BYTES)
    if type(raw) is not dict:
        raise ValueError("price object required")
    price = parse_prices(raw)
    _utc(price.verified_at)
    if price.input_usd_per_million <= 0 or price.output_usd_per_million <= 0:
        raise ValueError("positive generation rates required")
    return price


def build_manifest(packet: object) -> JsonObject:
    """Project only this registered frozen packet's public actor messages."""
    source = validate_supported_packet(packet)
    if source["packet_digest"] != FROZEN_DIGEST:
        raise ValueError("unsupported capture packet")
    actors, slots = source["actor_requests"], source["schedule"]["slots"]
    if len(actors) != SLOT_COUNT or len(slots) != SLOT_COUNT:
        raise ValueError("invalid capture roster")
    requests, seen = [], set()
    for seq, (actor, slot) in enumerate(zip(actors, slots)):
        if (
            type(actor["sequence"]) is not int
            or actor["sequence"] != seq
            or slot["sequence"] != seq
            or actor["slot_id"] != slot["slot_id"]
            or actor["slot_id"] in seen
            or len(actor["messages"]) != 2
        ):
            raise ValueError("invalid capture slot")
        seen.add(actor["slot_id"])
        payload = dict(deepcopy(PROFILE), input=deepcopy(actor["messages"]))
        requests.append(
            dict(
                sequence=seq,
                slot_id=actor["slot_id"],
                payload=payload,
                request_digest=content_digest(payload),
                count_payload={
                    key: deepcopy(payload[key]) for key in ("model", "input", "reasoning", "tools")
                },
            )
        )
    result = dict(
        schema_version="knowledge-utility-followup-requests-v1",
        packet_digest=FROZEN_DIGEST,
        profile=deepcopy(PROFILE),
        requests=requests,
    )
    result["manifest_digest"] = content_digest(result)
    validate_json_value(result, max_bytes=MAX_INPUT_BYTES - 1)
    return result


def prepare_capture(packet: object, prices: object | None = None) -> JsonObject:
    """Prepare requests and optional generation reservations without provider access."""
    manifest = build_manifest(packet)
    estimate = None
    if prices is not None:
        price = _prices(prices)
        reserve = reservation_micro_usd(price)
        estimate = dict(
            per_request_micro_usd=reserve,
            roster_micro_usd=SLOT_COUNT * reserve,
            budget_scope="generation_only",
            counting_policy=price.counting_policy,
            counting_cost_usd="0" if price.counting_policy == "no_charge" else None,
        )
    result = dict(manifest=manifest, reservation_estimate=estimate, token_counts="unknown")
    validate_json_value(result, max_bytes=MAX_INPUT_BYTES - 1)
    return result

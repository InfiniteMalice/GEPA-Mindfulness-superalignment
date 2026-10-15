"""Exact follow-up request identities and independently approved capture constraints."""

# Standard library
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from scripts.knowledge_utility_capture_records import (
    PriceRecord,
    budget_micro_usd,
    normalized_path,
    parse_prices,
    reservation_micro_usd,
)
from scripts.knowledge_utility_followup_records import (
    MAX_INPUT_BYTES,
    JsonObject,
    bounded_text,
    object_fields,
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


@dataclass(frozen=True)
class FollowupExecution:
    """Snapshot of declared approval and conservative generation reservations."""

    raw: JsonObject
    prices: PriceRecord
    budget_micro_usd: int
    reservation_micro_usd: int


def _text(value: object, cap: int = 1024) -> None:
    bounded_text(value, cap)
    if type(value) is not str or "\x00" in value:
        raise ValueError("NUL in reference")


def _provider_review(value: object) -> JsonObject:
    review = object_fields(
        value,
        "verified_at source_urls model_available profile_compatible counting_charge_reference",
    )
    _utc(review["verified_at"])
    _text(review["counting_charge_reference"])
    if review["model_available"] is not True or review["profile_compatible"] is not True:
        raise ValueError("provider assertions required")
    urls = review["source_urls"]
    if type(urls) is not list or not 1 <= len(urls) <= 10:
        raise ValueError("provider sources required")
    for url in urls:
        _text(url, 2048)
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc not in (
            "openai.com",
            "developers.openai.com",
            "platform.openai.com",
        ):
            raise ValueError("official provider source required")
    return review


def validate_execution(
    raw: object,
    manifest: JsonObject,
    run_dir: Path,
    *,
    live_commit: str | None = None,
    now: datetime | None = None,
) -> FollowupExecution:
    """Bind approval before side effects; absent live context permits historical export."""
    validate_json_value(raw, max_bytes=MAX_APPROVAL_BYTES)
    value = object_fields(
        raw,
        "schema_version run_id packet_digest manifest_digest implementation_commit run_directory "
        "profile prices provider_review approved_budget_usd approved_count_requests "
        "approved_generation_requests accept_unknown_counting_cost execution_authorized "
        "approval_reference execution_digest",
    )
    if (
        value["schema_version"] != "knowledge-utility-followup-execution-v1"
        or value["execution_authorized"] is not True
        or value["execution_digest"]
        != content_digest({key: item for key, item in value.items() if key != "execution_digest"})
    ):
        raise ValueError("invalid execution declaration")
    for key in ("run_id", "approval_reference"):
        _text(value[key])
    for key in ("approved_count_requests", "approved_generation_requests"):
        if type(value[key]) is not int or value[key] != SLOT_COUNT:
            raise ValueError("invalid approved request limit")
    for key in ("packet_digest", "manifest_digest", "profile"):
        if canonical_json(value[key]) != canonical_json(manifest[key]):
            raise ValueError("execution binding mismatch")
    commit = value["implementation_commit"]
    if type(commit) is not str or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("invalid implementation binding")
    _text(value["run_directory"], 4096)
    directory = Path(value["run_directory"])
    if not directory.is_absolute() or normalized_path(directory) != normalized_path(run_dir):
        raise ValueError("run directory mismatch")
    prices = _prices(value["prices"])
    review = _provider_review(value["provider_review"])
    accepted = value["accept_unknown_counting_cost"]
    if type(accepted) is not bool or accepted != (prices.counting_policy == "unverified_accepted"):
        raise ValueError("counting charge acceptance mismatch")
    reserve = reservation_micro_usd(prices)
    budget = budget_micro_usd(value["approved_budget_usd"])
    if not SLOT_COUNT * reserve <= budget <= 5_000_000:
        raise ValueError("budget outside full-roster limits")
    if (live_commit is None) != (now is None):
        raise ValueError("incomplete live validation context")
    if live_commit is not None and now is not None:
        if commit != live_commit or now.utcoffset() != timedelta(0):
            raise ValueError("invalid live revision or clock")
        for stamp in (prices.verified_at, review["verified_at"]):
            age = now - _utc(stamp)
            if not timedelta(0) <= age <= timedelta(hours=24):
                raise ValueError("provider evidence not fresh")
        if os.path.lexists(run_dir) or not run_dir.parent.is_dir():
            raise ValueError("fresh run directory required")
    return FollowupExecution(deepcopy(value), prices, budget, reserve)

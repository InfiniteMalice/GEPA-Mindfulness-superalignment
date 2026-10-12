"""Pilot-specific request identities, exact money, and durable capture records."""

# Standard library
from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

# Third-party
# Local
from evaluation.causal_records import canonical_json, content_digest
from gepa_mindfulness.training.causal_reward_records import validate_input_envelope
from scripts.score_knowledge_utility_pilot import SUPPORTED_PACKET_DIGEST
from synthetic_data.curriculum_records import decode_curriculum_json

JsonObject = dict[str, Any]
MODEL = "gpt-5.4-mini-2026-03-17"
MAX_FILE_BYTES = 4 * 1024 * 1024


def read_bounded_json(path: Path) -> JsonObject:
    """Reject oversized, deep, duplicate-key, or non-object artifact inputs."""
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("artifact exceeds byte limit")
    value = decode_curriculum_json(data.decode("utf-8"))
    validate_input_envelope(value)
    if type(value) is not dict:
        raise ValueError("artifact requires an object")
    return value


def build_manifest(packet: JsonObject) -> JsonObject:
    """Project only frozen public messages into independent provider requests."""
    validate_input_envelope(packet)
    digest = content_digest({k: v for k, v in packet.items() if k != "packet_digest"})
    if packet.get("packet_digest") != digest or digest != SUPPORTED_PACKET_DIGEST:
        raise ValueError("unsupported frozen packet")
    profile = dict(
        model=MODEL,
        reasoning={"effort": "none"},
        temperature=1,
        max_output_tokens=256,
        tools=[],
        store=False,
        service_tier="default",
        truncation="disabled",
        stream=False,
    )
    requests = []
    for sequence, actor in enumerate(packet["actor_requests"]):
        payload = dict(deepcopy(profile), input=deepcopy(actor["messages"]))
        # Count all token-bearing input, including reasoning configuration and tools.
        count_payload = {k: deepcopy(payload[k]) for k in ("model", "input", "reasoning", "tools")}
        requests.append(
            dict(
                sequence=sequence,
                slot_id=actor["slot_id"],
                payload=payload,
                request_digest=content_digest(payload),
                count_payload=count_payload,
            )
        )
    manifest = dict(
        schema_version="knowledge-utility-pilot-requests-v1",
        packet_digest=digest,
        profile=profile,
        requests=requests,
    )
    manifest["manifest_digest"] = content_digest(manifest)
    return manifest


def _money(value: object) -> Decimal:
    if (
        type(value) is not str
        or len(value) > 32
        or not re.fullmatch(r"[0-9]+(\.[0-9]{1,6})?", value)
    ):
        raise ValueError("money requires a bounded unsigned decimal string")
    return Decimal(value)


def budget_micro_usd(value: str) -> int:
    """Convert exactly without increasing the approved ceiling."""
    with localcontext() as ctx:
        ctx.prec = 50
        return int(_money(value) * 1_000_000)


@dataclass(frozen=True)
class PriceRecord:
    """Operator assertions about prices; URL syntax does not authenticate them."""

    model: str
    service_tier: str
    currency: str
    verified_at: str
    source_urls: tuple[str, ...]
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    counting_policy: str


def parse_prices(raw: JsonObject) -> PriceRecord:
    """Require exact generation rates and an explicit counting-charge policy."""
    if type(raw) is not dict or set(raw) != set(PriceRecord.__dataclass_fields__):
        raise ValueError("invalid price fields")
    if (raw["model"], raw["service_tier"], raw["currency"]) != (
        MODEL,
        "default",
        "USD",
    ):
        raise ValueError("unsupported price profile")
    if raw["counting_policy"] not in ("no_charge", "unverified_accepted"):
        raise ValueError("unsupported counting policy")
    stamp = raw["verified_at"]
    if type(stamp) is not str or not stamp.endswith("Z"):
        raise ValueError("price timestamp requires UTC")
    datetime.fromisoformat(stamp[:-1] + "+00:00")
    urls = raw["source_urls"]
    if type(urls) is not list or not 1 <= len(urls) <= 10:
        raise ValueError("price sources required")
    for url in urls:
        if type(url) is not str or len(url) > 2048:
            raise ValueError("invalid price source")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc not in (
            "openai.com",
            "developers.openai.com",
            "platform.openai.com",
        ):
            raise ValueError("official price source required")
    return PriceRecord(
        MODEL,
        "default",
        "USD",
        stamp,
        tuple(urls),
        _money(raw["input_usd_per_million"]),
        _money(raw["output_usd_per_million"]),
        raw["counting_policy"],
    )


def reservation_micro_usd(prices: PriceRecord) -> int:
    """Reserve uncached worst-case tokens, rounding the charge upward."""
    with localcontext() as ctx:
        ctx.prec = 50
        cost = 2048 * prices.input_usd_per_million + 256 * prices.output_usd_per_million
        return int(cost.to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class ExecutionRecord:
    """Validated wire binding with computed conservative accounting limits."""

    raw: JsonObject
    prices: PriceRecord
    budget_micro_usd: int
    reservation_micro_usd: int


def normalized_path(path: Path) -> str:
    """Resolve aliases before comparing an operator's absolute directory binding."""
    return os.path.normcase(str(path.resolve()))


def validate_execution(
    raw: JsonObject, manifest: JsonObject, run_dir: Path, implementation_commit: str
) -> ExecutionRecord:
    """Reject changed bindings before directory ownership or credential access."""
    fields = (
        "schema_version run_id packet_digest manifest_digest implementation_commit "
        "run_directory profile prices approved_budget_usd approval_reference"
    )
    if type(raw) is not dict or set(raw) != set(fields.split()):
        raise ValueError("invalid execution fields")
    if raw["schema_version"] != "knowledge-utility-pilot-execution-v1":
        raise ValueError("unsupported execution schema")
    for key in ("run_id", "approval_reference"):
        if type(raw[key]) is not str or not raw[key].strip() or len(raw[key]) > 1024:
            raise ValueError("execution reference required")
    for key in ("packet_digest", "manifest_digest", "profile"):
        # Python equality conflates booleans and numbers; bind the approved JSON instead.
        if canonical_json(raw[key]) != canonical_json(manifest[key]):
            raise ValueError("execution binding mismatch")
    if not re.fullmatch(r"[a-f0-9]{40}", implementation_commit):
        raise ValueError("invalid implementation commit")
    if raw["implementation_commit"] != implementation_commit:
        raise ValueError("implementation commit mismatch")
    directory = raw["run_directory"]
    if type(directory) is not str or not Path(directory).is_absolute():
        raise ValueError("absolute run directory required")
    if normalized_path(Path(directory)) != normalized_path(run_dir):
        raise ValueError("run directory mismatch")
    price = parse_prices(raw["prices"])
    return ExecutionRecord(
        deepcopy(raw),
        price,
        min(5_000_000, budget_micro_usd(raw["approved_budget_usd"])),
        reservation_micro_usd(price),
    )


def utc_now() -> str:
    """Use explicit UTC for audit records regardless of host timezone."""
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, data: JsonObject) -> None:
    # Preserve a partial intent on failure: its existence blocks an unsafe resend.
    with path.open("xb") as stream:
        stream.write((canonical_json(data) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def publish_json(path: Path, data: JsonObject) -> None:
    """Publish a complete artifact without ever unlinking a shared destination."""
    with TemporaryDirectory(prefix=".pilot-capture-", dir=path.parent) as staging:
        staged = Path(staging) / "artifact.json"
        _write_new(staged, data)
        os.link(staged, path)


@dataclass(frozen=True)
class RunStore:
    """Single-owner, single-use run directory containing immutable evidence."""

    root: Path
    identity: JsonObject

    @classmethod
    def create(cls, run_dir: Path, execution: ExecutionRecord, manifest: JsonObject) -> RunStore:
        """Claim a fresh directory before reading any credential or making a call."""
        run_dir.mkdir()
        identity = {
            key: execution.raw[key] for key in ("run_id", "packet_digest", "manifest_digest")
        }
        store = cls(run_dir, identity)
        for child in ("counts", "attempts", "results"):
            (run_dir / child).mkdir()
        store.write_record("execution.json", {"execution": execution.raw})
        store.write_record("requests.json", {"manifest": manifest})
        store.write_record("execution-started.json", {"utc": utc_now()})
        return store

    def write_record(self, relative: str, data: JsonObject) -> None:
        """Persist only fixed metadata names or one of the 72 numbered slots."""
        metadata = {"execution.json", "requests.json", "execution-started.json", "terminal.json"}
        match = re.fullmatch(
            r"(?:counts/(\d{3})-(?:intent|result)|(?:attempts|results)/(\d{3}))\.json", relative
        )
        if relative not in metadata and (
            match is None or int(match.group(1) or match.group(2)) >= 72
        ):
            raise ValueError("uncontrolled run filename")
        if set(data) & set(self.identity):
            raise ValueError("record cannot replace run identity")
        _write_new(self.root / relative, dict(data, **self.identity))

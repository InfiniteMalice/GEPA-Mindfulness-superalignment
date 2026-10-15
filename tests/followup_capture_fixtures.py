"""Retained packet and synthetic execution/provider data; never live authorization."""

# Standard library
import json
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts.prepare_knowledge_utility_followup import prepare_followup

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
COMMIT = "a" * 40
FROZEN_DIGEST = "c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1"


@lru_cache(maxsize=1)
def _packet():
    packet = prepare_followup()
    record = (
        Path(__file__).parent / "fixtures/knowledge_utility_followup/accepted_freeze_record.json"
    )
    packet.update(
        schema_version="knowledge-utility-followup-frozen-packet-v1",
        status="frozen_with_accepted_task_review",
        packet_frozen=True,
        freeze_record=json.loads(record.read_text(encoding="utf-8")),
    )
    packet["packet_digest"] = content_digest(
        {key: value for key, value in packet.items() if key != "packet_digest"}
    )
    assert packet["packet_digest"] == FROZEN_DIGEST
    return packet


def approved_packet():
    """Reconstruct the retained frozen content without overriding its registration."""
    return deepcopy(_packet())


def price_record():
    """Supply synthetic rates for exact-money tests, not current provider prices."""
    return dict(
        model="gpt-5.4-mini-2026-03-17",
        service_tier="default",
        currency="USD",
        verified_at="2026-10-15T12:00:00Z",
        source_urls=["https://developers.openai.com/api/docs/pricing"],
        input_usd_per_million="0.75",
        output_usd_per_million="4.5",
        counting_policy="no_charge",
    )

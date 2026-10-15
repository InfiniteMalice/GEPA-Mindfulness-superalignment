"""Retained packet and synthetic execution/provider data; never live authorization."""

# Standard library
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Third-party
# Local
from evaluation.causal_records import content_digest
from scripts.knowledge_utility_capture_transport import TransportResult
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


def seal_execution(raw):
    raw["execution_digest"] = content_digest(
        {key: value for key, value in raw.items() if key != "execution_digest"}
    )
    return raw


def execution_record(manifest, run_dir, *, budget="1.032192"):
    """Declare synthetic test authorization; never write an operational approval."""
    return seal_execution(
        dict(
            schema_version="knowledge-utility-followup-execution-v1",
            run_id="synthetic-run",
            packet_digest=manifest["packet_digest"],
            manifest_digest=manifest["manifest_digest"],
            implementation_commit=COMMIT,
            run_directory=os.path.normcase(str(run_dir.resolve())),
            profile=deepcopy(manifest["profile"]),
            prices=price_record(),
            provider_review=dict(
                verified_at="2026-10-15T12:00:00Z",
                source_urls=["https://developers.openai.com/api/docs/models"],
                model_available=True,
                profile_compatible=True,
                counting_charge_reference="synthetic assertion; not billing evidence",
            ),
            approved_budget_usd=budget,
            approved_count_requests=384,
            approved_generation_requests=384,
            accept_unknown_counting_cost=False,
            execution_authorized=True,
            approval_reference="synthetic:not-human-approval",
        )
    )


def transport_result(body):
    return TransportResult(
        200,
        "synthetic-request",
        json.dumps(body).encode(),
        None,
        False,
        False,
        NOW.isoformat(),
        NOW.isoformat(),
        0.01,
    )


def provider_result(text='{"answer":"A"}', *, status="completed"):
    return transport_result(
        dict(
            object="response",
            model="gpt-5.4-mini-2026-03-17",
            status=status,
            error=None,
            usage=dict(input_tokens=100, output_tokens=10),
            incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
            output=[
                dict(
                    type="message",
                    role="assistant",
                    status=status,
                    content=[dict(type="output_text", text=text)],
                )
            ],
        )
    )


class FakeTransport:
    """Observe real orchestration while replacing only the provider boundary."""

    def __init__(self, responder=None):
        self.calls = []
        self.responder = responder

    def post(self, endpoint, payload):
        index = len(self.calls)
        self.calls.append((endpoint, deepcopy(payload)))
        if self.responder is not None:
            return self.responder(endpoint, payload, index)
        if endpoint == "responses/input_tokens":
            return transport_result(
                dict(
                    object="response.input_tokens",
                    input_tokens=100,
                    usage=dict(input_tokens=100, output_tokens=0),
                )
            )
        return provider_result()

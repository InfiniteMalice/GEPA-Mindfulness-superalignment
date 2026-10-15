"""Exercise the retained operational freeze through the production registry."""

# Standard library
import json
from pathlib import Path

# Third-party
import pytest

# Local
from evaluation.causal_records import content_digest
from scripts.score_knowledge_utility_followup import main, score_followup
from tests.followup_scoring_fixtures import capture_batch, fresh_packet

FROZEN_DIGEST = "c565ce074a50dafb3c3cfc318aa73e16610298de2d6cc0275fafa60c9656d9a1"
RECORD_PATH = (
    Path(__file__).parent / "fixtures/knowledge_utility_followup/accepted_freeze_record.json"
)


@pytest.fixture
def approved_packet():
    packet = fresh_packet()
    packet.update(
        schema_version="knowledge-utility-followup-frozen-packet-v1",
        status="frozen_with_accepted_task_review",
        packet_frozen=True,
        freeze_record=json.loads(RECORD_PATH.read_text(encoding="utf-8")),
    )
    packet["packet_digest"] = content_digest(
        {key: value for key, value in packet.items() if key != "packet_digest"}
    )
    assert packet["packet_digest"] == FROZEN_DIGEST
    return packet


def test_operational_packet_scores_without_registry_override(approved_packet):
    report = score_followup(approved_packet, capture_batch(approved_packet))
    draft = fresh_packet()
    baseline = score_followup(draft, capture_batch(draft))
    assert report["schema_version"] == "knowledge-utility-followup-report-v2"
    assert report["packet_state"] == dict(
        status="frozen_with_accepted_task_review",
        packet_frozen=True,
        execution_authorized=False,
        pending_task_reviews=0,
    )
    assert report["task_review_provenance"] == dict(
        source_packet_digest=draft["packet_digest"],
        candidate_digest="e8a41152d1f647779a4facbfddc06c2defd61987377950bb345066df172561b9",
        decision_digest="4c012dc2300e2780af357651d62273bf991eeddcd32015a11050d1c9d68cc767",
        acceptance_recorded=True,
        maintainer_identity_authenticated=False,
        reviewer_identity_authenticated=False,
        blinding_authenticated=False,
    )
    for field in (
        "conditions",
        "contrasts",
        "observations",
        "limitations",
        "training_eligibility",
        "verified",
        "paired",
        "confers_authority",
        "optimizer_input",
        "interval",
    ):
        assert report[field] == baseline[field]


def test_operational_packet_cli_publishes_report(approved_packet, tmp_path):
    packet_path = tmp_path / "packet.json"
    capture_path = tmp_path / "captures.json"
    output_path = tmp_path / "report.json"
    packet_path.write_text(json.dumps(approved_packet), encoding="utf-8")
    capture_path.write_text(json.dumps(capture_batch(approved_packet)), encoding="utf-8")
    main(
        [
            "--packet",
            str(packet_path),
            "--captures",
            str(capture_path),
            "--output",
            str(output_path),
        ]
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report == score_followup(approved_packet, capture_batch(approved_packet))


@pytest.mark.parametrize("change", ["execution", "task", "decision", "limitation"])
def test_rehashed_changes_to_registered_packet_stay_unsupported(approved_packet, change):
    if change == "execution":
        approved_packet["execution_authorized"] = True
    elif change == "task":
        approved_packet["evaluator_cases"][0]["prompt"] += " changed"
    elif change == "decision":
        decision = approved_packet["freeze_record"]["decision"]
        decision["authorization_ref"] = "different decision"
        decision["decision_digest"] = content_digest(
            {key: value for key, value in decision.items() if key != "decision_digest"}
        )
    else:
        candidate = approved_packet["freeze_record"]["acceptance"]
        candidate["limitations"].pop()
        candidate["candidate_digest"] = content_digest(
            {key: value for key, value in candidate.items() if key != "candidate_digest"}
        )
    approved_packet["packet_digest"] = content_digest(
        {key: value for key, value in approved_packet.items() if key != "packet_digest"}
    )
    with pytest.raises(ValueError, match="unsupported packet"):
        score_followup(approved_packet, capture_batch(approved_packet))

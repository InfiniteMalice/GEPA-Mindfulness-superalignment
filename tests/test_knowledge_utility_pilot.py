"""The proposed pilot is reproducible preparation, never authorization or capture."""

# Standard library
import json
from collections import Counter
from pathlib import Path

# Third-party
import pytest

# Local
from evaluation.causal_records import canonical_json, content_digest
from evaluation.improvement_audit import validate_manifest
from evaluation.improvement_records import DatasetManifest
from scripts import prepare_knowledge_utility_pilot as pilot
from scripts.prepare_knowledge_utility_pilot import main, prepare_pilot
from synthetic_data.curriculum_records import UtilityStudy


def test_packet_is_deterministic_and_complete() -> None:
    """All four conditions share the same six tasks and three repeats."""
    packet = prepare_pilot()
    assert canonical_json(packet) == canonical_json(prepare_pilot())
    digest = packet.pop("packet_digest")
    assert digest == content_digest(packet)
    study = UtilityStudy.from_dict(packet["utility_study"])
    assert len(study.expected_slots) == 72
    assert len({slot.slot_id for slot in study.expected_slots}) == 72
    assert Counter(slot.condition for slot in study.expected_slots) == {
        "none": 18,
        "lesson_only": 18,
        "bank_only": 18,
        "bank_plus_lesson": 18,
    }
    assert len({slot.family_id for slot in study.expected_slots}) == 6
    assert study.observations == () and study.seed_policy == "unknown"
    assert all(slot.seed is None for slot in study.expected_slots)


def test_preparation_cannot_grant_review_or_execution_authority() -> None:
    """Authored proposals stay pending and all empirical outcomes stay missing."""
    packet = prepare_pilot()
    assert packet["execution_authorized"] is False
    assert packet["training_eligibility"] == "DEVELOPMENT"
    assert packet["optimizer_input"] is False and packet["confers_authority"] is False
    assert packet["lesson"]["review_status"] == "pending"
    assert packet["bank"]["entries"][0]["lesson"]["review_status"] == "pending"
    assert packet["curation_preview"]["proposed_bank_additions"] == []
    assert packet["curation_preview"]["lessons"][0]["disposition"] == "withheld"
    assert all(row["missing"] == 18 for row in packet["utility_preview"]["conditions"].values())
    assert all(not row["verified"] for row in packet["utility_preview"]["comparisons"].values())


def test_actor_requests_omit_evaluator_labels_and_preserve_controls() -> None:
    """Only knowledge text changes across conditions for the same public task."""
    packet = prepare_pilot()
    expected = {row["case_id"]: row for row in packet["evaluator_cases"]}
    requests = packet["actor_requests"]
    assert {row["slot_id"] for row in requests} == {
        row["slot_id"] for row in packet["utility_study"]["expected_slots"]
    }
    by_case = {}
    knowledge = {
        "none": "",
        "lesson_only": packet["lesson"]["summary"],
        "bank_only": packet["bank"]["entries"][0]["lesson"]["summary"],
        "bank_plus_lesson": packet["bank"]["entries"][0]["lesson"]["summary"]
        + "\n"
        + packet["lesson"]["summary"],
    }
    for request in requests:
        assert set(request) == {"slot_id", "case_id", "condition", "messages"}
        assert all(set(turn) == {"role", "content"} for turn in request["messages"])
        task = request["messages"][-1]["content"]
        assert task == expected[request["case_id"]]["prompt"]
        context = knowledge[request["condition"]]
        instruction = pilot.BASE_INSTRUCTION + (
            "\nKnowledge context:\n" + context if context else ""
        )
        assert request["messages"][0] == {"role": "system", "content": instruction}
        assert request["messages"][1]["role"] == "user"
        by_case.setdefault(request["case_id"], set()).add(task)
    assert all(len(prompts) == 1 for prompts in by_case.values())
    assert packet["limits"]["generation_requests"] == 72
    assert packet["limits"]["automatic_retries"] == 0


def test_source_and_audit_lineage_remain_separate_and_unverified() -> None:
    """Local partition checks do not certify external semantic independence."""
    manifest = DatasetManifest.from_dict(prepare_pilot()["manifest"])
    assert validate_manifest(manifest)
    assert not manifest.dependencies_known
    source_families = {c.family_id for c in manifest.cases if c.purpose == "synthetic_training"}
    audit_families = {c.family_id for c in manifest.cases if c.purpose == "independent_audit"}
    assert source_families.isdisjoint(audit_families)
    assert len(source_families) == 2 and len(audit_families) == 6


def test_authored_rubric_answers_are_hand_checked() -> None:
    """Fixed toy scenarios have explicit answers calculated from their public facts."""
    rows = prepare_pilot()["evaluator_cases"]
    assert {row["family_id"]: row["expected_answer"] for row in rows} == {
        "inventory": "6",
        "precedence": "A",
        "route_cost": "VIA",
        "intervals": "DISJOINT",
        "quorum": "FAIL",
        "units": "LONGER",
    }


def test_cli_refuses_to_overwrite_a_frozen_packet(tmp_path: Path) -> None:
    """Re-running preparation cannot silently replace the packet being reviewed."""
    output = tmp_path / "packet.json"
    main(["--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == prepare_pilot()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_bytes() == original


def test_changed_evaluator_answers_never_enter_actor_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator contamination changes bindings but cannot change public messages."""
    original = prepare_pilot()
    cases = [
        dict(case, expected_answer="EVALUATOR_ONLY_SENTINEL")
        for case in original["evaluator_cases"]
    ]
    monkeypatch.setattr(pilot, "_cases", lambda: cases)
    changed = prepare_pilot()
    assert changed["actor_requests"] == original["actor_requests"]
    assert changed["packet_digest"] != original["packet_digest"]
    before = original["utility_study"]["expected_slots"]
    after = changed["utility_study"]["expected_slots"]
    assert all(a["rubric_digest"] != b["rubric_digest"] for a, b in zip(before, after))

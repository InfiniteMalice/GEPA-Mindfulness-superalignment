"""Lineage and journal integrity precede any claim of independent improvement."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_improvement_records import DIGEST, REF, candidate, case, protocol

# Local
from evaluation.causal_records import content_digest
from evaluation.improvement_audit import independence_status, summarize_journal, validate_manifest
from evaluation.improvement_records import (
    AttemptEvent,
    AttemptJournal,
    CostEntry,
    DatasetManifest,
    EvaluationSlot,
    ExposureRecord,
    record_digest,
)


def test_five_partitions_and_transitive_clusters():
    cases = tuple(
        case(p, p)
        for p in (
            "synthetic_training",
            "optimizer_selection",
            "independent_audit",
            "final_test",
            "ood_combinations",
        )
    )
    definition = '{"combination":["novel","conditions"]}'
    cases = cases[:-1] + (
        replace(cases[-1], combination_digest=content_digest(json.loads(definition))),
    )
    manifest = DatasetManifest("data", cases, (REF,), (definition,))
    assert len(set(validate_manifest(manifest).values())) == 5
    linked = (case("root"), case("child", parent_ids=("root",)), case("sibling", family_id="child"))
    assert len(set(validate_manifest(DatasetManifest("data", linked)).values())) == 1


@pytest.mark.parametrize(
    "bad",
    [
        case("child", "final_test", parent_ids=("root",)),
        case("child", "final_test", family_id="root"),
        case("child", "final_test", content_digest=content_digest("root")),
        case("child", parent_ids=("missing",)),
        case("child", "ood_combinations"),
    ],
)
def test_contaminated_or_incomplete_lineage_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_manifest(DatasetManifest("data", (case("root"), bad)))


def test_cycles_duplicate_ids_and_shared_chains_are_rejected():
    for cases in (
        (case("a", parent_ids=("b",)), case("b", parent_ids=("a",))),
        (case("a"), case("a")),
        (
            case("a", transformation_ids=("chain",)),
            case("b", "final_test", transformation_ids=("chain",)),
        ),
    ):
        with pytest.raises(ValueError):
            validate_manifest(DatasetManifest("data", cases))


def journal_fixture():
    candidates = tuple(candidate(str(i)) for i in range(6))
    slots = tuple(
        EvaluationSlot(str(i), str(min(i, 4)), "case", "accuracy", "candidate", i, DIGEST, "base")
        for i in range(6)
    )
    spec, _ = protocol(candidates=candidates, slots=slots)
    events = []

    def event(cid, kind, **kwargs):
        ordinal = len(events)
        events.append(
            AttemptEvent(str(ordinal), ordinal, str(cid), "r1" if cid < 3 else "r2", kind, **kwargs)
        )

    for i in range(6):
        event(i, "proposal")
    for i in range(6):
        event(min(i, 4), "evaluation_started", slot_id=str(i))
    for i in range(4):
        event(
            i,
            "evaluation_finished",
            slot_id=str(i),
            costs=(CostEntry(0.5, "USD", "estimated", "USD"),),
        )
        event(i, "decision", terminal_status=("selected", "rejected", "failed", "cancelled")[i])
    return spec, AttemptJournal("journal", record_digest(spec), tuple(events), (REF,))


def test_all_attempts_costs_and_pending_results_remain_visible():
    spec, journal = journal_fixture()
    result = summarize_journal(spec, journal)
    assert (
        result["candidate_count"],
        result["evaluation_attempt_count"],
        result["selection_round_count"],
    ) == (5, 6, 2)
    assert len(result["candidates"]) == 6
    assert result["planned_unattempted"] == ["5"]
    assert result["cost"]["groups"][0]["known_total"] == 2
    assert result["cost"]["unmeasured_attempts"] == 2
    assert result["candidates"][2]["status"] == "failed"


@pytest.mark.parametrize("mutation", ["duplicate", "order", "foreign", "finish", "decision"])
def test_ambiguous_journal_is_rejected(mutation):
    spec, journal = journal_fixture()
    events = list(journal.events)
    if mutation == "duplicate":
        events[-1] = replace(events[-1], event_id=events[0].event_id)
    elif mutation == "order":
        events[-1] = replace(events[-1], ordinal=0)
    elif mutation == "foreign":
        events[6] = replace(events[6], candidate_id="5")
    elif mutation == "finish":
        events.append(AttemptEvent("extra", 99, "4", "r2", "evaluation_finished", "missing"))
    else:
        events.append(AttemptEvent("extra", 99, "0", "r1", "decision", terminal_status="failed"))
    with pytest.raises(ValueError):
        summarize_journal(spec, replace(journal, events=tuple(events)))


def test_audit_influence_propagates_to_descendants_even_after_freeze():
    spec, _ = protocol(
        candidates=(candidate("parent"), candidate("child", parent_candidate_id="parent"))
    )
    exposure = ExposureRecord(20, DIGEST, "independent_audit", ("parent",), (), "selection")
    assert (
        independence_status(spec, "child", "independent_audit", (exposure,))["independent"] is False
    )
    only_eval = replace(exposure, use="evaluation_only")
    assert (
        independence_status(spec, "child", "independent_audit", (only_eval,))["independent"] is True
    )
    with pytest.raises(ValueError):
        independence_status(
            spec, "child", "independent_audit", (replace(exposure, candidate_ids=("unknown",)),)
        )

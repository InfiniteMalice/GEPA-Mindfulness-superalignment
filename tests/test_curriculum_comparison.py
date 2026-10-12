"""Actual matched ablations and held-out independence require complete host evidence."""

# Standard library
import json
from dataclasses import replace

# Third-party
import pytest
from test_improvement_integration import fixture as audit_fixture

# Local
from curriculum_fixtures import comparison_inputs, curation_inputs
from evaluation.causal_records import canonical_json, content_digest
from evaluation.curriculum_comparison import audit_curriculum_comparison
from evaluation.improvement_audit import audit_improvement, independence_status
from evaluation.improvement_records import AttemptEvent, ExposureRecord, record_digest
from synthetic_data.curriculum_records import CurationExposure
from synthetic_data.lesson_curation import curate_lessons


def run(inputs=None, callback=lambda p: True):
    return audit_curriculum_comparison(
        **(inputs or comparison_inputs()),
        authenticate_manifest=callback,
        authenticate_audit=lambda request: True,
        enabled=True,
    )


def row(report, name="synthetic"):
    return next(arm for arm in report["arms"] if arm["arm_id"] == name)


def test_seven_arms_have_explicit_controls():
    report = run()
    assert len(report["arms"]) == 7
    assert row(report)["control_arm_id"] == "baseline"
    assert row(report)["isolated"] is True
    assert row(report, "knowledge")["ready"] is False
    assert report["deployment_eligibility"] == "not_assessed"
    inputs = comparison_inputs()
    inputs["arms"] = inputs["arms"][:-1]
    with pytest.raises(ValueError):
        run(inputs)


def test_planned_match_does_not_imply_actual_match():
    report = run()
    assert row(report)["planned_budget_match"] is True
    assert row(report)["actual_budget_match"] is None
    inputs = comparison_inputs()
    arms = [replace(a, actual_budget_json=a.planned_budget_json) for a in inputs["arms"]]
    budget = json.loads(arms[1].actual_budget_json)
    budget["train"]["compute"] += 1
    arms[1] = replace(arms[1], actual_budget_json=canonical_json(budget))
    inputs["arms"] = tuple(arms)
    assert row(run(inputs))["actual_budget_match"] is False


def test_unintended_component_change():
    inputs = comparison_inputs()
    inputs["arms"] = (
        inputs["arms"][0],
        replace(inputs["arms"][1], reward_digest=content_digest("changed")),
    ) + inputs["arms"][2:]
    assert row(run(inputs))["isolated"] is False
    assert "reward_digest_mismatch" in row(run(inputs))["reasons"]


@pytest.mark.parametrize("control_seeds", [(), (None,)])
def test_independent_comparison_requires_known_control_seeds(control_seeds):
    inputs = comparison_inputs()
    inputs["arms"] = tuple(
        replace(
            arm,
            seed_policy="independent",
            seeds=control_seeds if arm.arm_id == "baseline" else (7,),
            actual_budget_json=arm.planned_budget_json,
        )
        for arm in inputs["arms"]
    )
    result = row(run(inputs))
    assert result["isolated"] is False
    assert result["ready"] is False
    assert result["matched_execution"] is False
    assert "seed_policy_unverified" in result["reasons"]


def test_inherited_undeclared_component_prevents_isolation():
    inputs = comparison_inputs()
    arms = list(inputs["arms"])
    for index in (0, 1):
        components = json.loads(arms[index].components_json)
        components["laundering"] = content_digest("unexpected")
        arms[index] = replace(arms[index], components_json=canonical_json(components))
    inputs["arms"] = tuple(arms)
    assert row(run(inputs))["isolated"] is False


def test_control_result_matches_protocol_baseline():
    inputs = comparison_inputs()
    candidates = list(inputs["protocol"].candidates)
    candidates[1] = replace(candidates[1], baseline=candidates[0].baseline)
    inputs["protocol"] = replace(inputs["protocol"], candidates=tuple(candidates))
    inputs["journal"] = replace(
        inputs["journal"], protocol_digest=record_digest(inputs["protocol"])
    )
    assert row(run(inputs))["isolated"] is False


def test_knowledge_bank_ancestor_exposes_holdout():
    inputs = comparison_inputs()
    exposure = CurationExposure(
        ordinal=0,
        case_ids=("audit-child",),
        candidate_ids=("knowledge",),
        round_ids=(),
        source_json='{"bank_ancestor":"audit-child"}',
        evidence_refs=inputs["manifest"].evidence_refs,
    )
    inputs["curation_exposures"] = (exposure,)
    report = run(inputs)
    derived = tuple(ExposureRecord.from_dict(e) for e in report["derived_exposures"])
    status = independence_status(inputs["protocol"], "knowledge", "independent_audit", derived)
    assert status["independent"] is False
    assert report["inputs"]["curation_exposures"][0]["source_json"] == exposure.source_json


def knowledge_inputs():
    inputs = comparison_inputs()
    curation = curation_inputs()
    report = curate_lessons(**curation, authenticate=lambda p: True, enabled=True)
    inputs["arms"] = inputs["arms"][:-1] + (
        replace(inputs["arms"][-1], knowledge_report_json=canonical_json(report)),
    )
    inputs["curation_exposures"] = (
        CurationExposure(
            ordinal=0,
            case_ids=("case-1",),
            candidate_ids=("knowledge",),
            round_ids=(),
            source_json=canonical_json(report),
            evidence_refs=inputs["manifest"].evidence_refs,
        ),
    )
    return inputs


def test_knowledge_arm_requires_fresh_authentication():
    inputs = knowledge_inputs()
    assert row(run(inputs), "knowledge")["ready"] is True
    assert row(run(inputs, None), "knowledge")["ready"] is False
    inputs["as_of"] = "2026-10-12T00:00:00Z"
    assert row(run(inputs), "knowledge")["ready"] is False


def test_knowledge_missing_exposure_withholds():
    inputs = knowledge_inputs()
    inputs["curation_exposures"] = ()
    assert row(run(inputs), "knowledge")["ready"] is False


@pytest.mark.parametrize("callback", [None, lambda p: 1, lambda p: p.clear() or True])
def test_untrusted_manifest_cannot_endorse_comparison(callback):
    assert row(run(callback=callback))["isolated"] is False


def test_cancelled_attempt_and_severe_event_survive():
    inputs = comparison_inputs()
    protocol, manifest, journal, evidence = audit_fixture()
    extra = protocol.candidates[0]
    inputs["protocol"] = replace(protocol, candidates=inputs["protocol"].candidates + (extra,))
    inputs["manifest"] = manifest
    inputs["arms"] = tuple(replace(a, data_case_ids=("selection",)) for a in inputs["arms"])
    events = journal.events + (
        AttemptEvent("cancel-proposal", 20, "synthetic", "other", "proposal"),
        AttemptEvent("cancel", 21, "synthetic", "other", "decision", terminal_status="cancelled"),
    )
    inputs["journal"] = replace(
        journal, events=events, protocol_digest=record_digest(inputs["protocol"])
    )
    inputs["evidence"] = evidence
    expected = audit_improvement(
        inputs["protocol"],
        manifest,
        inputs["journal"],
        evidence,
        authenticate=lambda request: True,
        enabled=True,
    )
    report = run(inputs)
    assert report["improvement_audit"] == expected
    assert report["improvement_audit"]["severe_events"]
    assert any(
        c["status"] == "cancelled"
        for c in report["improvement_audit"]["optimization_progress"]["candidates"]
    )

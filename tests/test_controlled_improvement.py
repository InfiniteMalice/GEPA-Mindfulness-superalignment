"""Controlled improvement intake binds diagnostics without granting authority."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields, replace

import pytest
from test_model_harness_coevolution import _base, _events, _ref

from gepa_mindfulness.coevolution import CandidateComponent, CorrectionProposal
from gepa_mindfulness.controlled_improvement import TriageDiagnostics, assess_improvement
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.learning_surfaces import close_evaluation_epoch
from gepa_mindfulness.training.eligibility import require_training_eligible
from gepa_mindfulness.verification.epistemic_state import (
    Availability,
    EpistemicContext,
    EpistemicStateEstimate,
)
from gepa_mindfulness.verification.failure_graph import (
    FailureEdge,
    FailureGraph,
    FailureNode,
    FailureRelation,
    FailureRole,
    FailureRoleEvidence,
)
from mindful_trace_gepa.event_sequence import EvaluatedSystemVersion


def _supported(proposal, *, hypothesis=False):
    graph = proposal.failure_graph
    root = FailureNode(
        "failure:root",
        "event:prediction",
        "Public model error",
        "2026-09-10T11:59:56Z",
        (_ref("evidence:prediction"),),
    )
    relation = FailureRelation.HYPOTHESIZED if hypothesis else FailureRelation.CAUSAL
    edge = FailureEdge(
        root.failure_id,
        graph.nodes[0].failure_id,
        relation,
        () if hypothesis else ("verifier:causal",),
    )
    localization = replace(
        graph.localization,
        root_cause=root.failure_id,
        role_evidence=graph.localization.role_evidence
        + (FailureRoleEvidence(FailureRole.ROOT_CAUSE, root.failure_id, ("verifier:root",)),),
    )
    return replace(
        proposal, failure_graph=FailureGraph((root,) + graph.nodes, (edge,), localization)
    )


def _estimate(**changes):
    base = EpistemicStateEstimate(
        estimate_id="estimate:1",
        context=EpistemicContext("run:source", 0, EvaluatedSystemVersion("model:v1", "harness:v1")),
        estimator_version="public-unit-v1",
        action_id="action:source",
        prediction_commit_id="prediction:source",
        world_uncertainty=0.1,
        model_uncertainty=0.2,
        monitor_uncertainty=0.1,
        evidence_refs=(_ref(),),
        provenance=("observable-model-v1",),
    )
    return replace(base, **changes)


def _triage(**changes):
    base = TriageDiagnostics(
        protocol_id="triage:v1",
        evidence_refs=(_ref(),),
        severity=0.1,
        irreversibility=0.1,
        recurrence=0.1,
        ood_novelty=0.1,
        systemic_effect=0.1,
        autonomy_impact=0.1,
        reward_hacking_signal=0.1,
    )
    return replace(base, **changes)


@pytest.fixture
def source(tmp_path):
    store, history, proposal, *_ = _base(tmp_path)
    return store, history, _supported(proposal)


def _assess(source, **changes):
    args = dict(store=source[0], correction=source[2], estimate=_estimate(), triage=_triage())
    args.update(changes)
    return assess_improvement(**args, enabled=True)


def _catalogs(path):
    dumps = []
    for name in ("evaluation.sqlite", "coevolution.sqlite"):
        with sqlite3.connect(path / name) as db:
            dumps.append(tuple(db.iterdump()))
    return tuple(dumps)


def test_intake_is_read_only_and_correction_uses_existing_registration(source, tmp_path):
    before = _catalogs(tmp_path)
    result = _assess(source)
    assert result["route"] == "sandbox_review"
    assert result["priority"] == "routine"
    assert result["reasons"] == []
    assert result["root_cause_status"] == "supported"
    assert result["authority_granted"] is False
    assert _catalogs(tmp_path) == before
    assert _assess(source) == result
    json.dumps(result, allow_nan=False)
    proposal = CorrectionProposal.from_dict(result["correction"])
    candidate = source[0].register_candidate(
        candidate_id="candidate:intake",
        correction=proposal,
        artifact_digest="sha256:" + "b" * 64,
    )
    assert candidate.correction == source[2]
    assert candidate.rollback_target_epoch_id == "epoch:source"
    with pytest.raises(ValueError):
        source[0].consume_decision(result)


def test_training_restriction_applies_to_complete_envelope(source):
    result = _assess(source)
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        require_training_eligible(result)
    # Closed legacy schemas do not retain the envelope's eligibility label.
    require_training_eligible(result["correction"])
    require_training_eligible(result["estimate"])


@pytest.mark.parametrize("enabled", [False, None, 1, "yes"])
def test_disabled_or_invalid_enable_flag_never_reads_catalog(enabled):
    with pytest.raises(ValueError, match="enabled"):
        assess_improvement(None, None, None, None, enabled=enabled)


@pytest.mark.parametrize(
    "name,limit",
    [
        ("severity", 0.8),
        ("systemic_effect", 0.8),
        ("irreversibility", 0.5),
        ("autonomy_impact", 0.5),
        ("reward_hacking_signal", 0.5),
    ],
)
def test_consequence_thresholds_require_human_review(source, name, limit):
    below = _assess(source, triage=_triage(**{name: limit - 0.001}))
    assert below["route"] == "sandbox_review"
    result = _assess(source, triage=_triage(**{name: limit}))
    assert result["route"] == "human_review"
    assert result["priority"] == "urgent"
    assert result["correction"] is None
    assert f"high_{name}" in result["reasons"]


@pytest.mark.parametrize("name", ["recurrence", "ood_novelty"])
def test_recurrence_and_novelty_elevate_investigation_priority_without_reward(source, name):
    result = _assess(source, triage=_triage(**{name: 0.5}))
    assert result["priority"] == "elevated"
    assert result["route"] == "sandbox_review"
    assert "reward" not in result


@pytest.mark.parametrize("name", [f.name for f in fields(TriageDiagnostics)][2:])
def test_missing_triage_is_preserved_and_requires_investigation(source, name):
    result = _assess(source, triage=_triage(**{name: None}))
    assert result["route"] == "investigate"
    assert result["priority"] == "elevated"
    assert result["triage"][name] is None
    assert f"missing_{name}" in result["reasons"]


@pytest.mark.parametrize("name", ["world_uncertainty", "model_uncertainty", "monitor_uncertainty"])
@pytest.mark.parametrize("value", [None, 0.5, 1.0])
def test_unknown_or_elevated_uncertainty_cannot_release_correction(source, name, value):
    result = _assess(source, estimate=_estimate(**{name: value}))
    assert result["route"] == "investigate"
    assert result["correction"] is None
    assert result["estimate"][name] == value


def test_high_consequence_takes_precedence_over_missingness(source):
    result = _assess(source, triage=_triage(severity=0.8, recurrence=None))
    assert result["route"] == "human_review"
    assert "high_severity" in result["reasons"]
    assert "missing_recurrence" in result["reasons"]


def test_unavailable_estimate_is_investigation_not_zero_uncertainty(source):
    result = _assess(
        source,
        estimate=_estimate(
            world_uncertainty=None,
            model_uncertainty=None,
            monitor_uncertainty=None,
            status=Availability.UNAVAILABLE,
            evidence_refs=(),
        ),
    )
    assert result["route"] == "investigate"


@pytest.mark.parametrize("hypothesis", [False, True])
def test_root_cause_qualification_is_retained(tmp_path, hypothesis):
    store, history, proposal, *_ = _base(tmp_path)
    if hypothesis:
        proposal = _supported(proposal, hypothesis=True)
    result = _assess((store, history, proposal))
    assert result["root_cause_status"] == ("hypothesized" if hypothesis else None)
    assert result["route"] == "investigate"
    assert "root_cause_unqualified" in result["reasons"]


def test_multicomponent_edits_require_investigation(source):
    proposal = replace(source[2], changed_components=tuple(CandidateComponent))
    result = _assess(source, correction=proposal)
    assert result["route"] == "investigate"
    assert "multiple_components" in result["reasons"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("run_id", "other"),
        ("repeat_id", 1),
        ("system", EvaluatedSystemVersion("other", "harness:v1")),
        ("system", EvaluatedSystemVersion("model:v1", "other")),
    ],
)
def test_estimate_context_must_match_catalog_events(source, name, value):
    estimate = _estimate(context=replace(_estimate().context, **{name: value}))
    with pytest.raises(ValueError, match="context"):
        _assess(source, estimate=estimate)


@pytest.mark.parametrize("name", ["action_id", "prediction_commit_id"])
@pytest.mark.parametrize("value", [None, "other"])
def test_estimate_requires_exact_source_action_prediction(source, name, value):
    with pytest.raises(ValueError, match="action|prediction"):
        _assess(source, estimate=_estimate(**{name: value}))


@pytest.mark.parametrize(
    "kind", [EvidenceSourceKind.PRIVATE_REASONING, EvidenceSourceKind.LATENT_STATE]
)
def test_private_or_latent_estimate_evidence_is_rejected(source, kind):
    with pytest.raises(ValueError, match="observable"):
        _assess(
            source,
            estimate=_estimate(evidence_refs=(EvidenceReference("evidence:localized", kind),)),
        )


@pytest.mark.parametrize("target", ["estimate", "triage"])
def test_unrelated_evidence_is_rejected(source, target):
    value = (
        _estimate(evidence_refs=(_ref("other"),))
        if target == "estimate"
        else _triage(
            evidence_refs=(_ref("other"),),
        )
    )
    with pytest.raises(ValueError, match="localized"):
        _assess(source, **{target: value})


@pytest.mark.parametrize("value", [True, -1, 1.01, float("nan"), float("inf"), "0.1", 10**400])
def test_triage_rejects_invalid_numbers(value):
    with pytest.raises(ValueError):
        _triage(severity=value)


@pytest.mark.parametrize(
    "refs",
    [
        (),
        [_ref()],
        (_ref(),) * 33,
        (EvidenceReference("private", EvidenceSourceKind.PRIVATE_REASONING),),
    ],
)
def test_triage_requires_bounded_observable_references(refs):
    with pytest.raises(ValueError):
        _triage(evidence_refs=refs)


def test_mutated_typed_inputs_are_revalidated(source):
    triage = _triage()
    object.__setattr__(triage, "severity", True)
    with pytest.raises(ValueError):
        _assess(source, triage=triage)
    estimate = _estimate()
    object.__setattr__(estimate, "world_uncertainty", True)
    with pytest.raises(ValueError):
        _assess(source, estimate=estimate)
    proposal = source[2]
    object.__setattr__(proposal, "source_action_id", "forged")
    with pytest.raises(ValueError):
        _assess(source, correction=proposal)


@pytest.mark.parametrize("name", ["store", "correction", "estimate", "triage"])
def test_exact_input_types_are_required(source, name):
    with pytest.raises(ValueError):
        _assess(source, **{name: object()})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_trajectory_id", "absent"),
        ("source_epoch_id", "other"),
        ("trajectory_digest", "sha256:" + "f" * 64),
        ("source_action_id", "other"),
    ],
)
def test_correction_source_revalidates_catalog_binding(source, field, value):
    with pytest.raises(ValueError):
        source[0].correction_source_events(replace(source[2], **{field: value}))


def test_correction_events_are_detached_and_repeatable(source):
    assert source[0].correction_source_events(source[2]) == _events()
    assert source[0].correction_source_events(source[2])[0] is not _events()[0]


def test_corrupted_catalog_trajectory_fails_closed(source, tmp_path):
    with sqlite3.connect(tmp_path / "coevolution.sqlite") as db:
        payload = json.loads(db.execute("SELECT payload FROM trajectories").fetchone()[0])
        payload["events"][0]["run_id"] = "changed"
        db.execute("UPDATE trajectories SET payload = ?", (json.dumps(payload),))
    with pytest.raises(ValueError):
        _assess(source)


def test_source_versions_must_equal_closed_epoch(source):
    store = source[0]
    events = tuple(replace(event, model_version="other") for event in _events())
    binding = store.register_trajectory(
        trajectory_id="trajectory:other",
        source_epoch_id="epoch:source",
        events=events,
        event_evidence_refs=(
            ("event:prediction", _ref("evidence:prediction")),
            ("event:outcome", _ref()),
        ),
        source_evidence_refs=(_ref("evidence:prediction"), _ref()),
    )
    proposal = replace(
        source[2],
        source_trajectory_id=binding.trajectory_id,
        trajectory_digest=binding.trajectory_digest,
    )
    with pytest.raises(ValueError, match="versions"):
        store.correction_source_events(proposal)


def test_intake_does_not_reserve_or_bypass_later_candidate_epoch(source):
    result = _assess(source)
    close_evaluation_epoch(source[1])
    with pytest.raises(ValueError, match="open|empty|closed"):
        source[0].register_candidate(
            candidate_id="candidate:late",
            correction=CorrectionProposal.from_dict(result["correction"]),
            artifact_digest="sha256:" + "c" * 64,
        )


def test_evidence_instance_serializer_cannot_substitute_private_data(source):
    ref = _ref()
    object.__setattr__(ref, "to_dict", lambda: {"unexpected": "private"})
    result = _assess(
        source, estimate=_estimate(evidence_refs=(ref,)), triage=_triage(evidence_refs=(ref,))
    )
    assert result["triage"]["evidence_refs"] == [_ref().to_dict()]
    assert result["estimate"]["evidence_refs"] == [_ref().to_dict()]


def test_malformed_protocol_and_numeric_subclass_fail_closed():
    class PretendNumber(float):
        pass

    for protocol in ("", "x" * 129, "\ud800", 1):
        with pytest.raises(ValueError):
            _triage(protocol_id=protocol)
    with pytest.raises(ValueError):
        _triage(severity=PretendNumber(0.1))
    assert _triage(severity=0, recurrence=1).severity == 0.0

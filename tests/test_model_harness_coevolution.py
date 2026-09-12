"""Acceptance tests for durable, offline model and harness coevolution."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from evaluation import (
    BehaviorRecord,
    CaseIdentity,
    DiagnosticRecord,
    EpistemicRecord,
    OutcomeRecord,
    RobustnessIdentity,
    ScoreRecord,
    SystemIdentity,
    V5EvaluationRecord,
)
from gepa_mindfulness import (
    AcceptanceDecision,
    CandidateComponent,
    CandidateSystem,
    CandidateTargetClaim,
    CoevolutionStore,
    ComponentMetric,
    CorrectionProposal,
    CorrectionScope,
    EvaluationEpochStore,
    MetricAggregation,
    MetricDirection,
    MetricPolicy,
    MetricSpec,
    ProtectedSuiteManifest,
    TrajectoryBinding,
    ValidationBundle,
    ValidationSplit,
    append_epoch_record,
    begin_candidate_epoch,
    close_evaluation_epoch,
    decide_candidate_acceptance,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    FailureGraph,
    FailureLocalization,
    FailureNode,
    FailureRole,
    FailureRoleEvidence,
)
from mindful_trace_gepa import validate_action_bound_sequence
from mindful_trace_gepa.logging_schema import EventEnvelope

_NAMES = ("correctness", "calibration", "abstention", "epistemic_process", "total")


def _ref(identifier: str = "evidence:localized") -> EvidenceReference:
    return EvidenceReference(identifier, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _failure_graph() -> FailureGraph:
    return FailureGraph(
        (
            FailureNode(
                "failure:localized",
                "event:outcome",
                "The source action produced the wrong observable artifact.",
                "2026-09-10T12:00:00Z",
                (_ref(),),
            ),
        ),
        (),
        FailureLocalization(
            "failure:localized",
            None,
            None,
            (),
            "failure:localized",
            (
                FailureRoleEvidence(
                    FailureRole.FIRST_ANOMALY,
                    "failure:localized",
                    ("verifier:localization",),
                ),
                FailureRoleEvidence(
                    FailureRole.RECOVERABLE_UNTIL,
                    "failure:localized",
                    ("verifier:recovery-boundary",),
                ),
            ),
        ),
    )


def _events() -> tuple[EventEnvelope, ...]:
    common = {
        "run_id": "run:source",
        "repeat_id": 0,
        "model_version": "model:v1",
        "harness_version": "harness:v1",
    }
    prediction = EventEnvelope(
        "1.0",
        "event:prediction",
        "prediction_commit",
        "2026-09-10T11:59:56Z",
        payload={
            "prediction_commit_id": "prediction:source",
            "predicted_outcome": {"result": "expected"},
            "confidence": 0.8,
            "evidence_refs": ["evidence:prediction"],
        },
        evidence_refs=("evidence:prediction",),
        **common,
    )
    proposed = EventEnvelope(
        "1.0",
        "event:proposed",
        "action_proposed",
        "2026-09-10T11:59:57Z",
        payload={
            "action_id": "action:source",
            "action_class": "read",
            "reversible": True,
            "authorization_scope": "sandbox",
            "prediction_commit_id": "prediction:source",
        },
        action_id="action:source",
        parent_event_ids=(prediction.event_id,),
        authorization_scope="sandbox",
        **common,
    )
    executed = EventEnvelope(
        "1.0",
        "event:executed",
        "action_executed",
        "2026-09-10T11:59:58Z",
        payload=dict(proposed.payload),
        action_id="action:source",
        parent_event_ids=(proposed.event_id,),
        authorization_scope="sandbox",
        **common,
    )
    outcome = EventEnvelope(
        "1.0",
        "event:outcome",
        "outcome_observed",
        "2026-09-10T11:59:59Z",
        payload={
            "observation_id": "observation:source",
            "action_id": "action:source",
            "actual_outcome": {"result": "wrong"},
            "evidence_refs": ["evidence:localized"],
        },
        action_id="action:source",
        parent_event_ids=(executed.event_id,),
        evidence_refs=("evidence:localized",),
        **common,
    )
    verification = EventEnvelope(
        "1.0",
        "event:verification",
        "verification_result",
        "2026-09-10T12:00:00Z",
        payload={
            "verifier_id": "verifier:source",
            "verifier_version": "v1",
            "observation_id": "observation:source",
            "verified": True,
            "verifier_refs": ["verifier:localization", "verifier:recovery-boundary"],
        },
        parent_event_ids=(outcome.event_id,),
        verifier_refs=("verifier:localization", "verifier:recovery-boundary"),
        **common,
    )
    events = (prediction, proposed, executed, outcome, verification)
    validate_action_bound_sequence(events)
    return events


def _event_evidence(
    localized_kind: EvidenceSourceKind = EvidenceSourceKind.OBSERVABLE_OUTPUT,
) -> tuple[tuple[str, EvidenceReference], ...]:
    return (
        ("event:prediction", _ref("evidence:prediction")),
        (
            "event:outcome",
            EvidenceReference("evidence:localized", localized_kind),
        ),
    )


def _record(
    model: str,
    harness: str,
    repeat: int,
    *,
    total: float = 0.875,
    calibration: float = 0.8,
    passed: bool = True,
) -> V5EvaluationRecord:
    return V5EvaluationRecord(
        CaseIdentity(
            14,
            "17case-v5",
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        RobustnessIdentity("TOOL_ERROR", None),
        SystemIdentity(repeat, 4_200 + repeat, model, harness),
        EpistemicRecord(
            f"prediction:{repeat}",
            (f"evidence:{repeat}",),
            (f"verifier:{repeat}",),
            0.8,
        ),
        BehaviorRecord((f"action:{repeat}",), True, True),
        OutcomeRecord(
            (f"observation:{repeat}",),
            (f"outcome-verifier:{repeat}",),
            passed,
        ),
        ScoreRecord(1.0, calibration, 1.0, 0.7, total),
        DiagnosticRecord("diagnostic only", 0.1, 0.2),
    )


def _versions(components: tuple[CandidateComponent, ...]) -> tuple[str, str]:
    model = "model:v2" if CandidateComponent.MODEL in components else "model:v1"
    harness = "harness:v2" if CandidateComponent.HARNESS in components else "harness:v1"
    return model, harness


def _base(
    tmp_path: Path,
    components: tuple[CandidateComponent, ...] = (CandidateComponent.MODEL,),
) -> tuple[
    CoevolutionStore,
    object,
    CorrectionProposal,
    V5EvaluationRecord,
    tuple[V5EvaluationRecord, ...],
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    evaluation = EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod")
    history = evaluation.create_root(
        lineage_id="main",
        epoch_id="epoch:source",
        model_version="model:v1",
        harness_version="harness:v1",
    )
    source_held = _record("model:v1", "harness:v1", 0)
    source_protected = (
        _record("model:v1", "harness:v1", 1),
        _record("model:v1", "harness:v1", 2),
    )
    append_epoch_record(history, source_held)
    for record in source_protected:
        append_epoch_record(history, record)
    close_evaluation_epoch(history)
    model, harness = _versions(components)
    begin_candidate_epoch(
        history,
        epoch_id="epoch:candidate",
        model_version=model,
        harness_version=harness,
    )
    authority = CoevolutionStore(
        tmp_path / "coevolution.sqlite",
        "coevolution-prod",
        evaluation,
        lineage_id="main",
    )
    binding = authority.register_trajectory(
        trajectory_id="trajectory:1",
        source_epoch_id="epoch:source",
        events=_events(),
        event_evidence_refs=_event_evidence(),
        source_evidence_refs=(
            _ref("evidence:prediction"),
            _ref(),
        ),
    )
    proposal = CorrectionProposal(
        "correction:1",
        "trajectory:1",
        "action:source",
        "epoch:source",
        _failure_graph(),
        "failure:localized",
        ("verifier:localization", "verifier:recovery-boundary"),
        binding.source_evidence_refs,
        "Change only the component responsible for the localized failure.",
        (EvidenceReference("teacher:proposal", EvidenceSourceKind.PRIVATE_REASONING),),
        components,
        binding.trajectory_digest,
    )
    return authority, history, proposal, source_held, source_protected


def _flow(
    tmp_path: Path,
    components: tuple[CandidateComponent, ...] = (CandidateComponent.MODEL,),
    *,
    candidate_total: float = 0.9,
    primary: str = "total",
) -> tuple[CoevolutionStore, ValidationBundle]:
    authority, history, proposal, source_held, source_protected = _base(tmp_path, components)
    candidate = authority.register_candidate(
        candidate_id="candidate:1",
        correction=proposal,
        artifact_digest="sha256:" + "a" * 64,
    )
    model, harness = _versions(components)
    candidate_held = _record(model, harness, 0, total=candidate_total, calibration=0.7)
    candidate_protected = (
        _record(model, harness, 1, total=candidate_total),
        _record(model, harness, 2, total=candidate_total),
    )
    append_epoch_record(history, candidate_held)
    for record in candidate_protected:
        append_epoch_record(history, record)
    close_evaluation_epoch(history)
    source_held_receipt = authority.issue_source_validation_receipt(
        epoch_id="epoch:source",
        split=ValidationSplit.HELD_OUT,
        records=(source_held,),
    )
    source_protected_receipt = authority.issue_source_validation_receipt(
        epoch_id="epoch:source",
        split=ValidationSplit.PROTECTED,
        records=source_protected,
    )
    manifest = authority.register_protected_suite(
        suite_id="protected:v5", source_receipt=source_protected_receipt
    )
    policy = MetricPolicy(
        "metric-policy:v1",
        tuple(
            MetricSpec(
                name,
                (
                    MetricDirection.LOWER_IS_BETTER
                    if name == "calibration"
                    else MetricDirection.HIGHER_IS_BETTER
                ),
                0.0,
            )
            for name in _NAMES
        ),
        primary,
        MetricAggregation.ARITHMETIC_MEAN,
    )
    authority.register_metric_policy(policy)
    held = authority.issue_candidate_validation_receipt(
        candidate_id=candidate.candidate_id,
        split=ValidationSplit.HELD_OUT,
        records=(candidate_held,),
    )
    protected = authority.issue_candidate_validation_receipt(
        candidate_id=candidate.candidate_id,
        split=ValidationSplit.PROTECTED,
        records=candidate_protected,
    )
    comparison = authority.issue_metric_comparison(
        candidate_id=candidate.candidate_id,
        policy_id=policy.policy_id,
        source_receipt=source_held_receipt,
        candidate_receipt=held,
    )
    return authority, ValidationBundle(candidate, held, protected, comparison, manifest.suite_id)


@pytest.mark.parametrize(
    "components",
    [
        (CandidateComponent.MODEL,),
        (CandidateComponent.HARNESS,),
        (CandidateComponent.MODEL, CandidateComponent.HARNESS),
    ],
)
def test_model_harness_and_combined_candidates_use_new_offline_epoch(
    tmp_path: Path, components: tuple[CandidateComponent, ...]
) -> None:
    authority, bundle = _flow(tmp_path, components)
    decision = decide_candidate_acceptance(bundle, authority_store=authority)

    assert decision.accepted is True
    assert decision.candidate.changed_components == components
    assert decision.rollback_target_epoch_id == "epoch:source"
    assert decision.execute_candidate is False
    assert authority.evaluation_epochs()[-1].closed is True


def test_teacher_proposal_requires_exact_registered_localized_failure(tmp_path: Path) -> None:
    authority, _history, proposal, _held, _protected = _base(tmp_path)
    wholesale = proposal.to_dict()
    wholesale["scope"] = CorrectionScope.WHOLE_TRAJECTORY.value
    with pytest.raises(ValueError, match="localized"):
        CorrectionProposal.from_dict(wholesale)

    unrelated = proposal.to_dict()
    unrelated["source_action_id"] = "action:unrelated"
    with pytest.raises(ValueError, match="action"):
        authority.register_candidate(
            candidate_id="candidate:unrelated",
            correction=CorrectionProposal.from_dict(unrelated),
            artifact_digest="sha256:" + "a" * 64,
        )

    forged = proposal.to_dict()
    forged["localization_verifier_refs"] = ["verifier:fake-a", "verifier:fake-b"]
    failure_graph = cast(dict[str, object], forged["failure_graph"])
    localization = cast(dict[str, object], failure_graph["localization"])
    roles = cast(list[dict[str, object]], localization["role_evidence"])
    roles[0]["verifier_refs"] = ["verifier:fake-a"]
    roles[1]["verifier_refs"] = ["verifier:fake-b"]
    with pytest.raises(ValueError, match="verifier evidence"):
        authority.register_candidate(
            candidate_id="candidate:forged",
            correction=CorrectionProposal.from_dict(forged),
            artifact_digest="sha256:" + "b" * 64,
        )


def test_trajectory_binding_rejects_rebinding_and_inexact_evidence(tmp_path: Path) -> None:
    authority, _history, _proposal, _held, _protected = _base(tmp_path)
    with pytest.raises(ValueError, match="already|registered"):
        authority.register_trajectory(
            trajectory_id="trajectory:1",
            source_epoch_id="epoch:source",
            events=_events(),
            event_evidence_refs=_event_evidence(),
            source_evidence_refs=(_ref("evidence:prediction"), _ref()),
        )
    with pytest.raises(ValueError, match="source_evidence_refs"):
        authority.register_trajectory(
            trajectory_id="trajectory:2",
            source_epoch_id="epoch:source",
            events=_events(),
            event_evidence_refs=_event_evidence(),
            source_evidence_refs=(_ref(),),
        )


def test_trajectory_evidence_rejects_private_reasoning_laundering(tmp_path: Path) -> None:
    evaluation = EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod")
    history = evaluation.create_root(
        lineage_id="main",
        epoch_id="epoch:source",
        model_version="model:v1",
        harness_version="harness:v1",
    )
    close_evaluation_epoch(history)
    authority = CoevolutionStore(
        tmp_path / "coevolution.sqlite",
        "coevolution-prod",
        evaluation,
        lineage_id="main",
    )
    private = EvidenceReference("evidence:localized", EvidenceSourceKind.PRIVATE_REASONING)
    with pytest.raises(ValueError, match="observable|source_kind"):
        authority.register_trajectory(
            trajectory_id="trajectory:laundered",
            source_epoch_id="epoch:source",
            events=_events(),
            event_evidence_refs=_event_evidence(EvidenceSourceKind.PRIVATE_REASONING),
            source_evidence_refs=(_ref("evidence:prediction"), private),
        )


def test_candidate_target_registration_is_atomic_under_alias_race(tmp_path: Path) -> None:
    authority, _history, proposal, _held, _protected = _base(tmp_path)

    def register(identifier: str) -> object:
        try:
            return authority.register_candidate(
                candidate_id=identifier,
                correction=proposal,
                artifact_digest="sha256:" + identifier[-1] * 64,
            )
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(register, ("candidate:a", "candidate:b")))
    assert sum(type(item) is CandidateSystem for item in outcomes) == 1
    assert sum(type(item) is ValueError for item in outcomes) == 1


def test_evaluation_catalog_claim_is_idempotent_and_rejects_epoch_alias(tmp_path: Path) -> None:
    authority, _history, _proposal, _held, _protected = _base(tmp_path)
    evaluation = EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod")
    owner = authority.authority()
    correction_digest = "sha256:" + "c" * 64
    claim = evaluation.claim_candidate_target(
        lineage_id="main",
        epoch_id="epoch:candidate",
        candidate_id="candidate:1",
        artifact_digest="sha256:" + "a" * 64,
        coevolution_catalog_id=owner.catalog_id,
        coevolution_authority_domain=owner.authority_domain,
        correction_proposal_digest=correction_digest,
    )

    assert (
        evaluation.claim_candidate_target(
            lineage_id="main",
            epoch_id="epoch:candidate",
            candidate_id="candidate:1",
            artifact_digest="sha256:" + "a" * 64,
            coevolution_catalog_id=owner.catalog_id,
            coevolution_authority_domain=owner.authority_domain,
            correction_proposal_digest=correction_digest,
        )
        == claim
    )
    assert claim.coevolution_catalog_id == owner.catalog_id
    assert claim.coevolution_authority_domain == owner.authority_domain
    assert claim.correction_proposal_digest == correction_digest
    assert (
        EvaluationEpochStore(
            tmp_path / "evaluation.sqlite", "evaluation-prod"
        ).resolve_candidate_target_claim("candidate:1")
        == claim
    )
    tampered = claim.to_dict()
    tampered["correction_proposal_digest"] = "sha256:" + "d" * 64
    with pytest.raises(ValueError, match="digest|provenance"):
        CandidateTargetClaim.from_dict(tampered)
    with pytest.raises(ValueError, match="claim|alias|registered"):
        evaluation.claim_candidate_target(
            lineage_id="main",
            epoch_id="epoch:candidate",
            candidate_id="candidate:alias",
            artifact_digest="sha256:" + "b" * 64,
            coevolution_catalog_id=owner.catalog_id,
            coevolution_authority_domain=owner.authority_domain,
            correction_proposal_digest=correction_digest,
        )


def test_candidate_handoff_recovers_after_evaluation_claim_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, history, proposal, _held, _protected = _base(tmp_path)
    original = EvaluationEpochStore.claim_candidate_target
    claims = []

    def claim_then_crash(store: EvaluationEpochStore, **kwargs: object) -> object:
        claim = original(store, **kwargs)
        claims.append(claim)
        raise RuntimeError("simulated crash after evaluation claim")

    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", claim_then_crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        authority.register_candidate(
            candidate_id="candidate:1",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )
    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", original)
    append_epoch_record(history, _record("model:v2", "harness:v1", 0))
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch:later",
        model_version="model:v3",
        harness_version="harness:v1",
    )

    candidate = authority.register_candidate(
        candidate_id="candidate:1",
        correction=proposal,
        artifact_digest="sha256:" + "a" * 64,
    )
    assert candidate.epoch_revision == claims[0].epoch_revision
    assert (
        authority.register_candidate(
            candidate_id="candidate:1",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )
        == candidate
    )


def test_claim_only_recovery_rejects_changed_correction_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, history, proposal, _held, _protected = _base(tmp_path)
    original = EvaluationEpochStore.claim_candidate_target

    def claim_then_crash(store: EvaluationEpochStore, **kwargs: object) -> object:
        original(store, **kwargs)
        raise RuntimeError("simulated crash after evaluation claim")

    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", claim_then_crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        authority.register_candidate(
            candidate_id="candidate:1",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )
    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", original)
    append_epoch_record(history, _record("model:v2", "harness:v1", 0))
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch:later",
        model_version="model:v3",
        harness_version="harness:v1",
    )

    for field_name, changed in (
        ("proposal_id", "correction:changed"),
        ("teacher_correction", "A different teacher correction."),
    ):
        payload = proposal.to_dict()
        payload[field_name] = changed
        with pytest.raises(ValueError, match="claim|correction|proposal"):
            authority.register_candidate(
                candidate_id="candidate:1",
                correction=CorrectionProposal.from_dict(payload),
                artifact_digest="sha256:" + "a" * 64,
            )

    payload = proposal.to_dict()
    failure_graph = cast(dict[str, object], payload["failure_graph"])
    nodes = cast(list[dict[str, object]], failure_graph["nodes"])
    nodes[0]["summary"] = "A different localized failure description."
    with pytest.raises(ValueError, match="claim|correction|proposal"):
        authority.register_candidate(
            candidate_id="candidate:1",
            correction=CorrectionProposal.from_dict(payload),
            artifact_digest="sha256:" + "a" * 64,
        )


def test_claim_only_recovery_rejects_another_coevolution_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, _history, proposal, _held, _protected = _base(tmp_path)
    original = EvaluationEpochStore.claim_candidate_target

    def claim_then_crash(store: EvaluationEpochStore, **kwargs: object) -> object:
        original(store, **kwargs)
        raise RuntimeError("simulated crash after evaluation claim")

    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", claim_then_crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        authority.register_candidate(
            candidate_id="candidate:1",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )
    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", original)
    other = CoevolutionStore(
        tmp_path / "other-coevolution.sqlite",
        "other-coevolution",
        EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod"),
        lineage_id="main",
    )
    other.register_trajectory(
        trajectory_id="trajectory:1",
        source_epoch_id="epoch:source",
        events=_events(),
        event_evidence_refs=_event_evidence(),
        source_evidence_refs=(_ref("evidence:prediction"), _ref()),
    )
    with pytest.raises(ValueError, match="catalog|authority|owner|claim"):
        other.register_candidate(
            candidate_id="candidate:1",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )


def test_candidate_target_must_precede_records_in_new_open_epoch(tmp_path: Path) -> None:
    authority, history, proposal, _held, _protected = _base(tmp_path)
    append_epoch_record(history, _record("model:v2", "harness:v1", 0))
    with pytest.raises(ValueError, match="empty open"):
        authority.register_candidate(
            candidate_id="candidate:late",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )


def test_candidate_claim_closes_old_check_to_insert_interleaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority, history, proposal, _held, _protected = _base(tmp_path)
    original = EvaluationEpochStore.claim_candidate_target

    def interleaved_claim(store: EvaluationEpochStore, **kwargs: object) -> object:
        append_epoch_record(history, _record("model:v2", "harness:v1", 0))
        close_evaluation_epoch(history)
        return original(store, **kwargs)

    monkeypatch.setattr(EvaluationEpochStore, "claim_candidate_target", interleaved_claim)
    with pytest.raises(ValueError, match="empty open tip epoch"):
        authority.register_candidate(
            candidate_id="candidate:interleaved",
            correction=proposal,
            artifact_digest="sha256:" + "a" * 64,
        )


def test_record_derived_metrics_are_complete_directional_and_mutation_bound(
    tmp_path: Path,
) -> None:
    authority, bundle = _flow(tmp_path, primary="calibration")
    receipt = bundle.metric_receipt
    assert tuple(metric.name for metric in receipt.component_metrics) == _NAMES
    assert receipt.primary_metric_name == "calibration"
    assert bundle.primary_metric_name == "calibration"
    decision = decide_candidate_acceptance(bundle, authority_store=authority)
    assert decision.primary_metric_name == "calibration"
    assert decision.accepted is True
    assert receipt.component_metrics[-1].baseline_value == 0.875
    assert receipt.component_metrics[-1].candidate_value == 0.9

    object.__setattr__(receipt.component_metrics[-1], "baseline_value", 0.95)
    object.__setattr__(receipt.component_metrics[-1], "candidate_value", 1.0)
    with pytest.raises(ValueError, match="ComponentMetric|changed"):
        receipt.to_dict()
    with pytest.raises(ValueError, match="ComponentMetric|changed"):
        receipt.component_metrics[-1].is_non_worse()


def test_declared_primary_metric_can_authoritatively_reject_candidate(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path, candidate_total=0.8)
    decision = decide_candidate_acceptance(bundle, authority_store=authority)

    assert decision.accepted is False
    assert decision.primary_metric_name == "total"
    assert "worse" in decision.reason


@pytest.mark.parametrize("hostile", [True, float("nan"), 1, "1.0"])
def test_component_metric_rejects_bool_nan_and_coercion(hostile: object) -> None:
    with pytest.raises(ValueError, match="finite|float"):
        ComponentMetric(
            "total",
            cast(float, hostile),
            1.0,
            MetricDirection.HIGHER_IS_BETTER,
            0.0,
        )


def test_metric_policy_rejects_missing_duplicate_and_conflicting_schema() -> None:
    complete = tuple(MetricSpec(name, MetricDirection.HIGHER_IS_BETTER, 0.0) for name in _NAMES)
    for invalid in (complete[:-1], complete[:-1] + (complete[0],)):
        with pytest.raises(ValueError, match="every V5 score key"):
            MetricPolicy(
                "policy:invalid",
                invalid,
                "total",
                MetricAggregation.ARITHMETIC_MEAN,
            )


def test_failed_protected_result_cannot_receive_regression_receipt(tmp_path: Path) -> None:
    authority, history, proposal, _source_held, _source_protected = _base(tmp_path)
    candidate = authority.register_candidate(
        candidate_id="candidate:1",
        correction=proposal,
        artifact_digest="sha256:" + "a" * 64,
    )
    failed = _record("model:v2", "harness:v1", 1, passed=False)
    append_epoch_record(history, failed)
    close_evaluation_epoch(history)
    with pytest.raises(ValueError, match="pass"):
        authority.issue_candidate_validation_receipt(
            candidate_id=candidate.candidate_id,
            split=ValidationSplit.PROTECTED,
            records=(failed,),
        )


def test_protected_manifest_rejects_relabel_and_wrong_coverage(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        decide_candidate_acceptance(
            ValidationBundle(
                bundle.candidate,
                bundle.held_out_receipt,
                bundle.held_out_receipt,
                bundle.metric_receipt,
                bundle.protected_suite_id,
            ),
            authority_store=authority,
        )
    payload = bundle.to_dict()
    protected = dict(cast(dict[str, object], payload["protected_receipt"]))
    protected["split"] = ValidationSplit.HELD_OUT.value
    payload["protected_receipt"] = protected
    with pytest.raises(ValueError, match="protected|catalog"):
        decide_candidate_acceptance(ValidationBundle.from_dict(payload), authority_store=authority)

    candidate = bundle.candidate
    model, harness = candidate.model_version, candidate.harness_version
    subset = authority.issue_candidate_validation_receipt(
        candidate_id=candidate.candidate_id,
        split=ValidationSplit.PROTECTED,
        records=(_record(model, harness, 1, total=0.9),),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        decide_candidate_acceptance(
            ValidationBundle(
                candidate,
                bundle.held_out_receipt,
                subset,
                bundle.metric_receipt,
                bundle.protected_suite_id,
            ),
            authority_store=authority,
        )
    superset = authority.issue_candidate_validation_receipt(
        candidate_id=candidate.candidate_id,
        split=ValidationSplit.PROTECTED,
        records=(
            _record(model, harness, 1, total=0.9),
            _record(model, harness, 2, total=0.9),
            _record(model, harness, 0, total=0.9, calibration=0.7),
        ),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        decide_candidate_acceptance(
            ValidationBundle(
                candidate,
                bundle.held_out_receipt,
                superset,
                bundle.metric_receipt,
                bundle.protected_suite_id,
            ),
            authority_store=authority,
        )


def test_decision_requires_store_validation_and_is_single_consume(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path)
    before = authority.evaluation_epochs()
    decision = decide_candidate_acceptance(bundle, authority_store=authority)
    restored = AcceptanceDecision.from_dict(json.loads(json.dumps(decision.to_dict())))

    assert decision.is_authoritative is True
    assert restored.is_authoritative is False
    with pytest.raises(ValueError, match="authoritative|validated"):
        authority.consume_decision(restored)
    validated = authority.validate_decision(restored)
    assert authority.read_decision(decision.decision_id) == validated
    assert authority.consume_decision(validated) == validated
    with pytest.raises(ValueError, match="consumed"):
        authority.consume_decision(validated)
    assert authority.evaluation_epochs() == before


def test_identical_acceptance_input_has_one_decision_even_after_consume(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path)
    first = decide_candidate_acceptance(bundle, authority_store=authority)
    duplicate = decide_candidate_acceptance(bundle, authority_store=authority)

    assert duplicate.to_dict() == first.to_dict()
    authority.consume_decision(first)
    replay = decide_candidate_acceptance(bundle, authority_store=authority)
    assert replay.to_dict() == first.to_dict()
    with pytest.raises(ValueError, match="consumed"):
        authority.consume_decision(replay)


def test_concurrent_and_restarted_duplicate_decisions_resolve_one_token(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        decisions = tuple(
            executor.map(
                lambda _index: decide_candidate_acceptance(bundle, authority_store=authority),
                range(4),
            )
        )
    assert len({decision.decision_id for decision in decisions}) == 1
    assert len({decision.input_digest for decision in decisions}) == 1

    restarted = CoevolutionStore(
        tmp_path / "coevolution.sqlite",
        "coevolution-prod",
        EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod"),
        lineage_id="main",
    )
    duplicate = decide_candidate_acceptance(bundle, authority_store=restarted)
    assert duplicate.to_dict() == decisions[0].to_dict()


def test_json_tamper_and_cross_store_replay_are_rejected(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path / "one")
    other, _ = _flow(tmp_path / "two")
    decision = decide_candidate_acceptance(bundle, authority_store=authority)
    tampered = decision.to_dict()
    tampered["accepted"] = False
    with pytest.raises(ValueError, match="digest"):
        AcceptanceDecision.from_dict(tampered)

    coherent = decision.to_dict()
    coherent["accepted"] = False
    coherent["reason"] = "candidate primary metric is worse than the declared tolerance"
    coherent.pop("decision_digest")
    encoded = json.dumps(coherent, separators=(",", ":"), sort_keys=True).encode("utf-8")
    coherent["decision_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    forged = AcceptanceDecision.from_dict(coherent)
    assert forged.is_authoritative is False
    with pytest.raises(ValueError, match="canonical"):
        authority.validate_decision(forged)

    with pytest.raises(ValueError, match="catalog|absent|decision"):
        other.validate_decision(AcceptanceDecision.from_dict(decision.to_dict()))


def test_restart_preserves_target_manifest_metrics_and_decision(tmp_path: Path) -> None:
    authority, bundle = _flow(tmp_path)
    decision = decide_candidate_acceptance(bundle, authority_store=authority)
    restarted = CoevolutionStore(
        tmp_path / "coevolution.sqlite",
        "coevolution-prod",
        EvaluationEpochStore(tmp_path / "evaluation.sqlite", "evaluation-prod"),
        lineage_id="main",
    )
    assert restarted.read_candidate("candidate:1") == bundle.candidate
    assert restarted.read_decision(decision.decision_id).to_dict() == decision.to_dict()


def test_public_exports_include_authority_records() -> None:
    import gepa_mindfulness

    assert gepa_mindfulness.CoevolutionStore is CoevolutionStore
    assert gepa_mindfulness.TrajectoryBinding is TrajectoryBinding
    assert gepa_mindfulness.ProtectedSuiteManifest is ProtectedSuiteManifest
    assert gepa_mindfulness.AcceptanceDecision is AcceptanceDecision

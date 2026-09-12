"""Acceptance tests for controlled offline model and harness coevolution."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

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
    CandidateComponent,
    CandidateSystem,
    ComponentMetric,
    CorrectionProposal,
    CorrectionScope,
    EvaluationEpochStore,
    MetricDirection,
    ValidationBundle,
    ValidationSplit,
    append_epoch_record,
    begin_candidate_epoch,
    bind_candidate_system,
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


def _ref(identifier: str = "evidence:localized") -> EvidenceReference:
    return EvidenceReference(identifier, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _failure_graph() -> FailureGraph:
    node = FailureNode(
        "failure:localized",
        "action:source",
        "The source action produced the wrong observable artifact.",
        "2026-09-10T12:00:00Z",
        (_ref(),),
    )
    localization = FailureLocalization(
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
    )
    return FailureGraph((node,), (), localization)


def _proposal(
    *,
    changed_components: tuple[CandidateComponent, ...] = (CandidateComponent.MODEL,),
    scope: CorrectionScope = CorrectionScope.LOCALIZED_FAILURE,
    localized_failure_id: str = "failure:localized",
) -> CorrectionProposal:
    return CorrectionProposal(
        proposal_id="correction:1",
        source_trajectory_id="trajectory:1",
        source_action_id="action:source",
        source_epoch_id="epoch:source",
        failure_graph=_failure_graph(),
        localized_failure_id=localized_failure_id,
        localization_verifier_refs=(
            "verifier:localization",
            "verifier:recovery-boundary",
        ),
        source_evidence_refs=(_ref(),),
        teacher_correction="Change only the component responsible for the localized failure.",
        teacher_evidence_refs=(
            EvidenceReference("teacher:proposal", EvidenceSourceKind.PRIVATE_REASONING),
        ),
        changed_components=changed_components,
        scope=scope,
    )


def _record(
    *,
    model_version: str,
    harness_version: str,
    repeat_id: int,
    passed: bool = True,
) -> V5EvaluationRecord:
    return V5EvaluationRecord(
        case=CaseIdentity(
            14,
            "17case-v5",
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        robustness=RobustnessIdentity("TOOL_ERROR", None),
        system=SystemIdentity(
            repeat_id,
            4_200 + repeat_id,
            model_version,
            harness_version,
        ),
        epistemics=EpistemicRecord(
            f"prediction:{repeat_id}",
            (f"evidence:{repeat_id}",),
            (f"verifier:{repeat_id}",),
            0.8,
        ),
        behavior=BehaviorRecord((f"action:{repeat_id}",), True, True),
        outcome=OutcomeRecord(
            (f"observation:{repeat_id}",),
            (f"outcome-verifier:{repeat_id}",),
            passed,
        ),
        scores=ScoreRecord(1.0, 0.8, 1.0, 0.7, 0.875),
        diagnostics=DiagnosticRecord("diagnostic only", 0.1, 0.2),
    )


def _candidate_versions(
    components: tuple[CandidateComponent, ...],
) -> tuple[str, str]:
    model = "model:v2" if CandidateComponent.MODEL in components else "model:v1"
    harness = "harness:v2" if CandidateComponent.HARNESS in components else "harness:v1"
    return model, harness


def _prepared_candidate(
    tmp_path: Path,
    *,
    components: tuple[CandidateComponent, ...] = (CandidateComponent.MODEL,),
    candidate_id: str = "candidate:1",
) -> tuple[EvaluationEpochStore, Any, CandidateSystem]:
    store = EvaluationEpochStore(tmp_path / f"{candidate_id.replace(':', '-')}.sqlite", "prod")
    history = store.create_root(
        lineage_id=f"lineage:{candidate_id}",
        epoch_id="epoch:source",
        model_version="model:v1",
        harness_version="harness:v1",
    )
    close_evaluation_epoch(history)
    model, harness = _candidate_versions(components)
    begin_candidate_epoch(
        history,
        epoch_id=f"epoch:{candidate_id}",
        model_version=model,
        harness_version=harness,
    )
    candidate = bind_candidate_system(
        store,
        lineage_id=f"lineage:{candidate_id}",
        candidate_id=candidate_id,
        correction=_proposal(changed_components=components),
        artifact_digest="sha256:" + "a" * 64,
    )
    return store, history, candidate


def _validated_bundle(
    tmp_path: Path,
    *,
    components: tuple[CandidateComponent, ...] = (CandidateComponent.MODEL,),
    candidate_id: str = "candidate:1",
    metrics: tuple[ComponentMetric, ...] | None = None,
) -> tuple[EvaluationEpochStore, ValidationBundle]:
    store, history, candidate = _prepared_candidate(
        tmp_path,
        components=components,
        candidate_id=candidate_id,
    )
    model, harness = _candidate_versions(components)
    held_out = _record(model_version=model, harness_version=harness, repeat_id=0)
    protected = _record(model_version=model, harness_version=harness, repeat_id=1)
    append_epoch_record(history, held_out)
    append_epoch_record(history, protected)
    close_evaluation_epoch(history)
    target = candidate.validation_target()
    held_receipt = store.issue_validation_receipt(
        history,
        epoch_id=candidate.candidate_epoch_id,
        target=target,
        split=ValidationSplit.HELD_OUT,
        records=(held_out,),
    )
    protected_receipt = store.issue_validation_receipt(
        history,
        epoch_id=candidate.candidate_epoch_id,
        target=target,
        split=ValidationSplit.PROTECTED,
        records=(protected,),
    )
    declared_metrics = metrics or (
        ComponentMetric("accuracy", 0.8, 0.81, MetricDirection.HIGHER_IS_BETTER, 0.0),
        ComponentMetric("latency_ms", 100.0, 99.0, MetricDirection.LOWER_IS_BETTER, 1.0),
    )
    return store, ValidationBundle(
        candidate,
        held_receipt,
        protected_receipt,
        declared_metrics,
        declared_metrics[0].name,
    )


@pytest.mark.parametrize(
    "components",
    [
        (CandidateComponent.MODEL,),
        (CandidateComponent.HARNESS,),
        (CandidateComponent.MODEL, CandidateComponent.HARNESS),
    ],
)
def test_complete_offline_flow_accepts_model_harness_or_joint_candidates(
    tmp_path: Path,
    components: tuple[CandidateComponent, ...],
) -> None:
    """Catch valid component-specific candidate epochs being rejected or conflated."""

    store, bundle = _validated_bundle(tmp_path, components=components)

    decision = decide_candidate_acceptance(
        bundle,
        trusted_store=store,
        trusted_authority=store.authority(),
        required_protected_record_ids=bundle.protected_receipt.record_ids,
    )

    assert decision.accepted is True
    assert decision.rollback_target_epoch_id == "epoch:source"
    assert decision.component_metrics == bundle.component_metrics
    assert decision.primary_metric_name == "accuracy"
    assert decision.execute_candidate is False
    assert decision.reason == "candidate passed held-out and protected acceptance gates"


def test_teacher_prose_is_only_a_scoped_proposal_not_acceptance_authority(tmp_path: Path) -> None:
    """Catch persuasive teacher text granting acceptance without store-issued validation."""

    _, _, candidate = _prepared_candidate(tmp_path)
    assert candidate.correction.teacher_correction
    assert not hasattr(candidate.correction, "accepted")
    assert not hasattr(candidate.correction, "authorized")

    with pytest.raises(ValueError, match="held_out_receipt"):
        ValidationBundle(candidate, cast(Any, None), cast(Any, None), (), "accuracy")


def test_wholesale_or_unlocalized_correction_is_rejected() -> None:
    """Catch trajectory imitation or a failure label not bound to graph localization."""

    with pytest.raises(ValueError, match="localized failure"):
        _proposal(scope=CorrectionScope.WHOLE_TRAJECTORY)
    with pytest.raises(ValueError, match="localized_failure_id"):
        _proposal(localized_failure_id="failure:invented")
    with pytest.raises(ValueError, match="localization_verifier_refs"):
        values = _proposal().to_dict()
        values["localization_verifier_refs"] = ["verifier:invented"]
        CorrectionProposal.from_dict(values)


def test_correction_is_bound_to_source_action_evidence_and_changed_components() -> None:
    """Catch a correction detached from the exact observed source failure or edit scope."""

    payload = _proposal().to_dict()
    for field_name, value in (
        ("source_action_id", "action:other"),
        ("source_evidence_refs", [_ref("evidence:other").to_dict()]),
        ("changed_components", []),
        ("changed_components", ["model", "model"]),
    ):
        changed = dict(payload)
        changed[field_name] = value
        with pytest.raises(ValueError, match=field_name):
            CorrectionProposal.from_dict(changed)


def test_correction_requires_the_complete_exact_localization_verifier_set() -> None:
    """Catch partial role evidence making a multi-role localization look fully supported."""

    payload = _proposal().to_dict()
    payload["localization_verifier_refs"] = ["verifier:localization"]

    with pytest.raises(ValueError, match="localization_verifier_refs"):
        CorrectionProposal.from_dict(payload)


def test_candidate_requires_closed_source_and_store_issued_new_epoch(tmp_path: Path) -> None:
    """Catch model or harness mutation inside an open source episode or a forged epoch."""

    store = EvaluationEpochStore(tmp_path / "open.sqlite", "prod")
    store.create_root(
        lineage_id="lineage:open",
        epoch_id="epoch:source",
        model_version="model:v1",
        harness_version="harness:v1",
    )
    with pytest.raises(ValueError, match="source.*closed|candidate epoch"):
        bind_candidate_system(
            store,
            lineage_id="lineage:open",
            candidate_id="candidate:open",
            correction=_proposal(),
            artifact_digest="a" * 64,
        )

    history = store.open("lineage:open")
    close_evaluation_epoch(history)
    with pytest.raises(ValueError, match="candidate.*version"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch:reused",
            model_version="model:v1",
            harness_version="harness:v1",
        )


@pytest.mark.parametrize(
    ("metric", "accepted"),
    [
        (ComponentMetric("quality", 0.8, 0.79, MetricDirection.HIGHER_IS_BETTER, 0.01), True),
        (ComponentMetric("quality", 0.8, 0.78, MetricDirection.HIGHER_IS_BETTER, 0.01), False),
        (ComponentMetric("error", 0.2, 0.21, MetricDirection.LOWER_IS_BETTER, 0.01), True),
        (ComponentMetric("error", 0.2, 0.22, MetricDirection.LOWER_IS_BETTER, 0.01), False),
    ],
)
def test_primary_metric_uses_explicit_direction_and_tolerance(
    tmp_path: Path,
    metric: ComponentMetric,
    accepted: bool,
) -> None:
    """Catch a universal higher-is-better comparison or silently ignored tolerance."""

    store, bundle = _validated_bundle(tmp_path, metrics=(metric,))
    decision = decide_candidate_acceptance(
        bundle,
        trusted_store=store,
        trusted_authority=store.authority(),
        required_protected_record_ids=bundle.protected_receipt.record_ids,
    )
    assert decision.accepted is accepted
    assert decision.component_metrics == (metric,)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("baseline_value", True),
        ("baseline_value", 1),
        ("candidate_value", float("nan")),
        ("candidate_value", float("inf")),
        ("tolerance", -0.01),
        ("tolerance", False),
        ("direction", "higher_is_better"),
    ],
)
def test_metrics_reject_opaque_coercible_or_nonfinite_values(
    field_name: str,
    value: object,
) -> None:
    """Catch bool, integer, NaN, infinity, enum-string, or negative-tolerance metrics."""

    values: dict[str, object] = {
        "name": "quality",
        "baseline_value": 0.8,
        "candidate_value": 0.9,
        "direction": MetricDirection.HIGHER_IS_BETTER,
        "tolerance": 0.0,
    }
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        ComponentMetric(**cast(Any, values))


def test_bundle_requires_complete_unique_metrics_and_primary_metric(tmp_path: Path) -> None:
    """Catch scalar collapse, duplicate/conflicting components, or an undeclared primary metric."""

    _, valid = _validated_bundle(tmp_path)
    metric = ComponentMetric("quality", 0.8, 0.9, MetricDirection.HIGHER_IS_BETTER, 0.0)
    duplicate = ComponentMetric("quality", 0.8, 0.7, MetricDirection.LOWER_IS_BETTER, 0.0)
    with pytest.raises(ValueError, match="component_metrics"):
        ValidationBundle(
            valid.candidate,
            valid.held_out_receipt,
            valid.protected_receipt,
            (),
            "quality",
        )
    with pytest.raises(ValueError, match="unique|conflicting"):
        ValidationBundle(
            valid.candidate,
            valid.held_out_receipt,
            valid.protected_receipt,
            (metric, duplicate),
            "quality",
        )
    with pytest.raises(ValueError, match="primary_metric_name"):
        ValidationBundle(
            valid.candidate,
            valid.held_out_receipt,
            valid.protected_receipt,
            (metric,),
            "missing",
        )


def test_failed_and_incomplete_protected_regressions_cannot_accept(tmp_path: Path) -> None:
    """Catch partial or failing protected coverage being described as regression-safe."""

    store, history, candidate = _prepared_candidate(tmp_path)
    failed = _record(
        model_version=candidate.model_version,
        harness_version=candidate.harness_version,
        repeat_id=0,
        passed=False,
    )
    append_epoch_record(history, failed)
    close_evaluation_epoch(history)
    with pytest.raises(ValueError, match="every outcome.*pass"):
        store.issue_validation_receipt(
            history,
            epoch_id=candidate.candidate_epoch_id,
            target=candidate.validation_target(),
            split=ValidationSplit.PROTECTED,
            records=(failed,),
        )

    store, bundle = _validated_bundle(tmp_path, candidate_id="candidate:incomplete")
    required = (*bundle.protected_receipt.record_ids, "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="complete protected"):
        decide_candidate_acceptance(
            bundle,
            trusted_store=store,
            trusted_authority=store.authority(),
            required_protected_record_ids=required,
        )


def test_receipts_must_match_candidate_epoch_identity_lineage_and_split(tmp_path: Path) -> None:
    """Catch cross-candidate, cross-epoch, and split-relabel validation replay."""

    store, bundle = _validated_bundle(tmp_path, candidate_id="candidate:one")
    other_store, other = _validated_bundle(tmp_path, candidate_id="candidate:two")
    payload = bundle.to_dict()
    payload["protected_receipt"] = other.protected_receipt.to_dict()
    mixed = ValidationBundle.from_dict(payload)
    with pytest.raises(ValueError, match="candidate|target|lineage|epoch|catalog"):
        decide_candidate_acceptance(
            mixed,
            trusted_store=store,
            trusted_authority=store.authority(),
            required_protected_record_ids=mixed.protected_receipt.record_ids,
        )

    relabeled = bundle.protected_receipt.to_dict()
    relabeled["split"] = "held_out"
    payload = bundle.to_dict()
    payload["protected_receipt"] = relabeled
    with pytest.raises(ValueError, match="protected|catalog"):
        decide_candidate_acceptance(
            ValidationBundle.from_dict(payload),
            trusted_store=store,
            trusted_authority=store.authority(),
            required_protected_record_ids=bundle.protected_receipt.record_ids,
        )

    with pytest.raises(ValueError, match="authority|catalog"):
        decide_candidate_acceptance(
            bundle,
            trusted_store=other_store,
            trusted_authority=other_store.authority(),
            required_protected_record_ids=bundle.protected_receipt.record_ids,
        )


def test_records_snapshot_nested_inputs_and_revalidate_use_time_corruption(tmp_path: Path) -> None:
    """Catch caller mutation or frozen-record bypass changing an accepted candidate."""

    store, bundle = _validated_bundle(tmp_path)
    source = CandidateSystem.from_dict(bundle.candidate.to_dict())
    detached = ValidationBundle(
        source,
        bundle.held_out_receipt,
        bundle.protected_receipt,
        bundle.component_metrics,
        bundle.primary_metric_name,
    )
    object.__setattr__(source.correction.failure_graph.localization, "first_anomaly", "rewritten")
    assert detached.candidate.correction.failure_graph.localization.first_anomaly == (
        "failure:localized"
    )

    object.__setattr__(detached.protected_receipt, "target_digest", "sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="receipt|target|construction"):
        decide_candidate_acceptance(
            detached,
            trusted_store=store,
            trusted_authority=store.authority(),
            required_protected_record_ids=detached.protected_receipt.record_ids,
        )


def test_candidate_bundle_and_decision_json_round_trip_and_remain_frozen(tmp_path: Path) -> None:
    """Catch lossy audit persistence or mutable accepted decisions."""

    store, bundle = _validated_bundle(tmp_path)
    restored_bundle = ValidationBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
    decision = decide_candidate_acceptance(
        restored_bundle,
        trusted_store=store,
        trusted_authority=store.authority(),
        required_protected_record_ids=restored_bundle.protected_receipt.record_ids,
    )
    restored_decision = type(decision).from_dict(json.loads(json.dumps(decision.to_dict())))

    assert restored_bundle == bundle
    assert restored_decision == decision
    assert restored_decision.inputs_digest.startswith("sha256:")
    for record in (
        restored_bundle.candidate.correction,
        restored_bundle.candidate,
        restored_bundle,
        restored_decision,
    ):
        assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored_decision.accepted = False


def test_candidate_acceptance_is_a_nonexecuting_deterministic_audit_decision(
    tmp_path: Path,
) -> None:
    """Catch acceptance mutating the epoch lineage or producing unauditable replay variance."""

    store, bundle = _validated_bundle(tmp_path)
    before = store.open(bundle.candidate.lineage_id).snapshot()
    first = decide_candidate_acceptance(
        bundle,
        trusted_store=store,
        trusted_authority=store.authority(),
        required_protected_record_ids=bundle.protected_receipt.record_ids,
    )
    second = decide_candidate_acceptance(
        bundle,
        trusted_store=store,
        trusted_authority=store.authority(),
        required_protected_record_ids=bundle.protected_receipt.record_ids,
    )

    assert first == second
    assert first.inputs_digest == second.inputs_digest
    assert store.open(bundle.candidate.lineage_id).snapshot() == before


def test_public_exports_expose_controlled_coevolution_contract() -> None:
    """Catch the public API omitting the records needed by offline evolution callers."""

    import gepa_mindfulness

    assert gepa_mindfulness.CorrectionProposal is CorrectionProposal
    assert gepa_mindfulness.CandidateSystem is CandidateSystem
    assert gepa_mindfulness.ValidationBundle is ValidationBundle
    assert gepa_mindfulness.decide_candidate_acceptance is decide_candidate_acceptance

"""Durable verified-skill lifecycle acceptance tests."""

from __future__ import annotations

import json
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
    ConsolidationProvenance,
    EvaluationEpochStore,
    ExecutionEvidenceBundle,
    InstantiationProvenance,
    PruningProvenance,
    RefinementProvenance,
    SkillArtifact,
    SkillLifecycleState,
    SkillLifecycleStore,
    ValidationSplit,
    append_epoch_record,
    close_evaluation_epoch,
    transition_skill,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import (
    ArtifactObservation,
    LocalVerificationResult,
    RelationalVerificationResult,
    VerificationEvidenceBinding,
    WorldStateChange,
    make_local_verification_event,
    make_relational_verification_event,
)
from mindful_trace_gepa import (
    ActionRecord,
    OutcomeObservation,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_verification_result_event,
)
from mindful_trace_gepa.logging_schema import StructuredEventType


def _ref(identifier: str = "evidence:observed") -> EvidenceReference:
    return EvidenceReference(identifier, EvidenceSourceKind.OBSERVABLE_OUTPUT)


def _binding(name: str) -> VerificationEvidenceBinding:
    return VerificationEvidenceBinding(name, (_ref(),))


def _bundle(*, reversible: bool = False) -> ExecutionEvidenceBundle:
    common = {
        "run_id": "run-1",
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "case_version": "17case-v5",
        "case_id": 14,
        "stripe_id": "TOOL_ERROR",
        "repeat_id": 0,
        "seed": 42,
    }
    action = ActionRecord("action-1", "write", reversible, "workspace", "prediction-1")
    action_event = make_action_event(
        action,
        StructuredEventType.ACTION_EXECUTED,
        event_id="action-event-1",
        timestamp="2026-09-10T12:00:00Z",
        **common,
    )
    outcome = OutcomeObservation(
        "observation-1",
        "action-1",
        {"artifact_ref": "artifact:skill.py", "digest": "a" * 64},
        ("evidence:observed",),
    )
    outcome_event = make_outcome_observation_event(
        outcome,
        event_id="outcome-event-1",
        timestamp="2026-09-10T12:00:01Z",
        parent_event_ids=(action_event.event_id,),
        **common,
    )
    local = LocalVerificationResult(
        "action-1",
        True,
        True,
        True,
        True,
        True,
        True,
        (_ref(),),
        tuple(
            _binding(name)
            for name in (
                "executed",
                "arguments_valid",
                "schema_valid",
                "authorization_valid",
                "intended_operation_observed",
                "irreversible_action_permitted",
            )
        ),
    )
    relational = RelationalVerificationResult(
        "action-1",
        True,
        True,
        "none",
        True,
        True,
        True,
        False,
        (_ref(),),
        tuple(
            _binding(name)
            for name in (
                "task_fit",
                "dependencies_satisfied",
                "contradiction_status",
                "provenance_intact",
                "authorization_scope_valid",
                "claimed_outcome_supported",
            )
        ),
    )
    verifier_common = {
        **common,
        "parent_event_ids": (outcome_event.event_id,),
    }
    local_event = make_local_verification_event(
        local,
        verifier_refs=("verifier:local",),
        event_id="verification-local-1",
        timestamp="2026-09-10T12:00:02Z",
        **verifier_common,
    )
    relational_event = make_relational_verification_event(
        relational,
        verifier_refs=("verifier:relational",),
        event_id="verification-relational-1",
        timestamp="2026-09-10T12:00:03Z",
        **verifier_common,
    )
    observation = ArtifactObservation(
        "observation-1",
        "artifact:skill.py",
        "a" * 64,
        "2026-09-10T12:00:01Z",
        (_ref(),),
    )
    change = WorldStateChange("change-1", "action-1", None, observation)
    return ExecutionEvidenceBundle(
        action_event,
        outcome_event,
        change,
        local_event,
        relational_event,
    )


def _record(*, passed: bool = True) -> V5EvaluationRecord:
    return V5EvaluationRecord(
        CaseIdentity(
            14,
            "17case-v5",
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        RobustnessIdentity("TOOL_ERROR", None),
        SystemIdentity(0, 42, "model-v1", "harness-v1"),
        EpistemicRecord("prediction-1", ("evidence:observed",), ("verifier:v1",), 0.8),
        BehaviorRecord(("action-event-1",), False, False),
        OutcomeRecord(("outcome-event-1",), ("verification-relational-1",), passed),
        ScoreRecord(1.0, 0.8, 1.0, 1.0, 0.95),
        DiagnosticRecord("held-out result", 0.0, 0.0),
    )


def _store(tmp_path: Path) -> SkillLifecycleStore:
    return SkillLifecycleStore(tmp_path / "skills.sqlite", "skill-authority")


def _task_local(store: SkillLifecycleStore, skill_id: str) -> SkillArtifact:
    history = store.create_source(skill_id, "v1", (_ref(f"evidence:{skill_id}"),))
    verified = transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    return transition_skill(
        history,
        SkillLifecycleState.TASK_LOCAL,
        provenance=InstantiationProvenance(verified.artifact_id, f"task:{skill_id}"),
    )


def _main_history(tmp_path: Path) -> tuple[SkillLifecycleStore, Any, SkillArtifact]:
    store = _store(tmp_path)
    constituent = _task_local(store, "constituent")
    history = store.create_source("skill-1", "v1", (_ref("evidence:source"),))
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    family = transition_skill(
        history,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        provenance=ConsolidationProvenance((constituent.artifact_id,)),
    )
    local = transition_skill(
        history,
        SkillLifecycleState.TASK_LOCAL,
        provenance=InstantiationProvenance(family.artifact_id, "task:production"),
    )
    return store, history, local


def _validation_receipt(tmp_path: Path, candidate_version: str) -> tuple[Any, Any]:
    store = EvaluationEpochStore(tmp_path / "epochs.sqlite", "evaluation-authority")
    history = store.create_root(
        lineage_id="held-out",
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    record = _record()
    append_epoch_record(history, record)
    close_evaluation_epoch(history)
    receipt = store.issue_validation_receipt(
        history,
        epoch_id="epoch-1",
        candidate_version=candidate_version,
        split=ValidationSplit.HELD_OUT,
        records=(record,),
    )
    return store, receipt


def test_full_lifecycle_persists_structured_receipts_and_reopens(tmp_path: Path) -> None:
    store, history, _local = _main_history(tmp_path)
    executed = transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_bundle(),
    )
    credited = transition_skill(history, SkillLifecycleState.CREDITED)
    refined = transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
        provenance=RefinementProvenance("generalize verified behavior", (_ref(),)),
    )
    validation_store, receipt = _validation_receipt(tmp_path, "v2")
    validated = transition_skill(
        history,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        validation_store=validation_store,
        validation_receipt=receipt,
    )
    committed = transition_skill(
        history,
        SkillLifecycleState.COMMITTED,
        rollback_target_id=credited.artifact_id,
    )
    reopened = SkillLifecycleStore(
        tmp_path / "skills.sqlite",
        "skill-authority",
    ).open("skill-1")
    assert reopened.current() == committed
    assert executed.execution_receipt is not None
    assert validated.validation_receipt == receipt
    assert refined.supersedes == "v1"


def test_legacy_scalar_verification_never_creates_credit(tmp_path: Path) -> None:
    _store_value, history, _local = _main_history(tmp_path)
    legacy = _bundle()
    legacy_event = make_verification_result_event(
        VerificationResult("verifier", "v1", "observation-1", True, ("ref",)),
        event_id="legacy-verification",
        timestamp="2026-09-10T12:00:02Z",
        action_id="action-1",
    )
    object.__setattr__(legacy, "local_verification_event", legacy_event)
    with pytest.raises(ValueError, match="leveled verification"):
        transition_skill(
            history,
            SkillLifecycleState.EXECUTED,
            execution_evidence=legacy,
        )


@pytest.mark.parametrize(
    ("event_name", "field_name", "value"),
    [
        ("action_event", "event_id", 1),
        ("outcome_event", "timestamp", "tomorrow"),
        ("relational_verification_event", "case_id", True),
    ],
)
def test_bundle_rejects_hostile_event_scalars(
    event_name: str,
    field_name: str,
    value: object,
) -> None:
    bundle = _bundle()
    object.__setattr__(getattr(bundle, event_name), field_name, value)
    with pytest.raises(ValueError):
        ExecutionEvidenceBundle(
            *cast(Any, tuple(getattr(bundle, name) for name in bundle.__dataclass_fields__))
        )


def test_bundle_rejects_reversed_time_and_artifact_mismatch() -> None:
    bundle = _bundle()
    object.__setattr__(bundle.action_event, "timestamp", "2026-09-10T12:00:04Z")
    with pytest.raises(ValueError, match="causal order"):
        ExecutionEvidenceBundle(
            *cast(Any, tuple(getattr(bundle, name) for name in bundle.__dataclass_fields__))
        )
    bundle = _bundle()
    object.__setattr__(bundle.world_change.after_observation, "digest", "b" * 64)
    with pytest.raises(ValueError, match="artifact identity and digest"):
        ExecutionEvidenceBundle(
            *cast(Any, tuple(getattr(bundle, name) for name in bundle.__dataclass_fields__))
        )


def test_validation_receipts_reject_failed_open_cross_catalog_and_wrong_candidate(
    tmp_path: Path,
) -> None:
    failed = _record(passed=False)
    store = EvaluationEpochStore(tmp_path / "failed.sqlite", "eval")
    epoch = store.create_root(
        lineage_id="lineage",
        epoch_id="epoch",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    append_epoch_record(epoch, failed)
    with pytest.raises(ValueError, match="closed"):
        store.issue_validation_receipt(
            epoch,
            epoch_id="epoch",
            candidate_version="v2",
            split=ValidationSplit.HELD_OUT,
            records=(failed,),
        )
    close_evaluation_epoch(epoch)
    with pytest.raises(ValueError, match="outcome.*pass"):
        store.issue_validation_receipt(
            epoch,
            epoch_id="epoch",
            candidate_version="v2",
            split=ValidationSplit.HELD_OUT,
            records=(failed,),
        )
    validation_store, receipt = _validation_receipt(tmp_path, "wrong-version")
    _skill_store, history, _local = _main_history(tmp_path)
    transition_skill(history, SkillLifecycleState.EXECUTED, execution_evidence=_bundle())
    transition_skill(history, SkillLifecycleState.CREDITED)
    transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
        provenance=RefinementProvenance("change", (_ref(),)),
    )
    with pytest.raises(ValueError, match="candidate_version"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            validation_store=validation_store,
            validation_receipt=receipt,
        )
    other_store = EvaluationEpochStore(tmp_path / "other.sqlite", "evaluation-authority")
    with pytest.raises(ValueError, match="catalog"):
        other_store.validate_validation_receipt(receipt)
    other_domain = EvaluationEpochStore(tmp_path / "epochs.sqlite", "other-authority")
    with pytest.raises(ValueError, match="authority domain"):
        other_domain.validate_validation_receipt(receipt)
    payload = receipt.to_dict()
    payload["split"] = "training"
    with pytest.raises(ValueError, match="split"):
        type(receipt).from_dict(payload)


def test_durable_store_rejects_duplicate_roots_and_stale_handles(tmp_path: Path) -> None:
    first = _store(tmp_path)
    history = first.create_source("skill-1", "v1", (_ref(),))
    second = _store(tmp_path)
    stale = second.open("skill-1")
    with pytest.raises(ValueError, match="claimed"):
        second.create_source("skill-1", "v1", (_ref(),))
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(RuntimeError, match="stale revision"):
        stale.current()


def test_rollback_uses_strict_predecessor_id_and_pruning_provenance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    history = store.create_source("skill-1", "v1", (_ref(),))
    source = history.current()
    verified = transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="strict predecessor"):
        transition_skill(
            history,
            SkillLifecycleState.ROLLED_BACK,
            rollback_target_id=verified.artifact_id,
            provenance=PruningProvenance(verified.artifact_id, "retire", (_ref(),)),
        )
    rolled = transition_skill(
        history,
        SkillLifecycleState.ROLLED_BACK,
        rollback_target_id=source.artifact_id,
        provenance=PruningProvenance(verified.artifact_id, "retire", (_ref(),)),
    )
    assert rolled.rollback_target_id == source.artifact_id


def test_artifact_and_provenance_json_round_trip(tmp_path: Path) -> None:
    _store_value, history, _local = _main_history(tmp_path)
    restored = SkillArtifact.from_dict(json.loads(json.dumps(history.current().to_dict())))
    assert restored == history.current()


def test_use_time_mutation_of_artifact_and_validation_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    _store_value, history, _local = _main_history(tmp_path)
    artifact = history.current()
    object.__setattr__(artifact, "version", "forged")
    with pytest.raises(ValueError, match="construction binding"):
        artifact.to_dict()

    validation_store, receipt = _validation_receipt(tmp_path, "v2")
    object.__setattr__(receipt, "candidate_version", "forged")
    with pytest.raises(ValueError, match="construction binding"):
        validation_store.validate_validation_receipt(receipt)


def test_negative_local_or_relational_finding_blocks_execution() -> None:
    for field_name in ("executed", "claimed_outcome_supported"):
        bundle = _bundle()
        event_name = (
            "local_verification_event"
            if field_name == "executed"
            else "relational_verification_event"
        )
        event = getattr(bundle, event_name)
        payload = event.to_dict()["payload"]
        result = cast(dict[str, object], cast(dict[str, object], payload)["result"])
        result[field_name] = False
        object.__setattr__(event, "payload", cast(dict[str, object], payload))
        with pytest.raises(ValueError, match="verification must affirm"):
            ExecutionEvidenceBundle(
                bundle.action_event,
                bundle.outcome_event,
                bundle.world_change,
                bundle.local_verification_event,
                bundle.relational_verification_event,
            )


@pytest.mark.parametrize(
    "target",
    [
        SkillLifecycleState.CREDITED,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        SkillLifecycleState.COMMITTED,
    ],
)
def test_forbidden_gate_skips_are_rejected(target: SkillLifecycleState, tmp_path: Path) -> None:
    store = _store(tmp_path)
    history = store.create_source("skill-1", "v1", (_ref(),))
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="transition"):
        transition_skill(history, target)

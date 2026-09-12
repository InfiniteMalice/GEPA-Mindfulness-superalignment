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
    FamilySeedProvenance,
    InstantiationProvenance,
    PruningProvenance,
    RefinementProvenance,
    SkillArtifact,
    SkillLifecycleState,
    SkillLifecycleStore,
    ValidationSplit,
    ValidationTarget,
    append_epoch_record,
    close_evaluation_epoch,
    skill_artifact_digest,
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


def _binding(
    name: str,
    identifier: str = "evidence:observed",
) -> VerificationEvidenceBinding:
    return VerificationEvidenceBinding(name, (_ref(identifier),))


def _bundle(
    *,
    reversible: bool = False,
    finding_evidence: str = "evidence:observed",
    artifact_ref: str = "artifact:skill.py",
) -> ExecutionEvidenceBundle:
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
        {"artifact_ref": artifact_ref, "digest": "a" * 64},
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
        (_ref(finding_evidence),),
        tuple(
            _binding(name, finding_evidence)
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
        (_ref(finding_evidence),),
        tuple(
            _binding(name, finding_evidence)
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
        artifact_ref,
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


def _record(
    *,
    passed: bool = True,
    model_version: str = "model-v1",
    harness_version: str = "v2",
) -> V5EvaluationRecord:
    return V5EvaluationRecord(
        CaseIdentity(
            14,
            "17case-v5",
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        RobustnessIdentity("TOOL_ERROR", None),
        SystemIdentity(0, 42, model_version, harness_version),
        EpistemicRecord("prediction-1", ("evidence:observed",), ("verifier:v1",), 0.8),
        BehaviorRecord(("action-event-1",), False, False),
        OutcomeRecord(("outcome-event-1",), ("verification-relational-1",), passed),
        ScoreRecord(1.0, 0.8, 1.0, 1.0, 0.95),
        DiagnosticRecord("held-out result", 0.0, 0.0),
    )


def _evaluation_store(
    tmp_path: Path,
    *,
    filename: str = "epochs.sqlite",
    authority_domain: str = "evaluation-authority",
) -> EvaluationEpochStore:
    return EvaluationEpochStore(tmp_path / filename, authority_domain)


def _store(
    tmp_path: Path,
    *,
    evaluation_store: EvaluationEpochStore | None = None,
    allowed_lineages: tuple[str, ...] = ("held-out",),
) -> SkillLifecycleStore:
    trusted = evaluation_store or _evaluation_store(tmp_path)
    return SkillLifecycleStore(
        tmp_path / "skills.sqlite",
        "skill-authority",
        evaluation_store=trusted,
        allowed_evaluation_lineages=allowed_lineages,
    )


def _task_local(store: SkillLifecycleStore, skill_id: str) -> SkillArtifact:
    history = store.create_source(skill_id, "v1", (_ref(f"evidence:{skill_id}"),))
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    family = transition_skill(
        history,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        provenance=FamilySeedProvenance(
            history.current().artifact_id,
            "seed from verified evidence",
            (_ref(f"evidence:{skill_id}"),),
        ),
    )
    return transition_skill(
        history,
        SkillLifecycleState.TASK_LOCAL,
        provenance=InstantiationProvenance(family.artifact_id, f"task:{skill_id}"),
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


def _validation_receipt(
    tmp_path: Path,
    target: SkillArtifact,
    *,
    split: ValidationSplit = ValidationSplit.HELD_OUT,
    lineage_id: str = "held-out",
    target_version: str | None = None,
) -> tuple[Any, Any]:
    store = _evaluation_store(tmp_path)
    evaluated_version = target.version if target_version is None else target_version
    model_version = f"model:{lineage_id}"
    history = store.create_root(
        lineage_id=lineage_id,
        epoch_id=f"epoch:{lineage_id}",
        model_version=model_version,
        harness_version=evaluated_version,
    )
    record = _record(model_version=model_version, harness_version=evaluated_version)
    append_epoch_record(history, record)
    close_evaluation_epoch(history)
    receipt = store.issue_validation_receipt(
        history,
        epoch_id=f"epoch:{lineage_id}",
        target=ValidationTarget(
            target.artifact_id,
            target.skill_id,
            evaluated_version,
            skill_artifact_digest(target),
        ),
        split=split,
        records=(record,),
    )
    return store, receipt


def test_full_lifecycle_persists_structured_receipts_and_reopens(tmp_path: Path) -> None:
    store, history, _local = _main_history(tmp_path)
    executed = transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_bundle(artifact_ref=history.current().artifact_id),
    )
    credited = transition_skill(history, SkillLifecycleState.CREDITED)
    refined = transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
        provenance=RefinementProvenance("generalize verified behavior", (_ref(),)),
    )
    _validation_store, receipt = _validation_receipt(tmp_path, refined)
    validated = transition_skill(
        history,
        SkillLifecycleState.HELD_OUT_VALIDATED,
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
        evaluation_store=_evaluation_store(tmp_path),
        allowed_evaluation_lineages=("held-out",),
    ).open("skill-1")
    assert reopened.current() == committed
    assert executed.execution_receipt is not None
    assert tuple(
        item.finding_key for item in executed.execution_receipt.required_finding_evidence
    ) == (
        "local_execution:executed",
        "local_execution:intended_operation_observed",
        "relational_evidence:claimed_outcome_supported",
        "relational_evidence:provenance_intact",
    )
    assert validated.validation_receipt == receipt
    assert receipt.target_artifact_id == refined.artifact_id
    assert receipt.target_digest == skill_artifact_digest(refined)
    assert refined.supersedes == "v1"


def test_legacy_scalar_verification_never_creates_credit(tmp_path: Path) -> None:
    _store_value, history, _local = _main_history(tmp_path)
    legacy = _bundle(artifact_ref=history.current().artifact_id)
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


def test_execution_evidence_must_target_the_current_lifecycle_artifact(tmp_path: Path) -> None:
    store, history, _local = _main_history(tmp_path)
    unrelated = _task_local(store, "unrelated")

    with pytest.raises(ValueError, match="current lifecycle artifact"):
        transition_skill(
            history,
            SkillLifecycleState.EXECUTED,
            execution_evidence=_bundle(artifact_ref=unrelated.artifact_id),
        )

    transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_bundle(artifact_ref=history.current().artifact_id),
    )
    unrelated_history = store.open(unrelated.skill_id)
    with pytest.raises(ValueError, match="execution evidence.*already claimed"):
        transition_skill(
            unrelated_history,
            SkillLifecycleState.EXECUTED,
            execution_evidence=_bundle(artifact_ref=unrelated.artifact_id),
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


def test_validation_receipts_reject_failed_open_and_old_records_relabelled_as_candidate(
    tmp_path: Path,
) -> None:
    failed = _record(passed=False, harness_version="v2")
    store = EvaluationEpochStore(tmp_path / "failed.sqlite", "eval")
    epoch = store.create_root(
        lineage_id="lineage",
        epoch_id="epoch",
        model_version="model-v1",
        harness_version="v2",
    )
    append_epoch_record(epoch, failed)
    with pytest.raises(ValueError, match="closed"):
        store.issue_validation_receipt(
            epoch,
            epoch_id="epoch",
            target=ValidationTarget("artifact-1", "skill-1", "v2", "sha256:" + "a" * 64),
            split=ValidationSplit.HELD_OUT,
            records=(failed,),
        )
    close_evaluation_epoch(epoch)
    with pytest.raises(ValueError, match="outcome.*pass"):
        store.issue_validation_receipt(
            epoch,
            epoch_id="epoch",
            target=ValidationTarget("artifact-1", "skill-1", "v2", "sha256:" + "a" * 64),
            split=ValidationSplit.HELD_OUT,
            records=(failed,),
        )
    old_store = EvaluationEpochStore(tmp_path / "old.sqlite", "eval")
    old_history = old_store.create_root(
        lineage_id="old",
        epoch_id="old-epoch",
        model_version="model-v1",
        harness_version="v1",
    )
    old_record = _record(harness_version="v1")
    append_epoch_record(old_history, old_record)
    close_evaluation_epoch(old_history)
    with pytest.raises(ValueError, match="target version"):
        old_store.issue_validation_receipt(
            old_history,
            epoch_id="old-epoch",
            target=ValidationTarget("artifact-2", "skill-1", "v2", "sha256:" + "b" * 64),
            split=ValidationSplit.HELD_OUT,
            records=(old_record,),
        )


def test_lifecycle_pins_evaluation_catalog_domain_and_allowed_lineage(tmp_path: Path) -> None:
    trusted = _evaluation_store(tmp_path)
    _store(tmp_path, evaluation_store=trusted)
    rogue_catalog = _evaluation_store(tmp_path, filename="rogue.sqlite")
    with pytest.raises(ValueError, match="trusted evaluation authority"):
        _store(tmp_path, evaluation_store=rogue_catalog)
    rogue_domain = _evaluation_store(tmp_path, authority_domain="rogue-domain")
    with pytest.raises(ValueError, match="trusted evaluation authority"):
        _store(tmp_path, evaluation_store=rogue_domain)


def test_held_out_transition_requires_exact_target_split_and_allowed_lineage(
    tmp_path: Path,
) -> None:
    _skill_store, history, _local = _main_history(tmp_path)
    transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_bundle(artifact_ref=history.current().artifact_id),
    )
    transition_skill(history, SkillLifecycleState.CREDITED)
    refined = transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
        provenance=RefinementProvenance("change", (_ref(),)),
    )
    validation_store, protected = _validation_receipt(
        tmp_path,
        refined,
        split=ValidationSplit.PROTECTED,
    )
    with pytest.raises(ValueError, match="HELD_OUT"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            validation_receipt=protected,
        )

    evaluation_history = validation_store.open("held-out")
    validation_model = "model:held-out"
    validation_record = _record(model_version=validation_model, harness_version="v2")
    wrong_target = validation_store.issue_validation_receipt(
        evaluation_history,
        epoch_id="epoch:held-out",
        target=ValidationTarget(
            "another-artifact",
            refined.skill_id,
            refined.version,
            skill_artifact_digest(refined),
        ),
        split=ValidationSplit.HELD_OUT,
        records=(validation_record,),
    )
    with pytest.raises(ValueError, match="exact current lifecycle artifact"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            validation_receipt=wrong_target,
        )

    _rogue_store, rogue_lineage = _validation_receipt(
        tmp_path,
        refined,
        lineage_id="rogue-lineage",
        target_version="v3",
    )
    with pytest.raises(ValueError, match="allowed lineage"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            validation_receipt=rogue_lineage,
        )
    other_store = EvaluationEpochStore(tmp_path / "other.sqlite", "evaluation-authority")
    with pytest.raises(ValueError, match="catalog"):
        other_store.validate_validation_receipt(protected)
    other_domain = EvaluationEpochStore(tmp_path / "epochs.sqlite", "other-authority")
    with pytest.raises(ValueError, match="authority domain"):
        other_domain.validate_validation_receipt(protected)
    payload = protected.to_dict()
    payload["split"] = "training"
    with pytest.raises(ValueError, match="split"):
        type(protected).from_dict(payload)


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

    validation_store, receipt = _validation_receipt(tmp_path, history.current())
    object.__setattr__(receipt, "target_version", "forged")
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


def test_required_findings_cannot_cite_unrelated_observable_evidence() -> None:
    with pytest.raises(ValueError, match="canonical outcome observation evidence"):
        _bundle(finding_evidence="evidence:unrelated")


def test_direct_verified_to_task_local_and_wrong_family_parent_are_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    history = store.create_source("skill-direct", "v1", (_ref(),))
    verified = transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="transition"):
        transition_skill(
            history,
            SkillLifecycleState.TASK_LOCAL,
            provenance=InstantiationProvenance(verified.artifact_id, "task:forbidden"),
        )
    family = transition_skill(
        history,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        provenance=FamilySeedProvenance(
            verified.artifact_id,
            "seed from verified evidence",
            (_ref(),),
        ),
    )
    with pytest.raises(ValueError, match="current parent"):
        transition_skill(
            history,
            SkillLifecycleState.TASK_LOCAL,
            provenance=InstantiationProvenance(verified.artifact_id, "task:wrong-parent"),
        )
    local = transition_skill(
        history,
        SkillLifecycleState.TASK_LOCAL,
        provenance=InstantiationProvenance(family.artifact_id, "task:valid"),
    )
    assert local.state is SkillLifecycleState.TASK_LOCAL


def test_family_seed_requires_exact_current_verified_artifact_and_consolidation_is_nonempty(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    history = store.create_source("skill-seed", "v1", (_ref(),))
    source = history.current()
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="must not be empty"):
        ConsolidationProvenance(())
    with pytest.raises(ValueError, match="exact current verified"):
        transition_skill(
            history,
            SkillLifecycleState.PROCEDURAL_FAMILY,
            provenance=FamilySeedProvenance(
                source.artifact_id,
                "wrong source",
                (_ref(),),
            ),
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

"""Verified skill-lifecycle authority and provenance contracts."""

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
    EvaluationEpochHistory,
    EvaluationEpochStore,
    SkillArtifact,
    SkillExecutionEvidence,
    SkillLifecycleHistory,
    SkillLifecycleState,
    SkillLifecycleStore,
    append_epoch_record,
    close_evaluation_epoch,
    transition_skill,
)
from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
from gepa_mindfulness.verification import ArtifactObservation, WorldStateChange
from mindful_trace_gepa import (
    ActionRecord,
    OutcomeObservation,
    VerificationResult,
    make_action_event,
    make_outcome_observation_event,
    make_verification_result_event,
)
from mindful_trace_gepa.logging_schema import StructuredEventType


def _ref(reference_id: str = "evidence:source") -> EvidenceReference:
    return EvidenceReference(reference_id, EvidenceSourceKind.EXTERNAL_RECORD)


def _execution_evidence() -> SkillExecutionEvidence:
    common = {
        "run_id": "run-1",
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "case_version": "17case-v5",
        "case_id": 14,
        "stripe_id": "none",
        "repeat_id": 0,
        "seed": 42,
    }
    action = ActionRecord("action-1", "write", True, "workspace", "prediction-1")
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
        ("evidence:outcome",),
    )
    outcome_event = make_outcome_observation_event(
        outcome,
        event_id="outcome-event-1",
        timestamp="2026-09-10T12:00:01Z",
        parent_event_ids=(action_event.event_id,),
        **common,
    )
    verification = VerificationResult(
        "verifier-1",
        "v1",
        "observation-1",
        True,
        ("evidence:verification",),
    )
    verification_event = make_verification_result_event(
        verification,
        event_id="verification-event-1",
        timestamp="2026-09-10T12:00:02Z",
        action_id="action-1",
        parent_event_ids=(outcome_event.event_id,),
        **common,
    )
    observed = ArtifactObservation(
        "artifact-observation-1",
        "artifact:skill.py",
        "a" * 64,
        "2026-09-10T12:00:01Z",
        (_ref("evidence:outcome"),),
    )
    change = WorldStateChange("change-1", "action-1", None, observed)
    return SkillExecutionEvidence(action_event, outcome_event, verification_event, change)


def _record() -> V5EvaluationRecord:
    return V5EvaluationRecord(
        case=CaseIdentity(
            14,
            "17case-v5",
            "correct_high_stakes_clarifying_abstention",
            "Correct high-stakes clarifying abstention",
        ),
        robustness=RobustnessIdentity("TOOL_ERROR", None),
        system=SystemIdentity(0, 42, "model-v1", "harness-v1"),
        epistemics=EpistemicRecord(
            "prediction-event-1",
            ("evidence:request",),
            ("verifier:v1",),
            0.8,
        ),
        behavior=BehaviorRecord(("action-event-1",), False, False),
        outcome=OutcomeRecord(
            ("outcome-event-1",),
            ("verification-event-1",),
            True,
        ),
        scores=ScoreRecord(1.0, 0.8, 1.0, 1.0, 0.95),
        diagnostics=DiagnosticRecord("held-out success", 0.0, 0.0),
    )


def _closed_epoch(tmp_path: Path) -> tuple[EvaluationEpochHistory, V5EvaluationRecord]:
    record = _record()
    store = EvaluationEpochStore(tmp_path / "epochs.sqlite", "test-evaluation-authority")
    history = store.create_root(
        lineage_id="held-out",
        epoch_id="epoch-held-out",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    append_epoch_record(history, record)
    close_evaluation_epoch(history)
    return history, record


def _source() -> tuple[SkillLifecycleStore, SkillLifecycleHistory, SkillArtifact]:
    store = SkillLifecycleStore("test-skill-authority")
    history = store.create_source("skill-1", "v1", (_ref(),))
    return store, history, history.current()


def _advance_to_executed() -> tuple[SkillLifecycleHistory, SkillArtifact]:
    _store, history, artifact = _source()
    for state in (
        SkillLifecycleState.VERIFIED_SKILL,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        SkillLifecycleState.TASK_LOCAL,
    ):
        artifact = transition_skill(history, state)
    artifact = transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_execution_evidence(),
    )
    return history, artifact


def test_complete_lifecycle_requires_structured_credit_and_closed_held_out_records(
    tmp_path: Path,
) -> None:
    _store, history, artifact = _source()
    artifacts = [artifact]
    for state in (
        SkillLifecycleState.VERIFIED_SKILL,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        SkillLifecycleState.TASK_LOCAL,
    ):
        artifact = transition_skill(history, state)
        artifacts.append(artifact)
    artifact = transition_skill(
        history,
        SkillLifecycleState.EXECUTED,
        execution_evidence=_execution_evidence(),
    )
    artifacts.append(artifact)
    artifact = transition_skill(history, SkillLifecycleState.CREDITED)
    rollback_target = artifact
    artifacts.append(artifact)
    artifact = transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
    )
    artifacts.append(artifact)
    epoch_history, record = _closed_epoch(tmp_path)
    artifact = transition_skill(
        history,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        evaluation_history=epoch_history,
        evaluation_epoch_id="epoch-held-out",
        validation_records=(record,),
    )
    artifacts.append(artifact)
    artifact = transition_skill(
        history,
        SkillLifecycleState.COMMITTED,
        rollback_target=rollback_target,
    )
    artifacts.append(artifact)

    assert tuple(item.state for item in artifacts) == tuple(SkillLifecycleState)[:-1]
    assert artifact.version == "v2"
    assert artifact.supersedes == "v1"
    assert artifact.execution_event_ids == (
        "action-event-1",
        "outcome-event-1",
        "verification-event-1",
    )
    assert len(artifact.validation_record_ids) == 1
    assert history.current() == artifact


@pytest.mark.parametrize(
    "source_state",
    [
        SkillLifecycleState.SOURCE_EXPERIENCE,
        SkillLifecycleState.VERIFIED_SKILL,
        SkillLifecycleState.PROCEDURAL_FAMILY,
        SkillLifecycleState.TASK_LOCAL,
    ],
)
def test_every_skip_into_credited_is_forbidden(source_state: SkillLifecycleState) -> None:
    _store, history, _artifact = _source()
    paths = {
        SkillLifecycleState.SOURCE_EXPERIENCE: (),
        SkillLifecycleState.VERIFIED_SKILL: (SkillLifecycleState.VERIFIED_SKILL,),
        SkillLifecycleState.PROCEDURAL_FAMILY: (
            SkillLifecycleState.VERIFIED_SKILL,
            SkillLifecycleState.PROCEDURAL_FAMILY,
        ),
        SkillLifecycleState.TASK_LOCAL: (
            SkillLifecycleState.VERIFIED_SKILL,
            SkillLifecycleState.TASK_LOCAL,
        ),
    }
    for state in paths[source_state]:
        transition_skill(history, state)
    with pytest.raises(ValueError, match="transition"):
        transition_skill(history, SkillLifecycleState.CREDITED)


@pytest.mark.parametrize(
    "target",
    [SkillLifecycleState.HELD_OUT_VALIDATED, SkillLifecycleState.COMMITTED],
)
def test_forbidden_skips_into_validation_or_commit(target: SkillLifecycleState) -> None:
    _store, history, _artifact = _source()
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="transition"):
        transition_skill(history, target)


def test_generated_explanation_cannot_create_skill_credit() -> None:
    _store, history, _artifact = _source()
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    transition_skill(history, SkillLifecycleState.TASK_LOCAL)

    with pytest.raises(ValueError, match="structured execution evidence"):
        transition_skill(
            history,
            SkillLifecycleState.EXECUTED,
            execution_event_ids=cast(Any, ("I ran it", "it worked", "verified")),
        )


@pytest.mark.parametrize("mutation", ["action", "outcome", "verification", "world"])
def test_credit_evidence_rejects_broken_action_outcome_verification_bindings(
    mutation: str,
) -> None:
    evidence = _execution_evidence()
    if mutation == "action":
        object.__setattr__(evidence.action_event, "event_type", "action_proposed")
    elif mutation == "outcome":
        object.__setattr__(evidence.outcome_event, "parent_event_ids", ("other",))
    elif mutation == "verification":
        object.__setattr__(evidence.verification_event, "action_id", "other")
    else:
        object.__setattr__(evidence.world_change, "action_id", "other")

    _store, history, _artifact = _source()
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    transition_skill(history, SkillLifecycleState.TASK_LOCAL)
    with pytest.raises(ValueError):
        transition_skill(
            history,
            SkillLifecycleState.EXECUTED,
            execution_evidence=evidence,
        )


def test_credit_evidence_rejects_failed_verification_and_mismatched_artifact_digest() -> None:
    failed = _execution_evidence()
    failed_payload = dict(failed.verification_event.payload)
    failed_payload["verified"] = False
    object.__setattr__(failed.verification_event, "payload", failed_payload)
    with pytest.raises(ValueError, match="positively verify"):
        SkillExecutionEvidence(
            failed.action_event,
            failed.outcome_event,
            failed.verification_event,
            failed.world_change,
        )

    mismatch = _execution_evidence()
    outcome_payload = dict(mismatch.outcome_event.payload)
    outcome_payload["actual_outcome"] = {
        "artifact_ref": "artifact:skill.py",
        "digest": "b" * 64,
    }
    object.__setattr__(mismatch.outcome_event, "payload", outcome_payload)
    with pytest.raises(ValueError, match="artifact observation"):
        SkillExecutionEvidence(
            mismatch.action_event,
            mismatch.outcome_event,
            mismatch.verification_event,
            mismatch.world_change,
        )


def test_credit_evidence_rejects_hostile_event_identity_and_context_scalars() -> None:
    evidence = _execution_evidence()
    object.__setattr__(evidence.action_event, "event_id", 1)
    with pytest.raises(ValueError, match="event_id"):
        SkillExecutionEvidence(
            evidence.action_event,
            evidence.outcome_event,
            evidence.verification_event,
            evidence.world_change,
        )

    evidence = _execution_evidence()
    for event in (
        evidence.action_event,
        evidence.outcome_event,
        evidence.verification_event,
    ):
        object.__setattr__(event, "case_id", True)
    with pytest.raises(ValueError, match="case_id"):
        SkillExecutionEvidence(
            evidence.action_event,
            evidence.outcome_event,
            evidence.verification_event,
            evidence.world_change,
        )


def test_held_out_validation_rejects_open_detached_or_unlisted_records(tmp_path: Path) -> None:
    history, _artifact = _advance_to_executed()
    transition_skill(history, SkillLifecycleState.CREDITED)
    transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
    )
    record = _record()
    epoch_store = EvaluationEpochStore(tmp_path / "epochs.sqlite", "eval-authority")
    open_epoch = epoch_store.create_root(
        lineage_id="held-out",
        epoch_id="epoch-held-out",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    append_epoch_record(open_epoch, record)
    with pytest.raises(ValueError, match="closed"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            evaluation_history=open_epoch,
            evaluation_epoch_id="epoch-held-out",
            validation_records=(record,),
        )
    with pytest.raises(ValueError, match="EvaluationEpochHistory"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            evaluation_history=cast(Any, open_epoch.snapshot()),
            evaluation_epoch_id="epoch-held-out",
            validation_records=(record,),
        )
    close_evaluation_epoch(open_epoch)
    unlisted = V5EvaluationRecord.from_dict(record.to_dict())
    object.__setattr__(unlisted.scores, "total", 0.5)
    with pytest.raises(ValueError, match="not declared"):
        transition_skill(
            history,
            SkillLifecycleState.HELD_OUT_VALIDATED,
            evaluation_history=open_epoch,
            evaluation_epoch_id="epoch-held-out",
            validation_records=(unlisted,),
        )


def test_commit_requires_exact_prior_rollback_target_and_supersedes(tmp_path: Path) -> None:
    history, rollback_target = _advance_to_executed()
    rollback_target = transition_skill(history, SkillLifecycleState.CREDITED)
    transition_skill(
        history,
        SkillLifecycleState.REFINED,
        version="v2",
        supersedes="v1",
    )
    epoch_history, record = _closed_epoch(tmp_path)
    transition_skill(
        history,
        SkillLifecycleState.HELD_OUT_VALIDATED,
        evaluation_history=epoch_history,
        evaluation_epoch_id="epoch-held-out",
        validation_records=(record,),
    )
    forged = SkillArtifact(
        "skill-1",
        "v1",
        SkillLifecycleState.CREDITED,
        (_ref("evidence:forged"),),
        ("action-event-1", "outcome-event-1", "verification-event-1"),
    )
    with pytest.raises(ValueError, match="rollback target"):
        transition_skill(
            history,
            SkillLifecycleState.COMMITTED,
            rollback_target=forged,
        )
    committed = transition_skill(
        history,
        SkillLifecycleState.COMMITTED,
        rollback_target=rollback_target,
    )
    assert committed.supersedes == rollback_target.version


@pytest.mark.parametrize(
    "rollback_state",
    list(SkillLifecycleState)[1:-1],
)
def test_rollback_is_available_from_every_state_after_verified_skill(
    rollback_state: SkillLifecycleState,
    tmp_path: Path,
) -> None:
    _store, history, source = _source()
    verified = transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    target = verified
    if rollback_state is SkillLifecycleState.VERIFIED_SKILL:
        current = verified
    else:
        path = {
            SkillLifecycleState.PROCEDURAL_FAMILY: [SkillLifecycleState.PROCEDURAL_FAMILY],
            SkillLifecycleState.TASK_LOCAL: [SkillLifecycleState.TASK_LOCAL],
            SkillLifecycleState.EXECUTED: [SkillLifecycleState.TASK_LOCAL],
            SkillLifecycleState.CREDITED: [SkillLifecycleState.TASK_LOCAL],
            SkillLifecycleState.REFINED: [SkillLifecycleState.TASK_LOCAL],
            SkillLifecycleState.HELD_OUT_VALIDATED: [SkillLifecycleState.TASK_LOCAL],
            SkillLifecycleState.COMMITTED: [SkillLifecycleState.TASK_LOCAL],
        }[rollback_state]
        current = verified
        for state in path:
            current = transition_skill(history, state)
        if rollback_state.value in {
            "executed",
            "credited",
            "refined",
            "held_out_validated",
            "committed",
        }:
            current = transition_skill(
                history,
                SkillLifecycleState.EXECUTED,
                execution_evidence=_execution_evidence(),
            )
        if rollback_state.value in {"credited", "refined", "held_out_validated", "committed"}:
            current = transition_skill(history, SkillLifecycleState.CREDITED)
        if rollback_state.value in {"refined", "held_out_validated", "committed"}:
            current = transition_skill(
                history,
                SkillLifecycleState.REFINED,
                version="v2",
                supersedes="v1",
            )
        if rollback_state.value in {"held_out_validated", "committed"}:
            epoch_history, record = _closed_epoch(tmp_path)
            current = transition_skill(
                history,
                SkillLifecycleState.HELD_OUT_VALIDATED,
                evaluation_history=epoch_history,
                evaluation_epoch_id="epoch-held-out",
                validation_records=(record,),
            )
        if rollback_state is SkillLifecycleState.COMMITTED:
            current = transition_skill(
                history,
                SkillLifecycleState.COMMITTED,
                rollback_target=target,
            )
    rolled_back = transition_skill(
        history,
        SkillLifecycleState.ROLLED_BACK,
        rollback_target=target,
    )
    assert current.state is rollback_state
    assert rolled_back.supersedes == target.version
    assert source.skill_id == rolled_back.skill_id


def test_refinement_requires_a_new_nonreused_version_and_exact_supersedes() -> None:
    history, _artifact = _advance_to_executed()
    transition_skill(history, SkillLifecycleState.CREDITED)
    with pytest.raises(ValueError, match="version"):
        transition_skill(
            history,
            SkillLifecycleState.REFINED,
            version="v1",
            supersedes="v1",
        )
    with pytest.raises(ValueError, match="supersedes"):
        transition_skill(
            history,
            SkillLifecycleState.REFINED,
            version="v2",
            supersedes="other",
        )


def test_lifecycle_history_prevents_replay_and_detached_snapshot_authority() -> None:
    _store, history, source = _source()
    transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="transition"):
        transition_skill(history, SkillLifecycleState.VERIFIED_SKILL)
    with pytest.raises(ValueError, match="SkillLifecycleHistory"):
        transition_skill(cast(Any, source), SkillLifecycleState.VERIFIED_SKILL)


def test_store_rejects_skill_and_version_reuse_within_authority() -> None:
    store, _history, _source_artifact = _source()
    with pytest.raises(ValueError, match="skill_id"):
        store.create_source("skill-1", "v2", (_ref(),))


def test_artifact_is_frozen_slotted_exact_and_json_round_trips() -> None:
    artifact = SkillArtifact(
        "skill-1",
        "v1",
        SkillLifecycleState.SOURCE_EXPERIENCE,
        (_ref(),),
    )
    restored = SkillArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))
    assert restored == artifact
    assert not hasattr(artifact, "__dict__")
    with pytest.raises(FrozenInstanceError):
        artifact.version = "v2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("skill_id", " "),
        ("version", " v1 "),
        ("state", "source_experience"),
        ("source_refs", [_ref()]),
        ("execution_event_ids", ["action", "outcome", "verification"]),
        ("validation_record_ids", []),
        ("supersedes", 1),
    ],
)
def test_artifact_rejects_hostile_scalar_enum_and_container_values(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "skill_id": "skill-1",
        "version": "v1",
        "state": SkillLifecycleState.SOURCE_EXPERIENCE,
        "source_refs": (_ref(),),
        "execution_event_ids": (),
        "validation_record_ids": (),
        "supersedes": None,
    }
    values[field] = value
    with pytest.raises(ValueError):
        SkillArtifact(**cast(Any, values))


def test_artifact_rejects_duplicate_ordered_references_and_use_time_mutation() -> None:
    with pytest.raises(ValueError, match="source_refs"):
        SkillArtifact(
            "skill-1",
            "v1",
            SkillLifecycleState.SOURCE_EXPERIENCE,
            (_ref(), _ref()),
        )
    artifact = SkillArtifact(
        "skill-1",
        "v1",
        SkillLifecycleState.SOURCE_EXPERIENCE,
        (_ref(),),
    )
    object.__setattr__(artifact, "version", "v2")
    with pytest.raises(ValueError, match="construction binding"):
        artifact.to_dict()


def test_public_exports_are_available() -> None:
    import gepa_mindfulness

    assert gepa_mindfulness.SkillArtifact is SkillArtifact
    assert gepa_mindfulness.SkillLifecycleState is SkillLifecycleState
    assert gepa_mindfulness.transition_skill is transition_skill

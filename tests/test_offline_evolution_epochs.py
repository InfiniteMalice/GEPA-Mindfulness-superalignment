"""Evaluation-epoch boundaries for controlled offline evolution."""

from __future__ import annotations

import json
import sqlite3
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
    EvaluationEpoch,
    EvaluationEpochHistory,
    EvaluationEpochStore,
    append_epoch_record,
    begin_candidate_epoch,
    close_evaluation_epoch,
    evaluation_record_cell_id,
    evaluation_record_id,
    validate_epoch_record,
)


def _record(
    *,
    model_version: str = "model-v1",
    harness_version: str = "harness-v1",
) -> V5EvaluationRecord:
    return V5EvaluationRecord(
        case=CaseIdentity(
            case_id=14,
            case_version="17case-v5",
            case_key="correct_high_stakes_clarifying_abstention",
            case_title="Correct high-stakes clarifying abstention",
        ),
        robustness=RobustnessIdentity(stripe_id="TOOL_ERROR", subtype=None),
        system=SystemIdentity(
            repeat_id=0,
            seed=4_242,
            model_version=model_version,
            harness_version=harness_version,
        ),
        epistemics=EpistemicRecord(
            prediction_ref="event:prediction-14-0",
            evidence_refs=("evidence:request-14", "evidence:tool-failure-14"),
            verifier_refs=("verifier:tool-error-contract-v1",),
            confidence=0.82,
        ),
        behavior=BehaviorRecord(
            action_refs=("event:action-proposed-14-0",),
            abstained=True,
            requested_clarification=True,
        ),
        outcome=OutcomeRecord(
            observation_refs=("event:outcome-observed-14-0",),
            verifier_refs=("event:verification-result-14-0",),
            passed=True,
        ),
        scores=ScoreRecord(
            correctness=1.0,
            calibration=0.82,
            abstention=1.0,
            epistemic_process=0.75,
            total=0.8925,
        ),
        diagnostics=DiagnosticRecord(
            trace_summary="The tool failure was observed before a targeted question.",
            deception_signal=0.13,
            mechanistic_signal=0.44,
        ),
    )


def _epoch(
    *,
    epoch_id: str = "epoch-1",
    model_version: str = "model-v1",
    harness_version: str = "harness-v1",
    record_ids: tuple[str, ...] | None = None,
    record_cell_ids: tuple[str, ...] | None = None,
    closed: bool = False,
) -> EvaluationEpoch:
    declared_ids = (evaluation_record_id(_record()),) if record_ids is None else record_ids
    if record_cell_ids is not None:
        declared_cell_ids = record_cell_ids
    elif record_ids is None:
        declared_cell_ids = (evaluation_record_cell_id(_record()),)
    else:
        declared_cell_ids = tuple(
            "sha256:" + f"{index + 1:064x}" for index in range(len(declared_ids))
        )
    return EvaluationEpoch(
        epoch_id=epoch_id,
        model_version=model_version,
        harness_version=harness_version,
        record_ids=declared_ids,
        record_cell_ids=declared_cell_ids,
        closed=closed,
    )


def test_epoch_accepts_only_the_declared_exact_record_and_versions() -> None:
    record = _record()
    epoch = _epoch()

    validate_epoch_record(epoch, record)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("epoch_id", " "),
        ("epoch_id", " epoch-1 "),
        ("model_version", " "),
        ("model_version", " model-v1 "),
        ("harness_version", " "),
        ("harness_version", " harness-v1 "),
        ("record_ids", ("sha256:" + "A" * 64,)),
        ("record_ids", ("sha256:" + "1" * 64, "sha256:" + "1" * 64)),
        ("record_ids", ["sha256:" + "1" * 64]),
        ("closed", 0),
    ],
)
def test_epoch_rejects_invalid_exact_fields(field_name: str, value: object) -> None:
    values: dict[str, object] = {
        "epoch_id": "epoch-1",
        "model_version": "model-v1",
        "harness_version": "harness-v1",
        "record_ids": ("sha256:" + "1" * 64,),
        "record_cell_ids": ("sha256:" + "2" * 64,),
        "closed": False,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        EvaluationEpoch(**cast(Any, values))


@pytest.mark.parametrize("field_name", ["epoch_id", "model_version", "harness_version"])
def test_epoch_rejects_string_subclasses(field_name: str) -> None:
    class StringSubclass(str):
        pass

    values = _epoch().to_dict()
    values[field_name] = StringSubclass("apparently-valid")

    with pytest.raises(ValueError, match=field_name):
        EvaluationEpoch.from_dict(values)


def test_epoch_is_frozen_slotted_and_json_round_trips() -> None:
    epoch = _epoch(closed=True)

    restored = EvaluationEpoch.from_dict(json.loads(json.dumps(epoch.to_dict())))

    assert restored == epoch
    assert not hasattr(restored, "__dict__")
    with pytest.raises(FrozenInstanceError):
        restored.closed = False


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"epoch_id": "epoch-1"},
        {
            "epoch_id": "epoch-1",
            "model_version": "model-v1",
            "harness_version": "harness-v1",
            "record_ids": [],
            "record_cell_ids": [],
            "closed": False,
            "extra": True,
        },
    ],
)
def test_epoch_json_boundary_requires_an_exact_shape(payload: object) -> None:
    with pytest.raises(ValueError):
        EvaluationEpoch.from_dict(payload)


def test_epoch_rejects_model_and_harness_drift() -> None:
    epoch = _epoch()

    with pytest.raises(ValueError, match="model_version"):
        validate_epoch_record(epoch, _record(model_version="model-v2"))
    with pytest.raises(ValueError, match="harness_version"):
        validate_epoch_record(epoch, _record(harness_version="harness-v2"))


def test_epoch_rejects_a_record_that_was_not_declared() -> None:
    record = _record()
    epoch = _epoch(record_ids=("sha256:" + "0" * 64,))

    with pytest.raises(ValueError, match="not declared"):
        validate_epoch_record(epoch, record)


def test_record_identifier_binds_complete_content_and_versions() -> None:
    source = _record()
    changed = V5EvaluationRecord.from_dict(source.to_dict())
    object.__setattr__(changed.scores, "total", 0.4)

    assert evaluation_record_id(source) != evaluation_record_id(changed)
    assert evaluation_record_id(source) != evaluation_record_id(_record(model_version="model-v2"))


def test_validation_rejects_record_and_epoch_use_time_mutation() -> None:
    record = _record()
    epoch = _epoch()
    object.__setattr__(record.system, "model_version", " ")

    with pytest.raises(ValueError, match="model_version"):
        validate_epoch_record(epoch, record)

    record = _record()
    object.__setattr__(epoch, "record_ids", [evaluation_record_id(record)])
    with pytest.raises(ValueError, match="record_ids"):
        validate_epoch_record(epoch, record)


def test_online_record_append_keeps_frozen_system_versions(tmp_path: Path) -> None:
    record = _record()
    history = _history(tmp_path)

    updated = append_epoch_record(history, record)

    assert updated.record_ids == (evaluation_record_id(record),)
    assert updated.model_version == record.system.model_version
    assert updated.harness_version == record.system.harness_version
    validate_epoch_record(updated, record)


def test_online_append_rejects_closed_epochs_and_duplicate_records(tmp_path: Path) -> None:
    record = _record()
    history = _history(tmp_path)
    append_epoch_record(history, record)
    with pytest.raises(ValueError, match="duplicate"):
        append_epoch_record(history, record)
    close_evaluation_epoch(history)
    with pytest.raises(ValueError, match="closed"):
        append_epoch_record(history, _record())


def _history(tmp_path: Path, *, lineage_id: str = "lineage-1") -> EvaluationEpochHistory:
    store = EvaluationEpochStore(tmp_path / "epochs.sqlite", "test-authority")
    return store.create_root(
        lineage_id=lineage_id,
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )


def test_candidate_change_requires_a_closed_source_and_new_epoch(tmp_path: Path) -> None:
    history = _history(tmp_path)
    with pytest.raises(ValueError, match="source epoch.*closed"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-2",
            model_version="model-v2",
            harness_version="harness-v1",
        )

    close_evaluation_epoch(history)
    with pytest.raises(ValueError, match="epoch_id.*new"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-1",
            model_version="model-v2",
            harness_version="harness-v1",
        )


def test_candidate_change_requires_at_least_one_new_version(tmp_path: Path) -> None:
    history = _history(tmp_path)
    close_evaluation_epoch(history)

    with pytest.raises(ValueError, match="candidate.*version"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-2",
            model_version="model-v1",
            harness_version="harness-v1",
        )


@pytest.mark.parametrize(
    ("model_version", "harness_version", "expected_message"),
    [
        ("model-v1", "harness-v3", "model_version"),
        ("model-v3", "harness-v1", "harness_version"),
    ],
)
def test_candidate_cannot_reuse_a_prior_changed_component_version(
    tmp_path: Path,
    model_version: str,
    harness_version: str,
    expected_message: str,
) -> None:
    history = _history(tmp_path)
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v2",
    )
    close_evaluation_epoch(history)

    with pytest.raises(ValueError, match=expected_message):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-3",
            model_version=model_version,
            harness_version=harness_version,
        )


def test_candidate_epoch_has_new_identity_and_no_inherited_records(tmp_path: Path) -> None:
    history = _history(tmp_path)
    close_evaluation_epoch(history)

    candidate = begin_candidate_epoch(
        history,
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v1",
    )

    assert candidate == EvaluationEpoch(
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v1",
        record_ids=(),
        record_cell_ids=(),
        closed=False,
    )


def test_candidate_history_rejects_caller_supplied_containers() -> None:
    source = _epoch(record_ids=(), closed=True)

    with pytest.raises(ValueError, match="EvaluationEpochHistory"):
        begin_candidate_epoch(
            cast(Any, [source]),
            epoch_id="epoch-2",
            model_version="model-v2",
            harness_version="harness-v1",
        )


def test_epoch_helpers_reject_record_and_epoch_subclasses() -> None:
    class EpochSubclass(EvaluationEpoch):
        pass

    class RecordSubclass(V5EvaluationRecord):
        pass

    epoch = _epoch()
    record = _record()
    derived_epoch = EpochSubclass(
        epoch.epoch_id,
        epoch.model_version,
        epoch.harness_version,
        epoch.record_ids,
        epoch.record_cell_ids,
    )
    derived_record = RecordSubclass(
        record.case,
        record.robustness,
        record.system,
        record.epistemics,
        record.behavior,
        record.outcome,
        record.scores,
        record.diagnostics,
    )

    with pytest.raises(ValueError, match="exact EvaluationEpoch"):
        validate_epoch_record(derived_epoch, record)
    with pytest.raises(ValueError, match="exact V5EvaluationRecord"):
        validate_epoch_record(epoch, derived_record)


def test_record_identifiers_require_exact_lowercase_sha256() -> None:
    class StringSubclass(str):
        pass

    invalid_ids = (
        "sha256:" + "A" * 64,
        "sha256:1234",
        "sha256:" + "g" * 64,
        cast(str, StringSubclass("sha256:" + "1" * 64)),
    )
    for invalid in invalid_ids:
        with pytest.raises(ValueError, match="record_ids"):
            _epoch(record_ids=(invalid,))

    payload = _epoch(record_ids=("sha256:" + "1" * 64,)).to_dict()
    cast(list[object], payload["record_ids"])[0] = "sha256:" + "A" * 64
    with pytest.raises(ValueError, match="record_ids"):
        EvaluationEpoch.from_dict(payload)


def test_epoch_binding_rejects_coherent_epoch_and_record_mutation() -> None:
    record = _record()
    epoch = _epoch()
    object.__setattr__(epoch, "model_version", "model-v2")
    object.__setattr__(record.system, "model_version", "model-v2")
    object.__setattr__(epoch, "record_ids", (evaluation_record_id(record),))
    object.__setattr__(epoch, "record_cell_ids", (evaluation_record_cell_id(record),))

    with pytest.raises(ValueError, match="construction binding"):
        epoch.to_dict()
    with pytest.raises(ValueError, match="construction binding"):
        validate_epoch_record(epoch, record)


def test_duplicate_logical_cell_is_rejected_when_content_differs(tmp_path: Path) -> None:
    first = _record()
    history = _history(tmp_path)
    append_epoch_record(history, first)
    changed = V5EvaluationRecord.from_dict(first.to_dict())
    object.__setattr__(changed.scores, "total", 0.4)
    assert evaluation_record_id(changed) != evaluation_record_id(first)
    assert evaluation_record_cell_id(changed) == evaluation_record_cell_id(first)

    with pytest.raises(ValueError, match="logical evaluation cell"):
        append_epoch_record(history, changed)


def test_history_snapshot_is_detached_and_close_is_controlled(tmp_path: Path) -> None:
    history = _history(tmp_path)
    record = _record()
    appended = append_epoch_record(history, record)
    validate_epoch_record(appended, record)
    detached = history.snapshot()
    object.__setattr__(detached[0], "closed", True)
    with pytest.raises(ValueError, match="source epoch.*closed"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-2",
            model_version="model-v2",
            harness_version="harness-v1",
        )

    closed = close_evaluation_epoch(history)
    assert closed.closed is True
    with pytest.raises(ValueError, match="already closed"):
        close_evaluation_epoch(history)


def test_history_rejects_omission_reorder_fork_and_non_tip_authority(tmp_path: Path) -> None:
    history = _history(tmp_path)
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v1",
    )
    close_evaluation_epoch(history)
    lineage = history.snapshot()

    forged_histories = (lineage[:1], tuple(reversed(lineage)), (lineage[0], lineage[0]))
    for forged in forged_histories:
        with pytest.raises(ValueError, match="EvaluationEpochHistory"):
            begin_candidate_epoch(
                cast(Any, forged),
                epoch_id="epoch-3",
                model_version="model-v3",
                harness_version="harness-v1",
            )


def test_history_validates_every_prior_transition_and_global_version_reuse(tmp_path: Path) -> None:
    history = _history(tmp_path)
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v2",
    )
    close_evaluation_epoch(history)

    with pytest.raises(ValueError, match="model_version"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-3",
            model_version="model-v1",
            harness_version="harness-v3",
        )

    # Persisted historical entries are revalidated against the authority catalog before use.
    database = tmp_path / "epochs.sqlite"
    with sqlite3.connect(database) as connection:
        payload = connection.execute(
            "SELECT payload FROM epoch_lineages WHERE authority_domain = ? AND lineage_id = ?",
            ("test-authority", "lineage-1"),
        ).fetchone()[0]
        connection.execute(
            "UPDATE epoch_lineages SET payload = ? WHERE authority_domain = ? AND lineage_id = ?",
            (
                payload.replace('"model_version":"model-v1"', '"model_version":"model-v2"'),
                "test-authority",
                "lineage-1",
            ),
        )
    with pytest.raises(ValueError, match="catalog|lineage"):
        begin_candidate_epoch(
            history,
            epoch_id="epoch-3",
            model_version="model-v3",
            harness_version="harness-v3",
        )


def test_durable_store_reopens_complete_lineage_and_preserves_nonreuse(tmp_path: Path) -> None:
    database = tmp_path / "epochs.sqlite"
    first_store = EvaluationEpochStore(database, "production-authority")
    history = first_store.create_root(
        lineage_id="main",
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    append_epoch_record(history, _record())
    close_evaluation_epoch(history)
    begin_candidate_epoch(
        history,
        epoch_id="epoch-2",
        model_version="model-v2",
        harness_version="harness-v1",
    )
    close_evaluation_epoch(history)

    restarted_store = EvaluationEpochStore(database, "production-authority")
    resumed = restarted_store.open("main")
    assert tuple(epoch.epoch_id for epoch in resumed.snapshot()) == ("epoch-1", "epoch-2")
    with pytest.raises(ValueError, match="model_version"):
        begin_candidate_epoch(
            resumed,
            epoch_id="epoch-3",
            model_version="model-v1",
            harness_version="harness-v2",
        )


def test_store_exclusively_claims_roots_epochs_and_versions_per_authority(tmp_path: Path) -> None:
    database = tmp_path / "epochs.sqlite"
    store = EvaluationEpochStore(database, "shared-authority")
    store.create_root(
        lineage_id="main",
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    with pytest.raises(ValueError, match="lineage"):
        store.create_root(
            lineage_id="main",
            epoch_id="epoch-2",
            model_version="model-v2",
            harness_version="harness-v2",
        )
    with pytest.raises(ValueError, match="authority domain"):
        store.create_root(
            lineage_id="fork",
            epoch_id="epoch-1",
            model_version="model-v2",
            harness_version="harness-v2",
        )
    with pytest.raises(ValueError, match="authority domain"):
        store.create_root(
            lineage_id="fork",
            epoch_id="epoch-other",
            model_version="model-v1",
            harness_version="harness-other",
        )

    # Nonreuse is scoped to an injected authority domain, not a universal process registry.
    other_domain = EvaluationEpochStore(database, "independent-authority")
    other_domain.create_root(
        lineage_id="main",
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )


def test_store_handles_fail_stale_transitions_atomically(tmp_path: Path) -> None:
    database = tmp_path / "epochs.sqlite"
    store = EvaluationEpochStore(database, "production-authority")
    first = store.create_root(
        lineage_id="main",
        epoch_id="epoch-1",
        model_version="model-v1",
        harness_version="harness-v1",
    )
    stale = EvaluationEpochStore(database, "production-authority").open("main")
    append_epoch_record(first, _record())

    with pytest.raises(RuntimeError, match="stale revision"):
        close_evaluation_epoch(stale)
    assert store.open("main").snapshot()[0].closed is False


def test_public_mutation_rejects_raw_epochs_even_with_forged_cell_manifest() -> None:
    record = _record()
    forged = EvaluationEpoch(
        epoch_id="epoch-forged",
        model_version=record.system.model_version,
        harness_version=record.system.harness_version,
        record_ids=("sha256:" + "0" * 64,),
        record_cell_ids=(evaluation_record_cell_id(record),),
    )

    assert not hasattr(EvaluationEpochHistory, "enroll")
    with pytest.raises(ValueError, match="authoritative EvaluationEpochHistory"):
        append_epoch_record(cast(Any, forged), record)

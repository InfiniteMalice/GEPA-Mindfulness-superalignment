"""Atomic and fail-closed tests for the local RL checkpoint store."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

import gepa_mindfulness.training.checkpointing as checkpointing
from gepa_mindfulness.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSnapshot,
    LocalCheckpointStore,
)

DATASET_HASH = "a" * 64
CONFIG_HASH = "b" * 64


def _snapshot(
    *, global_step: int = 3, parent: str | None = "checkpoint-00000002"
) -> CheckpointSnapshot:
    return CheckpointSnapshot(
        global_step=global_step,
        algorithm_state={"updates": global_step, "target_kl": 0.1},
        scheduler_state=None,
        python_rng_state=random.getstate(),
        torch_cpu_rng_state=torch.get_rng_state().clone(),
        torch_cuda_rng_states=(),
        canonical_config={"runtime": {"backend": "pytorch", "device": "cpu"}},
        dataset_hash=DATASET_HASH,
        config_hash=CONFIG_HASH,
        parent_checkpoint=parent,
    )


@pytest.fixture
def backend_callbacks() -> tuple[list[bytes], object, object]:
    restored: list[bytes] = []

    def save_backend(path: Path) -> None:
        path.write_bytes(b"versioned-backend-payload")

    def load_backend(path: Path) -> None:
        restored.append(path.read_bytes())

    return restored, save_backend, load_backend


def test_checkpoint_round_trip_restores_step_rng_parent_and_state(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
) -> None:
    restored_backend, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_load=load_backend,
    )
    snapshot = _snapshot()

    manifest = store.save(snapshot)
    restored = store.load(
        manifest.path,
        expected_dataset_hash=DATASET_HASH,
        expected_config_hash=CONFIG_HASH,
    )

    assert restored.global_step == 3
    assert restored.artifact_hashes == manifest.artifact_hashes
    assert restored.algorithm_state == snapshot.algorithm_state
    assert restored.scheduler_state is None
    assert restored.python_rng_state == snapshot.python_rng_state
    assert torch.equal(restored.torch_cpu_rng_state, snapshot.torch_cpu_rng_state)
    assert restored.torch_cuda_rng_states == ()
    assert restored.canonical_config == snapshot.canonical_config
    assert restored.parent_checkpoint == "checkpoint-00000002"
    assert restored_backend == [b"versioned-backend-payload"]


def test_checkpoint_manifest_has_exact_versioned_compatibility_fields(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
) -> None:
    _, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, backend_load=load_backend)

    manifest = store.save(_snapshot(global_step=7, parent=None))
    payload = json.loads((manifest.path / "manifest.json").read_text(encoding="utf-8"))

    assert set(payload) == {
        "artifact_hashes",
        "checkpoint_id",
        "config_hash",
        "dataset_hash",
        "global_step",
        "parent_checkpoint",
        "schema_version",
    }
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["checkpoint_id"] == "checkpoint-00000007"
    assert payload["global_step"] == 7
    assert payload["parent_checkpoint"] is None
    assert set(payload["artifact_hashes"]) == {"backend.pt", "training_state.pt"}
    assert all(len(digest) == 64 for digest in payload["artifact_hashes"].values())


def test_checkpoint_save_failure_leaves_no_destination_or_temporary_directory(
    tmp_path: Path,
) -> None:
    def fail_after_partial_write(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("backend save failed")

    store = LocalCheckpointStore(tmp_path, backend_save=fail_after_partial_write)

    with pytest.raises(RuntimeError, match="backend save failed"):
        store.save(_snapshot(), checkpoint_id="chosen")

    assert not (tmp_path / "chosen").exists()
    assert list(tmp_path.glob(".chosen.tmp-*")) == []


def test_checkpoint_rejects_unrecognized_backend_artifacts_without_publication(
    tmp_path: Path,
) -> None:
    def save_extra_artifact(path: Path) -> None:
        path.write_bytes(b"payload")
        (path.parent / "unrecognized.bin").write_bytes(b"unexpected")

    store = LocalCheckpointStore(tmp_path, backend_save=save_extra_artifact)

    with pytest.raises(ValueError, match="directory fields"):
        store.save(_snapshot(), checkpoint_id="chosen")

    assert not (tmp_path / "chosen").exists()
    assert list(tmp_path.glob(".chosen.tmp-*")) == []


def test_checkpoint_cleanup_failure_preserves_the_primary_save_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_backend(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("primary backend save failure")

    def fail_cleanup(path: Path) -> None:
        raise OSError("secondary cleanup failure")

    monkeypatch.setattr(checkpointing.shutil, "rmtree", fail_cleanup)
    store = LocalCheckpointStore(tmp_path, backend_save=fail_backend)

    with pytest.raises(RuntimeError, match="cleanup") as caught:
        store.save(_snapshot(), checkpoint_id="chosen")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "primary backend save failure"
    assert not (tmp_path / "chosen").exists()


def test_checkpoint_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "chosen"
    destination.mkdir()
    marker = destination / "operator-data.txt"
    marker.write_text("keep", encoding="utf-8")
    store = LocalCheckpointStore(tmp_path, backend_save=lambda path: path.write_bytes(b"payload"))

    with pytest.raises(FileExistsError, match="already exists"):
        store.save(_snapshot(), checkpoint_id="chosen")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_artifact_tamper_fails_before_backend_restoration(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
) -> None:
    restored_backend, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, backend_load=load_backend)
    manifest = store.save(_snapshot())
    (manifest.path / "backend.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        store.load(manifest.path)

    assert restored_backend == []


@pytest.mark.parametrize("field", ["dataset_hash", "config_hash"])
def test_incompatible_hashes_fail_before_backend_restoration(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
    field: str,
) -> None:
    restored_backend, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, backend_load=load_backend)
    manifest = store.save(_snapshot())
    expectations = {
        "expected_dataset_hash": DATASET_HASH,
        "expected_config_hash": CONFIG_HASH,
    }
    expectations[f"expected_{field}"] = "c" * 64

    with pytest.raises(ValueError, match=field):
        store.load(manifest.path, **expectations)

    assert restored_backend == []


def test_checkpoint_rejects_unsafe_save_and_load_paths_before_callbacks(tmp_path: Path) -> None:
    callbacks: list[str] = []

    def save_backend(path: Path) -> None:
        callbacks.append("save")
        path.write_bytes(b"payload")

    def load_backend(path: Path) -> None:
        callbacks.append("load")

    store = LocalCheckpointStore(
        tmp_path / "root", backend_save=save_backend, backend_load=load_backend
    )

    with pytest.raises(ValueError, match="checkpoint_id"):
        store.save(_snapshot(), checkpoint_id="../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="checkpoint root"):
        store.load(outside)

    assert callbacks == []


def test_checkpoint_rejects_closed_schema_manifest_before_backend_restoration(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
) -> None:
    restored_backend, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, backend_load=load_backend)
    manifest = store.save(_snapshot())
    manifest_path = manifest.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unrecognized"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest fields"):
        store.load(manifest.path)

    assert restored_backend == []


def test_checkpoint_rejects_rehashed_incompatible_state_before_backend_restoration(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, object],
) -> None:
    restored_backend, save_backend, load_backend = backend_callbacks
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, backend_load=load_backend)
    manifest = store.save(_snapshot())
    state_path = manifest.path / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["global_step"] = 999
    torch.save(state, state_path)
    manifest_path = manifest.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_hashes"]["training_state.pt"] = LocalCheckpointStore.sha256(state_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="global_step"):
        store.load(manifest.path)

    assert restored_backend == []

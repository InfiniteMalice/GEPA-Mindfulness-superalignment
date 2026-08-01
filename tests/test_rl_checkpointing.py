"""Atomic and fail-closed tests for the local RL checkpoint store."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

import gepa_mindfulness.training.checkpointing as checkpointing
from gepa_mindfulness.training.backends.base import BackendCheckpointResult
from gepa_mindfulness.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointRNGTopology,
    CheckpointSnapshot,
    LocalCheckpointStore,
    RankRNGState,
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
        batch_cursor=7,
        rollout_cursor=9,
        parent_checkpoint=parent,
    )


@dataclass
class _BackendBytesCallbacks:
    restored: list[bytes]
    step: int = 3
    format_version: int = 1

    def preflight(self, payload: bytes) -> BackendCheckpointResult:
        assert payload == b"versioned-backend-payload"
        return BackendCheckpointResult(format_version=self.format_version, step=self.step)

    def load(self, payload: bytes) -> BackendCheckpointResult:
        self.restored.append(payload)
        return self.preflight(payload)

    def snapshot(self) -> object:
        return list(self.restored)

    def rollback(self, snapshot: object) -> None:
        assert isinstance(snapshot, list)
        self.restored[:] = snapshot


@pytest.fixture
def backend_callbacks() -> tuple[list[bytes], object, _BackendBytesCallbacks]:
    restored: list[bytes] = []

    def save_backend(path: Path) -> BackendCheckpointResult:
        path.write_bytes(b"versioned-backend-payload")
        return BackendCheckpointResult(format_version=1, step=3)

    return restored, save_backend, _BackendBytesCallbacks(restored)


def test_nonzero_rank_checkpoint_store_never_creates_or_truncates_output(tmp_path: Path) -> None:
    backend_calls: list[Path] = []

    def unexpected_backend_save(path: Path) -> BackendCheckpointResult:
        backend_calls.append(path)
        raise AssertionError("nonzero rank must not publish a backend checkpoint")

    store = LocalCheckpointStore(
        tmp_path,
        rank=1,
        world_size=2,
        backend_save=unexpected_backend_save,
    )

    assert store.save(_snapshot()) is None
    assert backend_calls == []
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_round_trip_restores_step_rng_parent_and_state(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
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
    assert restored.batch_cursor == 7
    assert restored.rollout_cursor == 9
    assert restored.parent_checkpoint == "checkpoint-00000002"
    assert restored_backend == [b"versioned-backend-payload"]


def test_distributed_checkpoint_load_selects_exact_rank_rng_state(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, save_backend, backend_bytes = backend_callbacks
    original = torch.get_rng_state()
    try:
        torch.manual_seed(101)
        rank_zero_cpu = torch.get_rng_state().clone()
        torch.manual_seed(202)
        rank_one_cpu = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(original)
    rank_zero_cuda = torch.zeros(8, dtype=torch.uint8)
    rank_one_cuda = torch.ones(8, dtype=torch.uint8)
    rank_states = {
        0: RankRNGState(
            python_rng_state=random.Random(101).getstate(),
            torch_cpu_rng_state=rank_zero_cpu,
            torch_cuda_rng_states=(rank_zero_cuda,),
        ),
        1: RankRNGState(
            python_rng_state=random.Random(202).getstate(),
            torch_cpu_rng_state=rank_one_cpu,
            torch_cuda_rng_states=(rank_one_cuda,),
        ),
    }
    snapshot = replace(
        _snapshot(),
        torch_cuda_rng_states=(rank_zero_cuda,),
        rank_rng_states=rank_states,
    )
    topology = CheckpointRNGTopology(
        device_type="cuda",
        cpu_state_length=len(rank_zero_cpu),
        cuda_state_lengths=(8,),
    )
    publisher = LocalCheckpointStore(
        tmp_path,
        rank=0,
        world_size=2,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
        rng_topology=topology,
    )
    manifest = publisher.save(snapshot)
    rank_one_loader = LocalCheckpointStore(
        tmp_path,
        rank=1,
        world_size=2,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
        rng_topology=topology,
    )

    restored = rank_one_loader.load(manifest.path)

    assert restored.python_rng_state == rank_states[1].python_rng_state
    assert torch.equal(restored.torch_cpu_rng_state, rank_one_cpu)
    assert len(restored.torch_cuda_rng_states) == 1
    assert torch.equal(restored.torch_cuda_rng_states[0], rank_one_cuda)


def test_distributed_checkpoint_rejects_missing_rank_rng_before_publication(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, save_backend, _ = backend_callbacks
    snapshot = replace(
        _snapshot(),
        rank_rng_states={
            0: RankRNGState(
                python_rng_state=random.Random(101).getstate(),
                torch_cpu_rng_state=torch.get_rng_state().clone(),
                torch_cuda_rng_states=(),
            )
        },
    )
    store = LocalCheckpointStore(
        tmp_path,
        rank=0,
        world_size=2,
        backend_save=save_backend,
    )

    with pytest.raises(ValueError, match="rank_rng_states.*world_size"):
        store.save(snapshot)

    assert list(tmp_path.iterdir()) == []


def test_distributed_checkpoint_validates_every_rank_rng_before_publication(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, save_backend, _ = backend_callbacks
    snapshot = replace(
        _snapshot(),
        rank_rng_states={
            0: RankRNGState(
                python_rng_state=random.Random(101).getstate(),
                torch_cpu_rng_state=torch.get_rng_state().clone(),
                torch_cuda_rng_states=(),
            ),
            1: RankRNGState(
                python_rng_state=random.Random(202).getstate(),
                torch_cpu_rng_state=torch.zeros(1, dtype=torch.uint8),
                torch_cuda_rng_states=(),
            ),
        },
    )
    store = LocalCheckpointStore(
        tmp_path,
        rank=0,
        world_size=2,
        backend_save=save_backend,
    )

    with pytest.raises(ValueError, match="rank 1.*CPU RNG state length"):
        store.save(snapshot)

    assert list(tmp_path.iterdir()) == []


def test_checkpoint_manifest_has_exact_versioned_compatibility_fields(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, _, backend_bytes = backend_callbacks

    def save_backend(path: Path) -> BackendCheckpointResult:
        path.write_bytes(b"versioned-backend-payload")
        return BackendCheckpointResult(format_version=1, step=7)

    backend_bytes.step = 7
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )

    manifest = store.save(_snapshot(global_step=7, parent=None))
    payload = json.loads((manifest.path / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        manifest.path / "training_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert set(payload) == {
        "artifact_hashes",
        "backend_format_version",
        "backend_step",
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
    assert payload["backend_format_version"] == 1
    assert payload["backend_step"] == 7
    assert state["backend_format_version"] == 1
    assert state["backend_step"] == 7
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
    def save_extra_artifact(path: Path) -> BackendCheckpointResult:
        path.write_bytes(b"payload")
        (path.parent / "unrecognized.bin").write_bytes(b"unexpected")
        return BackendCheckpointResult(format_version=1, step=3)

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
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
    manifest = store.save(_snapshot())
    (manifest.path / "backend.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        store.load(manifest.path)

    assert restored_backend == []


@pytest.mark.parametrize("field", ["dataset_hash", "config_hash"])
def test_incompatible_hashes_fail_before_backend_restoration(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
    field: str,
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
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

    def save_backend(path: Path) -> BackendCheckpointResult:
        callbacks.append("save")
        path.write_bytes(b"payload")
        return BackendCheckpointResult(format_version=1, step=3)

    def preflight_backend(payload: bytes) -> BackendCheckpointResult:
        callbacks.append("preflight")
        return BackendCheckpointResult(format_version=1, step=3)

    def load_backend(payload: bytes) -> BackendCheckpointResult:
        callbacks.append("load")
        return BackendCheckpointResult(format_version=1, step=3)

    store = LocalCheckpointStore(
        tmp_path / "root",
        backend_save=save_backend,
        backend_preflight=preflight_backend,
        backend_load_bytes=load_backend,
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
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
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
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
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


def test_checkpoint_rejects_wrong_cpu_rng_length_before_backend_save(tmp_path: Path) -> None:
    callbacks: list[str] = []

    def save_backend(path: Path) -> BackendCheckpointResult:
        callbacks.append("save")
        path.write_bytes(b"payload")
        return BackendCheckpointResult(format_version=1, step=3)

    store = LocalCheckpointStore(tmp_path, backend_save=save_backend)
    snapshot = replace(_snapshot(), torch_cpu_rng_state=torch.zeros(1, dtype=torch.uint8))

    with pytest.raises(ValueError, match="CPU RNG state length"):
        store.save(snapshot)

    assert callbacks == []


def test_checkpoint_rejects_cuda_rng_state_for_cpu_topology_before_backend_save(
    tmp_path: Path,
) -> None:
    callbacks: list[str] = []

    def save_backend(path: Path) -> BackendCheckpointResult:
        callbacks.append("save")
        path.write_bytes(b"payload")
        return BackendCheckpointResult(format_version=1, step=3)

    store = LocalCheckpointStore(tmp_path, backend_save=save_backend)
    snapshot = replace(_snapshot(), torch_cuda_rng_states=(torch.zeros(8, dtype=torch.uint8),))

    with pytest.raises(ValueError, match="CUDA RNG states"):
        store.save(snapshot)

    assert callbacks == []


@pytest.mark.parametrize(
    "cuda_states, match",
    [
        ((torch.zeros(8, dtype=torch.uint8),), "CUDA RNG state count"),
        (
            (
                torch.zeros(8, dtype=torch.uint8),
                torch.zeros(8, dtype=torch.uint8),
            ),
            "CUDA RNG state length",
        ),
    ],
)
def test_checkpoint_validates_explicit_cuda_rng_topology_before_backend_save(
    tmp_path: Path,
    cuda_states: tuple[torch.Tensor, ...],
    match: str,
) -> None:
    callbacks: list[str] = []

    def save_backend(path: Path) -> BackendCheckpointResult:
        callbacks.append("save")
        path.write_bytes(b"payload")
        return BackendCheckpointResult(format_version=1, step=3)

    topology = CheckpointRNGTopology(
        device_type="cuda",
        cpu_state_length=len(torch.get_rng_state()),
        cuda_state_lengths=(8, 9),
    )
    store = LocalCheckpointStore(tmp_path, backend_save=save_backend, rng_topology=topology)
    snapshot = replace(_snapshot(), torch_cuda_rng_states=cuda_states)

    with pytest.raises(ValueError, match=match):
        store.save(snapshot)

    assert callbacks == []


def test_checkpoint_rejects_rehashed_wrong_rng_length_before_backend_preflight(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    preflight_payloads: list[bytes] = []

    def preflight(payload: bytes) -> BackendCheckpointResult:
        preflight_payloads.append(payload)
        return backend_bytes.preflight(payload)

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=preflight,
        backend_load_bytes=backend_bytes.load,
    )
    manifest = store.save(_snapshot())
    state_path = manifest.path / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["rank_rng_states"][0]["torch_cpu_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(state, state_path)
    manifest_path = manifest.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_hashes"]["training_state.pt"] = LocalCheckpointStore.sha256(state_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="CPU RNG state length"):
        store.load(manifest.path)

    assert preflight_payloads == []
    assert restored_backend == []


def test_checkpoint_rejects_backend_step_mismatch_before_publication(tmp_path: Path) -> None:
    def save_backend(path: Path) -> BackendCheckpointResult:
        path.write_bytes(b"payload")
        return BackendCheckpointResult(format_version=1, step=2)

    store = LocalCheckpointStore(tmp_path, backend_save=save_backend)

    with pytest.raises(ValueError, match="backend step"):
        store.save(_snapshot(), checkpoint_id="chosen")

    assert not (tmp_path / "chosen").exists()


def test_checkpoint_preflight_step_mismatch_fails_before_backend_restore(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=lambda payload: BackendCheckpointResult(format_version=1, step=2),
        backend_load_bytes=backend_bytes.load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
    )
    manifest = store.save(_snapshot())

    with pytest.raises(ValueError, match="backend step"):
        store.load(manifest.path)

    assert restored_backend == []


def test_checkpoint_restore_step_mismatch_is_rejected(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks

    def mismatched_load(payload: bytes) -> BackendCheckpointResult:
        restored_backend.append(payload)
        return BackendCheckpointResult(format_version=1, step=2)

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=mismatched_load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
    )
    manifest = store.save(_snapshot())

    with pytest.raises(ValueError, match="backend step"):
        store.load(manifest.path)

    assert restored_backend == []


def test_checkpoint_load_uses_verified_backend_bytes_when_path_changes_after_preflight(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    checkpoint_path: Path | None = None

    def mutate_after_preflight(payload: bytes) -> BackendCheckpointResult:
        assert checkpoint_path is not None
        (checkpoint_path / "backend.pt").write_bytes(b"mutated-after-verification")
        return backend_bytes.preflight(payload)

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=mutate_after_preflight,
        backend_load_bytes=backend_bytes.load,
        backend_snapshot=backend_bytes.snapshot,
        backend_rollback=backend_bytes.rollback,
    )
    manifest = store.save(_snapshot())
    checkpoint_path = manifest.path

    store.load(manifest.path)

    assert restored_backend == [b"versioned-backend-payload"]


def test_checkpoint_rejects_symlink_source_before_backend_callbacks(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
    manifest = store.save(_snapshot())
    symlink = tmp_path / "checkpoint-link"
    try:
        os.symlink(manifest.path, symlink, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        store.load(symlink)

    assert restored_backend == []


@pytest.mark.parametrize("artifact", ["manifest.json", "backend.pt", "training_state.pt"])
def test_checkpoint_rejects_symlink_members_before_backend_callbacks(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
    artifact: str,
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=backend_bytes.load,
    )
    manifest = store.save(_snapshot())
    member = manifest.path / artifact
    moved = tmp_path / f"original-{artifact}"
    member.replace(moved)
    try:
        os.symlink(moved, member)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        store.load(manifest.path)

    assert restored_backend == []


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (BackendCheckpointResult(format_version=1, step=2), "backend step"),
        (BackendCheckpointResult(format_version=2, step=3), "format_version"),
    ],
)
def test_checkpoint_rolls_back_backend_mutation_when_restore_evidence_mismatches(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
    result: BackendCheckpointResult,
    match: str,
) -> None:
    _, save_backend, backend_bytes = backend_callbacks
    backend_state = {"value": "before"}

    def load(payload: bytes) -> BackendCheckpointResult:
        backend_state["value"] = "mutated"
        assert payload == b"versioned-backend-payload"
        return result

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=load,
        backend_snapshot=lambda: dict(backend_state),
        backend_rollback=lambda snapshot: backend_state.update(snapshot),
    )
    manifest = store.save(_snapshot())

    with pytest.raises(ValueError, match=match):
        store.load(manifest.path)

    assert backend_state == {"value": "before"}


def test_checkpoint_rolls_back_backend_mutation_and_preserves_loader_exception(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, save_backend, backend_bytes = backend_callbacks
    backend_state = {"value": "before"}
    primary = RuntimeError("loader exploded after mutation")

    def load(payload: bytes) -> BackendCheckpointResult:
        backend_state["value"] = "mutated"
        assert payload == b"versioned-backend-payload"
        raise primary

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=load,
        backend_snapshot=lambda: dict(backend_state),
        backend_rollback=lambda snapshot: backend_state.update(snapshot),
    )
    manifest = store.save(_snapshot())

    with pytest.raises(RuntimeError, match="loader exploded") as caught:
        store.load(manifest.path)

    assert caught.value is primary
    assert backend_state == {"value": "before"}


def test_checkpoint_preserves_primary_and_rollback_failure_diagnostics(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    _, save_backend, backend_bytes = backend_callbacks
    primary = RuntimeError("primary loader failure")

    def load(payload: bytes) -> BackendCheckpointResult:
        raise primary

    def rollback(snapshot: object) -> None:
        raise OSError("rollback diagnostics")

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=backend_bytes.preflight,
        backend_load_bytes=load,
        backend_snapshot=lambda: {"value": "before"},
        backend_rollback=rollback,
    )
    manifest = store.save(_snapshot())

    with pytest.raises(RuntimeError, match="rollback.*OSError.*rollback diagnostics") as caught:
        store.load(manifest.path)

    assert caught.value.__cause__ is primary


def test_checkpoint_requires_transaction_hooks_before_backend_preflight_or_restore(
    tmp_path: Path,
    backend_callbacks: tuple[list[bytes], object, _BackendBytesCallbacks],
) -> None:
    restored_backend, save_backend, backend_bytes = backend_callbacks
    preflight_payloads: list[bytes] = []

    def preflight(payload: bytes) -> BackendCheckpointResult:
        preflight_payloads.append(payload)
        return backend_bytes.preflight(payload)

    store = LocalCheckpointStore(
        tmp_path,
        backend_save=save_backend,
        backend_preflight=preflight,
        backend_load_bytes=backend_bytes.load,
    )
    manifest = store.save(_snapshot())

    with pytest.raises(RuntimeError, match="snapshot.*rollback"):
        store.load(manifest.path)

    assert preflight_payloads == []
    assert restored_backend == []

"""Atomic, hash-verified publication tests for prebuilt policy adapters."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import gepa_mindfulness.training.adapter_publication as publication_module
from gepa_mindfulness.training.adapter_publication import (
    AdapterCandidate,
    ArtifactHashError,
    LocalAdapterPublisher,
)
from gepa_mindfulness.training.policy_versions import PolicyVersion


def _spawn_publish(root: str, artifact: str, version: int, gate: object, queue: object) -> None:
    gate.wait()  # type: ignore[attr-defined]
    path = Path(artifact)
    candidate = AdapterCandidate(
        artifact_path=path,
        policy_version=PolicyVersion(version),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parent_policy_version=None,
        format_id="safetensors-v1",
        source_id="spawn-worker",
        model_id="tiny-model",
    )
    try:
        LocalAdapterPublisher(root).publish(candidate)
        queue.put(("ok", version))  # type: ignore[attr-defined]
    except Exception as exc:
        queue.put(("error", type(exc).__name__))  # type: ignore[attr-defined]


def _candidate(
    tmp_path: Path,
    version: int,
    *,
    parent: int | None,
    content: bytes | None = None,
    expected_sha256: str | None = None,
) -> AdapterCandidate:
    payload = content if content is not None else f"adapter-{version}".encode()
    artifact = tmp_path / f"candidate-{version}.safetensors"
    artifact.write_bytes(payload)
    digest = expected_sha256 or hashlib.sha256(payload).hexdigest()
    return AdapterCandidate(
        artifact_path=artifact,
        policy_version=PolicyVersion(version),
        expected_sha256=digest,
        parent_policy_version=None if parent is None else PolicyVersion(parent),
        format_id="safetensors-v1",
        source_id="pytorch-learner",
        model_id="tiny-model",
        metadata={"rank": 8, "scale": 0.25, "producer": "unit-test"},
    )


def test_successful_publication_is_versioned_hash_verified_and_atomic(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    candidate = _candidate(tmp_path, 1, parent=None)

    manifest = publisher.publish(candidate)

    assert manifest.policy_version == PolicyVersion(1)
    assert manifest.parent_policy_version is None
    assert manifest.artifact_sha256 == candidate.expected_sha256
    assert publisher.current() == manifest
    assert (publisher.root / manifest.artifact_path).read_bytes() == b"adapter-1"
    assert not any(path.name.startswith(".adapter-stage-") for path in publisher.root.iterdir())


def test_checksum_mismatch_preserves_previous_readable_manifest(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    current = publisher.publish(_candidate(tmp_path, 1, parent=None))
    corrupt = _candidate(tmp_path, 2, parent=1, expected_sha256="0" * 64)

    with pytest.raises(ArtifactHashError, match="candidate"):
        publisher.publish(corrupt)

    assert publisher.current() == current
    assert (publisher.root / current.artifact_path).read_bytes() == b"adapter-1"


@pytest.mark.parametrize(
    ("version", "parent", "message"),
    [
        (2, None, "parent"),
        (2, 0, "parent"),
        (1, 1, "newer"),
        (0, 1, "newer"),
        (3, 1, "next"),
        (2, 2, "parent"),
    ],
)
def test_parent_mismatch_replay_rollback_and_skip_are_rejected(
    tmp_path: Path,
    version: int,
    parent: int | None,
    message: str,
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    current = publisher.publish(_candidate(tmp_path, 1, parent=None))

    with pytest.raises(ValueError, match=message):
        publisher.publish(_candidate(tmp_path, version, parent=parent))

    assert publisher.current() == current


def test_first_publication_rejects_a_parent(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")

    with pytest.raises(ValueError, match="first.*parent"):
        publisher.publish(_candidate(tmp_path, 1, parent=0))

    assert publisher.current() is None


def test_destination_collision_is_rejected_without_overwrite(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    current = publisher.publish(_candidate(tmp_path, 1, parent=None))
    collision = publisher.root / "versions" / "2"
    collision.mkdir()
    marker = collision / "foreign"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="version"):
        publisher.publish(_candidate(tmp_path, 2, parent=1))

    assert marker.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="orphan"):
        publisher.current()
    assert (publisher.root / current.artifact_path).read_bytes() == b"adapter-1"


def test_interrupted_stage_copy_cleans_exact_stage_and_preserves_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    current = publisher.publish(_candidate(tmp_path, 1, parent=None))

    def interrupted_copy(source: Path, destination: Path) -> str:
        del source
        destination.write_bytes(b"partial")
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(LocalAdapterPublisher, "_copy_artifact", staticmethod(interrupted_copy))

    with pytest.raises(OSError, match="copy interruption"):
        publisher.publish(_candidate(tmp_path, 2, parent=1))

    assert publisher.current() == current
    assert not any(path.name.startswith(".adapter-stage-") for path in publisher.root.iterdir())


def test_interrupted_current_replace_preserves_previous_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    current = publisher.publish(_candidate(tmp_path, 1, parent=None))
    real_replace = publication_module.os.replace

    def interrupted_replace(source: str | bytes | os.PathLike[str], destination: object) -> None:
        if Path(destination) == publisher.root / "current.json":  # type: ignore[arg-type]
            raise OSError("simulated pointer interruption")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(publication_module.os, "replace", interrupted_replace)

    with pytest.raises(OSError, match="pointer interruption"):
        publisher.publish(_candidate(tmp_path, 2, parent=1))

    assert publisher.current() == current
    assert not (publisher.root / "versions" / "2").exists()
    assert not any(path.name.startswith(".current-") for path in publisher.root.iterdir())


def test_tampered_current_schema_and_paths_fail_closed(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    publisher.publish(_candidate(tmp_path, 1, parent=None))
    current_path = publisher.root / "current.json"
    payload = json.loads(current_path.read_text(encoding="utf-8"))

    payload["unexpected"] = True
    current_path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ValueError, match="fields"):
        publisher.current()

    payload.pop("unexpected")
    payload["artifact_path"] = "../secret"
    current_path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(ValueError, match="artifact_path"):
        publisher.current()


def test_tampered_artifact_hash_fails_closed(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    manifest = publisher.publish(_candidate(tmp_path, 1, parent=None))
    (publisher.root / manifest.artifact_path).write_bytes(b"tampered")

    with pytest.raises(ArtifactHashError, match="published"):
        publisher.current()


def test_current_missing_is_none_only_for_genuinely_empty_store(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    assert publisher.current() is None
    (publisher.root / "versions" / "orphan").mkdir()

    with pytest.raises(ValueError, match="current"):
        publisher.current()


def test_symlink_candidate_and_published_symlink_are_rejected(tmp_path: Path) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    target = tmp_path / "target.bin"
    target.write_bytes(b"adapter")
    link = tmp_path / "candidate-link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    candidate = AdapterCandidate(
        artifact_path=link,
        policy_version=PolicyVersion(1),
        expected_sha256=hashlib.sha256(b"adapter").hexdigest(),
        parent_policy_version=None,
        format_id="safetensors-v1",
        source_id="pytorch-learner",
        model_id="tiny-model",
    )

    with pytest.raises(ValueError, match="symlink|regular"):
        publisher.publish(candidate)

    manifest = publisher.publish(_candidate(tmp_path, 1, parent=None))
    artifact = publisher.root / manifest.artifact_path
    artifact.unlink()
    artifact.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|regular"):
        publisher.current()


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_sha256": "A" * 64},
        {"format_id": "../unsafe"},
        {"source_id": "bad\nsource"},
        {"model_id": "org/model"},
        {"metadata": {"nan": float("nan")}},
        {"metadata": {"bool-as-int": True}},
    ],
)
def test_candidate_rejects_noncanonical_or_unsafe_scalars(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "artifact_path": tmp_path / "adapter.bin",
        "policy_version": PolicyVersion(1),
        "expected_sha256": "0" * 64,
        "parent_policy_version": None,
        "format_id": "safetensors-v1",
        "source_id": "pytorch-learner",
        "model_id": "tiny-model",
        "metadata": {},
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        AdapterCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_version", [True, 1.0])
@pytest.mark.parametrize("target", ["current.json", "versions/1/manifest.json"])
def test_schema_version_rejects_bool_and_float(
    tmp_path: Path,
    schema_version: object,
    target: str,
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    publisher.publish(_candidate(tmp_path, 1, parent=None))
    path = publisher.root / target
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())

    with pytest.raises(ValueError, match="schema_version"):
        publisher.current()


def test_two_publishers_serialize_first_publication_without_orphan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "published"
    first = LocalAdapterPublisher(root)
    second = LocalAdapterPublisher(root)
    barrier = threading.Barrier(2)

    def publish(publisher: LocalAdapterPublisher, version: int) -> object:
        barrier.wait()
        return publisher.publish(_candidate(tmp_path, version, parent=None))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, first, 1), pool.submit(publish, second, 2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    current = first.current()
    assert current is not None
    assert [path.name for path in (root / "versions").iterdir()] == [
        current.policy_version.to_json()
    ]


@pytest.mark.parametrize("name", ["foreign.txt", ".adapter-stage-orphan", ".current-orphan.tmp"])
def test_current_rejects_nonempty_control_layout(tmp_path: Path, name: str) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    path = publisher.root / name
    if "stage" in name:
        path.mkdir()
    else:
        path.write_text("orphan", encoding="utf-8")

    with pytest.raises(ValueError, match="layout|current"):
        publisher.current()


def test_internal_artifact_symlink_and_version_directory_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    manifest = publisher.publish(_candidate(tmp_path, 1, parent=None))
    target = tmp_path / "target"
    target.write_bytes(b"adapter-1")
    artifact = publisher.root / manifest.artifact_path
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink|unsafe"):
        publisher.current()

    artifact.unlink()
    version = artifact.parent
    (version / "manifest.json").unlink()
    version.rmdir()
    version.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|unsafe|canonical"):
        publisher.current()


def test_parent_swap_between_check_and_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publisher = LocalAdapterPublisher(tmp_path / "published")
    manifest = publisher.publish(_candidate(tmp_path, 1, parent=None))
    version = (publisher.root / manifest.artifact_path).parent
    external = tmp_path / "external"
    shutil.copytree(version, external)
    real_open = publication_module._open_regular
    swapped = False

    def swapping_open(path: Path, field_name: str) -> object:
        nonlocal swapped
        if not swapped and path.name == "manifest.json":
            swapped = True
            moved = tmp_path / "original-version"
            version.rename(moved)
            external.rename(version)
        return real_open(path, field_name)

    monkeypatch.setattr(publication_module, "_open_regular", swapping_open)
    with pytest.raises(ValueError, match="changed|unsafe"):
        publisher.current()


def test_spawned_publishers_serialize_first_publication(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "published"
    LocalAdapterPublisher(root)
    artifacts = []
    for version in (1, 2):
        path = tmp_path / f"spawn-{version}.bin"
        path.write_bytes(f"spawn-{version}".encode())
        artifacts.append(path)
    gate = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(target=_spawn_publish, args=(str(root), str(path), version, gate, queue))
        for version, path in zip((1, 2), artifacts, strict=True)
    ]
    for process in processes:
        process.start()
    gate.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    results = [queue.get(timeout=5), queue.get(timeout=5)]
    assert sorted(result[0] for result in results) == ["error", "ok"]
    current = LocalAdapterPublisher(root).current()
    assert current is not None
    assert len(tuple((root / "versions").iterdir())) == 1
    assert not any(
        path.name.startswith((".adapter-stage-", ".current-")) for path in root.iterdir()
    )

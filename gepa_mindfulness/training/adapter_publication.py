"""Atomic local publication for already-produced versioned adapter artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from ._windows_file_lock import windows_byte_lock
from .policy_versions import PolicyVersion

ADAPTER_SCHEMA_VERSION = 1
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_path",
        "artifact_sha256",
        "artifact_size",
        "format_id",
        "manifest_path",
        "metadata",
        "model_id",
        "parent_policy_version",
        "policy_version",
        "schema_version",
        "source_id",
    }
)
_MAX_MANIFEST_BYTES = 65_536
_MAX_METADATA_ENTRIES = 32
_MAX_METADATA_STRING = 256
_MAX_METADATA_INTEGER = 2**63 - 1
_MAX_METADATA_FLOAT = 1.0e15
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class ArtifactHashError(ValueError):
    """Raised when candidate or published bytes fail SHA-256 verification."""


@dataclass(frozen=True, slots=True)
class AdapterCandidate:
    """One prebuilt adapter and the evidence required to publish it unchanged."""

    artifact_path: Path
    policy_version: PolicyVersion
    expected_sha256: str
    parent_policy_version: PolicyVersion | None
    format_id: str
    source_id: str
    model_id: str
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise TypeError("artifact_path must be a pathlib.Path")
        _validate_policy_version(self.policy_version, "policy_version")
        _validate_optional_policy_version(
            self.parent_policy_version,
            "parent_policy_version",
        )
        _validate_sha256(self.expected_sha256, "expected_sha256")
        _validate_id(self.format_id, "format_id")
        _validate_id(self.source_id, "source_id")
        _validate_id(self.model_id, "model_id")
        object.__setattr__(self, "metadata", MappingProxyType(_validate_metadata(self.metadata)))


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    """Closed immutable metadata for one verified, published adapter version."""

    policy_version: PolicyVersion
    parent_policy_version: PolicyVersion | None
    artifact_path: str
    manifest_path: str
    artifact_sha256: str
    artifact_size: int
    format_id: str
    source_id: str
    model_id: str
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_policy_version(self.policy_version, "policy_version")
        _validate_optional_policy_version(
            self.parent_policy_version,
            "parent_policy_version",
        )
        expected_artifact, expected_manifest = _version_paths(self.policy_version)
        if self.artifact_path != expected_artifact:
            raise ValueError("adapter artifact_path is not the canonical version path")
        if self.manifest_path != expected_manifest:
            raise ValueError("adapter manifest_path is not the canonical version path")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if type(self.artifact_size) is not int or self.artifact_size <= 0:
            raise ValueError("artifact_size must be a positive integer")
        _validate_id(self.format_id, "format_id")
        _validate_id(self.source_id, "source_id")
        _validate_id(self.model_id, "model_id")
        if type(self.schema_version) is not int or self.schema_version != ADAPTER_SCHEMA_VERSION:
            raise ValueError(f"adapter schema_version must be {ADAPTER_SCHEMA_VERSION}")
        object.__setattr__(self, "metadata", MappingProxyType(_validate_metadata(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        """Return the canonical closed JSON representation."""
        return {
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size": self.artifact_size,
            "format_id": self.format_id,
            "manifest_path": self.manifest_path,
            "metadata": dict(self.metadata),
            "model_id": self.model_id,
            "parent_policy_version": (
                None if self.parent_policy_version is None else self.parent_policy_version.to_json()
            ),
            "policy_version": self.policy_version.to_json(),
            "schema_version": self.schema_version,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, payload: object) -> AdapterManifest:
        """Validate a closed manifest mapping without consulting the filesystem."""
        if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
            raise ValueError("adapter manifest fields are missing or unrecognized")
        parent_value = payload["parent_policy_version"]
        parent = None if parent_value is None else PolicyVersion.from_json(parent_value)
        return cls(
            policy_version=PolicyVersion.from_json(payload["policy_version"]),
            parent_policy_version=parent,
            artifact_path=payload["artifact_path"],
            manifest_path=payload["manifest_path"],
            artifact_sha256=payload["artifact_sha256"],
            artifact_size=payload["artifact_size"],
            format_id=payload["format_id"],
            source_id=payload["source_id"],
            model_id=payload["model_id"],
            metadata=payload["metadata"],
            schema_version=payload["schema_version"],
        )


class LocalAdapterPublisher:
    """Publish immutable adapter versions before atomically advancing current.json."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        _require_directory(self.root, "adapter publication root")
        self.versions = self.root / "versions"
        self.versions.mkdir(exist_ok=True)
        _require_directory(self.versions, "adapter versions directory")

    def publish(self, candidate: AdapterCandidate) -> AdapterManifest:
        """Verify and atomically publish one already-produced adapter artifact."""
        with self._transaction_lock():
            return self._publish_locked(candidate)

    def _publish_locked(self, candidate: AdapterCandidate) -> AdapterManifest:
        self._validate_store()
        if type(candidate) is not AdapterCandidate:
            raise TypeError("candidate must be an AdapterCandidate")
        candidate_digest, candidate_size = _sha256_regular(
            candidate.artifact_path,
            "candidate adapter",
        )
        if candidate_digest != candidate.expected_sha256:
            raise ArtifactHashError("candidate adapter failed SHA-256 verification")
        current = self._current_unlocked(validate_version_layout=False)
        self._validate_chain(candidate, current)
        artifact_path, manifest_path = _version_paths(candidate.policy_version)
        destination = self.root / Path(manifest_path).parent
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("adapter version destination already exists")
        self._current_unlocked()

        stage = self.root / f".adapter-stage-{uuid.uuid4().hex}"
        pointer = self.root / f".current-{uuid.uuid4().hex}.tmp"
        published_version = False
        try:
            stage.mkdir(mode=0o700)
            staged_artifact = stage / "adapter.bin"
            copied_digest = self._copy_artifact(candidate.artifact_path, staged_artifact)
            if copied_digest != candidate.expected_sha256:
                raise ArtifactHashError("staged adapter failed SHA-256 verification")
            manifest = AdapterManifest(
                policy_version=candidate.policy_version,
                parent_policy_version=candidate.parent_policy_version,
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_sha256=candidate.expected_sha256,
                artifact_size=candidate_size,
                format_id=candidate.format_id,
                source_id=candidate.source_id,
                model_id=candidate.model_id,
                metadata=candidate.metadata,
            )
            _write_json_exclusive(stage / "manifest.json", manifest.to_dict())
            staged_digest, staged_size = _sha256_regular(staged_artifact, "staged adapter")
            if staged_digest != manifest.artifact_sha256 or staged_size != manifest.artifact_size:
                raise ArtifactHashError("staged adapter changed before publication")
            os.rename(stage, destination)
            published_version = True
            _write_json_exclusive(pointer, manifest.to_dict())
            os.replace(pointer, self.root / "current.json")
            return manifest
        except BaseException:
            self._remove_pointer(pointer)
            self._remove_owned_directory(stage, stage=True)
            if published_version:
                self._remove_owned_directory(destination, stage=False)
            raise

    def current(self) -> AdapterManifest | None:
        """Return the fully verified current manifest, or None for a genuinely empty store."""
        with self._transaction_lock():
            return self._current_unlocked()

    def current_artifact(self) -> tuple[AdapterManifest, bytes]:
        """Return one current manifest with the exact verified artifact bytes it identifies."""
        with self._transaction_lock():
            manifest, payload = self._read_current_unlocked(retain_artifact=True)
            if manifest is None:
                raise ValueError("adapter publication store has no current artifact")
            return manifest, payload

    def _current_unlocked(
        self,
        *,
        validate_version_layout: bool = True,
    ) -> AdapterManifest | None:
        manifest, _ = self._read_current_unlocked(
            validate_version_layout=validate_version_layout,
            retain_artifact=False,
        )
        return manifest

    def _read_current_unlocked(
        self,
        *,
        validate_version_layout: bool = True,
        retain_artifact: bool,
    ) -> tuple[AdapterManifest | None, bytes]:
        self._validate_store()
        allowed_root = {".publication.lock", "current.json", "versions"}
        if any(path.name not in allowed_root for path in self.root.iterdir()):
            raise ValueError("adapter publication root layout contains unknown control state")
        current_path = self.root / "current.json"
        if not current_path.exists() and not current_path.is_symlink():
            if any(self.versions.iterdir()):
                raise ValueError("adapter versions exist without a current manifest")
            return None, b""
        current_payload = _read_canonical_json(current_path, "current adapter manifest")
        manifest = AdapterManifest.from_dict(current_payload)
        version_values: list[PolicyVersion] = []
        for version_path in self.versions.iterdir():
            _require_directory(version_path, "adapter version directory")
            version_values.append(PolicyVersion.from_json(version_path.name))
        if validate_version_layout and (
            not version_values or max(version_values) != manifest.policy_version
        ):
            raise ValueError("adapter versions contain an orphan or omit the current version")
        artifact_payload, artifact_digest, artifact_size = _contained_read(
            self.root,
            manifest.artifact_path,
            "artifact_path",
            retain=retain_artifact,
        )
        version_payload = _contained_bytes(
            self.root,
            manifest.manifest_path,
            "manifest_path",
        )
        stored_payload = _decode_canonical_json(version_payload, "version adapter manifest")
        stored_manifest = AdapterManifest.from_dict(stored_payload)
        if stored_manifest != manifest:
            raise ValueError("current adapter manifest does not match its version manifest")
        if artifact_digest != manifest.artifact_sha256 or artifact_size != manifest.artifact_size:
            raise ArtifactHashError("published adapter failed SHA-256 verification")
        return manifest, artifact_payload

    @contextmanager
    def _transaction_lock(self) -> Iterator[None]:
        key = os.path.abspath(self.root)
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with thread_lock:
            lock_path = self.root / ".publication.lock"
            if lock_path.exists() or lock_path.is_symlink():
                before = lock_path.lstat()
                if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
                    raise ValueError("adapter publication lock must be a regular file")
            else:
                before = None
            flags = (
                os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(lock_path, flags, 0o600)
            try:
                opened = os.fstat(descriptor)
                after = lock_path.lstat()
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _is_link_or_reparse(after)
                    or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
                    or before is not None
                    and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                ):
                    raise ValueError("adapter publication lock changed or is unsafe")
                if opened.st_size == 0:
                    os.write(descriptor, b"0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                with _descriptor_lock(descriptor):
                    yield
            finally:
                os.close(descriptor)

    def _validate_store(self) -> None:
        _require_directory(self.root, "adapter publication root")
        _require_directory(self.versions, "adapter versions directory")

    @staticmethod
    def _validate_chain(
        candidate: AdapterCandidate,
        current: AdapterManifest | None,
    ) -> None:
        if current is None:
            if candidate.parent_policy_version is not None:
                raise ValueError("first adapter publication must not name a parent")
            return
        if candidate.parent_policy_version != current.policy_version:
            raise ValueError("adapter parent must match the current policy version")
        if candidate.policy_version <= current.policy_version:
            raise ValueError("adapter policy version must be newer than current")
        if candidate.policy_version.value != current.policy_version.value + 1:
            raise ValueError("adapter policy version must be the next linear version")

    @staticmethod
    def _copy_artifact(source: Path, destination: Path) -> str:
        source_stat, source_stream = _open_regular(source, "candidate adapter")
        digest = hashlib.sha256()
        with source_stream:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            destination_descriptor = os.open(destination, flags, 0o600)
            try:
                output = os.fdopen(destination_descriptor, "wb", closefd=True)
            except BaseException:
                try:
                    os.close(destination_descriptor)
                except OSError:
                    pass
                raise
            with output:
                while True:
                    chunk = source_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        after = source.stat(follow_symlinks=False)
        if (source_stat.st_dev, source_stat.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("candidate adapter changed while it was copied")
        return digest.hexdigest()

    def _remove_pointer(self, path: Path) -> None:
        if path.parent != self.root or not path.name.startswith(".current-"):
            raise RuntimeError("refusing to clean an unsafe adapter pointer path")
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _remove_owned_directory(self, path: Path, *, stage: bool) -> None:
        expected_parent = self.root if stage else self.versions
        if path.parent != expected_parent:
            raise RuntimeError("refusing to clean an unsafe adapter directory path")
        if stage and not path.name.startswith(".adapter-stage-"):
            raise RuntimeError("refusing to clean an unowned adapter staging path")
        if not path.exists() and not path.is_symlink():
            return
        info = path.lstat()
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("refusing to clean an unsafe adapter directory")
        allowed = {"adapter.bin", "manifest.json"}
        children = tuple(path.iterdir())
        if any(child.name not in allowed for child in children):
            raise RuntimeError("refusing to clean unexpected adapter publication content")
        for child in children:
            child_info = child.lstat()
            if stat.S_ISDIR(child_info.st_mode) and not _is_link_or_reparse(child_info):
                raise RuntimeError("refusing to recursively clean adapter publication content")
            child.unlink()
        path.rmdir()


def _version_paths(version: PolicyVersion) -> tuple[str, str]:
    component = version.to_json()
    return f"versions/{component}/adapter.bin", f"versions/{component}/manifest.json"


def _validate_policy_version(value: object, field_name: str) -> PolicyVersion:
    if type(value) is not PolicyVersion:
        raise TypeError(f"{field_name} must be a PolicyVersion")
    return value


def _validate_optional_policy_version(
    value: object,
    field_name: str,
) -> PolicyVersion | None:
    if value is None:
        return None
    return _validate_policy_version(value, field_name)


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA-256 digest")
    return value


def _validate_id(value: object, field_name: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{field_name} must be one safe identifier without path separators")
    return value


def _validate_metadata(value: object) -> dict[str, str | int | float]:
    if not isinstance(value, Mapping) or len(value) > _MAX_METADATA_ENTRIES:
        raise ValueError(f"metadata must be a mapping with at most {_MAX_METADATA_ENTRIES} entries")
    result: dict[str, str | int | float] = {}
    for key, item in value.items():
        validated_key = _validate_id(key, "metadata key")
        if type(item) is str:
            if len(item) > _MAX_METADATA_STRING or any(ord(character) < 32 for character in item):
                raise ValueError("metadata strings must be bounded and contain no controls")
            result[validated_key] = item
        elif type(item) is int:
            if abs(item) > _MAX_METADATA_INTEGER:
                raise ValueError("metadata integers must be bounded")
            result[validated_key] = item
        elif type(item) is float:
            if not math.isfinite(item) or abs(item) > _MAX_METADATA_FLOAT:
                raise ValueError("metadata floats must be finite and bounded")
            result[validated_key] = item
        else:
            raise TypeError("metadata values must be strings, integers, or finite floats")
    return result


def _require_directory(path: Path, field_name: str) -> None:
    info = path.lstat()
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{field_name} must be a real local directory")


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse)


def _open_regular(path: Path, field_name: str) -> tuple[os.stat_result, BinaryIO]:
    try:
        before = path.lstat()
        if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{field_name} must be a regular file, not a symlink")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            os.close(descriptor)
            raise ValueError(f"{field_name} changed while it was opened")
        return before, os.fdopen(descriptor, "rb")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{field_name} is missing or unsafe") from exc


def _sha256_regular(path: Path, field_name: str) -> tuple[str, int]:
    before, stream = _open_regular(path, field_name)
    digest = hashlib.sha256()
    size = 0
    with stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    after = path.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError(f"{field_name} changed while it was hashed")
    if size <= 0:
        raise ValueError(f"{field_name} must not be empty")
    return digest.hexdigest(), size


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    serialized = _canonical_json(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    text = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def _read_canonical_json(path: Path, field_name: str) -> object:
    _, stream = _open_regular(path, field_name)
    with stream:
        payload_bytes = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(payload_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"{field_name} exceeds the bounded manifest size")
    return _decode_canonical_json(payload_bytes, field_name)


def _decode_canonical_json(payload_bytes: bytes, field_name: str) -> object:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} is not valid canonical JSON") from exc
    if not isinstance(payload, Mapping) or _canonical_json(payload) != payload_bytes:
        raise ValueError(f"{field_name} is not valid canonical JSON")
    return payload


def _contained_parts(root: Path, relative: str, field_name: str) -> tuple[Path, tuple[Path, ...]]:
    if type(relative) is not str:
        raise ValueError(f"{field_name} must be a canonical relative path")
    parts = Path(relative).parts
    if (
        len(parts) != 3
        or parts[0] != "versions"
        or parts[2]
        not in {
            "adapter.bin",
            "manifest.json",
        }
    ):
        raise ValueError(f"{field_name} escapes the adapter publication root")
    versions = root / "versions"
    version = versions / parts[1]
    parents = (root, versions, version)
    for parent in parents:
        _require_directory(parent, f"{field_name} parent directory")
    return version / parts[2], parents


def _contained_read(
    root: Path,
    relative: str,
    field_name: str,
    *,
    retain: bool,
    limit: int | None = None,
) -> tuple[bytes, str, int]:
    candidate, parents = _contained_parts(root, relative, field_name)
    parent_stats = tuple(parent.lstat() for parent in parents)
    leaf_stat, stream = _open_regular(candidate, field_name)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    with stream:
        while True:
            read_size = 1024 * 1024
            if limit is not None:
                read_size = min(read_size, limit + 1 - size)
            chunk = stream.read(read_size)
            if not chunk:
                break
            if retain:
                chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
            if limit is not None and size > limit:
                break
    paths = (*parents, candidate)
    before_stats = (*parent_stats, leaf_stat)
    for path, before in zip(paths, before_stats, strict=True):
        after = path.lstat()
        if (
            _is_link_or_reparse(after)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
        ):
            raise ValueError(f"{field_name} parent or leaf changed while read")
    return b"".join(chunks), digest.hexdigest(), size


def _contained_bytes(root: Path, relative: str, field_name: str) -> bytes:
    payload, _, _ = _contained_read(
        root, relative, field_name, retain=True, limit=_MAX_MANIFEST_BYTES
    )
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"{field_name} exceeds the bounded manifest size")
    return payload


def _contained_hash(root: Path, relative: str, field_name: str) -> tuple[str, int]:
    _, digest, size = _contained_read(root, relative, field_name, retain=False)
    return digest, size


@contextmanager
def _descriptor_lock(descriptor: int) -> Iterator[None]:
    if os.name == "nt":
        with windows_byte_lock(descriptor):
            yield
        return

    import fcntl

    getattr(fcntl, "flock")(descriptor, getattr(fcntl, "LOCK_EX"))
    try:
        yield
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
        getattr(fcntl, "flock")(descriptor, getattr(fcntl, "LOCK_UN"))


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "AdapterCandidate",
    "AdapterManifest",
    "ArtifactHashError",
    "LocalAdapterPublisher",
]

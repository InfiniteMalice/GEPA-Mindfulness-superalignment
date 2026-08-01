"""Closed-schema JSON and JSONL records shared by RL execution backends."""

from __future__ import annotations

import importlib
import json
import math
import os
import stat
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Iterator, Literal, Protocol, cast

from gepa_mindfulness.training.trajectory import Trajectory

LOG_SCHEMA_VERSION = 1
MetricScope = Literal["aggregate", "group", "response"]


class _FcntlLike(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


_METRIC_NAMES = frozenset(
    {
        "benign_creativity",
        "entropy",
        "exploit_disclosure",
        "feedback_integrity",
        "gepa_alignment",
        "gradient_norm",
        "group_advantage_mean",
        "group_reward_mean",
        "group_reward_std",
        "group_skipped",
        "group_size",
        "hallucination",
        "honesty",
        "kl",
        "learning_rate",
        "long_horizon_agency",
        "objective_fidelity",
        "optimizer_step",
        "paraconsistent_truth",
        "policy_loss",
        "reality_contact",
        "repair_quality",
        "response_reward",
        "skill_transfer",
        "task_success",
        "total_reward",
        "value_loss",
    }
)
_RUN_MANIFEST_FIELDS = frozenset(
    {
        "actor_backend",
        "adapter",
        "algorithm",
        "backend",
        "checkpoint_parent",
        "config_hash",
        "dataset_hash",
        "device_capabilities",
        "learner_backend",
        "model",
        "reference_model",
        "run_id",
        "schema_version",
        "seed",
        "software_versions",
        "start_time",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "actor_backend",
        "backend",
        "global_step",
        "learner_backend",
        "metrics",
        "policy_version",
        "record_id",
        "run_id",
        "schema_version",
        "scope",
        "timestamp",
    }
)
_TRAJECTORY_FIELDS = frozenset(
    {
        "actor_backend",
        "backend",
        "global_step",
        "learner_backend",
        "policy_version",
        "record_id",
        "run_id",
        "schema_version",
        "timestamp",
        "trajectory",
    }
)
_TRAJECTORY_PAYLOAD_REQUIRED_FIELDS = frozenset(
    {
        "adapter_identifier",
        "advantage",
        "backend_name",
        "backend_version",
        "case_id",
        "model_identifier",
        "old_log_probs",
        "policy_version",
        "prompt",
        "prompt_token_ids",
        "reference_log_probs",
        "response",
        "response_token_ids",
        "return",
        "reward_component_evidence",
        "reward_components",
        "reward_total",
        "sampling_parameters",
        "seed",
        "trace_references",
        "trajectory_id",
        "value_predictions",
    }
)
_TRAJECTORY_PAYLOAD_OPTIONAL_FIELDS = frozenset({"evidence_references"})


@dataclass(frozen=True)
class RunManifest:
    """One immutable run identity and compatibility record."""

    run_id: str
    algorithm: str
    backend: str
    actor_backend: str
    learner_backend: str
    model: str
    reference_model: str | None
    adapter: str | None
    dataset_hash: str
    config_hash: str
    seed: int
    software_versions: Mapping[str, object]
    device_capabilities: Mapping[str, object]
    start_time: str
    checkpoint_parent: str | None
    schema_version: int = LOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "algorithm",
            "backend",
            "actor_backend",
            "learner_backend",
            "model",
            "start_time",
        ):
            _required_string(getattr(self, field_name), field_name)
        _optional_string(self.reference_model, "reference_model")
        _optional_string(self.adapter, "adapter")
        _optional_string(self.checkpoint_parent, "checkpoint_parent")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_sha256(self.config_hash, "config_hash")
        _non_negative_integer(self.seed, "seed")
        if self.schema_version != LOG_SCHEMA_VERSION:
            raise ValueError(f"run manifest schema_version must be {LOG_SCHEMA_VERSION}")
        versions = _json_mapping(self.software_versions, "software_versions")
        capabilities = _json_mapping(self.device_capabilities, "device_capabilities")
        object.__setattr__(self, "software_versions", MappingProxyType(versions))
        object.__setattr__(self, "device_capabilities", MappingProxyType(capabilities))

    def to_dict(self) -> dict[str, object]:
        """Return the exact run-manifest JSON schema."""
        return {
            "actor_backend": self.actor_backend,
            "adapter": self.adapter,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "checkpoint_parent": self.checkpoint_parent,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "device_capabilities": dict(self.device_capabilities),
            "learner_backend": self.learner_backend,
            "model": self.model,
            "reference_model": self.reference_model,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "software_versions": dict(self.software_versions),
            "start_time": self.start_time,
        }

    @classmethod
    def from_mapping(cls, payload: object) -> "RunManifest":
        if not isinstance(payload, Mapping) or set(payload) != _RUN_MANIFEST_FIELDS:
            raise ValueError("run manifest fields are missing or unrecognized")
        return cls(**dict(payload))


@dataclass(frozen=True)
class MetricRecord:
    """One aggregate, prompt-group, or response-level metric record."""

    record_id: str
    run_id: str
    timestamp: str
    global_step: int
    scope: MetricScope
    backend: str
    actor_backend: str
    learner_backend: str
    policy_version: str
    metrics: Mapping[str, float]
    schema_version: int = LOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "run_id",
            "timestamp",
            "backend",
            "actor_backend",
            "learner_backend",
            "policy_version",
        ):
            _required_string(getattr(self, field_name), field_name)
        _non_negative_integer(self.global_step, "global_step")
        if self.scope not in {"aggregate", "group", "response"}:
            raise ValueError("metric scope must be 'aggregate', 'group', or 'response'")
        if self.schema_version != LOG_SCHEMA_VERSION:
            raise ValueError(f"metric schema_version must be {LOG_SCHEMA_VERSION}")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            _required_string(name, "metric name")
            if name not in _METRIC_NAMES:
                raise ValueError(f"metric name {name!r} is not in schema version 1")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("metric values must be finite JSON numbers")
            metrics[name] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_backend": self.actor_backend,
            "backend": self.backend,
            "global_step": self.global_step,
            "learner_backend": self.learner_backend,
            "metrics": dict(self.metrics),
            "policy_version": self.policy_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "scope": self.scope,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_mapping(cls, payload: object) -> "MetricRecord":
        if not isinstance(payload, Mapping) or set(payload) != _METRIC_FIELDS:
            raise ValueError("metric record fields are missing or unrecognized")
        return cls(**dict(payload))


@dataclass(frozen=True)
class TrajectoryRecord:
    """One trajectory plus run, backend, learner, and policy identity."""

    record_id: str
    run_id: str
    timestamp: str
    global_step: int
    backend: str
    actor_backend: str
    learner_backend: str
    policy_version: str
    trajectory: Trajectory
    schema_version: int = LOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "run_id",
            "timestamp",
            "backend",
            "actor_backend",
            "learner_backend",
            "policy_version",
        ):
            _required_string(getattr(self, field_name), field_name)
        _non_negative_integer(self.global_step, "global_step")
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("trajectory must be a Trajectory")
        if self.trajectory.backend_name and self.trajectory.backend_name != self.backend:
            raise ValueError("trajectory backend_name must match the record backend")
        if (
            self.trajectory.policy_version is not None
            and self.trajectory.policy_version != self.policy_version
        ):
            raise ValueError("trajectory policy_version must match the record policy_version")
        if self.schema_version != LOG_SCHEMA_VERSION:
            raise ValueError(f"trajectory schema_version must be {LOG_SCHEMA_VERSION}")
        _validate_json_value(self.trajectory.to_dict(), "trajectory")

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_backend": self.actor_backend,
            "backend": self.backend,
            "global_step": self.global_step,
            "learner_backend": self.learner_backend,
            "policy_version": self.policy_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "trajectory": self.trajectory.to_dict(),
        }

    @classmethod
    def from_mapping(cls, payload: object) -> "TrajectoryRecord":
        if not isinstance(payload, Mapping) or set(payload) != _TRAJECTORY_FIELDS:
            raise ValueError("trajectory record fields are missing or unrecognized")
        values = dict(payload)
        trajectory_payload = values["trajectory"]
        if not isinstance(trajectory_payload, Mapping) or (
            not _TRAJECTORY_PAYLOAD_REQUIRED_FIELDS.issubset(trajectory_payload)
            or set(trajectory_payload)
            - _TRAJECTORY_PAYLOAD_REQUIRED_FIELDS
            - _TRAJECTORY_PAYLOAD_OPTIONAL_FIELDS
        ):
            raise ValueError("trajectory fields are missing or unrecognized")
        values["trajectory"] = Trajectory.from_dict(trajectory_payload)
        return cls(**values)


class JSONLLoggingSink:
    """Append validated run records while suppressing duplicate record IDs."""

    _locks_guard = threading.Lock()
    _path_locks: dict[Path, threading.RLock] = {}

    def __init__(self, directory: Path, *, rank: int = 0) -> None:
        if not isinstance(directory, Path):
            raise TypeError("logging directory must be a pathlib.Path")
        _non_negative_integer(rank, "rank")
        self.directory = directory.resolve(strict=False)
        self.rank = rank

    def start_run(self, manifest: RunManifest | Mapping[str, object]) -> bool:
        """Transactionally initialize one rank-zero run under a stream lock."""
        parsed = (
            manifest if isinstance(manifest, RunManifest) else RunManifest.from_mapping(manifest)
        )
        if self.rank != 0:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "run_manifest.json"
        metrics_path = self.directory / "metrics.jsonl"
        trajectories_path = self.directory / "trajectories.jsonl"
        with self._path_lock(metrics_path):
            with _open_regular_stream(metrics_path, create=True) as metrics:
                with _exclusive_stream_lock(metrics):
                    with self._path_lock(trajectories_path):
                        with _open_regular_stream(trajectories_path, create=True) as trajectories:
                            with _exclusive_stream_lock(trajectories):
                                return self._start_locked(
                                    parsed,
                                    manifest_path,
                                    metrics,
                                    trajectories,
                                )

    def log_metrics(self, record: MetricRecord | Mapping[str, object]) -> bool:
        """Append one metric record; only rank zero writes aggregate metrics."""
        parsed = record if isinstance(record, MetricRecord) else MetricRecord.from_mapping(record)
        if parsed.scope == "aggregate" and self.rank != 0:
            return False
        manifest = self._require_run(parsed)
        return self._append_unique(
            self.directory / "metrics.jsonl",
            parsed.to_dict(),
            manifest,
        )

    def log_trajectory(self, record: TrajectoryRecord | Mapping[str, object]) -> bool:
        """Append one evidence-preserving trajectory record from any rank."""
        parsed = (
            record
            if isinstance(record, TrajectoryRecord)
            else TrajectoryRecord.from_mapping(record)
        )
        manifest = self._require_run(parsed)
        return self._append_unique(
            self.directory / "trajectories.jsonl",
            parsed.to_dict(),
            manifest,
        )

    def _require_run(self, record: MetricRecord | TrajectoryRecord) -> RunManifest:
        manifest_path = self.directory / "run_manifest.json"
        manifest = RunManifest.from_mapping(self._read_json(manifest_path, "run manifest"))
        if manifest.run_id != record.run_id:
            raise ValueError("log record run_id does not match the run manifest")
        self._validate_record_provenance(record, manifest)
        return manifest

    def _append_unique(
        self,
        path: Path,
        payload: Mapping[str, object],
        manifest: RunManifest,
    ) -> bool:
        _validate_json_value(payload, "record")
        record_id = payload.get("record_id")
        _required_string(record_id, "record_id")
        serialized = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        with self._path_lock(path):
            with _open_regular_stream(path, create=False) as stream, _exclusive_stream_lock(stream):
                existing_records = self._existing_records(stream, path, manifest)
                if record_id in existing_records:
                    if existing_records[record_id] == payload:
                        return False
                    raise ValueError("record_id already identifies a different payload")
                stream.seek(0, os.SEEK_END)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        return True

    @classmethod
    def _existing_records(
        cls,
        stream: BinaryIO,
        path: Path,
        manifest: RunManifest,
    ) -> dict[str, object]:
        records: dict[str, object] = {}
        try:
            stream.seek(0)
            content = stream.read()
            if content and not content.endswith(b"\n"):
                raise ValueError("nonempty JSONL stream must end with a final newline")
            lines = content.decode("utf-8").splitlines()
        except ValueError:
            raise
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"JSONL stream is unreadable: {path}") from error
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL stream has an invalid record at line {line_number}"
                ) from error
            if path.name == "metrics.jsonl":
                record: MetricRecord | TrajectoryRecord = MetricRecord.from_mapping(payload)
            elif path.name == "trajectories.jsonl":
                record = TrajectoryRecord.from_mapping(payload)
            else:
                raise ValueError(f"unrecognized JSONL stream: {path}")
            cls._validate_record_provenance(record, manifest)
            validated_payload = record.to_dict()
            previous = records.get(record.record_id)
            if previous is not None and previous != validated_payload:
                raise ValueError("record_id already identifies a different payload")
            records[record.record_id] = validated_payload
        return records

    @staticmethod
    def _validate_record_provenance(
        record: MetricRecord | TrajectoryRecord,
        manifest: RunManifest,
    ) -> None:
        if record.run_id != manifest.run_id:
            raise ValueError("log record run_id does not match the run manifest")
        if any(
            getattr(manifest, field) != getattr(record, field)
            for field in ("backend", "actor_backend", "learner_backend")
        ):
            raise ValueError("log record backend provenance does not match the run manifest")

    @classmethod
    def _path_lock(cls, path: Path) -> threading.RLock:
        resolved = path.resolve(strict=False)
        with cls._locks_guard:
            return cls._path_locks.setdefault(resolved, threading.RLock())

    @staticmethod
    def _write_atomic_file(path: Path, payload: Mapping[str, object]) -> None:
        serialized = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _read_json(path: Path, field_name: str) -> object:
        try:
            with _open_regular_stream(path, create=False) as stream:
                stream.seek(0)
                return json.loads(stream.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{field_name} is missing, unreadable, or invalid JSON") from error

    def _start_locked(
        self,
        manifest: RunManifest,
        manifest_path: Path,
        metrics: BinaryIO,
        trajectories: BinaryIO,
    ) -> bool:
        existing = self._optional_manifest(manifest_path)
        if existing is not None:
            if existing.to_dict() != manifest.to_dict():
                raise ValueError("logging directory already contains a different run manifest")
            self._validate_existing_stream(metrics, self.directory / "metrics.jsonl", existing)
            self._validate_existing_stream(
                trajectories,
                self.directory / "trajectories.jsonl",
                existing,
            )
            return False
        for stream, name in (
            (metrics, "metrics.jsonl"),
            (trajectories, "trajectories.jsonl"),
        ):
            stream.seek(0, os.SEEK_END)
            if stream.tell() != 0:
                raise ValueError(f"nonempty stale {name} exists before the run manifest")
        temporary = self.directory / f".run_manifest.json.tmp-{uuid.uuid4().hex}"
        try:
            self._write_atomic_file(temporary, manifest.to_dict())
            try:
                os.link(temporary, manifest_path)
            except FileExistsError:
                winner = RunManifest.from_mapping(self._read_json(manifest_path, "run manifest"))
                if winner.to_dict() != manifest.to_dict():
                    raise ValueError("logging directory already contains a different run manifest")
                return False
            return True
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _optional_manifest(self, path: Path) -> RunManifest | None:
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        return RunManifest.from_mapping(self._read_json(path, "run manifest"))

    def _validate_existing_stream(
        self,
        stream: BinaryIO,
        path: Path,
        manifest: RunManifest,
    ) -> None:
        self._existing_records(stream, path, manifest)


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _json_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a JSON object with string keys")
    normalized = dict(value)
    _validate_json_value(normalized, field_name)
    return normalized


def _validate_json_value(value: object, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} JSON object keys must be strings")
        for key, item in value.items():
            _validate_json_value(item, f"{field_name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item, f"{field_name}[]")
        return
    raise TypeError(f"{field_name} contains a value that is not JSON-serializable")


@contextmanager
def _open_regular_stream(path: Path, *, create: bool) -> Iterator[BinaryIO]:
    """Open one stable regular file while rejecting symlinks before and after open."""
    before: os.stat_result | None
    try:
        before = path.lstat()
    except FileNotFoundError:
        before = None
    if before is not None and stat.S_ISLNK(before.st_mode):
        raise ValueError(f"logging path must not be a symlink: {path}")
    if before is not None and not stat.S_ISREG(before.st_mode):
        raise ValueError(f"logging path must be a regular file: {path}")
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError(f"logging path is missing, unreadable, or unsafe: {path}") from error
    try:
        after_path = path.lstat()
        opened = os.fstat(descriptor)
        if stat.S_ISLNK(after_path.st_mode):
            raise ValueError(f"logging path must not be a symlink: {path}")
        if not stat.S_ISREG(opened.st_mode) or (
            after_path.st_dev,
            after_path.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"logging path changed while it was opened: {path}")
        if before is not None and (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError(f"logging path changed while it was opened: {path}")
        with os.fdopen(descriptor, "r+b", closefd=False) as stream:
            yield stream
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_stream_lock(stream: BinaryIO) -> Iterator[None]:
    """Hold one cross-process exclusive lock without creating extra run artifacts."""
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = cast(_FcntlLike, importlib.import_module("fcntl"))

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "JSONLLoggingSink",
    "LOG_SCHEMA_VERSION",
    "MetricRecord",
    "RunManifest",
    "TrajectoryRecord",
]

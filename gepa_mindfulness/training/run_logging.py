"""Closed-schema JSON and JSONL records shared by RL execution backends."""

from __future__ import annotations

import importlib
import json
import math
import os
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
        "group_size",
        "hallucination",
        "honesty",
        "kl",
        "learning_rate",
        "long_horizon_agency",
        "objective_fidelity",
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
        values["trajectory"] = Trajectory.from_dict(values["trajectory"])
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
        """Create one rank-zero manifest and the two shared JSONL streams."""
        parsed = (
            manifest if isinstance(manifest, RunManifest) else RunManifest.from_mapping(manifest)
        )
        if self.rank != 0:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "run_manifest.json"
        with self._path_lock(manifest_path):
            if manifest_path.exists():
                existing = self._read_json(manifest_path, "run manifest")
                if existing == parsed.to_dict():
                    return False
                raise ValueError("logging directory already contains a different run manifest")
            temporary = self.directory / f".run_manifest.tmp-{uuid.uuid4().hex}"
            try:
                self._write_atomic_file(temporary, parsed.to_dict())
                temporary.replace(manifest_path)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise
            for name in ("metrics.jsonl", "trajectories.jsonl"):
                (self.directory / name).touch(exist_ok=True)
        return True

    def log_metrics(self, record: MetricRecord | Mapping[str, object]) -> bool:
        """Append one metric record; only rank zero writes aggregate metrics."""
        parsed = record if isinstance(record, MetricRecord) else MetricRecord.from_mapping(record)
        if parsed.scope == "aggregate" and self.rank != 0:
            return False
        self._require_run(parsed.run_id)
        return self._append_unique(self.directory / "metrics.jsonl", parsed.to_dict())

    def log_trajectory(self, record: TrajectoryRecord | Mapping[str, object]) -> bool:
        """Append one evidence-preserving trajectory record from any rank."""
        parsed = (
            record
            if isinstance(record, TrajectoryRecord)
            else TrajectoryRecord.from_mapping(record)
        )
        self._require_run(parsed.run_id)
        return self._append_unique(self.directory / "trajectories.jsonl", parsed.to_dict())

    def _require_run(self, run_id: str) -> None:
        manifest_path = self.directory / "run_manifest.json"
        manifest = RunManifest.from_mapping(self._read_json(manifest_path, "run manifest"))
        if manifest.run_id != run_id:
            raise ValueError("log record run_id does not match the run manifest")

    def _append_unique(self, path: Path, payload: Mapping[str, object]) -> bool:
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
            with path.open("a+b") as stream, _exclusive_stream_lock(stream):
                existing_records = self._existing_records(stream, path)
                if record_id in existing_records:
                    if existing_records[record_id] == payload:
                        return False
                    raise ValueError("record_id already identifies a different payload")
                stream.seek(0, os.SEEK_END)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        return True

    @staticmethod
    def _existing_records(stream: BinaryIO, path: Path) -> dict[str, object]:
        records: dict[str, object] = {}
        try:
            stream.seek(0)
            lines = stream.read().decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"JSONL stream is unreadable: {path}") from error
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL stream has an invalid record at line {line_number}"
                ) from error
            if not isinstance(payload, Mapping) or not isinstance(payload.get("record_id"), str):
                raise ValueError(f"JSONL stream has an invalid record at line {line_number}")
            records[payload["record_id"]] = payload
        return records

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
        with path.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _read_json(path: Path, field_name: str) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{field_name} is missing, unreadable, or invalid JSON") from error


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

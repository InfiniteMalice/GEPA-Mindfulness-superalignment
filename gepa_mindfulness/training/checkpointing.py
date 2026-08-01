"""Atomic local checkpoint publication above backend-owned payload formats."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import torch

from gepa_mindfulness.training.backends.base import BackendCheckpointResult

CHECKPOINT_SCHEMA_VERSION = 3
_ARTIFACT_NAMES = frozenset({"backend.pt", "training_state.pt"})
_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MANIFEST_FIELDS = frozenset(
    {
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
)
_STATE_FIELDS = frozenset(
    {
        "algorithm_state",
        "batch_cursor",
        "backend_format_version",
        "backend_step",
        "canonical_config",
        "config_hash",
        "dataset_hash",
        "global_step",
        "parent_checkpoint",
        "python_rng_state",
        "rank_rng_states",
        "rollout_cursor",
        "scheduler_state",
        "schema_version",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
)

BackendSave = Callable[[Path], BackendCheckpointResult]
BackendBytes = Callable[[bytes], BackendCheckpointResult]
BackendSnapshot = Callable[[], object]
BackendRollback = Callable[[object], None]


@dataclass(frozen=True)
class CheckpointRNGTopology:
    """Closed CPU/CUDA RNG-state shape selected for one checkpoint process."""

    device_type: Literal["cpu", "cuda"]
    cpu_state_length: int
    cuda_state_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("checkpoint RNG device_type must be 'cpu' or 'cuda'")
        _validate_positive_integer(self.cpu_state_length, "cpu_state_length")
        if not isinstance(self.cuda_state_lengths, tuple):
            raise TypeError("cuda_state_lengths must be a tuple")
        for index, length in enumerate(self.cuda_state_lengths):
            _validate_positive_integer(length, f"cuda_state_lengths[{index}]")
        if self.device_type == "cpu" and self.cuda_state_lengths:
            raise ValueError("CPU checkpoint topology cannot include CUDA RNG states")
        if self.device_type == "cuda" and not self.cuda_state_lengths:
            raise ValueError("CUDA checkpoint topology requires at least one CUDA RNG state")

    @classmethod
    def current(cls, *, device_type: Literal["cpu", "cuda"] = "cpu") -> "CheckpointRNGTopology":
        """Capture exact RNG lengths for the selected current process topology."""
        cpu_length = len(torch.get_rng_state())
        if device_type == "cpu":
            return cls(device_type="cpu", cpu_state_length=cpu_length)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG topology requires an available CUDA runtime")
        lengths = tuple(
            len(torch.cuda.get_rng_state(index)) for index in range(torch.cuda.device_count())
        )
        return cls(
            device_type="cuda",
            cpu_state_length=cpu_length,
            cuda_state_lengths=lengths,
        )


@dataclass(frozen=True)
class RankRNGState:
    """One rank's exact Python, Torch CPU, and visible CUDA RNG state."""

    python_rng_state: object
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        _validate_python_rng_state(self.python_rng_state)
        cpu_state = _validate_rng_tensor(self.torch_cpu_rng_state, "torch_cpu_rng_state")
        if not isinstance(self.torch_cuda_rng_states, (list, tuple)):
            raise TypeError("torch_cuda_rng_states must be a sequence of byte tensors")
        cuda_states = tuple(
            _validate_rng_tensor(value, f"torch_cuda_rng_states[{index}]")
            for index, value in enumerate(self.torch_cuda_rng_states)
        )
        object.__setattr__(self, "torch_cpu_rng_state", cpu_state)
        object.__setattr__(self, "torch_cuda_rng_states", cuda_states)


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Engine-owned state saved beside one opaque backend checkpoint payload."""

    global_step: int
    algorithm_state: Mapping[str, object]
    scheduler_state: Mapping[str, object] | None
    python_rng_state: object
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]
    canonical_config: Mapping[str, object]
    dataset_hash: str
    config_hash: str
    rank_rng_states: Mapping[int, RankRNGState] | None = None
    batch_cursor: int = 0
    rollout_cursor: int = 0
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.global_step, "global_step")
        _validate_non_negative_integer(self.batch_cursor, "batch_cursor")
        _validate_non_negative_integer(self.rollout_cursor, "rollout_cursor")
        algorithm_state = _state_mapping(self.algorithm_state, "algorithm_state")
        scheduler_state = (
            None
            if self.scheduler_state is None
            else _state_mapping(self.scheduler_state, "scheduler_state")
        )
        canonical_config = _state_mapping(self.canonical_config, "canonical_config")
        _validate_python_rng_state(self.python_rng_state)
        cpu_rng = _validate_rng_tensor(self.torch_cpu_rng_state, "torch_cpu_rng_state")
        if not isinstance(self.torch_cuda_rng_states, (list, tuple)):
            raise TypeError("torch_cuda_rng_states must be a sequence of byte tensors")
        cuda_rng = tuple(
            _validate_rng_tensor(state, f"torch_cuda_rng_states[{index}]")
            for index, state in enumerate(self.torch_cuda_rng_states)
        )
        rank_rng_states = (
            None
            if self.rank_rng_states is None
            else MappingProxyType(_validated_rank_rng_states(self.rank_rng_states))
        )
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_sha256(self.config_hash, "config_hash")
        _validate_optional_string(self.parent_checkpoint, "parent_checkpoint")
        object.__setattr__(self, "algorithm_state", MappingProxyType(algorithm_state))
        object.__setattr__(
            self,
            "scheduler_state",
            None if scheduler_state is None else MappingProxyType(scheduler_state),
        )
        object.__setattr__(self, "canonical_config", MappingProxyType(canonical_config))
        object.__setattr__(self, "torch_cpu_rng_state", cpu_rng)
        object.__setattr__(self, "torch_cuda_rng_states", cuda_rng)
        object.__setattr__(self, "rank_rng_states", rank_rng_states)


@dataclass(frozen=True)
class CheckpointManifest:
    """Verified checkpoint compatibility metadata and artifact digests."""

    path: Path
    checkpoint_id: str
    global_step: int
    backend_format_version: int
    backend_step: int
    dataset_hash: str
    config_hash: str
    parent_checkpoint: str | None
    artifact_hashes: Mapping[str, str]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("checkpoint manifest path must be a pathlib.Path")
        _validate_checkpoint_id(self.checkpoint_id)
        _validate_non_negative_integer(self.global_step, "global_step")
        _validate_positive_integer(self.backend_format_version, "backend_format_version")
        _validate_non_negative_integer(self.backend_step, "backend_step")
        if self.backend_step != self.global_step:
            raise ValueError("checkpoint backend step must match global_step")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_sha256(self.config_hash, "config_hash")
        _validate_optional_string(self.parent_checkpoint, "parent_checkpoint")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"checkpoint schema_version must be {CHECKPOINT_SCHEMA_VERSION}")
        if not isinstance(self.artifact_hashes, Mapping):
            raise TypeError("artifact_hashes must be a mapping")
        hashes = dict(self.artifact_hashes)
        if set(hashes) != _ARTIFACT_NAMES:
            raise ValueError("checkpoint artifact_hashes must name the exact checkpoint artifacts")
        for name, digest in hashes.items():
            if not isinstance(name, str):
                raise TypeError("checkpoint artifact names must be strings")
            _validate_sha256(digest, f"artifact_hashes.{name}")
        object.__setattr__(self, "artifact_hashes", MappingProxyType(hashes))

    def to_dict(self) -> dict[str, object]:
        """Return the closed serialized manifest schema."""
        return {
            "artifact_hashes": dict(self.artifact_hashes),
            "backend_format_version": self.backend_format_version,
            "backend_step": self.backend_step,
            "checkpoint_id": self.checkpoint_id,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "global_step": self.global_step,
            "parent_checkpoint": self.parent_checkpoint,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, path: Path, payload: object) -> "CheckpointManifest":
        """Validate one closed manifest object without consulting artifact contents."""
        if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
            raise ValueError("checkpoint manifest fields are missing or unrecognized")
        return cls(
            path=path,
            checkpoint_id=payload["checkpoint_id"],
            global_step=payload["global_step"],
            backend_format_version=payload["backend_format_version"],
            backend_step=payload["backend_step"],
            dataset_hash=payload["dataset_hash"],
            config_hash=payload["config_hash"],
            parent_checkpoint=payload["parent_checkpoint"],
            artifact_hashes=payload["artifact_hashes"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Validated engine state and evidence from an operator-selected checkpoint."""

    manifest: CheckpointManifest
    algorithm_state: Mapping[str, object]
    scheduler_state: Mapping[str, object] | None
    python_rng_state: object
    torch_cpu_rng_state: torch.Tensor
    torch_cuda_rng_states: tuple[torch.Tensor, ...]
    canonical_config: Mapping[str, object]
    batch_cursor: int
    rollout_cursor: int
    backend_result: BackendCheckpointResult | None = None

    @property
    def path(self) -> Path:
        return self.manifest.path

    @property
    def global_step(self) -> int:
        return self.manifest.global_step

    @property
    def artifact_hashes(self) -> Mapping[str, str]:
        return self.manifest.artifact_hashes

    @property
    def dataset_hash(self) -> str:
        return self.manifest.dataset_hash

    @property
    def config_hash(self) -> str:
        return self.manifest.config_hash

    @property
    def parent_checkpoint(self) -> str | None:
        return self.manifest.parent_checkpoint


class LocalCheckpointStore:
    """Publish and validate local checkpoint directories without owning backend state."""

    def __init__(
        self,
        root: Path,
        *,
        backend_save: BackendSave | None = None,
        backend_preflight: BackendBytes | None = None,
        backend_load_bytes: BackendBytes | None = None,
        backend_snapshot: BackendSnapshot | None = None,
        backend_rollback: BackendRollback | None = None,
        rng_topology: CheckpointRNGTopology | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("checkpoint root must be a pathlib.Path")
        if backend_save is not None and not callable(backend_save):
            raise TypeError("backend_save must be callable")
        if backend_preflight is not None and not callable(backend_preflight):
            raise TypeError("backend_preflight must be callable")
        if backend_load_bytes is not None and not callable(backend_load_bytes):
            raise TypeError("backend_load_bytes must be callable")
        if backend_snapshot is not None and not callable(backend_snapshot):
            raise TypeError("backend_snapshot must be callable")
        if backend_rollback is not None and not callable(backend_rollback):
            raise TypeError("backend_rollback must be callable")
        if (backend_snapshot is None) != (backend_rollback is None):
            raise ValueError("backend_snapshot and backend_rollback must be configured together")
        if rng_topology is not None and not isinstance(rng_topology, CheckpointRNGTopology):
            raise TypeError("rng_topology must be a CheckpointRNGTopology")
        _validate_non_negative_integer(rank, "rank")
        _validate_positive_integer(world_size, "world_size")
        if rank >= world_size:
            raise ValueError("rank must be within world_size")
        self.root = root.resolve(strict=False)
        self.rank = rank
        self.world_size = world_size
        self.backend_save = backend_save
        self.backend_preflight = backend_preflight
        self.backend_load_bytes = backend_load_bytes
        self.backend_snapshot = backend_snapshot
        self.backend_rollback = backend_rollback
        self.rng_topology = rng_topology or CheckpointRNGTopology.current()

    def save(
        self,
        snapshot: CheckpointSnapshot,
        *,
        checkpoint_id: str | None = None,
    ) -> CheckpointManifest | None:
        """Verify and atomically publish one new sibling checkpoint directory."""
        if not isinstance(snapshot, CheckpointSnapshot):
            raise TypeError("snapshot must be a CheckpointSnapshot")
        if self.rank != 0:
            return None
        if self.backend_save is None:
            raise RuntimeError("backend_save is required to create a checkpoint")
        rank_rng_states = self._snapshot_rank_rng_states(snapshot)
        if set(rank_rng_states) != set(range(self.world_size)):
            raise ValueError("checkpoint rank_rng_states must exactly cover world_size")
        for rank, state in rank_rng_states.items():
            try:
                self._validate_rng_topology(
                    state.torch_cpu_rng_state,
                    state.torch_cuda_rng_states,
                )
            except ValueError as error:
                raise ValueError(
                    f"checkpoint rank {rank} has invalid RNG state: {error}"
                ) from error
        self._validate_rng_topology(
            snapshot.torch_cpu_rng_state,
            snapshot.torch_cuda_rng_states,
        )
        selected_id = checkpoint_id or f"checkpoint-{snapshot.global_step:08d}"
        _validate_checkpoint_id(selected_id)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / selected_id
        if destination.exists():
            raise FileExistsError(f"checkpoint destination already exists: {destination}")
        temporary = self.root / f".{selected_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            backend_path = temporary / "backend.pt"
            backend_result = _validate_backend_result(
                self.backend_save(backend_path),
                expected_step=snapshot.global_step,
                field_name="backend save",
            )
            backend_payload = self._read_regular_bytes(backend_path, "backend.pt")
            state_path = temporary / "training_state.pt"
            state_buffer = io.BytesIO()
            torch.save(self._snapshot_payload(snapshot, backend_result), state_buffer)
            state_payload = state_buffer.getvalue()
            self._write_bytes(state_path, state_payload)
            hashes = {
                "backend.pt": self.sha256_bytes(backend_payload),
                "training_state.pt": self.sha256_bytes(state_payload),
            }
            manifest = CheckpointManifest(
                path=destination,
                checkpoint_id=selected_id,
                global_step=snapshot.global_step,
                backend_format_version=backend_result.format_version,
                backend_step=backend_result.step,
                dataset_hash=snapshot.dataset_hash,
                config_hash=snapshot.config_hash,
                parent_checkpoint=snapshot.parent_checkpoint,
                artifact_hashes=hashes,
            )
            self._write_json(temporary / "manifest.json", manifest.to_dict())
            self._load_validated(
                temporary,
                expected_checkpoint_id=selected_id,
                expected_dataset_hash=None,
                expected_config_hash=None,
            )
            temporary.replace(destination)
            return CheckpointManifest.from_dict(destination, manifest.to_dict())
        except Exception as error:
            try:
                self._remove_temporary(temporary)
            except Exception:
                raise RuntimeError(
                    "checkpoint save failed and temporary cleanup also failed"
                ) from error
            raise

    def load(
        self,
        source: Path,
        *,
        expected_dataset_hash: str | None = None,
        expected_config_hash: str | None = None,
    ) -> RestoredCheckpoint:
        """Validate every artifact before invoking the backend restoration callback."""
        checkpoint = self._operator_selected_path(source)
        restored, artifacts = self._load_validated(
            checkpoint,
            expected_checkpoint_id=checkpoint.name,
            expected_dataset_hash=expected_dataset_hash,
            expected_config_hash=expected_config_hash,
        )
        if (
            self.backend_preflight is None
            or self.backend_load_bytes is None
            or self.backend_snapshot is None
            or self.backend_rollback is None
        ):
            raise RuntimeError(
                "backend preflight, load, snapshot, and rollback callbacks are required to restore"
            )
        backend_payload = artifacts["backend.pt"]
        _validate_backend_result(
            self.backend_preflight(backend_payload),
            expected_step=restored.manifest.backend_step,
            expected_format=restored.manifest.backend_format_version,
            field_name="backend preflight",
        )
        transaction_snapshot = self.backend_snapshot()
        try:
            backend_result = _validate_backend_result(
                self.backend_load_bytes(backend_payload),
                expected_step=restored.manifest.backend_step,
                expected_format=restored.manifest.backend_format_version,
                field_name="backend restore",
            )
        except Exception as error:
            try:
                self.backend_rollback(transaction_snapshot)
            except Exception as rollback_error:
                details = f"{type(rollback_error).__name__}: {rollback_error}"
                raise RuntimeError(
                    f"backend restore failed and rollback also failed: {details}"
                ) from error
            raise
        return RestoredCheckpoint(
            manifest=restored.manifest,
            algorithm_state=restored.algorithm_state,
            scheduler_state=restored.scheduler_state,
            python_rng_state=restored.python_rng_state,
            torch_cpu_rng_state=restored.torch_cpu_rng_state,
            torch_cuda_rng_states=restored.torch_cuda_rng_states,
            canonical_config=restored.canonical_config,
            batch_cursor=restored.batch_cursor,
            rollout_cursor=restored.rollout_cursor,
            backend_result=backend_result,
        )

    @staticmethod
    def sha256(path: Path) -> str:
        """Return the lowercase SHA-256 digest of one regular artifact."""
        return LocalCheckpointStore.sha256_bytes(
            LocalCheckpointStore._read_regular_bytes(path, path.name)
        )

    @staticmethod
    def sha256_bytes(payload: bytes) -> str:
        """Return the lowercase SHA-256 digest of an immutable artifact payload."""
        return hashlib.sha256(payload).hexdigest()

    def _operator_selected_path(self, source: Path) -> Path:
        if not isinstance(source, Path):
            raise TypeError("checkpoint source must be a pathlib.Path")
        try:
            source_status = source.lstat()
        except OSError as error:
            raise ValueError("checkpoint source must be an existing regular directory") from error
        if stat.S_ISLNK(source_status.st_mode):
            raise ValueError("checkpoint source must not be a symlink")
        resolved = source.resolve(strict=False)
        if resolved.parent != self.root:
            raise ValueError("checkpoint source must be a direct child of the checkpoint root")
        if not stat.S_ISDIR(source_status.st_mode):
            raise ValueError("checkpoint source must be an existing regular directory")
        try:
            resolved_status = resolved.lstat()
        except OSError as error:
            raise ValueError("checkpoint source changed while it was resolved") from error
        if (source_status.st_dev, source_status.st_ino) != (
            resolved_status.st_dev,
            resolved_status.st_ino,
        ):
            raise ValueError("checkpoint source changed while it was resolved")
        return resolved

    def _load_validated(
        self,
        checkpoint: Path,
        *,
        expected_checkpoint_id: str,
        expected_dataset_hash: str | None,
        expected_config_hash: str | None,
    ) -> tuple[RestoredCheckpoint, Mapping[str, bytes]]:
        try:
            directory_fields = {item.name for item in checkpoint.iterdir()}
        except OSError as error:
            raise ValueError("checkpoint directory is unreadable") from error
        expected_fields = {*_ARTIFACT_NAMES, "manifest.json"}
        if directory_fields != expected_fields:
            raise ValueError("checkpoint directory fields are missing or unrecognized")
        try:
            artifact_payloads = {
                name: self._read_regular_bytes(checkpoint / name, name)
                for name in (*sorted(_ARTIFACT_NAMES), "manifest.json")
            }
            payload = json.loads(artifact_payloads["manifest.json"].decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint manifest is unreadable or invalid JSON") from error
        manifest = CheckpointManifest.from_dict(checkpoint, payload)
        if manifest.checkpoint_id != expected_checkpoint_id:
            raise ValueError("checkpoint manifest checkpoint_id does not match its directory")
        self._validate_expected_hash(
            manifest.dataset_hash,
            expected_dataset_hash,
            "dataset_hash",
        )
        self._validate_expected_hash(
            manifest.config_hash,
            expected_config_hash,
            "config_hash",
        )
        for name, expected_digest in manifest.artifact_hashes.items():
            if self.sha256_bytes(artifact_payloads[name]) != expected_digest:
                raise ValueError(f"checkpoint artifact {name} failed SHA-256 verification")
        state = self._load_state(artifact_payloads["training_state.pt"])
        self._validate_state_matches_manifest(state, manifest)
        rank_rng_states = _restored_rank_rng_states(state["rank_rng_states"])
        if set(rank_rng_states) != set(range(self.world_size)):
            raise ValueError("checkpoint rank_rng_states do not match world_size")
        selected_rng = rank_rng_states[self.rank]
        self._validate_rng_topology(
            selected_rng.torch_cpu_rng_state,
            selected_rng.torch_cuda_rng_states,
        )
        restored = RestoredCheckpoint(
            manifest=manifest,
            algorithm_state=MappingProxyType(
                _state_mapping(state["algorithm_state"], "algorithm_state")
            ),
            scheduler_state=self._restored_scheduler_state(state["scheduler_state"]),
            python_rng_state=selected_rng.python_rng_state,
            torch_cpu_rng_state=selected_rng.torch_cpu_rng_state,
            torch_cuda_rng_states=selected_rng.torch_cuda_rng_states,
            canonical_config=MappingProxyType(
                _state_mapping(state["canonical_config"], "canonical_config")
            ),
            batch_cursor=_validate_non_negative_integer(state["batch_cursor"], "batch_cursor"),
            rollout_cursor=_validate_non_negative_integer(
                state["rollout_cursor"], "rollout_cursor"
            ),
        )
        return restored, MappingProxyType(artifact_payloads)

    @staticmethod
    def _snapshot_payload(
        snapshot: CheckpointSnapshot,
        backend_result: BackendCheckpointResult,
    ) -> dict[str, object]:
        return {
            "algorithm_state": dict(snapshot.algorithm_state),
            "batch_cursor": snapshot.batch_cursor,
            "backend_format_version": backend_result.format_version,
            "backend_step": backend_result.step,
            "canonical_config": dict(snapshot.canonical_config),
            "config_hash": snapshot.config_hash,
            "dataset_hash": snapshot.dataset_hash,
            "global_step": snapshot.global_step,
            "parent_checkpoint": snapshot.parent_checkpoint,
            "python_rng_state": snapshot.python_rng_state,
            "rank_rng_states": {
                rank: {
                    "python_rng_state": state.python_rng_state,
                    "torch_cpu_rng_state": state.torch_cpu_rng_state,
                    "torch_cuda_rng_states": state.torch_cuda_rng_states,
                }
                for rank, state in LocalCheckpointStore._snapshot_rank_rng_states(snapshot).items()
            },
            "rollout_cursor": snapshot.rollout_cursor,
            "scheduler_state": (
                None if snapshot.scheduler_state is None else dict(snapshot.scheduler_state)
            ),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "torch_cpu_rng_state": snapshot.torch_cpu_rng_state,
            "torch_cuda_rng_states": snapshot.torch_cuda_rng_states,
        }

    @staticmethod
    def _snapshot_rank_rng_states(snapshot: CheckpointSnapshot) -> Mapping[int, RankRNGState]:
        if snapshot.rank_rng_states is not None:
            return snapshot.rank_rng_states
        return {
            0: RankRNGState(
                python_rng_state=snapshot.python_rng_state,
                torch_cpu_rng_state=snapshot.torch_cpu_rng_state,
                torch_cuda_rng_states=snapshot.torch_cuda_rng_states,
            )
        }

    @staticmethod
    def _load_state(payload_bytes: bytes) -> Mapping[str, object]:
        try:
            payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError) as error:
            raise ValueError("checkpoint training state is unreadable or unsafe") from error
        if not isinstance(payload, Mapping) or set(payload) != _STATE_FIELDS:
            raise ValueError("checkpoint training state fields are missing or unrecognized")
        return payload

    def _validate_rng_topology(
        self,
        cpu_state: torch.Tensor,
        cuda_states: tuple[torch.Tensor, ...],
    ) -> None:
        if len(cpu_state) != self.rng_topology.cpu_state_length:
            raise ValueError("checkpoint CPU RNG state length does not match the selected topology")
        if self.rng_topology.device_type == "cpu" and cuda_states:
            raise ValueError("checkpoint CUDA RNG states are forbidden for a CPU topology")
        if len(cuda_states) != len(self.rng_topology.cuda_state_lengths):
            raise ValueError("checkpoint CUDA RNG state count does not match the selected topology")
        for index, (state, expected_length) in enumerate(
            zip(cuda_states, self.rng_topology.cuda_state_lengths, strict=True)
        ):
            if len(state) != expected_length:
                raise ValueError(
                    f"checkpoint CUDA RNG state length for device {index} does not match topology"
                )

    @staticmethod
    def _read_regular_bytes(path: Path, field_name: str) -> bytes:
        try:
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(f"checkpoint {field_name} must not be a symlink")
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"checkpoint {field_name} must be a regular file")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                after = os.fstat(descriptor)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    raise ValueError(f"checkpoint {field_name} changed while it was opened")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    return stream.read()
            finally:
                os.close(descriptor)
        except ValueError:
            raise
        except OSError as error:
            raise ValueError(f"checkpoint {field_name} is missing or unsafe") from error

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _validate_state_matches_manifest(
        state: Mapping[str, object],
        manifest: CheckpointManifest,
    ) -> None:
        expected = {
            "backend_format_version": manifest.backend_format_version,
            "backend_step": manifest.backend_step,
            "config_hash": manifest.config_hash,
            "dataset_hash": manifest.dataset_hash,
            "global_step": manifest.global_step,
            "parent_checkpoint": manifest.parent_checkpoint,
            "schema_version": manifest.schema_version,
        }
        for field_name, expected_value in expected.items():
            if state[field_name] != expected_value:
                raise ValueError(
                    f"checkpoint training state {field_name} does not match the manifest"
                )
        _validate_python_rng_state(state["python_rng_state"])
        _restored_rank_rng_states(state["rank_rng_states"])
        cuda_states = state["torch_cuda_rng_states"]
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError("checkpoint torch_cuda_rng_states must be a sequence")

    @staticmethod
    def _restored_scheduler_state(value: object) -> Mapping[str, object] | None:
        if value is None:
            return None
        return MappingProxyType(_state_mapping(value, "scheduler_state"))

    @staticmethod
    def _validate_expected_hash(
        actual: str,
        expected: str | None,
        field_name: str,
    ) -> None:
        if expected is None:
            return
        _validate_sha256(expected, f"expected_{field_name}")
        if actual != expected:
            raise ValueError(f"checkpoint {field_name} is incompatible")

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        serialized = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        LocalCheckpointStore._write_bytes(path, (serialized + "\n").encode("utf-8"))

    def _remove_temporary(self, temporary: Path) -> None:
        resolved = temporary.resolve(strict=False)
        if resolved.parent != self.root or not resolved.name.startswith("."):
            raise RuntimeError("refusing to remove an unsafe temporary checkpoint path")
        if resolved.exists():
            shutil.rmtree(resolved)


def _validate_checkpoint_id(value: object) -> str:
    if not isinstance(value, str) or not _CHECKPOINT_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError("checkpoint_id must be one safe local path component")
    return value


def _validate_non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_backend_result(
    result: object,
    *,
    expected_step: int,
    field_name: str,
    expected_format: int | None = None,
) -> BackendCheckpointResult:
    if not isinstance(result, BackendCheckpointResult):
        raise TypeError(f"{field_name} must return BackendCheckpointResult")
    _validate_positive_integer(result.format_version, f"{field_name} format_version")
    _validate_non_negative_integer(result.step, f"{field_name} step")
    if result.step != expected_step:
        raise ValueError(f"{field_name} backend step does not match global step")
    if expected_format is not None and result.format_version != expected_format:
        raise ValueError(f"{field_name} format_version does not match the manifest")
    return result


def _validate_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or null")
    return value


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_rng_tensor(value: object, field_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.uint8 or value.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional byte tensor")
    return value.detach().cpu().clone()


def _validate_python_rng_state(value: object) -> None:
    if not isinstance(value, tuple):
        raise ValueError("python_rng_state is incompatible")
    try:
        random.Random().setstate(value)
    except (TypeError, ValueError) as error:
        raise ValueError("python_rng_state is incompatible") from error


def _validated_rank_rng_states(
    value: Mapping[int, RankRNGState],
) -> dict[int, RankRNGState]:
    if not isinstance(value, Mapping):
        raise TypeError("rank_rng_states must be a mapping")
    result: dict[int, RankRNGState] = {}
    for rank, state in value.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise TypeError("rank_rng_states keys must be non-negative integers")
        if not isinstance(state, RankRNGState):
            raise TypeError("rank_rng_states values must be RankRNGState instances")
        result[rank] = RankRNGState(
            python_rng_state=state.python_rng_state,
            torch_cpu_rng_state=state.torch_cpu_rng_state,
            torch_cuda_rng_states=state.torch_cuda_rng_states,
        )
    if not result:
        raise ValueError("rank_rng_states must not be empty")
    return result


def _restored_rank_rng_states(value: object) -> dict[int, RankRNGState]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint rank_rng_states must be a mapping")
    result: dict[int, RankRNGState] = {}
    for rank, payload in value.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("checkpoint rank_rng_states keys must be non-negative integers")
        if not isinstance(payload, Mapping) or set(payload) != {
            "python_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
        }:
            raise ValueError("checkpoint rank_rng_states entries are malformed")
        result[rank] = RankRNGState(
            python_rng_state=payload["python_rng_state"],
            torch_cpu_rng_state=payload["torch_cpu_rng_state"],
            torch_cuda_rng_states=payload["torch_cuda_rng_states"],
        )
    if not result:
        raise ValueError("checkpoint rank_rng_states must not be empty")
    return result


def _state_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a mapping with string keys")
    return {key: _state_value(item, f"{field_name}.{key}") for key, item in value.items()}


def _state_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return _state_mapping(value, field_name)
    if isinstance(value, list):
        return [_state_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, tuple):
        return tuple(_state_value(item, f"{field_name}[]") for item in value)
    raise TypeError(f"{field_name} contains an unsupported checkpoint value")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManifest",
    "CheckpointRNGTopology",
    "CheckpointSnapshot",
    "LocalCheckpointStore",
    "RankRNGState",
    "RestoredCheckpoint",
]

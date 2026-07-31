"""Atomic local checkpoint publication above backend-owned payload formats."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 1
_ARTIFACT_NAMES = frozenset({"backend.pt", "training_state.pt"})
_CHECKPOINT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_hashes",
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
        "canonical_config",
        "config_hash",
        "dataset_hash",
        "global_step",
        "parent_checkpoint",
        "python_rng_state",
        "scheduler_state",
        "schema_version",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
)

BackendSave = Callable[[Path], object]
BackendLoad = Callable[[Path], object]


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
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.global_step, "global_step")
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


@dataclass(frozen=True)
class CheckpointManifest:
    """Verified checkpoint compatibility metadata and artifact digests."""

    path: Path
    checkpoint_id: str
    global_step: int
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
    backend_result: object = None

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
        backend_load: BackendLoad | None = None,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("checkpoint root must be a pathlib.Path")
        if backend_save is not None and not callable(backend_save):
            raise TypeError("backend_save must be callable")
        if backend_load is not None and not callable(backend_load):
            raise TypeError("backend_load must be callable")
        self.root = root.resolve(strict=False)
        self.backend_save = backend_save
        self.backend_load = backend_load

    def save(
        self,
        snapshot: CheckpointSnapshot,
        *,
        checkpoint_id: str | None = None,
    ) -> CheckpointManifest:
        """Verify and atomically publish one new sibling checkpoint directory."""
        if not isinstance(snapshot, CheckpointSnapshot):
            raise TypeError("snapshot must be a CheckpointSnapshot")
        if self.backend_save is None:
            raise RuntimeError("backend_save is required to create a checkpoint")
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
            self.backend_save(backend_path)
            if not backend_path.is_file() or backend_path.is_symlink():
                raise RuntimeError("backend_save must create one regular backend.pt artifact")
            state_path = temporary / "training_state.pt"
            torch.save(self._snapshot_payload(snapshot), state_path)
            hashes = {name: self.sha256(temporary / name) for name in sorted(_ARTIFACT_NAMES)}
            manifest = CheckpointManifest(
                path=destination,
                checkpoint_id=selected_id,
                global_step=snapshot.global_step,
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
        restored = self._load_validated(
            checkpoint,
            expected_checkpoint_id=checkpoint.name,
            expected_dataset_hash=expected_dataset_hash,
            expected_config_hash=expected_config_hash,
        )
        if self.backend_load is None:
            raise RuntimeError("backend_load is required to restore a checkpoint")
        backend_result = self.backend_load(checkpoint / "backend.pt")
        return RestoredCheckpoint(
            manifest=restored.manifest,
            algorithm_state=restored.algorithm_state,
            scheduler_state=restored.scheduler_state,
            python_rng_state=restored.python_rng_state,
            torch_cpu_rng_state=restored.torch_cpu_rng_state,
            torch_cuda_rng_states=restored.torch_cuda_rng_states,
            canonical_config=restored.canonical_config,
            backend_result=backend_result,
        )

    @staticmethod
    def sha256(path: Path) -> str:
        """Return the lowercase SHA-256 digest of one regular artifact."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _operator_selected_path(self, source: Path) -> Path:
        if not isinstance(source, Path):
            raise TypeError("checkpoint source must be a pathlib.Path")
        resolved = source.resolve(strict=False)
        if resolved.parent != self.root:
            raise ValueError("checkpoint source must be a direct child of the checkpoint root")
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValueError("checkpoint source must be an existing regular directory")
        return resolved

    def _load_validated(
        self,
        checkpoint: Path,
        *,
        expected_checkpoint_id: str,
        expected_dataset_hash: str | None,
        expected_config_hash: str | None,
    ) -> RestoredCheckpoint:
        try:
            directory_fields = {item.name for item in checkpoint.iterdir()}
        except OSError as error:
            raise ValueError("checkpoint directory is unreadable") from error
        expected_fields = {*_ARTIFACT_NAMES, "manifest.json"}
        if directory_fields != expected_fields:
            raise ValueError("checkpoint directory fields are missing or unrecognized")
        manifest_path = checkpoint / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            artifact = checkpoint / name
            if artifact.parent != checkpoint or not artifact.is_file() or artifact.is_symlink():
                raise ValueError(f"checkpoint artifact {name} is missing or unsafe")
            if self.sha256(artifact) != expected_digest:
                raise ValueError(f"checkpoint artifact {name} failed SHA-256 verification")
        state = self._load_state(checkpoint / "training_state.pt")
        self._validate_state_matches_manifest(state, manifest)
        return RestoredCheckpoint(
            manifest=manifest,
            algorithm_state=MappingProxyType(
                _state_mapping(state["algorithm_state"], "algorithm_state")
            ),
            scheduler_state=self._restored_scheduler_state(state["scheduler_state"]),
            python_rng_state=state["python_rng_state"],
            torch_cpu_rng_state=_validate_rng_tensor(
                state["torch_cpu_rng_state"],
                "torch_cpu_rng_state",
            ),
            torch_cuda_rng_states=tuple(
                _validate_rng_tensor(item, f"torch_cuda_rng_states[{index}]")
                for index, item in enumerate(state["torch_cuda_rng_states"])
            ),
            canonical_config=MappingProxyType(
                _state_mapping(state["canonical_config"], "canonical_config")
            ),
        )

    @staticmethod
    def _snapshot_payload(snapshot: CheckpointSnapshot) -> dict[str, object]:
        return {
            "algorithm_state": dict(snapshot.algorithm_state),
            "canonical_config": dict(snapshot.canonical_config),
            "config_hash": snapshot.config_hash,
            "dataset_hash": snapshot.dataset_hash,
            "global_step": snapshot.global_step,
            "parent_checkpoint": snapshot.parent_checkpoint,
            "python_rng_state": snapshot.python_rng_state,
            "scheduler_state": (
                None if snapshot.scheduler_state is None else dict(snapshot.scheduler_state)
            ),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "torch_cpu_rng_state": snapshot.torch_cpu_rng_state,
            "torch_cuda_rng_states": snapshot.torch_cuda_rng_states,
        }

    @staticmethod
    def _load_state(path: Path) -> Mapping[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError) as error:
            raise ValueError("checkpoint training state is unreadable or unsafe") from error
        if not isinstance(payload, Mapping) or set(payload) != _STATE_FIELDS:
            raise ValueError("checkpoint training state fields are missing or unrecognized")
        return payload

    @staticmethod
    def _validate_state_matches_manifest(
        state: Mapping[str, object],
        manifest: CheckpointManifest,
    ) -> None:
        expected = {
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
        path.write_text(serialized + "\n", encoding="utf-8")

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
    "CheckpointSnapshot",
    "LocalCheckpointStore",
    "RestoredCheckpoint",
]

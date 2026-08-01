"""Canonical capability-first reinforcement-learning execution engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, cast

from .capability import (
    BackendCapabilities,
    Capability,
    CapabilityEvidence,
    CapabilityState,
)
from .contracts import (
    RewardProvider,
    RewardRequest,
    RLAlgorithm,
    RolloutBackend,
    TrainablePolicyBackend,
)
from .runtime_config import DistributedRuntimeConfig, RLRunConfig
from .trajectory import PolicyEvaluation, RolloutRequest, Trajectory, TrajectoryBatch

if TYPE_CHECKING:
    from torch import nn

    from .backends.base import BackendCheckpointResult, TokenizerLike
    from .checkpointing import RankRNGState

EngineMode = Literal["train", "resume", "collect", "evaluate"]


class EngineDependencyError(RuntimeError):
    """Raised when a required local runtime dependency or binding is unavailable."""


@dataclass(frozen=True)
class EngineResult:
    """Observable result of one canonical engine invocation."""

    mode: EngineMode
    global_step: int
    trajectory_count: int
    checkpoint_parent: str | None = None
    checkpoint: object | None = None
    parameter_checksum_before: str | None = None
    parameter_checksum_after: str | None = None
    parameters_updated: bool | None = None
    policy_parameter_checksum_before: str | None = None
    policy_parameter_checksum_after: str | None = None
    policy_parameters_updated: bool | None = None
    log_directory: Path | None = None
    run_id: str | None = None
    dataset_hash: str | None = None
    trajectories: tuple[Trajectory, ...] = ()
    evaluation_artifacts: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        checkpoint_payload = _serialized_artifact(self.checkpoint)
        return {
            "checkpoint": checkpoint_payload,
            "checkpoint_parent": self.checkpoint_parent,
            "dataset_hash": self.dataset_hash,
            "evaluation_artifacts": _serialized_artifact(dict(self.evaluation_artifacts or {})),
            "global_step": self.global_step,
            "log_directory": None if self.log_directory is None else str(self.log_directory),
            "mode": self.mode,
            "parameter_checksum_after": self.parameter_checksum_after,
            "parameter_checksum_before": self.parameter_checksum_before,
            "parameters_updated": self.parameters_updated,
            "policy_parameter_checksum_after": self.policy_parameter_checksum_after,
            "policy_parameter_checksum_before": self.policy_parameter_checksum_before,
            "policy_parameters_updated": self.policy_parameters_updated,
            "run_id": self.run_id,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "trajectory_count": self.trajectory_count,
        }


@dataclass(frozen=True)
class RewardAssessment:
    """One scalar reward with auditable components and external-record evidence."""

    total: float
    components: Mapping[str, float]
    evidence: Mapping[str, tuple[object, ...]]
    references: tuple[object, ...]
    breakdown: object | None = None


@dataclass(frozen=True)
class PreparedBatch:
    """A usable algorithm batch plus scored and skipped rollout evidence."""

    batch: TrajectoryBatch | None
    trajectories: tuple[Trajectory, ...]
    skipped: tuple[Trajectory, ...] = ()
    group_metrics: tuple[Mapping[str, float], ...] = ()


@dataclass(frozen=True)
class _DatasetSnapshot:
    payload: bytes
    sha256: str
    requests: tuple[RolloutRequest, ...]
    pairs: Mapping[str, Mapping[str, object]]


class CapabilityProvider(Protocol):
    """Model-free runtime capability detector."""

    def detect(self, config: RLRunConfig) -> BackendCapabilities: ...


class DatasetProvider(Protocol):
    """Lazily materialize rollout requests after capability validation."""

    def materialize(self, mode: EngineMode) -> Sequence[RolloutRequest]: ...


class BatchPreparer(Protocol):
    """Adapt scored trajectories to an algorithm/backend batch."""

    def prepare(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[object, ...],
        algorithm: RLAlgorithm | None,
    ) -> TrajectoryBatch | PreparedBatch: ...


class CheckpointCoordinator(Protocol):
    """Engine-facing transaction boundary around the local checkpoint store."""

    def load(self, path: Path) -> object: ...

    def save(self, global_step: int, parent_checkpoint: str | None) -> object: ...


class RunLogger(Protocol):
    """Engine-facing boundary around the structured JSONL sink."""

    def start(
        self,
        mode: EngineMode,
        global_step: int,
        checkpoint_parent: str | None,
        capabilities: BackendCapabilities,
    ) -> None: ...

    def trajectories(self, values: tuple[Trajectory, ...], global_step: int) -> None: ...

    def metrics(self, rewards: tuple[object, ...], global_step: int) -> None: ...


BackendFactory = Callable[[RLRunConfig], RolloutBackend]
AlgorithmFactory = Callable[[RLRunConfig, TrainablePolicyBackend], RLAlgorithm]
DatasetFactory = Callable[[RLRunConfig], DatasetProvider]
RewardFactory = Callable[[RLRunConfig], RewardProvider]
BatchPreparerFactory = Callable[[RLRunConfig], BatchPreparer]
CheckpointFactory = Callable[
    [RLRunConfig, TrainablePolicyBackend],
    CheckpointCoordinator,
]
LoggerFactory = Callable[[RLRunConfig], RunLogger]


def required_capabilities(config: RLRunConfig, mode: EngineMode) -> frozenset[Capability]:
    """Return requirements without constructing an algorithm or model."""
    required = {Capability.SUPPORTS_GENERATION}
    if mode in {"evaluate", "train", "resume"}:
        required.update(
            {
                Capability.SUPPORTS_REFERENCE_LOG_PROBS,
                Capability.SUPPORTS_TOKEN_LOG_PROBS,
            }
        )
        if config.algorithm.name == "ppo":
            required.add(Capability.SUPPORTS_VALUE_HEAD)
    if mode in {"train", "resume"}:
        required.update(
            {
                Capability.SUPPORTS_BACKWARD,
                Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
                Capability.SUPPORTS_OPTIMIZER_STEP,
            }
        )
    if config.runtime.device.startswith("cuda"):
        required.add(Capability.SUPPORTS_CUDA)
    if config.runtime.precision != "fp32":
        required.add(Capability.SUPPORTS_MIXED_PRECISION)
    if _distributed_runtime(config).strategy != "none":
        required.add(Capability.SUPPORTS_DISTRIBUTED_TRAINING)
    return frozenset(required)


def _factory_requirements(factory: object, config: RLRunConfig) -> frozenset[Capability]:
    provider = getattr(factory, "required_capabilities", None)
    if not callable(provider):
        return frozenset()
    required = provider(config)
    try:
        values = frozenset(required)
    except TypeError as error:
        raise TypeError("factory required_capabilities must return an iterable") from error
    if not all(isinstance(item, Capability) for item in values):
        raise TypeError("factory required_capabilities must contain Capability values")
    return values


class SystemCapabilityProvider:
    """Detect local dependencies and device support without loading a model."""

    def detect(self, config: RLRunConfig) -> BackendCapabilities:
        torch_available = importlib.util.find_spec("torch") is not None
        transformers_available = importlib.util.find_spec("transformers") is not None
        implementation_available = torch_available and transformers_available
        cuda_available, cuda_evidence = self._cuda_support(config, torch_available)
        mixed_available, mixed_evidence = self._mixed_precision_support(
            config,
            torch_available=torch_available,
            cuda_available=cuda_available,
        )
        distributed_available, distributed_evidence = self._distributed_support(
            config,
            torch_available=torch_available,
        )
        capabilities: dict[Capability, CapabilityEvidence] = {}
        implemented = {
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
            Capability.SUPPORTS_GENERATION,
            Capability.SUPPORTS_OPTIMIZER_STEP,
            Capability.SUPPORTS_REFERENCE_LOG_PROBS,
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            Capability.SUPPORTS_VALUE_HEAD,
        }
        for capability in Capability:
            supported = capability in implemented and implementation_available
            if capability is Capability.SUPPORTS_CUDA:
                supported = implementation_available and cuda_available
            if capability is Capability.SUPPORTS_MIXED_PRECISION:
                supported = implementation_available and mixed_available
            if capability is Capability.SUPPORTS_DISTRIBUTED_TRAINING:
                supported = implementation_available and distributed_available
            state = CapabilityState.SUPPORTED if supported else CapabilityState.UNSUPPORTED
            capabilities[capability] = CapabilityEvidence(
                state=state,
                evidence=self._evidence(
                    capability,
                    supported=supported,
                    torch_available=torch_available,
                    transformers_available=transformers_available,
                    cuda_evidence=cuda_evidence,
                    mixed_evidence=mixed_evidence,
                    distributed_evidence=distributed_evidence,
                ),
            )
        return BackendCapabilities(
            backend_name=(
                "torch_cuda" if config.runtime.device.startswith("cuda") else "torch_portable"
            ),
            backend_version=self._package_version("torch"),
            capabilities=capabilities,
        )

    @staticmethod
    def _cuda_support(config: RLRunConfig, torch_available: bool) -> tuple[bool, str]:
        if not config.runtime.device.startswith("cuda"):
            return False, "CUDA was not selected for this invocation."
        if not re.fullmatch(r"cuda(?::[0-9]+)?", config.runtime.device):
            return False, "Configure runtime.device as 'cuda' or 'cuda:<non-negative index>'."
        if not torch_available:
            return False, "Install the 'train' extra to provide torch."
        try:
            torch_module = import_module("torch")
            if not bool(torch_module.cuda.is_available()):
                return False, "Install/configure a CUDA-enabled torch runtime and visible device."
            device_count = int(torch_module.cuda.device_count())
            raw_index = config.runtime.device.partition(":")[2]
            index = int(raw_index) if raw_index else 0
            if index >= device_count:
                return (
                    False,
                    f"Configure cuda:{index} only when device_count is greater than {index}; "
                    f"detected {device_count} device(s).",
                )
            return True, f"CUDA device cuda:{index} is available among {device_count} device(s)."
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            return False, f"Configure a usable CUDA runtime ({type(error).__name__}: {error})."

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unavailable"

    @staticmethod
    def _mixed_precision_support(
        config: RLRunConfig,
        *,
        torch_available: bool,
        cuda_available: bool,
    ) -> tuple[bool, str]:
        if config.runtime.precision == "fp32":
            return False, "FP32 was selected, so automatic mixed precision is disabled."
        if not torch_available:
            return False, "Install the 'train' extra to provide torch."
        if not cuda_available:
            return False, "Configure a usable CUDA device before selecting mixed precision."
        if config.runtime.precision == "fp16":
            return (
                True,
                f"CUDA device {config.runtime.device} supports selected FP16 autocast and "
                "loss scaling.",
            )
        try:
            torch_module = import_module("torch")
            raw_index = config.runtime.device.partition(":")[2]
            index = int(raw_index) if raw_index else 0
            with torch_module.cuda.device(index):
                bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            if bf16_supported:
                return (
                    True,
                    f"PyTorch reports selected BF16 support for CUDA device cuda:{index}.",
                )
            return False, "Select FP32 or FP16 because PyTorch reports no BF16 support."
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            return False, f"BF16 support detection failed ({type(error).__name__}: {error})."

    @staticmethod
    def _distributed_support(
        config: RLRunConfig,
        *,
        torch_available: bool,
    ) -> tuple[bool, str]:
        topology = _distributed_runtime(config)
        if topology.strategy == "none":
            return False, "Single-process execution was selected."
        if not torch_available:
            return False, "Install the 'train' extra to provide torch.distributed."
        try:
            distributed = import_module("torch").distributed
            if not bool(distributed.is_available()):
                return False, "Use a PyTorch build with torch.distributed support."
            if not bool(distributed.is_initialized()):
                return False, "Initialize torch.distributed before distributed RL preflight."
            world_size = int(distributed.get_world_size())
            rank = int(distributed.get_rank())
            if world_size != topology.world_size or rank != topology.rank:
                return (
                    False,
                    "Initialized process group does not match configuration: "
                    f"world_size={world_size}, rank={rank}.",
                )
            return (
                True,
                f"Initialized {topology.strategy} process group has "
                f"world_size={world_size}, rank={rank}, local_rank={topology.local_rank}.",
            )
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            return False, f"Distributed support detection failed ({type(error).__name__}: {error})."

    @staticmethod
    def _evidence(
        capability: Capability,
        *,
        supported: bool,
        torch_available: bool,
        transformers_available: bool,
        cuda_evidence: str,
        mixed_evidence: str,
        distributed_evidence: str,
    ) -> str:
        if supported and capability is Capability.SUPPORTS_CUDA:
            return cuda_evidence
        if supported and capability is Capability.SUPPORTS_MIXED_PRECISION:
            return mixed_evidence
        if supported and capability is Capability.SUPPORTS_DISTRIBUTED_TRAINING:
            return distributed_evidence
        if supported:
            return f"Local torch/transformers runtime supports {capability.value}."
        if not torch_available:
            return "Install the 'train' extra to provide torch."
        if not transformers_available:
            return "Install the 'train' extra to provide transformers."
        if capability is Capability.SUPPORTS_CUDA:
            return cuda_evidence
        if capability is Capability.SUPPORTS_MIXED_PRECISION:
            return mixed_evidence
        if capability is Capability.SUPPORTS_DISTRIBUTED_TRAINING:
            return distributed_evidence
        return f"Configure a backend that explicitly supports {capability.value}."


class RLTrainingEngine:
    """Compose canonical RL contracts behind capability-first lifecycle methods."""

    def __init__(
        self,
        config: RLRunConfig,
        *,
        capability_provider: CapabilityProvider | None = None,
        backend_factory: BackendFactory | None = None,
        algorithm_factory: AlgorithmFactory | None = None,
        dataset_factory: DatasetFactory | None = None,
        reward_factory: RewardFactory | None = None,
        batch_preparer_factory: BatchPreparerFactory | None = None,
        checkpoint_factory: CheckpointFactory | None = None,
        logger_factory: LoggerFactory | None = None,
    ) -> None:
        if not isinstance(config, RLRunConfig):
            raise TypeError("config must be an RLRunConfig")
        if config.runtime.backend == "llama-cpp-vulkan" and (
            capability_provider is None or backend_factory is None
        ):
            raise ValueError(
                "runtime.backend='llama-cpp-vulkan' requires "
                "build_llama_cpp_engine(config, endpoint)"
            )
        self.config = config
        self.capability_provider = capability_provider or SystemCapabilityProvider()
        self.backend_factory = backend_factory or _default_backend_factory
        self.algorithm_factory = algorithm_factory or _default_algorithm_factory
        self._default_dataset = dataset_factory is None
        self._default_reward = reward_factory is None
        self._default_checkpoint = checkpoint_factory is None
        self._default_logger = logger_factory is None
        self.dataset_factory = dataset_factory
        self.reward_factory = reward_factory
        self.batch_preparer_factory = batch_preparer_factory or (
            lambda value: _DefaultBatchPreparer(value.algorithm)
        )
        self.checkpoint_factory = checkpoint_factory
        self.logger_factory = logger_factory

    def train(self, *, max_steps: int | None = None) -> EngineResult:
        """Run one training invocation with a relative optimizer-step budget.

        ``None`` uses the positive configured budget. Zero initializes and closes the run without
        rollout or optimization. Negative invocation budgets are invalid.
        """
        return self._execute("train", max_steps=max_steps)

    def resume(self, checkpoint: Path, *, max_steps: int | None = None) -> EngineResult:
        """Resume and add at most the relative optimizer-step budget to the restored step."""
        if not isinstance(checkpoint, Path):
            raise TypeError("resume checkpoint must be a pathlib.Path")
        return self._execute("resume", checkpoint=checkpoint, max_steps=max_steps)

    def collect(self) -> EngineResult:
        return self._execute("collect")

    def evaluate(self) -> EngineResult:
        return self._execute("evaluate")

    def _execute(
        self,
        mode: EngineMode,
        *,
        checkpoint: Path | None = None,
        max_steps: int | None = None,
    ) -> EngineResult:
        step_budget = self._step_budget(max_steps)
        requirements = set(required_capabilities(self.config, mode))
        requirements.update(_factory_requirements(self.backend_factory, self.config))
        if mode in {"train", "resume"}:
            requirements.update(_factory_requirements(self.algorithm_factory, self.config))
        if _distributed_runtime(self.config).strategy == "none":
            return self._execute_configured(
                mode,
                checkpoint=checkpoint,
                step_budget=step_budget,
                requirements=requirements,
            )
        from .backends.torch_cuda import distributed_runtime_context

        with distributed_runtime_context(self.config):
            return self._execute_configured(
                mode,
                checkpoint=checkpoint,
                step_budget=step_budget,
                requirements=requirements,
            )

    def _execute_configured(
        self,
        mode: EngineMode,
        *,
        checkpoint: Path | None,
        step_budget: int,
        requirements: set[Capability],
    ) -> EngineResult:
        target_step = step_budget
        detected = self.capability_provider.detect(self.config)
        detected.require(requirements)
        snapshot = _capture_dataset_snapshot(
            self.config,
            require_pairs=self._default_dataset and mode in {"evaluate", "train", "resume"},
            materialize=self._default_dataset,
        )
        default_reward = (
            _PairRewardProvider(self.config, snapshot)
            if self._default_reward and mode in {"evaluate", "train", "resume"}
            else None
        )
        _seed_process(self.config)
        backend = self.backend_factory(self.config)
        global_step = 0
        batch_cursor = 0
        rollout_cursor = 0
        resume_parent: str | None = None
        trajectory_count = 0
        latest_checkpoint: object | None = None
        result_trajectories: tuple[Trajectory, ...] = ()
        evaluation_artifacts: dict[str, object] = {}
        primary_error: BaseException | None = None
        try:
            initial_backend_requirements = set(requirements)
            if mode == "collect":
                initial_backend_requirements.discard(Capability.SUPPORTS_GENERATION)
            backend.capabilities().require(initial_backend_requirements)
            algorithm = self._algorithm(mode, cast(TrainablePolicyBackend, backend))
            dataset = (
                _SnapshotDataset(snapshot)
                if self._default_dataset
                else cast(DatasetFactory, self.dataset_factory)(self.config)
            )
            reward_provider = (
                default_reward if self._default_reward else self._reward_provider(mode)
            )
            batch_preparer = self._batch_preparer(mode)
            checkpoint_coordinator = (
                _LocalCheckpointCoordinator(self.config, backend, snapshot)
                if self._default_checkpoint and mode in {"train", "resume"}
                else self._checkpoint(mode, backend)
            )
            logger = (
                _JSONLRunLogger(self.config, snapshot.sha256)
                if self._default_logger
                else cast(LoggerFactory, self.logger_factory)(self.config)
            )
            if mode == "resume":
                if checkpoint is None:  # pragma: no cover - public resume enforces this
                    raise ValueError("resume requires an operator-selected checkpoint")
                if checkpoint_coordinator is None:  # pragma: no cover - mode selection invariant
                    raise RuntimeError("checkpoint coordinator is unavailable")
                transaction = _capture_resume_transaction(backend, algorithm)
                try:
                    restored = checkpoint_coordinator.load(checkpoint)
                    global_step = self._restored_step(restored)
                    resume_parent = self._restored_checkpoint_id(restored)
                    batch_cursor = _restored_cursor(
                        restored,
                        "batch_cursor",
                        fallback=global_step * self.config.algorithm.gradient_accumulation_steps,
                    )
                    rollout_cursor = _restored_cursor(
                        restored,
                        "rollout_cursor",
                        fallback=batch_cursor,
                    )
                    latest_checkpoint = getattr(restored, "manifest", None)
                    _preflight_engine_state(restored, algorithm)
                    self._restore_engine_state(restored, algorithm)
                except BaseException as restore_error:
                    _rollback_resume_transaction(transaction, backend, algorithm, restore_error)
                    raise
                target_step = global_step + step_budget
            requests = tuple(dataset.materialize(mode))
            if not requests:
                raise ValueError("RL dataset materialized no rollout requests")
            logger.start(mode, global_step, resume_parent, detected)
            checksum_before = _parameter_checksum(backend)
            policy_checksum_before = _policy_parameter_checksum(backend)
            if mode == "collect":
                selected = self._rollout_requests(requests, global_step, rollout_index=0)
                trajectories = tuple(backend.generate(selected))
                _validate_rollout_output(self.config, selected, trajectories)
                backend.capabilities().require(requirements)
                logger.trajectories(trajectories, global_step)
                trajectory_count = len(trajectories)
                result_trajectories = trajectories
            elif mode == "evaluate":
                selected = self._rollout_requests(requests, global_step, rollout_index=0)
                trajectories = tuple(backend.generate(selected))
                _validate_rollout_output(self.config, selected, trajectories)
                scored, rewards = self._score(trajectories, reward_provider)
                if batch_preparer is None:  # pragma: no cover - mode selection invariant
                    raise RuntimeError("batch preparer is unavailable")
                prepared = _as_prepared(batch_preparer.prepare(scored, rewards, None), scored)
                if prepared.batch is None:
                    raise ValueError("evaluation produced no usable scored trajectories")
                evaluation = backend.evaluate(prepared.batch)
                logger.trajectories(scored, global_step)
                logger.metrics(rewards, global_step)
                trajectory_count = len(scored)
                result_trajectories = scored
                evaluation_artifacts["policy_evaluation"] = evaluation
            else:
                (
                    global_step,
                    trajectory_count,
                    result_trajectories,
                    latest_checkpoint,
                    evaluation_artifacts,
                ) = self._train_loop(
                    backend=backend,
                    algorithm=cast(RLAlgorithm, algorithm),
                    requests=requests,
                    reward_provider=cast(RewardProvider, reward_provider),
                    batch_preparer=cast(BatchPreparer, batch_preparer),
                    checkpoint=cast(CheckpointCoordinator, checkpoint_coordinator),
                    logger=logger,
                    global_step=global_step,
                    parent_checkpoint=resume_parent,
                    target_step=target_step,
                    batch_cursor=batch_cursor,
                    rollout_cursor=rollout_cursor,
                )
            checksum_after = _parameter_checksum(backend)
            policy_checksum_after = _policy_parameter_checksum(backend)
            return EngineResult(
                mode=mode,
                global_step=global_step,
                trajectory_count=trajectory_count,
                checkpoint_parent=resume_parent,
                checkpoint=latest_checkpoint,
                parameter_checksum_before=checksum_before,
                parameter_checksum_after=checksum_after,
                parameters_updated=(
                    None
                    if checksum_before is None or checksum_after is None
                    else checksum_before != checksum_after
                ),
                policy_parameter_checksum_before=policy_checksum_before,
                policy_parameter_checksum_after=policy_checksum_after,
                policy_parameters_updated=(
                    None
                    if policy_checksum_before is None or policy_checksum_after is None
                    else policy_checksum_before != policy_checksum_after
                ),
                log_directory=_logger_directory(logger),
                run_id=_logger_run_id(logger),
                dataset_hash=snapshot.sha256,
                trajectories=result_trajectories,
                evaluation_artifacts=evaluation_artifacts,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                backend.close()
            except BaseException as close_error:
                if primary_error is None:
                    raise
                diagnostic = (
                    "Backend close also failed: " f"{type(close_error).__name__}: {close_error}"
                )
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(diagnostic)
                else:  # pragma: no cover - Python 3.10 compatibility
                    primary_error.__cause__ = close_error

    def _train_loop(
        self,
        *,
        backend: TrainablePolicyBackend,
        algorithm: RLAlgorithm,
        requests: tuple[RolloutRequest, ...],
        reward_provider: RewardProvider,
        batch_preparer: BatchPreparer,
        checkpoint: CheckpointCoordinator,
        logger: RunLogger,
        global_step: int,
        parent_checkpoint: str | None,
        target_step: int,
        batch_cursor: int,
        rollout_cursor: int,
    ) -> tuple[int, int, tuple[Trajectory, ...], object | None, dict[str, object]]:
        trajectory_count = 0
        next_parent = parent_checkpoint
        latest_checkpoint: object | None = None
        all_trajectories: list[Trajectory] = []
        artifacts: dict[str, object] = {}
        batches = tuple(_chunks(requests, self.config.algorithm.batch_size))
        while global_step < target_step:
            backend.zero_grad()
            accumulated = 0
            attempts = 0
            accumulation_steps = self.config.algorithm.gradient_accumulation_steps
            window_limit = len(batches) * accumulation_steps
            step_losses: list[object] = []
            step_evaluations: list[PolicyEvaluation] = []
            while accumulated < accumulation_steps and attempts < window_limit:
                request_batch = batches[batch_cursor % len(batches)]
                batch_cursor += 1
                attempts += 1
                selected = self._rollout_requests(
                    request_batch,
                    global_step,
                    rollout_index=rollout_cursor,
                )
                rollout_cursor += 1
                trajectories = tuple(backend.generate(selected))
                _validate_rollout_output(self.config, selected, trajectories)
                scored, rewards = self._score(trajectories, reward_provider)
                prepared = _as_prepared(
                    batch_preparer.prepare(scored, rewards, algorithm),
                    scored,
                )
                logger.trajectories(scored, global_step)
                logger.metrics(rewards, global_step)
                _log_group_metrics(
                    logger,
                    prepared.group_metrics,
                    global_step=global_step,
                    rollout_index=rollout_cursor - 1,
                )
                trajectory_count += len(scored)
                all_trajectories.extend(scored)
                if prepared.batch is None:
                    continue
                evaluation = backend.evaluate(prepared.batch)
                loss = algorithm.compute_loss(prepared.batch, evaluation)
                total_loss = getattr(loss, "total_loss", loss)
                try:
                    scaled_loss = total_loss / accumulation_steps
                except TypeError:
                    scaled_loss = total_loss
                backend.backward(scaled_loss)
                accumulated += 1
                step_losses.append(loss)
                step_evaluations.append(evaluation)
            if accumulated < accumulation_steps:
                backend.zero_grad()
                break
            step_result = backend.optimizer_step()
            updated = getattr(step_result, "updated", True)
            if not isinstance(updated, bool):
                raise ValueError("backend optimizer updated evidence must be a boolean")
            if not updated:
                backend.zero_grad()
                continue
            expected_step = global_step + 1
            actual_step = getattr(step_result, "step", None)
            if actual_step != expected_step:
                raise ValueError("backend optimizer step does not match engine global step")
            global_step = expected_step
            _log_training_step(
                logger,
                losses=tuple(step_losses),
                evaluations=tuple(step_evaluations),
                step_result=step_result,
                global_step=global_step,
                learning_rate=self.config.algorithm.learning_rate,
            )
            artifacts = {
                "losses": tuple(step_losses),
                "evaluations": tuple(step_evaluations),
                "optimizer_step": step_result,
            }
            if self._should_checkpoint(global_step, target_step):
                _configure_checkpoint_state(
                    checkpoint,
                    algorithm,
                    batch_cursor=batch_cursor,
                    rollout_cursor=rollout_cursor,
                )
                saved = checkpoint.save(global_step, next_parent)
                latest_checkpoint = saved
                next_parent = getattr(saved, "checkpoint_id", next_parent)
        return (
            global_step,
            trajectory_count,
            tuple(all_trajectories),
            latest_checkpoint,
            artifacts,
        )

    def _algorithm(
        self,
        mode: EngineMode,
        backend: TrainablePolicyBackend,
    ) -> RLAlgorithm | None:
        if mode not in {"train", "resume"}:
            return None
        algorithm = self.algorithm_factory(self.config, backend)
        backend.capabilities().require(algorithm.required_capabilities())
        return algorithm

    def _reward_provider(self, mode: EngineMode) -> RewardProvider | None:
        if mode not in {"evaluate", "train", "resume"}:
            return None
        return cast(RewardFactory, self.reward_factory)(self.config)

    def _batch_preparer(self, mode: EngineMode) -> BatchPreparer | None:
        if mode not in {"evaluate", "train", "resume"}:
            return None
        return self.batch_preparer_factory(self.config)

    def _checkpoint(
        self,
        mode: EngineMode,
        backend: TrainablePolicyBackend,
    ) -> CheckpointCoordinator | None:
        if mode not in {"train", "resume"}:
            return None
        return cast(CheckpointFactory, self.checkpoint_factory)(self.config, backend)

    @staticmethod
    def _score(
        trajectories: tuple[Trajectory, ...],
        provider: RewardProvider | None,
    ) -> tuple[tuple[Trajectory, ...], tuple[object, ...]]:
        if provider is None:  # pragma: no cover - mode selection enforces a provider
            raise RuntimeError("reward provider is unavailable")
        results = tuple(
            provider.score(
                RewardRequest(
                    trajectory=trajectory,
                    observable_references=tuple(
                        reference
                        for reference in trajectory.evidence_references
                        if reference.is_observable
                    ),
                )
            )
            for trajectory in trajectories
        )
        scored = tuple(
            _bind_reward(trajectory, result)
            for trajectory, result in zip(trajectories, results, strict=True)
        )
        return scored, results

    def _should_checkpoint(self, global_step: int, target_step: int) -> bool:
        return global_step % self.config.checkpoint.save_steps == 0 or global_step == target_step

    def _step_budget(self, value: int | None) -> int:
        budget = self.config.algorithm.max_steps if value is None else value
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("max_steps must be an integer")
        if value is None and budget <= 0:
            raise ValueError("configured algorithm.max_steps must be positive")
        if budget < 0:
            raise ValueError("max_steps must be non-negative")
        return budget

    def _rollout_requests(
        self,
        requests: Sequence[RolloutRequest],
        global_step: int,
        *,
        rollout_index: int,
    ) -> tuple[RolloutRequest, ...]:
        sample_count = (
            self.config.algorithm.group_size if self.config.algorithm.name == "grpo" else 1
        )
        return tuple(
            replace(
                request,
                num_samples=sample_count,
                policy_version=f"policy-{global_step}",
                seed=self.config.seed + rollout_index + index,
                sampling_parameters={
                    **dict(request.sampling_parameters),
                    "do_sample": self.config.policy.do_sample,
                    "temperature": self.config.policy.temperature,
                    "top_p": self.config.policy.top_p,
                },
            )
            for index, request in enumerate(requests)
        )

    @staticmethod
    def _restore_engine_state(restored: object, algorithm: RLAlgorithm | None) -> None:
        algorithm_state = getattr(restored, "algorithm_state", None)
        _restore_mapping_state(
            algorithm,
            algorithm_state,
            "algorithm",
            allow_empty_without_loader=True,
        )
        scheduler_state = getattr(restored, "scheduler_state", None)
        scheduler = getattr(algorithm, "scheduler", None)
        _restore_mapping_state(
            scheduler,
            scheduler_state,
            "scheduler",
            allow_empty_without_loader=False,
        )
        python_state = getattr(restored, "python_rng_state", None)
        if python_state is not None:
            random.setstate(python_state)
        cpu_state = getattr(restored, "torch_cpu_rng_state", None)
        cuda_states = getattr(restored, "torch_cuda_rng_states", ())
        if cpu_state is not None:
            torch_module = import_module("torch")
            torch_module.set_rng_state(cpu_state)
            if cuda_states:
                torch_module.cuda.set_rng_state_all(list(cuda_states))

    @staticmethod
    def _restored_step(restored: object) -> int:
        value = getattr(restored, "global_step", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("restored checkpoint global_step is invalid")
        return value

    @staticmethod
    def _restored_checkpoint_id(restored: object) -> str:
        manifest = getattr(restored, "manifest", None)
        value = getattr(manifest, "checkpoint_id", None)
        if not isinstance(value, str) or not value:
            raise ValueError("restored checkpoint identity is invalid")
        return value


class _SnapshotDataset:
    def __init__(self, snapshot: _DatasetSnapshot) -> None:
        self.snapshot = snapshot

    def materialize(self, mode: EngineMode) -> Sequence[RolloutRequest]:
        return self.snapshot.requests


def _capture_dataset_snapshot(
    config: RLRunConfig,
    *,
    require_pairs: bool,
    materialize: bool,
) -> _DatasetSnapshot:
    path = Path(config.dataset.train_path)
    if not config.dataset.train_path:
        if materialize:
            raise EngineDependencyError("dataset.train_path is required for RL execution")
        payload = b""
        return _DatasetSnapshot(payload, hashlib.sha256(payload).hexdigest(), (), {})
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EngineDependencyError(f"Unable to read RL dataset snapshot: {path}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if not materialize:
        return _DatasetSnapshot(payload, digest, (), {})
    if config.dataset.format == "text":
        if require_pairs:
            raise EngineDependencyError(
                "train/evaluate requires authored JSONL chosen/rejected pair metadata"
            )
        prompts = payload.decode("utf-8").splitlines()
        text_requests = tuple(
            RolloutRequest(prompt=prompt.strip(), case_id=f"line-{index}")
            for index, prompt in enumerate(prompts, start=1)
            if prompt.strip()
        )
        return _DatasetSnapshot(payload, digest, text_requests, {})

    from .adapters.pair_records import validate_pair_record

    pairs: dict[str, Mapping[str, object]] = {}
    pair_requests: list[RolloutRequest] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: dataset must be UTF-8 JSONL") from error
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        pair = validate_pair_record(parsed, path, line_number)
        record_id = cast(str, pair["record_id"])
        chosen = cast(str, pair["chosen"])
        rejected = cast(str, pair["rejected"])
        if chosen.strip().casefold() == rejected.strip().casefold():
            raise ValueError(f"{path}:{line_number}: chosen/rejected normalize to the same output")
        if record_id in pairs:
            raise ValueError(f"{path}:{line_number}: duplicate record_id {record_id!r}")
        frozen_pair = cast(Mapping[str, object], _freeze_json_value(pair))
        pairs[record_id] = frozen_pair
        pair_requests.append(
            RolloutRequest(
                prompt=cast(str, pair["prompt"]),
                case_id=record_id,
                metadata=frozen_pair,
            )
        )
    if require_pairs and not pair_requests:
        raise EngineDependencyError("RL dataset contains no authored preference pairs")
    return _DatasetSnapshot(
        payload,
        digest,
        tuple(pair_requests),
        MappingProxyType(pairs),
    )


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


class _PairRewardProvider:
    def __init__(self, config: RLRunConfig, snapshot: _DatasetSnapshot) -> None:
        if not snapshot.pairs:
            raise EngineDependencyError(
                "configured reward provider requires authored chosen/rejected pair metadata"
            )
        from gepa_mindfulness.core.rewards import (
            GEPARewardCalculator,
            HallucinationConfig,
            RewardWeights,
        )

        self.config = config
        self.snapshot = snapshot
        self.weights = RewardWeights(
            alpha=config.reward.alpha,
            beta=config.reward.beta,
            gamma=config.reward.gamma,
            delta=config.reward.delta,
        ).normalized()
        self.calculator = GEPARewardCalculator(
            weights=self.weights,
            hallucination=HallucinationConfig(
                confidence_threshold=0.75,
                confident_wrong_penalty=-1.0,
                uncertain_wrong_penalty=-0.5,
                appropriate_abstention_reward=0.5,
                lazy_abstention_penalty=-1.0,
            ),
        )

    def score(self, request: RewardRequest) -> RewardAssessment:
        from gepa_mindfulness.core.evidence import EvidenceReference, EvidenceSourceKind
        from gepa_mindfulness.core.reward_integrity import (
            COMPONENT_NAMES,
            RewardIntegrityCalculator,
            RewardObservation,
        )

        record_id = request.trajectory.case_id
        pair = self.snapshot.pairs.get(record_id or "")
        if pair is None:
            raise ValueError(f"trajectory case_id {record_id!r} has no authored scoring pair")
        chosen = cast(str, pair["chosen"])
        rejected = cast(str, pair["rejected"])
        normalized = request.trajectory.response.strip().casefold()
        matched = "chosen" if normalized == chosen.strip().casefold() else None
        if normalized == rejected.strip().casefold():
            matched = "rejected"
        references = (
            EvidenceReference(
                f"{self.snapshot.sha256}:{record_id}:chosen",
                EvidenceSourceKind.EXTERNAL_RECORD,
            ),
            EvidenceReference(
                f"{self.snapshot.sha256}:{record_id}:rejected",
                EvidenceSourceKind.EXTERNAL_RECORD,
            ),
        )
        base = self.calculator.compute_reward(
            response=request.trajectory.response,
            reference_answers=(chosen,),
            gepa_scores=None,
            imperatives=None,
            confidence=1.0,
            trace_summary={},
        )

        if matched is None:
            integrity_components = {name: 0.0 for name in COMPONENT_NAMES}
            components = {
                **integrity_components,
                "reward_integrity_aggregate": 0.0,
                "task_success": base.task_success,
                "gepa_alignment": base.gepa_alignment,
                "honesty": base.honesty,
                "hallucination": base.hallucination,
                "paraconsistent_truth": base.paraconsistent_truth,
            }
            evidence = {name: references for name, value in components.items() if value < 0.0}
            return RewardAssessment(base.total, components, evidence, references, base)
        authored = cast(
            Mapping[str, float],
            pair[f"{matched}_reward_components"],
        )
        evidence = {name: references for name, value in authored.items() if float(value) < 0.0}
        observation = RewardObservation(
            **dict(authored),
            observable_evidence=evidence,
            observable_references=references,
        )
        integrity = RewardIntegrityCalculator().compute(observation)
        total = base.total
        if self.config.reward.integrity_overlay_enabled:
            total += self.config.reward.overlay_weight * integrity.aggregate
        components = {
            **dict(integrity.components),
            "reward_integrity_aggregate": integrity.aggregate,
            "task_success": base.task_success,
            "gepa_alignment": base.gepa_alignment,
            "honesty": base.honesty,
            "hallucination": base.hallucination,
            "paraconsistent_truth": base.paraconsistent_truth,
        }
        full_evidence = dict(evidence)
        for name, value in components.items():
            if value < 0.0:
                full_evidence[name] = references
        return RewardAssessment(total, components, full_evidence, references, integrity)


class _DefaultBatchPreparer:
    def __init__(self, config: object) -> None:
        self.config = config

    def prepare(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[object, ...],
        algorithm: RLAlgorithm | None,
    ) -> PreparedBatch:
        values = tuple(_reward_value(reward) for reward in rewards)
        if len(values) != len(trajectories):
            raise ValueError("rewards must align with trajectories")
        if not values:
            raise ValueError("at least one scored trajectory is required")
        if getattr(self.config, "name", "ppo") == "grpo":
            return self._prepare_grpo(trajectories, values, algorithm)
        return self._prepare_ppo(trajectories, values)

    def _prepare_grpo(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[float, ...],
        algorithm: RLAlgorithm | None,
    ) -> PreparedBatch:
        compute_advantages = getattr(algorithm, "compute_group_advantages", None)
        if not callable(compute_advantages):
            raise TypeError("GRPO algorithm must expose compute_group_advantages")
        grouped: dict[str, list[tuple[Trajectory, float]]] = {}
        for trajectory, reward in zip(trajectories, rewards, strict=True):
            group_id = trajectory.case_id or trajectory.prompt
            grouped.setdefault(group_id, []).append((trajectory, reward))
        prepared: list[Trajectory] = []
        skipped: list[Trajectory] = []
        group_metrics: list[Mapping[str, float]] = []
        for group in grouped.values():
            group_rewards = tuple(reward for _, reward in group)
            advantages = compute_advantages(group_rewards)
            mean = math.fsum(group_rewards) / len(group_rewards)
            variance = math.fsum((reward - mean) ** 2 for reward in group_rewards) / len(
                group_rewards
            )
            group_metrics.append(
                {
                    "group_advantage_mean": (
                        0.0 if advantages is None else math.fsum(advantages) / len(advantages)
                    ),
                    "group_reward_mean": mean,
                    "group_reward_std": math.sqrt(variance),
                    "group_skipped": 1.0 if advantages is None else 0.0,
                    "group_size": float(len(group_rewards)),
                }
            )
            if advantages is None:
                skipped.extend(trajectory for trajectory, _ in group)
                continue
            for (trajectory, _), advantage in zip(group, advantages, strict=True):
                width = _response_width(trajectory)
                prepared.append(
                    replace(
                        trajectory,
                        advantage=tuple(float(advantage) for _ in range(width)),
                        returns=None,
                    )
                )
        if not prepared:
            return PreparedBatch(None, (), tuple(skipped), tuple(group_metrics))
        return PreparedBatch(
            _trajectory_batch(tuple(prepared)),
            tuple(prepared),
            tuple(skipped),
            tuple(group_metrics),
        )

    def _prepare_ppo(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[float, ...],
    ) -> PreparedBatch:
        gamma = float(getattr(self.config, "gamma", 0.99))
        gae_lambda = float(getattr(self.config, "gae_lambda", 0.95))
        prepared: list[Trajectory] = []
        for trajectory, reward in zip(trajectories, rewards, strict=True):
            token_ids = trajectory.response_token_ids
            if token_ids is None or not token_ids:
                raise ValueError("response_token_ids are required for engine evaluation")
            width = len(token_ids)
            old_values = trajectory.value_predictions
            if old_values is None or len(old_values) != width:
                raise ValueError("PPO requires one value prediction per response token")
            terminal_rewards = [0.0 for _ in range(width)]
            terminal_rewards[-1] = reward
            continuation = [1.0 for _ in range(width)]
            continuation[-1] = 0.0
            next_values = [*old_values[1:], 0.0]
            running = 0.0
            reversed_advantages: list[float] = []
            for index in range(width - 1, -1, -1):
                delta = (
                    terminal_rewards[index]
                    + gamma * next_values[index] * continuation[index]
                    - old_values[index]
                )
                running = delta + gamma * gae_lambda * continuation[index] * running
                reversed_advantages.append(running)
            advantages = tuple(reversed(reversed_advantages))
            returns = tuple(
                advantage + value for advantage, value in zip(advantages, old_values, strict=True)
            )
            prepared.append(
                replace(
                    trajectory,
                    reward_total=reward,
                    advantage=advantages,
                    returns=returns,
                )
            )
        prepared_tuple = tuple(prepared)
        return PreparedBatch(_trajectory_batch(prepared_tuple), prepared_tuple)


def _response_width(trajectory: Trajectory) -> int:
    token_ids = trajectory.response_token_ids
    if token_ids is None or not token_ids:
        raise ValueError("response_token_ids are required for engine evaluation")
    return len(token_ids)


def _trajectory_batch(trajectories: tuple[Trajectory, ...]) -> TrajectoryBatch:
    width = max(_response_width(trajectory) for trajectory in trajectories)
    masks = tuple(
        tuple(index < _response_width(trajectory) for index in range(width))
        for trajectory in trajectories
    )
    return TrajectoryBatch(trajectories, masks)


def _as_prepared(
    value: TrajectoryBatch | PreparedBatch,
    scored: tuple[Trajectory, ...],
) -> PreparedBatch:
    if isinstance(value, PreparedBatch):
        return value
    if not isinstance(value, TrajectoryBatch):
        raise TypeError("batch preparer must return TrajectoryBatch or PreparedBatch")
    return PreparedBatch(value, value.trajectories or scored)


def _bind_reward(trajectory: Trajectory, result: object) -> Trajectory:
    total = _reward_value(result)
    if not isinstance(result, RewardAssessment):
        return replace(trajectory, reward_total=total)
    references = tuple(result.references)
    evidence = {name: tuple(values) for name, values in result.evidence.items()}
    return replace(
        trajectory,
        reward_total=total,
        reward_components=dict(result.components),
        reward_component_evidence=evidence,
        evidence_references=references,
    )


def _chunks(
    values: Sequence[RolloutRequest],
    size: int,
) -> Sequence[tuple[RolloutRequest, ...]]:
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))


def _validate_rollout_output(
    config: RLRunConfig,
    requests: Sequence[RolloutRequest],
    trajectories: tuple[Trajectory, ...],
) -> None:
    if config.algorithm.name != "grpo":
        return
    expected: dict[str, RolloutRequest] = {}
    for request in requests:
        if request.case_id is None or not request.case_id:
            raise ValueError("GRPO rollout requests require non-empty unique case IDs")
        if request.case_id in expected:
            raise ValueError("GRPO rollout requests require non-empty unique case IDs")
        expected[request.case_id] = request
    counts = dict.fromkeys(expected, 0)
    for trajectory in trajectories:
        request = expected.get(trajectory.case_id or "")
        if request is None:
            raise ValueError("GRPO backend output contains an unknown or missing case ID")
        if trajectory.prompt != request.prompt:
            raise ValueError("GRPO backend output prompt does not match its requested group")
        if trajectory.policy_version != request.policy_version:
            raise ValueError("GRPO backend output policy version does not match its request")
        counts[request.case_id] += 1
    expected_size = config.algorithm.group_size
    if len(trajectories) != len(expected) * expected_size or any(
        count != expected_size for count in counts.values()
    ):
        raise ValueError(
            "GRPO backend output must contain exactly algorithm.group_size trajectories "
            "for each requested case ID"
        )


def _parameter_checksum(backend: TrainablePolicyBackend) -> str | None:
    provider = getattr(backend, "parameter_checksum", None)
    if not callable(provider):
        return None
    value = provider()
    if not isinstance(value, str) or not value:
        raise ValueError("backend parameter_checksum must return a non-empty string")
    return value


def _policy_parameter_checksum(backend: TrainablePolicyBackend) -> str | None:
    provider = getattr(backend, "policy_parameter_checksum", None)
    if not callable(provider):
        return None
    value = provider()
    if not isinstance(value, str) or not value:
        raise ValueError("backend policy_parameter_checksum must return a non-empty string")
    return value


def _logger_directory(logger: RunLogger) -> Path | None:
    value = getattr(logger, "directory", None)
    return value if isinstance(value, Path) else None


def _logger_run_id(logger: RunLogger) -> str | None:
    value = getattr(logger, "run_id", None)
    return value if isinstance(value, str) and value else None


def _configure_checkpoint_state(
    checkpoint: CheckpointCoordinator,
    algorithm: RLAlgorithm,
    *,
    batch_cursor: int,
    rollout_cursor: int,
) -> None:
    configure = getattr(checkpoint, "set_training_state", None)
    if not callable(configure):
        return
    state_provider = getattr(algorithm, "state_dict", None)
    algorithm_state = state_provider() if callable(state_provider) else {}
    scheduler = getattr(algorithm, "scheduler", None)
    scheduler_state_provider = getattr(scheduler, "state_dict", None)
    scheduler_state = scheduler_state_provider() if callable(scheduler_state_provider) else None
    configure(
        algorithm_state=algorithm_state,
        scheduler_state=scheduler_state,
        batch_cursor=batch_cursor,
        rollout_cursor=rollout_cursor,
    )


def _restore_mapping_state(
    owner: object,
    state: object,
    name: str,
    *,
    allow_empty_without_loader: bool,
) -> None:
    if state is None:
        return
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint {name}_state must be a mapping or null")
    if not state and allow_empty_without_loader:
        return
    loader = getattr(owner, "load_state_dict", None)
    if not callable(loader):
        raise ValueError(f"checkpoint {name}_state is present but {name} has no load_state_dict")
    try:
        loader(dict(state))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"checkpoint {name}_state is incompatible") from error


@dataclass(frozen=True)
class _ResumeTransaction:
    backend_state: object | None
    algorithm_state: object | None
    scheduler_state: object | None
    python_rng_state: tuple[object, ...]
    torch_module: object | None
    torch_cpu_rng_state: object | None
    torch_cuda_rng_states: tuple[object, ...]


def _capture_resume_transaction(
    backend: TrainablePolicyBackend,
    algorithm: RLAlgorithm | None,
) -> _ResumeTransaction:
    backend_capture = getattr(backend, "capture_checkpoint_restore_state", None)
    backend_state = backend_capture() if callable(backend_capture) else None
    algorithm_state = _state_snapshot(algorithm)
    scheduler = getattr(algorithm, "scheduler", None)
    scheduler_state = _state_snapshot(scheduler)
    torch_module: object | None = None
    cpu_state: object | None = None
    cuda_states: tuple[object, ...] = ()
    if importlib.util.find_spec("torch") is not None:
        torch_module = import_module("torch")
        get_cpu_state = getattr(torch_module, "get_rng_state", None)
        if callable(get_cpu_state):
            cpu_state = _clone_state(get_cpu_state())
        cuda = getattr(torch_module, "cuda", None)
        get_cuda_states = getattr(cuda, "get_rng_state_all", None)
        if callable(get_cuda_states):
            cuda_states = tuple(_clone_state(state) for state in get_cuda_states())
    return _ResumeTransaction(
        backend_state=backend_state,
        algorithm_state=algorithm_state,
        scheduler_state=scheduler_state,
        python_rng_state=random.getstate(),
        torch_module=torch_module,
        torch_cpu_rng_state=cpu_state,
        torch_cuda_rng_states=cuda_states,
    )


def _state_snapshot(owner: object | None) -> object | None:
    provider = getattr(owner, "state_dict", None)
    return deepcopy(provider()) if callable(provider) else None


def _clone_state(value: object) -> object:
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else deepcopy(value)


def _preflight_engine_state(restored: object, algorithm: RLAlgorithm | None) -> None:
    _preflight_mapping_loader(
        algorithm, getattr(restored, "algorithm_state", None), "algorithm", True
    )
    scheduler = getattr(algorithm, "scheduler", None)
    _preflight_mapping_loader(
        scheduler,
        getattr(restored, "scheduler_state", None),
        "scheduler",
        False,
    )
    python_state = getattr(restored, "python_rng_state", None)
    if python_state is not None:
        try:
            random.Random().setstate(python_state)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint python_rng_state is incompatible") from error


def _preflight_mapping_loader(
    owner: object,
    state: object,
    name: str,
    allow_empty_without_loader: bool,
) -> None:
    if state is None:
        return
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint {name}_state must be a mapping or null")
    if not state and allow_empty_without_loader:
        return
    if not callable(getattr(owner, "load_state_dict", None)):
        raise ValueError(f"checkpoint {name}_state is present but {name} has no load_state_dict")


def _rollback_resume_transaction(
    transaction: _ResumeTransaction,
    backend: TrainablePolicyBackend,
    algorithm: RLAlgorithm | None,
    primary_error: BaseException,
) -> None:
    failures: list[str] = []

    def attempt(label: str, callback: Callable[[], None]) -> None:
        try:
            callback()
        except BaseException as rollback_error:
            failures.append(f"{label}: {type(rollback_error).__name__}: {rollback_error}")

    torch_module = transaction.torch_module
    if torch_module is not None:
        if transaction.torch_cpu_rng_state is not None:
            set_cpu_state = getattr(torch_module, "set_rng_state", None)
            if callable(set_cpu_state):
                attempt(
                    "torch CPU RNG rollback",
                    lambda: set_cpu_state(transaction.torch_cpu_rng_state),
                )
        cuda = getattr(torch_module, "cuda", None)
        set_cuda_states = getattr(cuda, "set_rng_state_all", None)
        if callable(set_cuda_states):
            attempt(
                "torch CUDA RNG rollback",
                lambda: set_cuda_states(list(transaction.torch_cuda_rng_states)),
            )
    attempt("Python RNG rollback", lambda: random.setstate(transaction.python_rng_state))
    scheduler = getattr(algorithm, "scheduler", None)
    if transaction.scheduler_state is not None:
        attempt(
            "scheduler rollback",
            lambda: _restore_mapping_state(
                scheduler,
                transaction.scheduler_state,
                "scheduler rollback",
                allow_empty_without_loader=False,
            ),
        )
    if transaction.algorithm_state is not None:
        attempt(
            "algorithm rollback",
            lambda: _restore_mapping_state(
                algorithm,
                transaction.algorithm_state,
                "algorithm rollback",
                allow_empty_without_loader=True,
            ),
        )
    backend_rollback = getattr(backend, "rollback_checkpoint_restore_state", None)
    if transaction.backend_state is not None and callable(backend_rollback):
        attempt("backend rollback", lambda: backend_rollback(transaction.backend_state))
    if failures:
        diagnostic = "Resume rollback also failed: " + "; ".join(failures)
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(diagnostic)
        else:  # pragma: no cover - Python 3.10 compatibility
            primary_error.__context__ = RuntimeError(diagnostic)


def _restored_cursor(restored: object, name: str, *, fallback: int) -> int:
    value = getattr(restored, name, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"restored checkpoint {name} is invalid")
    return value


def _log_group_metrics(
    logger: RunLogger,
    metrics: tuple[Mapping[str, float], ...],
    *,
    global_step: int,
    rollout_index: int,
) -> None:
    handler = getattr(logger, "groups", None)
    if callable(handler) and metrics:
        handler(metrics, global_step=global_step, rollout_index=rollout_index)


def _log_training_step(
    logger: RunLogger,
    *,
    losses: tuple[object, ...],
    evaluations: tuple[PolicyEvaluation, ...],
    step_result: object,
    global_step: int,
    learning_rate: float,
) -> None:
    handler = getattr(logger, "training_step", None)
    if callable(handler):
        handler(
            losses=losses,
            evaluations=evaluations,
            step_result=step_result,
            global_step=global_step,
            learning_rate=learning_rate,
        )


def _serialized_artifact(value: object | None) -> object | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _serialized_artifact(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _serialized_artifact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialized_artifact(item) for item in value]
    if hasattr(value, "__dict__"):
        return _serialized_artifact(vars(value))
    return repr(value)


def _reward_value(result: object) -> float:
    value = result if isinstance(result, (int, float)) else getattr(result, "total", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward provider must return a numeric value or object with total")
    return float(value)


def _seed_process(config: RLRunConfig) -> None:
    random.seed(config.seed)
    if importlib.util.find_spec("torch") is None:
        return
    torch_module = import_module("torch")
    torch_module.manual_seed(config.seed)
    if config.runtime.device.startswith("cuda"):
        torch_module.cuda.manual_seed_all(config.seed)


def _default_backend_factory(config: RLRunConfig) -> TrainablePolicyBackend:
    if config.runtime.device.startswith("cuda"):
        from .backends.torch_cuda import create_cuda_backend

        return create_cuda_backend(
            config,
            lambda: _load_local_transformers_assets(config.policy.model_name),
        )
    policy_model, tokenizer = _load_local_transformers_assets(config.policy.model_name)
    from .backends.torch_policy import TorchPolicyBackend

    return TorchPolicyBackend(
        policy_model=policy_model,
        tokenizer=tokenizer,
        device=config.runtime.device,
        learning_rate=config.algorithm.learning_rate,
        max_new_tokens=config.policy.max_new_tokens,
        max_grad_norm=config.algorithm.max_grad_norm,
        model_identifier=config.policy.model_name,
    )


def _load_local_transformers_assets(model_name: str) -> tuple[nn.Module, TokenizerLike]:
    try:
        transformers = import_module("transformers")
    except (ImportError, OSError) as error:
        raise EngineDependencyError("Install the 'train' extra to provide transformers.") from error
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        policy_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise EngineDependencyError(
            f"Model {model_name!r} is not available locally; no network access is attempted."
        ) from error
    return policy_model, tokenizer


def _default_algorithm_factory(
    config: RLRunConfig,
    backend: TrainablePolicyBackend,
) -> RLAlgorithm:
    from .backends.base import TorchTensorOps

    if config.algorithm.name == "ppo":
        from .algorithms.ppo import PPOAlgorithm

        return PPOAlgorithm.from_runtime_config(TorchTensorOps(), config.algorithm)
    from .algorithms.grpo import GRPOAlgorithm

    return GRPOAlgorithm.from_runtime_config(TorchTensorOps(), config.algorithm)


class _LocalCheckpointCoordinator:
    def __init__(
        self,
        config: RLRunConfig,
        backend: TrainablePolicyBackend,
        snapshot: _DatasetSnapshot,
    ) -> None:
        import torch

        from .checkpointing import CheckpointRNGTopology, LocalCheckpointStore

        self.config = config
        self.backend = backend
        self.snapshot = snapshot
        self.algorithm_state: Mapping[str, object] = {}
        self.scheduler_state: Mapping[str, object] | None = None
        self.batch_cursor = 0
        self.rollout_cursor = 0
        device_type = cast(
            Literal["cpu", "cuda"],
            "cuda" if config.runtime.device.startswith("cuda") else "cpu",
        )
        topology = CheckpointRNGTopology.current(device_type=device_type)
        try:
            backend_save = cast(
                "Callable[[Path], BackendCheckpointResult]",
                getattr(backend, "save_checkpoint"),
            )
            preflight = cast(
                "Callable[[bytes], BackendCheckpointResult]",
                getattr(backend, "preflight_checkpoint_bytes"),
            )
            load_bytes = cast(
                "Callable[[bytes], BackendCheckpointResult]",
                getattr(backend, "load_checkpoint_bytes"),
            )
            backend_snapshot = cast(
                "Callable[[], object]",
                getattr(backend, "capture_checkpoint_restore_state"),
            )
            rollback = cast(
                "Callable[[object], None]",
                getattr(backend, "rollback_checkpoint_restore_state"),
            )
        except AttributeError as error:
            raise EngineDependencyError(
                "Backend does not expose the Task 4 checkpoint transaction interface."
            ) from error
        self._torch = torch
        self._store_type = LocalCheckpointStore
        self._backend_save = backend_save
        self._backend_preflight = preflight
        self._backend_load_bytes = load_bytes
        self._backend_snapshot = backend_snapshot
        self._backend_rollback = rollback
        self._rng_topology = topology
        self.store = LocalCheckpointStore(
            Path(config.checkpoint.output_dir),
            backend_save=backend_save,
            backend_preflight=preflight,
            backend_load_bytes=load_bytes,
            backend_snapshot=backend_snapshot,
            backend_rollback=rollback,
            rng_topology=topology,
            rank=_distributed_runtime(config).rank,
            world_size=_distributed_runtime(config).world_size,
        )

    def load(self, path: Path) -> object:
        selected_store = self._store_type(
            path.parent,
            backend_save=self._backend_save,
            backend_preflight=self._backend_preflight,
            backend_load_bytes=self._backend_load_bytes,
            backend_snapshot=self._backend_snapshot,
            backend_rollback=self._backend_rollback,
            rng_topology=self._rng_topology,
            rank=_distributed_runtime(self.config).rank,
            world_size=_distributed_runtime(self.config).world_size,
        )
        return selected_store.load(
            path,
            expected_dataset_hash=self.snapshot.sha256,
            expected_config_hash=_config_hash(self.config),
        )

    def set_training_state(
        self,
        *,
        algorithm_state: Mapping[str, object],
        scheduler_state: Mapping[str, object] | None,
        batch_cursor: int,
        rollout_cursor: int,
    ) -> None:
        self.algorithm_state = dict(algorithm_state)
        self.scheduler_state = None if scheduler_state is None else dict(scheduler_state)
        self.batch_cursor = batch_cursor
        self.rollout_cursor = rollout_cursor

    def save(self, global_step: int, parent_checkpoint: str | None) -> object:
        from .checkpointing import CheckpointSnapshot, RankRNGState

        config_payload = asdict(self.config)
        cpu_rng = self._torch.get_rng_state().clone()
        cuda_rng = (
            tuple(self._torch.cuda.get_rng_state_all())
            if self.config.runtime.device.startswith("cuda")
            else ()
        )
        local_rng = RankRNGState(
            python_rng_state=random.getstate(),
            torch_cpu_rng_state=cpu_rng,
            torch_cuda_rng_states=cuda_rng,
        )
        rank_rng_states = _gather_rank_rng_states(self.config, local_rng)
        snapshot = CheckpointSnapshot(
            global_step=global_step,
            algorithm_state=self.algorithm_state,
            scheduler_state=self.scheduler_state,
            python_rng_state=local_rng.python_rng_state,
            torch_cpu_rng_state=cpu_rng,
            torch_cuda_rng_states=cuda_rng,
            canonical_config=config_payload,
            dataset_hash=self.snapshot.sha256,
            config_hash=_config_hash(self.config),
            rank_rng_states=rank_rng_states,
            batch_cursor=self.batch_cursor,
            rollout_cursor=self.rollout_cursor,
            parent_checkpoint=parent_checkpoint,
        )
        return self.store.save(snapshot)


class _JSONLRunLogger:
    def __init__(self, config: RLRunConfig, dataset_hash: str) -> None:
        from .run_logging import JSONLLoggingSink

        self.config = config
        self.run_id = f"rl-{uuid.uuid4().hex}"
        self.directory = Path(config.logging.log_dir) / self.run_id
        self.sink = JSONLLoggingSink(
            self.directory,
            rank=_distributed_runtime(config).rank,
        )
        self.dataset_hash = dataset_hash
        self.backend_name = "torch_portable"
        self._record_counter = 0

    def start(
        self,
        mode: EngineMode,
        global_step: int,
        checkpoint_parent: str | None,
        capabilities: BackendCapabilities,
    ) -> None:
        from datetime import datetime, timezone

        from .run_logging import RunManifest

        self.backend_name = capabilities.backend_name
        evidence = {
            capability.value: {
                "evidence": item.evidence,
                "state": item.state.value,
            }
            for capability, item in capabilities.capabilities.items()
        }
        manifest = RunManifest(
            run_id=self.run_id,
            algorithm=self.config.algorithm.name,
            backend=self.backend_name,
            actor_backend=self.backend_name,
            learner_backend=self.backend_name,
            model=self.config.policy.model_name,
            reference_model=self.config.policy.model_name,
            adapter=None,
            dataset_hash=self.dataset_hash,
            config_hash=_config_hash(self.config),
            seed=self.config.seed,
            software_versions={"backend": capabilities.backend_version},
            device_capabilities={"mode": mode, "capabilities": evidence},
            start_time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            checkpoint_parent=checkpoint_parent,
        )
        self.sink.start_run(manifest)

    def trajectories(self, values: tuple[Trajectory, ...], global_step: int) -> None:
        from datetime import datetime, timezone

        from .run_logging import TrajectoryRecord

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for index, trajectory in enumerate(values):
            self._record_counter += 1
            policy_version = trajectory.policy_version or f"policy-{global_step}"
            record = TrajectoryRecord(
                record_id=(
                    f"trajectory-{global_step}-{self._record_counter}-{index}-"
                    f"{trajectory.trajectory_id}"
                ),
                run_id=self.run_id,
                timestamp=timestamp,
                global_step=global_step,
                backend=self.backend_name,
                actor_backend=self.backend_name,
                learner_backend=self.backend_name,
                policy_version=policy_version,
                trajectory=trajectory,
            )
            self.sink.log_trajectory(record)

    def metrics(self, rewards: tuple[object, ...], global_step: int) -> None:
        from datetime import datetime, timezone

        from .run_logging import MetricRecord

        values = tuple(_reward_value(reward) for reward in rewards)
        mean = sum(values) / len(values) if values else 0.0
        component_names = {
            name
            for reward in rewards
            if isinstance(reward, RewardAssessment)
            for name in reward.components
        }
        component_means = {
            name: math.fsum(
                float(reward.components.get(name, 0.0))
                for reward in rewards
                if isinstance(reward, RewardAssessment)
            )
            / max(sum(isinstance(reward, RewardAssessment) for reward in rewards), 1)
            for name in component_names
        }
        self._record_counter += 1
        record = MetricRecord(
            record_id=f"metrics-{global_step}-{self._record_counter}",
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            global_step=global_step,
            scope="aggregate",
            backend=self.backend_name,
            actor_backend=self.backend_name,
            learner_backend=self.backend_name,
            policy_version=f"policy-{global_step}",
            metrics=_distributed_mean_scalars(
                self.config,
                {"total_reward": mean, **component_means},
            ),
        )
        self.sink.log_metrics(record)
        for index, reward in enumerate(rewards):
            self._record_counter += 1
            response_metrics = {"response_reward": _reward_value(reward)}
            if isinstance(reward, RewardAssessment):
                response_metrics.update(reward.components)
            self.sink.log_metrics(
                MetricRecord(
                    record_id=f"response-{global_step}-{index}-{self._record_counter}",
                    run_id=self.run_id,
                    timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    global_step=global_step,
                    scope="response",
                    backend=self.backend_name,
                    actor_backend=self.backend_name,
                    learner_backend=self.backend_name,
                    policy_version=f"policy-{global_step}",
                    metrics=response_metrics,
                )
            )

    def training_step(
        self,
        *,
        losses: tuple[object, ...],
        evaluations: tuple[PolicyEvaluation, ...],
        step_result: object,
        global_step: int,
        learning_rate: float,
    ) -> None:
        from datetime import datetime, timezone

        from .run_logging import MetricRecord

        metrics: dict[str, float] = {
            "learning_rate": learning_rate,
            "optimizer_step": float(getattr(step_result, "step", global_step)),
        }
        gradient_norm = _scalar(getattr(step_result, "gradient_norm", None))
        if gradient_norm is not None:
            metrics["pre_clip_gradient_norm"] = gradient_norm
        for field, name in (
            ("policy_loss", "policy_loss"),
            ("value_loss", "value_loss"),
            ("entropy", "entropy"),
            ("kl", "kl"),
        ):
            values = [
                value
                for loss in losses
                if (value := _scalar(getattr(loss, field, None))) is not None
            ]
            if values:
                metrics[name] = math.fsum(values) / len(values)
        self._record_counter += 1
        self.sink.log_metrics(
            MetricRecord(
                record_id=f"step-{global_step}-{self._record_counter}",
                run_id=self.run_id,
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                global_step=global_step,
                scope="aggregate",
                backend=self.backend_name,
                actor_backend=self.backend_name,
                learner_backend=self.backend_name,
                policy_version=f"policy-{global_step - 1}",
                metrics=_distributed_mean_scalars(self.config, metrics),
            )
        )

    def groups(
        self,
        values: tuple[Mapping[str, float], ...],
        *,
        global_step: int,
        rollout_index: int,
    ) -> None:
        from datetime import datetime, timezone

        from .run_logging import MetricRecord

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for index, group in enumerate(values):
            self.sink.log_metrics(
                MetricRecord(
                    record_id=f"group-{global_step}-{rollout_index}-{index}",
                    run_id=self.run_id,
                    timestamp=timestamp,
                    global_step=global_step,
                    scope="group",
                    backend=self.backend_name,
                    actor_backend=self.backend_name,
                    learner_backend=self.backend_name,
                    policy_version=f"policy-{global_step}",
                    metrics=group,
                )
            )


def _config_payload(config: RLRunConfig) -> bytes:
    payload = asdict(config)
    payload.pop("checkpoint", None)
    payload.pop("logging", None)
    dataset = cast(dict[str, object], payload["dataset"])
    payload["dataset"] = {"format": dataset["format"]}
    algorithm = cast(dict[str, object], payload["algorithm"])
    algorithm.pop("max_steps", None)
    runtime = cast(dict[str, object], payload["runtime"])
    distributed = cast(dict[str, object], runtime["distributed"])
    if distributed["strategy"] != "none":
        runtime["device"] = "cuda:<local_rank>"
        distributed.pop("rank", None)
        distributed.pop("local_rank", None)
    serialized = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized.encode("utf-8")


def _config_hash(config: RLRunConfig) -> str:
    return hashlib.sha256(_config_payload(config)).hexdigest()


def _distributed_mean_scalars(
    config: RLRunConfig,
    metrics: Mapping[str, float],
) -> dict[str, float]:
    values = {name: float(value) for name, value in metrics.items()}
    topology = _distributed_runtime(config)
    if topology.strategy == "none":
        return values
    import torch

    gathered_names: list[object] = [None] * topology.world_size
    torch.distributed.all_gather_object(gathered_names, tuple(sorted(values)))
    if not all(
        isinstance(item, tuple) and all(isinstance(name, str) for name in item)
        for item in gathered_names
    ):
        raise RuntimeError("distributed metric key agreement returned malformed names")
    names = sorted({name for item in gathered_names for name in cast(tuple[str, ...], item)})
    value_tensor = torch.tensor(
        [values.get(name, 0.0) for name in names],
        dtype=torch.float64,
        device=config.runtime.device,
    )
    presence_tensor = torch.tensor(
        [1.0 if name in values else 0.0 for name in names],
        dtype=torch.float64,
        device=config.runtime.device,
    )
    torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(presence_tensor, op=torch.distributed.ReduceOp.SUM)
    return {
        name: float(total / presence)
        for name, total, presence in zip(
            names,
            value_tensor.cpu().tolist(),
            presence_tensor.cpu().tolist(),
            strict=True,
        )
        if presence > 0.0
    }


def _gather_rank_rng_states(
    config: RLRunConfig,
    local_state: "RankRNGState",
    *,
    distributed: object | None = None,
) -> Mapping[int, "RankRNGState"]:
    from .checkpointing import RankRNGState

    if not isinstance(local_state, RankRNGState):
        raise TypeError("local_state must be a RankRNGState")
    topology = _distributed_runtime(config)
    if topology.strategy == "none":
        return MappingProxyType({0: local_state})
    collective = import_module("torch").distributed if distributed is None else distributed
    gather = getattr(collective, "all_gather_object", None)
    if not callable(gather):
        raise RuntimeError("torch.distributed all_gather_object is required for RNG checkpointing")
    gathered: list[object] = [None] * topology.world_size
    gather(gathered, local_state)
    if not all(isinstance(state, RankRNGState) for state in gathered):
        raise RuntimeError("distributed RNG gather returned malformed rank state")
    return MappingProxyType(
        {rank: cast(RankRNGState, state) for rank, state in enumerate(gathered)}
    )


def _distributed_runtime(config: RLRunConfig) -> DistributedRuntimeConfig:
    """Return single-process defaults for deliberately malformed legacy test objects."""
    value = getattr(config.runtime, "distributed", None)
    return value if isinstance(value, DistributedRuntimeConfig) else DistributedRuntimeConfig()


def _scalar(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    mean = getattr(value, "mean", None)
    if callable(mean):
        value = mean()
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if isinstance(converted, (int, float)) and not isinstance(converted, bool):
            return float(converted)
    return None


def build_default_engine(config: RLRunConfig) -> RLTrainingEngine:
    """Build the local-only canonical engine without loading its model yet."""
    return RLTrainingEngine(config)


def build_llama_cpp_engine(config: RLRunConfig, endpoint: str) -> RLTrainingEngine:
    """Build the common engine around one operator-supplied local llama.cpp endpoint."""
    if config.runtime.backend != "llama-cpp-vulkan":
        raise ValueError("llama.cpp engine requires runtime.backend='llama-cpp-vulkan'")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("llama.cpp collection requires a non-empty endpoint")

    from .backends.llama_cpp_vulkan import (
        LlamaCppServerClient,
        LlamaCppVulkanBackend,
        detect_llama_cpp_runtime,
    )

    validation_client = LlamaCppServerClient(endpoint)
    validation_client.close()

    class LlamaCppCapabilityProvider:
        def detect(self, selected: RLRunConfig) -> BackendCapabilities:
            del selected
            return detect_llama_cpp_runtime(endpoint=endpoint)

    def backend_factory(selected: RLRunConfig) -> RolloutBackend:
        return LlamaCppVulkanBackend(
            endpoint,
            max_new_tokens=selected.policy.max_new_tokens,
            model_identifier=selected.policy.model_name,
        )

    return RLTrainingEngine(
        config,
        capability_provider=LlamaCppCapabilityProvider(),
        backend_factory=backend_factory,
    )


__all__ = [
    "CapabilityProvider",
    "EngineDependencyError",
    "EngineMode",
    "EngineResult",
    "RLTrainingEngine",
    "SystemCapabilityProvider",
    "build_default_engine",
    "build_llama_cpp_engine",
    "required_capabilities",
]

"""Canonical capability-first reinforcement-learning execution engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from .capability import (
    BackendCapabilities,
    Capability,
    CapabilityEvidence,
    CapabilityState,
)
from .contracts import RewardProvider, RewardRequest, RLAlgorithm, TrainablePolicyBackend
from .runtime_config import RLRunConfig
from .trajectory import RolloutRequest, Trajectory, TrajectoryBatch

if TYPE_CHECKING:
    from .backends.base import BackendCheckpointResult

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

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_parent": self.checkpoint_parent,
            "global_step": self.global_step,
            "mode": self.mode,
            "trajectory_count": self.trajectory_count,
        }


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
    ) -> TrajectoryBatch: ...


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


BackendFactory = Callable[[RLRunConfig], TrainablePolicyBackend]
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
    return frozenset(required)


class SystemCapabilityProvider:
    """Detect local dependencies and device support without loading a model."""

    def detect(self, config: RLRunConfig) -> BackendCapabilities:
        torch_available = importlib.util.find_spec("torch") is not None
        transformers_available = importlib.util.find_spec("transformers") is not None
        implementation_available = torch_available and transformers_available
        cuda_available = (
            self._cuda_available(torch_available)
            if config.runtime.device.startswith("cuda")
            else False
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
            state = CapabilityState.SUPPORTED if supported else CapabilityState.UNSUPPORTED
            capabilities[capability] = CapabilityEvidence(
                state=state,
                evidence=self._evidence(
                    capability,
                    supported=supported,
                    torch_available=torch_available,
                    transformers_available=transformers_available,
                ),
            )
        return BackendCapabilities(
            backend_name="torch_portable",
            backend_version=self._package_version("torch"),
            capabilities=capabilities,
        )

    @staticmethod
    def _cuda_available(torch_available: bool) -> bool:
        if not torch_available:
            return False
        try:
            torch_module = import_module("torch")
            return bool(torch_module.cuda.is_available())
        except (ImportError, OSError, RuntimeError):
            return False

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unavailable"

    @staticmethod
    def _evidence(
        capability: Capability,
        *,
        supported: bool,
        torch_available: bool,
        transformers_available: bool,
    ) -> str:
        if supported:
            return f"Local torch/transformers runtime supports {capability.value}."
        if not torch_available:
            return "Install the 'train' extra to provide torch."
        if not transformers_available:
            return "Install the 'train' extra to provide transformers."
        if capability is Capability.SUPPORTS_CUDA:
            return "Install/configure a CUDA-enabled torch runtime and visible device."
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
        self.config = config
        self.capability_provider = capability_provider or SystemCapabilityProvider()
        self.backend_factory = backend_factory or _default_backend_factory
        self.algorithm_factory = algorithm_factory or _default_algorithm_factory
        self.dataset_factory = dataset_factory or (lambda value: _LocalDataset(value))
        self.reward_factory = reward_factory or (lambda value: _RecordedRewardProvider())
        self.batch_preparer_factory = batch_preparer_factory or (
            lambda value: _DefaultBatchPreparer()
        )
        self.checkpoint_factory = checkpoint_factory or _default_checkpoint_factory
        self.logger_factory = logger_factory or (lambda value: _JSONLRunLogger(value))

    def train(self) -> EngineResult:
        return self._execute("train")

    def resume(self, checkpoint: Path) -> EngineResult:
        if not isinstance(checkpoint, Path):
            raise TypeError("resume checkpoint must be a pathlib.Path")
        return self._execute("resume", checkpoint=checkpoint)

    def collect(self) -> EngineResult:
        return self._execute("collect")

    def evaluate(self) -> EngineResult:
        return self._execute("evaluate")

    def _execute(
        self,
        mode: EngineMode,
        *,
        checkpoint: Path | None = None,
    ) -> EngineResult:
        requirements = required_capabilities(self.config, mode)
        detected = self.capability_provider.detect(self.config)
        detected.require(requirements)
        backend = self.backend_factory(self.config)
        global_step = 0
        resume_parent: str | None = None
        trajectory_count = 0
        try:
            backend.capabilities().require(requirements)
            algorithm = self._algorithm(mode, backend)
            dataset = self.dataset_factory(self.config)
            reward_provider = self._reward_provider(mode)
            batch_preparer = self._batch_preparer(mode)
            checkpoint_coordinator = self._checkpoint(mode, backend)
            logger = self.logger_factory(self.config)
            if mode == "resume":
                if checkpoint is None:  # pragma: no cover - public resume enforces this
                    raise ValueError("resume requires an operator-selected checkpoint")
                if checkpoint_coordinator is None:  # pragma: no cover - mode selection invariant
                    raise RuntimeError("checkpoint coordinator is unavailable")
                restored = checkpoint_coordinator.load(checkpoint)
                global_step = self._restored_step(restored)
                resume_parent = self._restored_checkpoint_id(restored)
            requests = tuple(dataset.materialize(mode))
            if not requests:
                raise ValueError("RL dataset materialized no rollout requests")
            logger.start(mode, global_step, resume_parent, detected)
            if mode == "collect":
                trajectories = tuple(backend.generate(requests))
                logger.trajectories(trajectories, global_step)
                trajectory_count = len(trajectories)
            elif mode == "evaluate":
                trajectories = tuple(backend.generate(requests))
                rewards = self._score(trajectories, reward_provider)
                if batch_preparer is None:  # pragma: no cover - mode selection invariant
                    raise RuntimeError("batch preparer is unavailable")
                batch = batch_preparer.prepare(trajectories, rewards, None)
                backend.evaluate(batch)
                logger.trajectories(trajectories, global_step)
                logger.metrics(rewards, global_step)
                trajectory_count = len(trajectories)
            else:
                global_step, trajectory_count = self._train_loop(
                    backend=backend,
                    algorithm=cast(RLAlgorithm, algorithm),
                    requests=requests,
                    reward_provider=cast(RewardProvider, reward_provider),
                    batch_preparer=cast(BatchPreparer, batch_preparer),
                    checkpoint=cast(CheckpointCoordinator, checkpoint_coordinator),
                    logger=logger,
                    global_step=global_step,
                    parent_checkpoint=resume_parent,
                )
            return EngineResult(
                mode=mode,
                global_step=global_step,
                trajectory_count=trajectory_count,
                checkpoint_parent=resume_parent,
            )
        finally:
            backend.close()

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
    ) -> tuple[int, int]:
        trajectory_count = 0
        next_parent = parent_checkpoint
        while global_step < self.config.algorithm.max_steps:
            trajectories = tuple(backend.generate(requests))
            rewards = self._score(trajectories, reward_provider)
            batch = batch_preparer.prepare(trajectories, rewards, algorithm)
            evaluation = backend.evaluate(batch)
            loss = algorithm.compute_loss(batch, evaluation)
            total_loss = getattr(loss, "total_loss", loss)
            backend.zero_grad()
            backend.backward(total_loss)
            step_result = backend.optimizer_step()
            expected_step = global_step + 1
            actual_step = getattr(step_result, "step", None)
            if actual_step != expected_step:
                raise ValueError("backend optimizer step does not match engine global step")
            global_step = expected_step
            logger.trajectories(trajectories, global_step)
            logger.metrics(rewards, global_step)
            trajectory_count += len(trajectories)
            if self._should_checkpoint(global_step):
                saved = checkpoint.save(global_step, next_parent)
                next_parent = getattr(saved, "checkpoint_id", next_parent)
        return global_step, trajectory_count

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
        return self.reward_factory(self.config)

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
        return self.checkpoint_factory(self.config, backend)

    @staticmethod
    def _score(
        trajectories: tuple[Trajectory, ...],
        provider: RewardProvider | None,
    ) -> tuple[object, ...]:
        if provider is None:  # pragma: no cover - mode selection enforces a provider
            raise RuntimeError("reward provider is unavailable")
        return tuple(
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

    def _should_checkpoint(self, global_step: int) -> bool:
        return (
            global_step % self.config.checkpoint.save_steps == 0
            or global_step == self.config.algorithm.max_steps
        )

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


class _LocalDataset:
    def __init__(self, config: RLRunConfig) -> None:
        self.config = config

    def materialize(self, mode: EngineMode) -> Sequence[RolloutRequest]:
        path = Path(self.config.dataset.train_path)
        if not self.config.dataset.train_path:
            raise EngineDependencyError("dataset.train_path is required for RL execution")
        if self.config.dataset.format == "text":
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            return tuple(
                RolloutRequest(prompt=prompt, case_id=f"line-{index}")
                for index, prompt in enumerate(prompts, start=1)
                if prompt
            )
        from .adapters.synthetic_cases import iter_json_object_lines

        requests: list[RolloutRequest] = []
        for line_number, row in iter_json_object_lines(path):
            prompt = row.get("prompt") or row.get("query")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number}: prompt or query is required")
            case_id = row.get("id")
            requests.append(
                RolloutRequest(
                    prompt=prompt,
                    case_id=case_id if isinstance(case_id, str) else f"line-{line_number}",
                    metadata=row,
                )
            )
        return tuple(requests)


class _RecordedRewardProvider:
    def score(self, request: RewardRequest) -> float:
        reward = request.trajectory.reward_total
        if reward is None:
            raise EngineDependencyError(
                "Generated trajectory has no recorded reward; configure an observable "
                "RewardProvider for train/evaluate."
            )
        return reward


class _DefaultBatchPreparer:
    def prepare(
        self,
        trajectories: tuple[Trajectory, ...],
        rewards: tuple[object, ...],
        algorithm: RLAlgorithm | None,
    ) -> TrajectoryBatch:
        values = tuple(_reward_value(reward) for reward in rewards)
        if len(values) != len(trajectories):
            raise ValueError("rewards must align with trajectories")
        if not values:
            raise ValueError("at least one scored trajectory is required")
        compute_advantages = (
            getattr(algorithm, "compute_group_advantages", None) if algorithm is not None else None
        )
        if callable(compute_advantages):
            computed = compute_advantages(values)
            advantages = values if computed is None else tuple(computed)
        else:
            advantages = values
        prepared: list[Trajectory] = []
        masks: list[tuple[bool, ...]] = []
        for trajectory, reward, advantage in zip(
            trajectories,
            values,
            advantages,
            strict=True,
        ):
            token_ids = trajectory.response_token_ids
            if token_ids is None or not token_ids:
                raise ValueError("response_token_ids are required for engine evaluation")
            width = len(token_ids)
            prepared.append(
                replace(
                    trajectory,
                    reward_total=reward,
                    advantage=tuple(advantage for _ in range(width)),
                    returns=tuple(reward for _ in range(width)),
                )
            )
            masks.append(tuple(True for _ in range(width)))
        return TrajectoryBatch(tuple(prepared), tuple(masks))


def _reward_value(result: object) -> float:
    value = result if isinstance(result, (int, float)) else getattr(result, "total", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reward provider must return a numeric value or object with total")
    return float(value)


def _default_backend_factory(config: RLRunConfig) -> TrainablePolicyBackend:
    try:
        transformers = import_module("transformers")
    except (ImportError, OSError) as error:
        raise EngineDependencyError("Install the 'train' extra to provide transformers.") from error
    model_name = config.policy.model_name
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
    from .backends.torch_policy import TorchPolicyBackend

    return TorchPolicyBackend(
        policy_model=policy_model,
        tokenizer=tokenizer,
        device=config.runtime.device,
        learning_rate=config.algorithm.learning_rate,
        max_new_tokens=config.policy.max_new_tokens,
        model_identifier=model_name,
    )


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
    def __init__(self, config: RLRunConfig, backend: TrainablePolicyBackend) -> None:
        import torch

        from .checkpointing import CheckpointRNGTopology, LocalCheckpointStore

        self.config = config
        self.backend = backend
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
            snapshot = cast(
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
        self.store = LocalCheckpointStore(
            Path(config.checkpoint.output_dir),
            backend_save=backend_save,
            backend_preflight=preflight,
            backend_load_bytes=load_bytes,
            backend_snapshot=snapshot,
            backend_rollback=rollback,
            rng_topology=topology,
        )

    def load(self, path: Path) -> object:
        return self.store.load(
            path,
            expected_dataset_hash=_dataset_hash(self.config),
            expected_config_hash=_config_hash(self.config),
        )

    def save(self, global_step: int, parent_checkpoint: str | None) -> object:
        from .checkpointing import CheckpointSnapshot

        config_payload = asdict(self.config)
        cpu_rng = self._torch.get_rng_state().clone()
        cuda_rng = (
            tuple(self._torch.cuda.get_rng_state_all())
            if self.config.runtime.device.startswith("cuda")
            else ()
        )
        snapshot = CheckpointSnapshot(
            global_step=global_step,
            algorithm_state={"algorithm": self.config.algorithm.name},
            scheduler_state=None,
            python_rng_state=random.getstate(),
            torch_cpu_rng_state=cpu_rng,
            torch_cuda_rng_states=cuda_rng,
            canonical_config=config_payload,
            dataset_hash=_dataset_hash(self.config),
            config_hash=_config_hash(self.config),
            parent_checkpoint=parent_checkpoint,
        )
        return self.store.save(snapshot)


def _default_checkpoint_factory(
    config: RLRunConfig,
    backend: TrainablePolicyBackend,
) -> CheckpointCoordinator:
    return _LocalCheckpointCoordinator(config, backend)


class _JSONLRunLogger:
    def __init__(self, config: RLRunConfig) -> None:
        from .run_logging import JSONLLoggingSink

        self.config = config
        self.sink = JSONLLoggingSink(Path(config.logging.log_dir), rank=0)
        self.run_id = f"rl-{uuid.uuid4().hex}"
        self.backend_name = "torch_portable"

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
            dataset_hash=_dataset_hash(self.config),
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
            policy_version = trajectory.policy_version or f"policy-{global_step}"
            record = TrajectoryRecord(
                record_id=f"trajectory-{global_step}-{index}-{trajectory.trajectory_id}",
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
        record = MetricRecord(
            record_id=f"metrics-{global_step}",
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            global_step=global_step,
            scope="aggregate",
            backend=self.backend_name,
            actor_backend=self.backend_name,
            learner_backend=self.backend_name,
            policy_version=f"policy-{global_step}",
            metrics={"total_reward": mean},
        )
        self.sink.log_metrics(record)


def _config_payload(config: RLRunConfig) -> bytes:
    serialized = json.dumps(
        asdict(config),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized.encode("utf-8")


def _config_hash(config: RLRunConfig) -> str:
    return hashlib.sha256(_config_payload(config)).hexdigest()


def _dataset_hash(config: RLRunConfig) -> str:
    path = Path(config.dataset.train_path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(config.dataset.train_path.encode("utf-8")).hexdigest()


def build_default_engine(config: RLRunConfig) -> RLTrainingEngine:
    """Build the local-only canonical engine without loading its model yet."""
    return RLTrainingEngine(config)


__all__ = [
    "CapabilityProvider",
    "EngineDependencyError",
    "EngineMode",
    "EngineResult",
    "RLTrainingEngine",
    "SystemCapabilityProvider",
    "build_default_engine",
    "required_capabilities",
]

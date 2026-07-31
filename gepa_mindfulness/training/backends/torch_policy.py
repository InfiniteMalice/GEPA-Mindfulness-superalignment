"""Portable in-memory PyTorch causal-language-model policy backend."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Sequence, cast

import torch
from torch import nn
from torch.nn import functional as functional

from gepa_mindfulness.training.capability import (
    BackendCapabilities,
    Capability,
    CapabilityEvidence,
    CapabilityState,
)
from gepa_mindfulness.training.trajectory import (
    PolicyEvaluation,
    RolloutRequest,
    Trajectory,
    TrajectoryBatch,
)

from .base import BackendCheckpointResult, OptimizerStepResult, TokenizerLike, TorchTensorOps

TrainingMode = Literal["full", "lora"]
_CHECKPOINT_FORMAT_VERSION = 1
_MISSING_CHECKPOINT_VALUE = object()


@dataclass(frozen=True)
class _BackendStateSnapshot:
    policy_state: dict[str, torch.Tensor]
    reference_state: dict[str, torch.Tensor]
    value_head_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    step: int
    max_new_tokens: int
    learning_rate: float
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor | None
    policy_training: bool
    reference_training: bool
    value_head_training: bool


class TorchPolicyBackend:
    """Generate, evaluate, and update a causal LM using public RL records."""

    backend_name = "torch_portable"

    def __init__(
        self,
        *,
        policy_model: nn.Module,
        tokenizer: TokenizerLike,
        reference_model: nn.Module | None = None,
        value_head: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
        learning_rate: float = 1e-5,
        max_new_tokens: int = 256,
        training_mode: TrainingMode = "full",
        model_identifier: str | None = None,
        adapter_identifier: str | None = None,
    ) -> None:
        self.device = self._validated_device(device)
        self.training_mode = self._validated_training_mode(training_mode)
        self.max_new_tokens = self._positive_integer(max_new_tokens, "max_new_tokens")
        self.learning_rate = self._positive_number(learning_rate, "learning_rate")
        self.tokenizer = tokenizer
        self.policy_model = policy_model.to(self.device)
        if self.training_mode == "full":
            self.policy_model.requires_grad_(True)
        else:
            self._validate_lora_policy(self.policy_model)
        if not any(parameter.requires_grad for parameter in self.policy_model.parameters()):
            raise ValueError("policy_model must expose at least one trainable parameter")

        if reference_model is not None:
            policy_parameters = {id(parameter) for parameter in self.policy_model.parameters()}
            if any(
                id(parameter) in policy_parameters for parameter in reference_model.parameters()
            ):
                raise ValueError("reference_model must not share parameters with policy_model")
        reference = deepcopy(policy_model) if reference_model is None else reference_model
        self.reference_model = reference.to(self.device)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        hidden_size = self._hidden_size(self.policy_model)
        self.value_head = value_head if value_head is not None else nn.Linear(hidden_size, 1)
        self.value_head = self.value_head.to(device=self.device, dtype=self._model_dtype())
        self.value_head.requires_grad_(True)

        parameters = [
            *self.policy_parameters(),
            *(parameter for parameter in self.value_head.parameters() if parameter.requires_grad),
        ]
        if not parameters:
            raise ValueError("backend requires trainable policy or value-head parameters")
        if optimizer is not None:
            self._validate_optimizer(optimizer, parameters)
        self.optimizer = optimizer or torch.optim.AdamW(parameters, lr=self.learning_rate)
        self.model_identifier = model_identifier or self._model_identifier(policy_model)
        self.adapter_identifier = adapter_identifier
        if self.training_mode == "lora" and self.adapter_identifier is None:
            self.adapter_identifier = "peft-lora"
        self.tensor_ops = TorchTensorOps()
        self._step = 0

    def generate(self, requests: Sequence[RolloutRequest]) -> Sequence[Trajectory]:
        """Generate trajectories with response-only token-level model evidence."""
        trajectories: list[Trajectory] = []
        for request_index, request in enumerate(requests):
            prompt_ids, sample_count, parameters = self._validated_request(request)
            max_new_tokens = parameters.pop("max_new_tokens", self.max_new_tokens)
            max_new_tokens = self._positive_integer(max_new_tokens, "max_new_tokens")
            self._validate_sampling_parameters(parameters)
            for sample_index in range(sample_count):
                policy_was_training = self.policy_model.training
                value_head_was_training = self.value_head.training
                self.policy_model.eval()
                self.value_head.eval()
                try:
                    with torch.no_grad():
                        response_ids = self._generate_response(
                            prompt_ids,
                            max_new_tokens=max_new_tokens,
                            sampling_parameters=parameters,
                            seed=request.seed,
                            sample_index=sample_index,
                        )
                        log_probs, reference_log_probs, values, _ = self._evaluate_tokens(
                            prompt_ids,
                            response_ids,
                        )
                finally:
                    self.policy_model.train(policy_was_training)
                    self.value_head.train(value_head_was_training)
                trajectory_id = self._trajectory_id(request, request_index, sample_index)
                effective_parameters = {
                    "max_new_tokens": max_new_tokens,
                    **parameters,
                }
                trajectories.append(
                    Trajectory(
                        trajectory_id=trajectory_id,
                        case_id=request.case_id,
                        prompt=request.prompt,
                        response=self.tokenizer.decode(
                            response_ids,
                            skip_special_tokens=True,
                        ),
                        prompt_token_ids=prompt_ids,
                        response_token_ids=response_ids,
                        old_log_probs=self._float_tuple(log_probs),
                        reference_log_probs=self._float_tuple(reference_log_probs),
                        value_predictions=self._float_tuple(values),
                        sampling_parameters=effective_parameters,
                        backend_name=self.backend_name,
                        backend_version=torch.__version__,
                        model_identifier=self.model_identifier,
                        adapter_identifier=self.adapter_identifier,
                        policy_version=request.policy_version,
                        seed=request.seed,
                    )
                )
        return trajectories

    def evaluate(self, batch: TrajectoryBatch) -> PolicyEvaluation:
        """Evaluate response tokens under the trainable, reference, and value models."""
        rows, masks, width = self._validated_batch(batch)
        log_prob_rows: list[torch.Tensor] = []
        reference_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        entropy_rows: list[torch.Tensor] = []
        for (prompt_ids, response_ids), mask in zip(rows, masks):
            if not any(mask[: len(response_ids)]):
                raise ValueError("each trajectory requires at least one selected response token")
            log_probs, reference_log_probs, values, entropy = self._evaluate_tokens(
                prompt_ids,
                response_ids,
            )
            padding = width - len(response_ids)
            log_prob_rows.append(functional.pad(log_probs, (0, padding)))
            reference_rows.append(functional.pad(reference_log_probs, (0, padding)))
            value_rows.append(functional.pad(values, (0, padding)))
            entropy_rows.append(functional.pad(entropy, (0, padding)))
        return PolicyEvaluation(
            log_probs=torch.stack(log_prob_rows),
            reference_log_probs=torch.stack(reference_rows),
            value_predictions=torch.stack(value_rows),
            entropy=torch.stack(entropy_rows),
        )

    def backward(self, loss: object) -> None:
        """Accumulate gradients from one scalar differentiable loss."""
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor")
        if loss.numel() != 1:
            raise ValueError("loss must contain exactly one scalar value")
        if not loss.requires_grad:
            raise ValueError("loss must require gradients")
        loss.backward()

    def optimizer_step(self) -> OptimizerStepResult:
        """Apply accumulated gradients and return monotonic step evidence."""
        trainable = [*self.policy_parameters(), *self.value_head.parameters()]
        if not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("optimizer_step requires accumulated gradients")
        self.optimizer.step()
        self._step += 1
        return OptimizerStepResult(step=self._step)

    def zero_grad(self) -> None:
        """Clear all optimizer-owned gradients."""
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, destination: Path) -> BackendCheckpointResult:
        """Serialize backend-owned training state without store-level orchestration."""
        destination = self._checkpoint_path(destination, must_exist=False)
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None
        )
        payload = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "backend_name": self.backend_name,
            "model_identifier": self.model_identifier,
            "adapter_identifier": self.adapter_identifier,
            "training_mode": self.training_mode,
            "policy_state": self.policy_model.state_dict(),
            "reference_state": self.reference_model.state_dict(),
            "value_head_state": self.value_head.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self._step,
            "max_new_tokens": self.max_new_tokens,
            "learning_rate": self.learning_rate,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
        }
        try:
            torch.save(payload, destination)
        except (OSError, RuntimeError, TypeError, pickle.PickleError) as error:
            raise RuntimeError(f"failed to save backend checkpoint: {destination}") from error
        return BackendCheckpointResult(
            format_version=_CHECKPOINT_FORMAT_VERSION,
            step=self._step,
        )

    def load_checkpoint(self, source: Path) -> BackendCheckpointResult:
        """Restore a strictly compatible checkpoint through CPU-safe deserialization."""
        source = self._checkpoint_path(source, must_exist=True)
        try:
            raw_payload = torch.load(source, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
            raise ValueError(f"failed to load backend checkpoint: {source}") from error
        payload = self._validated_checkpoint_payload(raw_payload)
        policy_state = self._validated_module_state(
            self.policy_model,
            payload["policy_state"],
            "policy_state",
        )
        reference_state = self._validated_module_state(
            self.reference_model,
            payload["reference_state"],
            "reference_state",
        )
        value_head_state = self._validated_module_state(
            self.value_head,
            payload["value_head_state"],
            "value_head_state",
        )
        optimizer_state = self._validated_optimizer_state(payload["optimizer_state"])
        snapshot = self._capture_backend_state()
        try:
            self.policy_model.load_state_dict(policy_state, strict=True)
            self.reference_model.load_state_dict(reference_state, strict=True)
            self.value_head.load_state_dict(value_head_state, strict=True)
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()
            self._step = cast(int, payload["step"])
            self.max_new_tokens = cast(int, payload["max_new_tokens"])
            self.learning_rate = float(cast(float, payload["learning_rate"]))
            torch.set_rng_state(cast(torch.Tensor, payload["cpu_rng_state"]))
            cuda_rng_state = payload["cuda_rng_state"]
            if self.device.type == "cuda" and isinstance(cuda_rng_state, torch.Tensor):
                torch.cuda.set_rng_state(cuda_rng_state, self.device)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
        except Exception as error:
            rollback_errors = self._rollback_backend_state(snapshot)
            failure_message = "backend checkpoint contains incompatible training state"
            if rollback_errors:
                failure_message += (
                    "; checkpoint rollback encountered secondary errors: "
                    + "; ".join(type(item).__name__ for item in rollback_errors)
                )
            raise ValueError(failure_message) from error
        return BackendCheckpointResult(
            format_version=_CHECKPOINT_FORMAT_VERSION,
            step=self._step,
        )

    def policy_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the policy parameters that this backend is allowed to update."""
        return tuple(
            parameter for parameter in self.policy_model.parameters() if parameter.requires_grad
        )

    def capabilities(self) -> BackendCapabilities:
        """Return explicit evidence for every public backend capability."""
        supported = {
            Capability.SUPPORTS_GENERATION,
            Capability.SUPPORTS_TOKEN_LOG_PROBS,
            Capability.SUPPORTS_REFERENCE_LOG_PROBS,
            Capability.SUPPORTS_BACKWARD,
            Capability.SUPPORTS_OPTIMIZER_STEP,
            Capability.SUPPORTS_VALUE_HEAD,
        }
        if self.training_mode == "full":
            supported.add(Capability.SUPPORTS_FULL_WEIGHT_TRAINING)
        else:
            supported.add(Capability.SUPPORTS_LORA_TRAINING)
        if self.device.type == "cuda":
            supported.add(Capability.SUPPORTS_CUDA)
        evidence = {
            capability: CapabilityEvidence(
                state=(
                    CapabilityState.SUPPORTED
                    if capability in supported
                    else CapabilityState.UNSUPPORTED
                ),
                evidence=self._capability_evidence(capability, capability in supported),
            )
            for capability in Capability
        }
        return BackendCapabilities(
            backend_name=self.backend_name,
            backend_version=torch.__version__,
            capabilities=evidence,
        )

    def close(self) -> None:
        """Release no-op in-memory backend resources."""

    @staticmethod
    def _validate_optimizer(
        optimizer: torch.optim.Optimizer,
        intended_parameters: Sequence[nn.Parameter],
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        intended_ids = [id(parameter) for parameter in intended_parameters]
        optimizer_parameters = [
            parameter for group in optimizer.param_groups for parameter in group.get("params", ())
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if (
            len(intended_ids) != len(set(intended_ids))
            or len(optimizer_ids) != len(set(optimizer_ids))
            or set(optimizer_ids) != set(intended_ids)
        ):
            raise ValueError(
                "injected optimizer must exactly cover trainable policy and value-head parameters"
            )

    @staticmethod
    def _validate_lora_policy(policy_model: nn.Module) -> None:
        peft_config = getattr(policy_model, "peft_config", None)
        if not isinstance(peft_config, Mapping) or not peft_config:
            raise ValueError("training_mode='lora' requires a PEFT LoRA adapter model")
        if not all(
            TorchPolicyBackend._peft_config_is_lora(config) for config in peft_config.values()
        ):
            raise ValueError("PEFT LoRA mode requires every adapter config to identify LoRA")
        trainable_names = [
            name for name, parameter in policy_model.named_parameters() if parameter.requires_grad
        ]
        adapter_only = all("lora_" in name or "modules_to_save" in name for name in trainable_names)
        if not trainable_names or not adapter_only:
            raise ValueError(
                "PEFT LoRA mode requires only LoRA adapter or modules_to_save trainables"
            )

    @staticmethod
    def _peft_config_is_lora(config: object) -> bool:
        markers: tuple[object, ...]
        if isinstance(config, (str, Enum)):
            markers = (config,)
        elif isinstance(config, Mapping):
            markers = tuple(
                config[marker_name]
                for marker_name in ("peft_type", "adapter_type")
                if marker_name in config
            )
        else:
            markers = tuple(
                marker
                for marker_name in ("peft_type", "adapter_type")
                if (marker := getattr(config, marker_name, _MISSING_CHECKPOINT_VALUE))
                is not _MISSING_CHECKPOINT_VALUE
            )
        return bool(markers) and all(
            TorchPolicyBackend._peft_marker_is_lora(marker) for marker in markers
        )

    @staticmethod
    def _peft_marker_is_lora(marker: object) -> bool:
        if isinstance(marker, Enum):
            marker = marker.value
        return isinstance(marker, str) and marker.strip().casefold() == "lora"

    @staticmethod
    def _checkpoint_path(path: Path, *, must_exist: bool) -> Path:
        if not isinstance(path, Path):
            raise TypeError("checkpoint path must be a pathlib.Path")
        if must_exist and not path.is_file():
            raise ValueError(f"backend checkpoint does not exist or is not a file: {path}")
        if not must_exist and path.exists() and path.is_dir():
            raise ValueError(f"backend checkpoint destination is a directory: {path}")
        return path

    def _validated_checkpoint_payload(self, raw_payload: object) -> Mapping[str, object]:
        if not isinstance(raw_payload, Mapping):
            raise ValueError("backend checkpoint must contain a mapping payload")
        if raw_payload.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"checkpoint format_version must be {_CHECKPOINT_FORMAT_VERSION}")
        required_fields = {
            "adapter_identifier",
            "backend_name",
            "cpu_rng_state",
            "cuda_rng_state",
            "format_version",
            "learning_rate",
            "max_new_tokens",
            "model_identifier",
            "optimizer_state",
            "policy_state",
            "reference_state",
            "step",
            "training_mode",
            "value_head_state",
        }
        if set(raw_payload) != required_fields:
            raise ValueError("backend checkpoint fields are missing or unrecognized")
        expected_metadata = {
            "adapter_identifier": self.adapter_identifier,
            "backend_name": self.backend_name,
            "model_identifier": self.model_identifier,
            "training_mode": self.training_mode,
        }
        for field_name, expected in expected_metadata.items():
            if raw_payload[field_name] != expected:
                raise ValueError(f"backend checkpoint {field_name} is incompatible")
        step = raw_payload["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("backend checkpoint step must be a non-negative integer")
        self._positive_integer(raw_payload["max_new_tokens"], "checkpoint max_new_tokens")
        self._positive_number(raw_payload["learning_rate"], "checkpoint learning_rate")
        self._validated_rng_state(
            raw_payload["cpu_rng_state"],
            "cpu_rng_state",
            expected_state=torch.get_rng_state(),
        )
        cuda_rng_state = raw_payload["cuda_rng_state"]
        if self.device.type == "cuda":
            if cuda_rng_state is None:
                raise ValueError("backend checkpoint cuda_rng_state is required for a CUDA backend")
            self._validated_rng_state(
                cuda_rng_state,
                "cuda_rng_state",
                expected_state=torch.cuda.get_rng_state(self.device),
            )
        elif cuda_rng_state is not None:
            raise ValueError("backend checkpoint cuda_rng_state must be None for a CPU backend")
        return raw_payload

    @staticmethod
    def _validated_rng_state(
        value: object,
        field_name: str,
        *,
        expected_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.uint8 or value.ndim != 1:
            raise ValueError(f"backend checkpoint {field_name} must be a byte tensor")
        if expected_state is not None and value.shape != expected_state.shape:
            raise ValueError(f"backend checkpoint {field_name} length is incompatible")
        return value

    @staticmethod
    def _validated_module_state(
        module: nn.Module,
        value: object,
        field_name: str,
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(value, Mapping) or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in value.items()
        ):
            raise ValueError(f"backend checkpoint {field_name} must be a tensor state mapping")
        state = cast(Mapping[str, torch.Tensor], value)
        current_state = module.state_dict()
        if set(state) != set(current_state):
            raise ValueError(f"backend checkpoint {field_name} keys are incompatible")
        for name, current_tensor in current_state.items():
            saved_tensor = state[name]
            if (
                saved_tensor.shape != current_tensor.shape
                or saved_tensor.dtype != current_tensor.dtype
            ):
                raise ValueError(
                    f"backend checkpoint {field_name}.{name} shape or dtype is incompatible"
                )
        return state

    def _validated_optimizer_state(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {"param_groups", "state"}:
            raise ValueError("backend checkpoint optimizer_state is malformed")
        saved_groups = value["param_groups"]
        current_optimizer_state = self.optimizer.state_dict()
        current_groups = current_optimizer_state["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
            raise ValueError("backend checkpoint optimizer parameter groups are incompatible")
        parameters_by_id: dict[int, nn.Parameter] = {}
        for saved_group, current_group, live_group in zip(
            saved_groups,
            current_groups,
            self.optimizer.param_groups,
        ):
            if not isinstance(saved_group, dict) or not isinstance(saved_group.get("params"), list):
                raise ValueError("backend checkpoint optimizer parameter groups are malformed")
            self._validate_optimizer_live_structure(saved_group, current_group)
            saved_parameter_ids = saved_group["params"]
            current_parameter_ids = current_group["params"]
            live_parameters = live_group.get("params")
            if (
                not isinstance(current_parameter_ids, list)
                or not isinstance(live_parameters, list)
                or saved_parameter_ids != current_parameter_ids
                or len(current_parameter_ids) != len(live_parameters)
                or not all(
                    isinstance(parameter_id, int) and not isinstance(parameter_id, bool)
                    for parameter_id in saved_parameter_ids
                )
                or not all(isinstance(parameter, nn.Parameter) for parameter in live_parameters)
            ):
                raise ValueError("backend checkpoint optimizer parameters are incompatible")
            for parameter_id, parameter in zip(current_parameter_ids, live_parameters):
                if parameter_id in parameters_by_id:
                    raise ValueError("backend checkpoint optimizer parameters are duplicated")
                parameters_by_id[parameter_id] = parameter
        if not isinstance(value["state"], dict):
            raise ValueError("backend checkpoint optimizer state must be a mapping")
        saved_state = value["state"]
        current_state = current_optimizer_state["state"]
        for parameter_id, parameter_state in saved_state.items():
            if (
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                or parameter_id not in parameters_by_id
                or not isinstance(parameter_state, Mapping)
            ):
                raise ValueError("backend checkpoint optimizer state is incompatible")
            current_parameter_state = current_state.get(
                parameter_id,
                _MISSING_CHECKPOINT_VALUE,
            )
            self._validate_optimizer_parameter_state(
                parameter_state,
                parameters_by_id[parameter_id],
                current_parameter_state,
            )
        return value

    @classmethod
    def _validate_optimizer_parameter_state(
        cls,
        value: object,
        parameter: nn.Parameter,
        current_value: object = _MISSING_CHECKPOINT_VALUE,
    ) -> None:
        if current_value is not _MISSING_CHECKPOINT_VALUE:
            cls._validate_optimizer_live_structure(value, current_value)
            return
        if isinstance(value, torch.Tensor):
            compatible = value.ndim == 0 or value.shape == parameter.shape
            if not compatible:
                raise ValueError("backend checkpoint optimizer state tensor is incompatible")
            return
        if isinstance(value, Mapping):
            for item in value.values():
                cls._validate_optimizer_parameter_state(
                    item,
                    parameter,
                )
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                cls._validate_optimizer_parameter_state(item, parameter)

    @classmethod
    def _validate_optimizer_live_structure(cls, value: object, current_value: object) -> None:
        if isinstance(current_value, torch.Tensor):
            if not isinstance(value, torch.Tensor) or (
                value.shape != current_value.shape or value.dtype != current_value.dtype
            ):
                raise ValueError("backend checkpoint optimizer state tensor is incompatible")
            return
        if isinstance(current_value, Mapping):
            if not isinstance(value, Mapping) or set(value) != set(current_value):
                raise ValueError("backend checkpoint optimizer state mapping is incompatible")
            for key, item in current_value.items():
                cls._validate_optimizer_live_structure(value[key], item)
            return
        if isinstance(current_value, list):
            if not isinstance(value, list) or len(value) != len(current_value):
                raise ValueError("backend checkpoint optimizer state list is incompatible")
            for saved_item, live_item in zip(value, current_value):
                cls._validate_optimizer_live_structure(saved_item, live_item)
            return
        if isinstance(current_value, tuple):
            if not isinstance(value, tuple) or len(value) != len(current_value):
                raise ValueError("backend checkpoint optimizer state tuple is incompatible")
            for saved_item, live_item in zip(value, current_value):
                cls._validate_optimizer_live_structure(saved_item, live_item)
            return
        if type(value) is not type(current_value):
            raise ValueError("backend checkpoint optimizer state leaf is incompatible")

    def _capture_backend_state(self) -> _BackendStateSnapshot:
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
        )
        return _BackendStateSnapshot(
            policy_state=self._cloned_module_state(self.policy_model),
            reference_state=self._cloned_module_state(self.reference_model),
            value_head_state=self._cloned_module_state(self.value_head),
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            step=self._step,
            max_new_tokens=self.max_new_tokens,
            learning_rate=self.learning_rate,
            cpu_rng_state=torch.get_rng_state().clone(),
            cuda_rng_state=cuda_rng_state,
            policy_training=self.policy_model.training,
            reference_training=self.reference_model.training,
            value_head_training=self.value_head.training,
        )

    @staticmethod
    def _cloned_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}

    def _rollback_backend_state(
        self,
        snapshot: _BackendStateSnapshot,
    ) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        self._attempt_rollback(
            errors,
            lambda: self.policy_model.load_state_dict(snapshot.policy_state, strict=True),
        )
        self._attempt_rollback(
            errors,
            lambda: self.reference_model.load_state_dict(snapshot.reference_state, strict=True),
        )
        self._attempt_rollback(
            errors,
            lambda: self.value_head.load_state_dict(snapshot.value_head_state, strict=True),
        )
        self._attempt_rollback(
            errors,
            lambda: self.optimizer.load_state_dict(snapshot.optimizer_state),
        )
        self._attempt_rollback(errors, self._move_optimizer_state_to_device)
        self._step = snapshot.step
        self.max_new_tokens = snapshot.max_new_tokens
        self.learning_rate = snapshot.learning_rate
        self._attempt_rollback(errors, lambda: torch.set_rng_state(snapshot.cpu_rng_state))
        if self.device.type == "cuda" and snapshot.cuda_rng_state is not None:
            self._attempt_rollback(
                errors,
                lambda: torch.cuda.set_rng_state(snapshot.cuda_rng_state, self.device),
            )
        self._attempt_rollback(
            errors,
            lambda: self.policy_model.train(snapshot.policy_training),
        )
        self._attempt_rollback(
            errors,
            lambda: self.reference_model.train(snapshot.reference_training),
        )
        self._attempt_rollback(
            errors,
            lambda: self.value_head.train(snapshot.value_head_training),
        )
        return tuple(errors)

    @staticmethod
    def _attempt_rollback(errors: list[Exception], operation: Callable[[], object]) -> None:
        try:
            operation()
        except Exception as error:
            errors.append(error)

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for name, value in state.items():
                state[name] = self._move_checkpoint_value(value)

    def _move_checkpoint_value(self, value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_checkpoint_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_checkpoint_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_checkpoint_value(item) for item in value)
        return value

    def _evaluate_tokens(
        self,
        prompt_ids: tuple[int, ...],
        response_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)
        outputs = self.policy_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits, hidden = self._validated_outputs(outputs, input_ids.shape[1], "policy_model")
        if hidden is None:  # pragma: no cover - enforced by _validated_outputs
            raise AssertionError("policy hidden states must be available")
        selected, entropy = self._response_statistics(logits, input_ids, len(prompt_ids))
        predictor_hidden = hidden[:, :-1, :][:, len(prompt_ids) - 1 :]
        raw_values = self.value_head(predictor_hidden)
        expected_value_shape = (1, len(response_ids), 1)
        if not isinstance(raw_values, torch.Tensor) or raw_values.shape != expected_value_shape:
            raise RuntimeError(
                "value_head must return shape [batch, response_tokens, 1] aligned with responses"
            )
        values = raw_values.squeeze(-1).squeeze(0)

        with torch.no_grad():
            reference_outputs = self.reference_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            reference_logits, _ = self._validated_outputs(
                reference_outputs,
                input_ids.shape[1],
                "reference_model",
                require_hidden=False,
            )
            reference_selected, _ = self._response_statistics(
                reference_logits,
                input_ids,
                len(prompt_ids),
            )
        return selected.squeeze(0), reference_selected.squeeze(0), values, entropy.squeeze(0)

    @staticmethod
    def _response_statistics(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        selected = token_log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        entropy = -(token_log_probs.exp() * token_log_probs).sum(dim=-1)
        return selected[:, prompt_length - 1 :], entropy[:, prompt_length - 1 :]

    @staticmethod
    def _validated_outputs(
        outputs: object,
        sequence_length: int,
        model_name: str,
        *,
        require_hidden: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError(f"{model_name} must return rank-three logits")
        if logits.shape[:2] != (1, sequence_length):
            raise RuntimeError(f"{model_name} logits must align with the input sequence")
        hidden_states = getattr(outputs, "hidden_states", None)
        if not require_hidden:
            return logits, None
        if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
            raise RuntimeError(f"{model_name} must return hidden states for the value head")
        hidden = hidden_states[-1]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"{model_name} must return rank-three hidden states")
        if hidden.shape[:2] != (1, sequence_length):
            raise RuntimeError(f"{model_name} hidden states must align with the input sequence")
        return logits, hidden

    def _generate_response(
        self,
        prompt_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        sampling_parameters: dict[str, object],
        seed: int | None,
        sample_index: int,
    ) -> tuple[int, ...]:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        rng_devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed + sample_index)
            with torch.no_grad():
                generated = self.policy_model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    **sampling_parameters,
                )
        if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
            raise RuntimeError("policy_model.generate must return a rank-two tensor")
        if generated.shape[0] != 1 or generated.shape[1] <= len(prompt_ids):
            raise RuntimeError("policy_model.generate must return at least one response token")
        if not torch.equal(generated[0, : len(prompt_ids)], input_ids[0]):
            raise RuntimeError("policy_model.generate must preserve the prompt-token prefix")
        return tuple(int(token_id) for token_id in generated[0, len(prompt_ids) :].tolist())

    def _validated_request(
        self,
        request: RolloutRequest,
    ) -> tuple[tuple[int, ...], int, dict[str, object]]:
        if not isinstance(request, RolloutRequest):
            raise TypeError("requests must contain RolloutRequest values")
        if not request.prompt or not request.prompt.strip():
            raise ValueError("rollout prompt must not be empty")
        sample_count = self._positive_integer(request.num_samples, "num_samples")
        prompt_ids = self._token_ids(
            self.tokenizer.encode(request.prompt, add_special_tokens=False),
            "prompt_token_ids",
        )
        if not prompt_ids:
            raise ValueError("tokenizer must produce at least one prompt token")
        if not isinstance(request.sampling_parameters, dict):
            parameters = dict(request.sampling_parameters)
        else:
            parameters = request.sampling_parameters.copy()
        return prompt_ids, sample_count, parameters

    def _validated_batch(
        self,
        batch: TrajectoryBatch,
    ) -> tuple[
        tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        tuple[tuple[bool, ...], ...],
        int,
    ]:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError("batch must be a TrajectoryBatch")
        if not batch.trajectories:
            raise ValueError("evaluation requires at least one trajectory")
        if batch.response_token_masks is None:
            raise ValueError("response_token_masks are required for evaluation")
        if len(batch.response_token_masks) != len(batch.trajectories):
            raise ValueError("response_token_masks must align with trajectories")
        rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for trajectory in batch.trajectories:
            if trajectory.prompt_token_ids is None:
                raise ValueError("trajectory.prompt_token_ids are required for evaluation")
            if trajectory.response_token_ids is None:
                raise ValueError("trajectory.response_token_ids are required for evaluation")
            prompt_ids = self._token_ids(trajectory.prompt_token_ids, "prompt_token_ids")
            response_ids = self._token_ids(trajectory.response_token_ids, "response_token_ids")
            if not prompt_ids:
                raise ValueError("trajectory.prompt_token_ids must not be empty")
            if not response_ids:
                raise ValueError("trajectory.response_token_ids must not be empty")
            rows.append((prompt_ids, response_ids))
        width = max(len(response_ids) for _, response_ids in rows)
        masks: list[tuple[bool, ...]] = []
        for (_, response_ids), raw_mask in zip(rows, batch.response_token_masks):
            if len(raw_mask) != width or not all(isinstance(value, bool) for value in raw_mask):
                raise ValueError("each response token mask must be boolean and batch-aligned")
            mask = tuple(raw_mask)
            if any(mask[len(response_ids) :]):
                raise ValueError("response token mask cannot select padding positions")
            masks.append(mask)
        return tuple(rows), tuple(masks), width

    @staticmethod
    def _token_ids(values: Sequence[int], field_name: str) -> tuple[int, ...]:
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            raise ValueError(f"{field_name} must contain non-negative integer token IDs")
        return tuple(values)

    @staticmethod
    def _validate_sampling_parameters(parameters: dict[str, object]) -> None:
        temperature = parameters.get("temperature")
        if temperature is not None:
            TorchPolicyBackend._positive_number(temperature, "temperature")
        top_p = parameters.get("top_p")
        if top_p is not None:
            top_p_value = TorchPolicyBackend._positive_number(top_p, "top_p")
            if top_p_value > 1.0:
                raise ValueError("top_p must not exceed 1.0")
        do_sample = parameters.get("do_sample")
        if do_sample is not None and not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean")

    @staticmethod
    def _positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _positive_number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _validated_device(device: str | torch.device) -> torch.device:
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"invalid torch device: {device!r}") from error
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("portable backend device must be CPU or CUDA")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _validated_training_mode(training_mode: str) -> TrainingMode:
        if training_mode not in {"full", "lora"}:
            raise ValueError("training_mode must be 'full' or 'lora'")
        return cast(TrainingMode, training_mode)

    @staticmethod
    def _hidden_size(model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attribute in ("hidden_size", "n_embd"):
            value = getattr(config, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise ValueError("policy_model.config must expose a positive hidden_size or n_embd")

    def _model_dtype(self) -> torch.dtype:
        for parameter in self.policy_model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.get_default_dtype()

    @staticmethod
    def _model_identifier(model: nn.Module) -> str:
        config = getattr(model, "config", None)
        identifier = getattr(config, "_name_or_path", None)
        return (
            identifier if isinstance(identifier, str) and identifier else model.__class__.__name__
        )

    @staticmethod
    def _float_tuple(values: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(value) for value in values.detach().cpu().tolist())

    @staticmethod
    def _trajectory_id(
        request: RolloutRequest,
        request_index: int,
        sample_index: int,
    ) -> str:
        request_part = request.case_id or f"request-{request_index}"
        version_part = request.policy_version or "unversioned"
        return f"{request_part}-{version_part}-{sample_index}"

    def _capability_evidence(self, capability: Capability, supported: bool) -> str:
        if supported:
            if capability is Capability.SUPPORTS_CUDA:
                return f"Policy, reference, value head, and tensors use {self.device}."
            if capability is Capability.SUPPORTS_LORA_TRAINING:
                return "Policy has PEFT config evidence and adapter-only trainable parameters."
            if capability is Capability.SUPPORTS_FULL_WEIGHT_TRAINING:
                return "All policy-model parameters are optimizer-eligible."
            return f"{capability.value} is implemented by TorchPolicyBackend."
        return f"{capability.value} is not enabled by this backend configuration."


__all__ = ["TorchPolicyBackend", "TrainingMode"]

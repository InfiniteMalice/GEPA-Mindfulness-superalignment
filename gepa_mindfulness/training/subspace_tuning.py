"""Opt-in CPU applied-update SST; no reward or model-admission authority."""

# Standard library
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SSTConfig:
    """Bound a standalone experiment; budgets are repository controls, not paper settings."""

    enabled: bool = False
    rank_fraction: float = 0.99
    refresh_every: int | None = None
    fallback: str = "error"
    max_matrix_elements: int = 1_048_576
    max_total_elements: int = 4_194_304

    def __post_init__(self) -> None:
        """Reject coercible booleans, nonfinite ranks and implicit fallbacks."""
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        rank = self.rank_fraction
        if type(rank) not in (int, float) or not 0 <= rank <= 1 or not math.isfinite(rank):
            raise ValueError("rank_fraction must be finite and within [0, 1]")
        if self.refresh_every is not None and (
            type(self.refresh_every) is not int or self.refresh_every < 1
        ):
            raise ValueError("refresh_every must be a positive integer or None")
        if self.fallback not in ("error", "freeze"):
            raise ValueError("fallback must be error or freeze")
        for value in (self.max_matrix_elements, self.max_total_elements):
            if type(value) is not int or not 1 <= value <= 16_777_216:
                raise ValueError("element budgets must be positive bounded integers")

    def to_dict(self) -> dict[str, object]:
        """Serialize all settings with a strict schema version."""
        return {"schema_version": "sst-config-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> SSTConfig:
        """Restore an exact configuration without ignoring unknown fields."""
        keys = {"schema_version", *cls.__dataclass_fields__}
        if type(value) is not dict or set(value) != keys:
            raise ValueError("invalid SST configuration fields")
        if value["schema_version"] != "sst-config-v1":
            raise ValueError("unsupported SST configuration version")
        return cls(**{key: item for key, item in value.items() if key != "schema_version"})


def create_sst_adapter(
    optimizer: object,
    targets: object,
    *,
    reference_id: str,
    config: SSTConfig = SSTConfig(),
) -> SSTStepAdapter | None:
    """Return no adapter by default, without inspecting tensors or importing PyTorch."""
    if type(config) is not SSTConfig:
        raise ValueError("expected exact SSTConfig")
    if not config.enabled:
        return None
    return SSTStepAdapter(optimizer, targets, reference_id=reference_id, config=config)


class SSTStepAdapter:
    """Replace selected native optimizer updates with their free-to-free blocks.

    The trusted host supplies already-admitted training and owns the native optimizer.
    This research adapter is separate from the canonical backend and its checkpoints.
    It supports CPU dense full-weight float32/float64 only. No closures or AMP are used.
    """

    def __init__(
        self, optimizer: object, targets: object, *, reference_id: str, config: SSTConfig
    ) -> None:
        """Capture fixed reference matrices after validating explicit optimizer ownership."""
        import torch

        if type(config) is not SSTConfig or not config.enabled:
            raise ValueError("an enabled exact SSTConfig is required")
        if type(reference_id) is not str or not reference_id.strip() or len(reference_id) > 512:
            raise ValueError("bounded reference model/checkpoint identity required")
        if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            raise ValueError("only native CPU SGD, Adam and AdamW are implemented")
        if not isinstance(targets, Mapping) or not 1 <= len(targets) <= 256:
            raise ValueError("explicit bounded named targets required")
        if any(type(name) is not str or not name.strip() or len(name) > 256 for name in targets):
            raise ValueError("invalid parameter name")
        self.optimizer = optimizer
        self.config = config
        self.reference_id = reference_id
        self.targets = dict(sorted(targets.items()))
        self._parameters = self._validate_parameters()
        owned = {id(parameter) for parameter in self._parameters}
        if any(
            id(parameter) not in owned or not parameter.requires_grad
            for parameter in targets.values()
        ):
            raise ValueError("targets must be trainable parameters owned by the optimizer")
        if len({id(parameter) for parameter in targets.values()}) != len(targets):
            raise ValueError("targets cannot alias one parameter")
        self._references = {
            name: parameter.detach().clone() for name, parameter in self.targets.items()
        }
        self._bases = {name: self._basis(value) for name, value in self._references.items()}
        self._step = 0
        self._basis_step = 0

    def _validate_parameters(self) -> tuple[Any, ...]:
        """Reject modes where a local transactional CPU step is not implemented."""
        import torch

        parameters = []
        storage_ids = set()
        total = 0
        for group in self.optimizer.param_groups:
            if any(group.get(flag, False) for flag in ("fused", "capturable", "differentiable")):
                raise ValueError("fused, capturable and differentiable optimizers are unsupported")
            for parameter in group["params"]:
                if (
                    type(parameter) is not torch.nn.Parameter
                    or parameter.device.type != "cpu"
                    or parameter.dtype not in (torch.float32, torch.float64)
                    or parameter.layout != torch.strided
                    or not parameter.is_contiguous()
                    or parameter.numel() == 0
                ):
                    raise ValueError("dense nonempty CPU float32/float64 Parameters required")
                storage = parameter.untyped_storage()
                if (
                    parameter.storage_offset() != 0
                    or storage.nbytes() != parameter.numel() * parameter.element_size()
                    or storage.data_ptr() in storage_ids
                ):
                    raise ValueError(
                        "shared storage, views and duplicate optimizer parameters forbidden"
                    )
                storage_ids.add(storage.data_ptr())
                total += parameter.numel()
                if total > self.config.max_total_elements:
                    raise ValueError("optimizer parameter snapshot budget exceeded")
                if not torch.isfinite(parameter).all().item():
                    raise ValueError("optimizer parameters must be finite")
                parameters.append(parameter)
        if not parameters:
            raise ValueError("optimizer has no parameters")
        return tuple(parameters)

    def _basis(self, reference: Any, frozen_reason: str | None = None) -> dict[str, Any]:
        """Compute exact float64 SVD and expose decomposition accuracy, or freeze explicitly."""
        import torch

        reason = frozen_reason
        if reference.ndim != 2 or reference.numel() > self.config.max_matrix_elements:
            reason = reason or "unsupported matrix shape or matrix element budget"
        if reason is None:
            try:
                matrix = reference.to(torch.float64)
                left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
                residual = float(torch.linalg.vector_norm((left * singular) @ right - matrix))
                scale = max(float(torch.linalg.vector_norm(matrix)), 1e-300)
                rank = math.floor(self.config.rank_fraction * min(matrix.shape))
                result = {
                    "left": left[:, :rank].clone(),
                    "right": right[:rank].T.contiguous(),
                    "svd_relative_residual": residual / scale,
                    "rank": rank,
                    "reason": None,
                }
                if not all(torch.isfinite(result[key]).all().item() for key in ("left", "right")):
                    raise RuntimeError("nonfinite singular vectors")
                if not math.isfinite(result["svd_relative_residual"]):
                    raise RuntimeError("nonfinite SVD residual")
                return result
            except RuntimeError as error:
                reason = f"decomposition unavailable: {type(error).__name__}"
        if self.config.fallback != "freeze":
            raise ValueError(reason)
        return {
            "left": None,
            "right": None,
            "rank": None,
            "svd_relative_residual": None,
            "reason": reason,
        }

    def _project(self, delta: Any, basis: dict[str, Any]) -> Any:
        """Apply two-sided projection to the optimizer delta, never to its input gradient."""
        import torch

        delta = delta.to(torch.float64)
        if basis["reason"] is not None:
            return torch.zeros_like(delta)
        left, right = basis["left"], basis["right"]
        projected = delta - left @ (left.T @ delta)
        return projected - (projected @ right) @ right.T

    def step(self) -> dict[str, object]:
        """Run and project one step; restore parameters and native state on failure."""
        import torch

        current = self._validate_parameters()
        if tuple(map(id, current)) != tuple(map(id, self._parameters)):
            raise ValueError("optimizer parameter groups changed after reference capture")
        if not any(parameter.grad is not None for parameter in current):
            raise ValueError("no gradients available for an SST optimizer step")
        for parameter in current:
            gradient = parameter.grad
            if gradient is not None and (
                gradient.layout != torch.strided or not torch.isfinite(gradient).all().item()
            ):
                raise ValueError("dense finite gradients required")
        before = [parameter.detach().clone() for parameter in current]
        native_state = copy.deepcopy(self.optimizer.state_dict())
        by_id = {id(parameter): original for parameter, original in zip(current, before)}
        metrics = {}
        next_references, next_bases = self._references, self._bases
        refresh = (
            self.config.refresh_every is not None
            and (self._step + 1) % self.config.refresh_every == 0
        )
        try:
            self.optimizer.step()
            with torch.no_grad():
                for name, parameter in self.targets.items():
                    original = by_id[id(parameter)]
                    raw = parameter.detach().to(torch.float64) - original.to(torch.float64)
                    if not torch.isfinite(raw).all().item():
                        raise ValueError("nonfinite optimizer update")
                    basis = self._bases[name]
                    if basis["reason"] is not None or basis["rank"] == min(parameter.shape):
                        parameter.copy_(original)
                    elif basis["rank"] != 0:
                        projected = self._project(raw, basis)
                        parameter.copy_(
                            (original.to(torch.float64) + projected).to(parameter.dtype)
                        )
                    actual = parameter.to(torch.float64) - original.to(torch.float64)
                    norm = float(torch.linalg.vector_norm(actual))
                    left, right = basis["left"], basis["right"]
                    metrics[name] = {
                        "status": "frozen" if basis["reason"] else "projected",
                        "reason": basis["reason"],
                        "protected_rank": basis["rank"],
                        "svd_relative_residual": basis["svd_relative_residual"],
                        "raw_update_norm": float(torch.linalg.vector_norm(raw)),
                        "applied_update_norm": norm,
                        "left_leakage": (
                            float(torch.linalg.vector_norm(left.T @ actual))
                            if left is not None
                            else 0.0
                        ),
                        "right_leakage": (
                            float(torch.linalg.vector_norm(actual @ right))
                            if right is not None
                            else 0.0
                        ),
                    }
                if not all(torch.isfinite(parameter).all().item() for parameter in current):
                    raise ValueError("nonfinite parameter after optimizer update")
                if refresh:
                    next_references = {
                        name: parameter.detach().clone() for name, parameter in self.targets.items()
                    }
                    next_bases = {
                        name: self._basis(value) for name, value in next_references.items()
                    }
                report = {
                    "schema_version": "sst-step-v1",
                    "training_eligibility": "DEVELOPMENT",
                    "authority_granted": False,
                    "source_reference": "REF-SST",
                    "implementation_version": "sst-cpu-v1",
                    "runtime_version": str(torch.__version__),
                    "reference_id": self.reference_id,
                    "config": self.config.to_dict(),
                    "optimizer": type(self.optimizer).__name__,
                    "step": self._step + 1,
                    "basis_step": self._basis_step,
                    "refreshed": refresh,
                    "unselected_parameter_count": len(current) - len(self.targets),
                    "targets": metrics,
                }
                json.dumps(report, allow_nan=False)
        except Exception:
            with torch.no_grad():
                for parameter, original in zip(current, before):
                    parameter.copy_(original)
            self.optimizer.load_state_dict(native_state)
            raise
        self._references, self._bases = next_references, next_bases
        self._step += 1
        if refresh:
            self._basis_step = self._step
        return report

    def state_dict(self) -> dict[str, object]:
        """Snapshot references and counters separately from model and native optimizer state."""
        return {
            "schema_version": "sst-state-v1",
            "reference_id": self.reference_id,
            "config": self.config.to_dict(),
            "step": self._step,
            "basis_step": self._basis_step,
            "references": {name: value.clone() for name, value in self._references.items()},
            "frozen": {
                name: basis["reason"] for name, basis in self._bases.items() if basis["reason"]
            },
        }

    def load_state_dict(self, value: object) -> None:
        """Validate every checkpoint field before replacing adapter state; grant no authority."""
        import torch

        keys = {
            "schema_version",
            "reference_id",
            "config",
            "step",
            "basis_step",
            "references",
            "frozen",
        }
        if (
            type(value) is not dict
            or set(value) != keys
            or value["schema_version"] != "sst-state-v1"
        ):
            raise ValueError("invalid SST checkpoint schema")
        if (
            value["reference_id"] != self.reference_id
            or SSTConfig.from_dict(value["config"]) != self.config
        ):
            raise ValueError("SST checkpoint reference or configuration mismatch")
        step, basis_step = value["step"], value["basis_step"]
        if type(step) is not int or step < 0 or type(basis_step) is not int:
            raise ValueError("invalid SST step counters")
        interval = self.config.refresh_every
        if basis_step != (step - step % interval if interval else 0):
            raise ValueError("SST basis counter does not match refresh cadence")
        references, frozen = value["references"], value["frozen"]
        if type(references) is not dict or set(references) != set(self.targets):
            raise ValueError("SST checkpoint target mismatch")
        if type(frozen) is not dict or not set(frozen) <= set(self.targets):
            raise ValueError("invalid frozen target records")
        if any(
            type(reason) is not str or not 1 <= len(reason) <= 512 for reason in frozen.values()
        ):
            raise ValueError("invalid frozen target reason")
        restored = {}
        for name, reference in references.items():
            target = self.targets[name]
            if (
                type(reference) is not torch.Tensor
                or reference.shape != target.shape
                or reference.dtype != target.dtype
                or reference.device.type != "cpu"
                or reference.layout != torch.strided
                or not torch.isfinite(reference).all().item()
            ):
                raise ValueError("invalid SST reference tensor")
            restored[name] = reference.detach().clone()
        bases = {
            name: self._basis(reference, frozen.get(name)) for name, reference in restored.items()
        }
        self._references, self._bases = restored, bases
        self._step, self._basis_step = step, basis_step

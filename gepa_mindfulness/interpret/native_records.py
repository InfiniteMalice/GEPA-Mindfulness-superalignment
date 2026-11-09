"""Experimental native-component observations; all records remain diagnostic and inert."""

# Standard library
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

# Local
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_enum,
    _read_records,
    _record,
    array,
    digest,
    enum,
    integer,
    text,
    texts,
)


class HookKind(str, Enum):
    MLP_PRE = "mlp_pre"
    MLP_POST = "mlp_post"
    MLP_UP = "mlp_up"
    MLP_PRODUCT = "mlp_product"
    ATTENTION_HEAD = "attention_head"
    MLP_RESIDUAL = "mlp_residual"
    ATTENTION_RESIDUAL = "attention_residual"
    RESIDUAL_STREAM = "residual_stream"


def index(value: object, *, maximum: int = 1_000_000) -> None:
    integer(value)
    if type(value) is not int or not 0 <= value < maximum:
        raise ValueError("index outside supported bounds")


def finite(value: object) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("expected exact finite float")


def indices(value: object, maximum: int) -> tuple[int, ...]:
    result = array(value, maximum, required=True)
    for item in result:
        index(item)
    if len(set(result)) != len(result):
        raise ValueError("duplicate indices")
    return result


@dataclass(frozen=True)
class NativeModelIdentity(CVARecord):
    model_id: str
    revision: str
    architecture: str
    state_digest: str
    torch_version: str
    transformers_version: str
    parameter_count: int

    schema_version = "native-model-identity-v1"

    def __post_init__(self) -> None:
        for field in (
            "model_id",
            "revision",
            "architecture",
            "torch_version",
            "transformers_version",
        ):
            text(getattr(self, field), field)
        digest(self.state_digest)
        integer(self.parameter_count)
        if self.parameter_count <= 0:
            raise ValueError("positive parameter_count required")


@dataclass(frozen=True)
class HookSpec(CVARecord):
    architecture: str
    layer: int
    kind: HookKind
    module_path: str
    location: str
    transformation: str
    normalization: str
    width: int
    heads: int
    head_dim: int
    upstream_paths: tuple[str, ...]
    downstream_paths: tuple[str, ...]

    schema_version = "native-hook-spec-v1"
    restorers = {
        "kind": lambda v: _read_enum(v, HookKind),
        "upstream_paths": _read_array,
        "downstream_paths": _read_array,
    }

    def __post_init__(self) -> None:
        for field in ("architecture", "module_path", "transformation", "normalization"):
            text(getattr(self, field), field, 2048)
        index(self.layer, maximum=1024)
        enum(self.kind, HookKind)
        if self.location not in ("input", "output", "output[0]"):
            raise ValueError("unsupported hook location")
        for value in (self.width, self.heads, self.head_dim):
            index(value)
        if self.width == 0:
            raise ValueError("hook requires nonzero tensor width")
        if self.kind is HookKind.ATTENTION_HEAD:
            if self.heads == 0 or self.head_dim == 0 or self.heads * self.head_dim != self.width:
                raise ValueError("attention hook requires exact head geometry")
        elif self.heads or self.head_dim:
            raise ValueError("head geometry applies only to attention-head hooks")
        texts(self.upstream_paths, required=True)
        texts(self.downstream_paths, required=True)


@dataclass(frozen=True)
class NativeInput(CVARecord):
    """Permitted prediction context; never contains target activations or held-out outputs."""

    example_id: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    schema_version = "native-public-input-v1"
    restorers = {"token_ids": _read_array, "attention_mask": _read_array}

    def __post_init__(self) -> None:
        text(self.example_id, "example_id")
        for item in array(self.token_ids, 128, required=True):
            index(item)
        array(self.attention_mask, 128, required=True)
        if len(self.token_ids) != len(self.attention_mask):
            raise ValueError("mask does not match tokens")
        if any(type(v) is not int or v not in (0, 1) for v in self.attention_mask):
            raise ValueError("mask requires exact binary integers")
        if not any(self.attention_mask):
            raise ValueError("empty input attention is unsupported")


@dataclass(frozen=True)
class NativeMeasurement(CVARecord):
    model_digest: str
    hook_digest: str
    input_digest: str
    example_id: str
    token_position: int
    component_index: int
    activation: tuple[float, ...]
    context_tokens: tuple[int, ...]
    attention_mask: tuple[int, ...]

    schema_version = "native-measurement-v1"
    restorers = {
        "activation": _read_array,
        "context_tokens": _read_array,
        "attention_mask": _read_array,
    }

    def __post_init__(self) -> None:
        for value in (self.model_digest, self.hook_digest, self.input_digest):
            digest(value)
        context = NativeInput(self.example_id, self.context_tokens, self.attention_mask)
        if context.digest != self.input_digest:
            raise ValueError("measurement input digest does not match recorded context")
        index(self.token_position, maximum=len(self.context_tokens))
        if not self.attention_mask[self.token_position]:
            raise ValueError("measurement at a masked token")
        index(self.component_index)
        for item in array(self.activation, 512, required=True):
            finite(item)


@dataclass(frozen=True)
class NativeBatch(CVARecord):
    model: NativeModelIdentity
    hook: HookSpec
    inputs: tuple[NativeInput, ...]
    component_indices: tuple[int, ...]
    token_positions: tuple[int, ...]
    tensor_shape: tuple[int, ...]
    dtype: str
    device: str
    measurements: tuple[NativeMeasurement, ...]
    evidence_status: str = "DISCOVERED_CORRELATION"

    schema_version = "native-measurement-batch-v1"
    restorers = {
        "model": NativeModelIdentity.from_dict,
        "hook": HookSpec.from_dict,
        "inputs": lambda v: _read_records(v, NativeInput),
        "component_indices": _read_array,
        "token_positions": _read_array,
        "tensor_shape": _read_array,
        "measurements": lambda v: _read_records(v, NativeMeasurement),
    }

    def __post_init__(self) -> None:
        model = _record(self.model, NativeModelIdentity)
        hook = _record(self.hook, HookSpec)
        if self.evidence_status != "DISCOVERED_CORRELATION":
            raise ValueError("native observation cannot self-certify predictive or causal status")
        if model.architecture != hook.architecture:
            raise ValueError("hook/model architecture mismatch")
        contexts = tuple(_record(i, NativeInput) for i in array(self.inputs, 8, required=True))
        indexed = {i.example_id: i for i in contexts}
        if len(indexed) != len(contexts):
            raise ValueError("duplicate example identities")
        comps = indices(self.component_indices, 64)
        positions = indices(self.token_positions, 128)
        width = hook.heads if hook.kind is HookKind.ATTENTION_HEAD else hook.width
        if any(c >= width for c in comps):
            raise ValueError("component outside hook geometry")
        shape = array(self.tensor_shape, 3, required=True)
        for value in shape:
            index(value)
        if len(shape) != 3 or shape[0] != len(contexts) or shape[2] != hook.width:
            raise ValueError("incorrect captured tensor shape")
        if any(len(i.token_ids) != shape[1] for i in contexts) or max(positions) >= shape[1]:
            raise ValueError("incorrect token geometry")
        text(self.dtype, "dtype")
        text(self.device, "device")
        if self.dtype not in ("torch.float16", "torch.float32", "torch.float64", "torch.bfloat16"):
            raise ValueError("unsupported activation dtype")
        if re.fullmatch(r"cpu|cuda:[0-9]+", self.device) is None:
            raise ValueError("unsupported activation device")
        expected = {
            (i.example_id, p, c)
            for i in contexts
            for p in positions
            for c in comps
            if i.attention_mask[p]
        }
        observed = set()
        model_digest, hook_digest = model.digest, hook.digest
        for raw in array(self.measurements, 2048, required=True):
            measurement = _record(raw, NativeMeasurement)
            key = (measurement.example_id, measurement.token_position, measurement.component_index)
            context = indexed.get(measurement.example_id)
            if key in observed or key not in expected or context is None:
                raise ValueError("foreign or duplicate measurement coordinates")
            if (
                measurement.model_digest != model_digest
                or measurement.hook_digest != hook_digest
                or measurement.input_digest != context.digest
            ):
                raise ValueError("measurement provenance mismatch")
            expected_width = hook.head_dim if hook.kind is HookKind.ATTENTION_HEAD else 1
            if len(measurement.activation) != expected_width:
                raise ValueError("measurement component shape mismatch")
            observed.add(key)
        if observed != expected:
            raise ValueError("missing measurement coordinates")


@dataclass(frozen=True)
class SamplingThresholds(CVARecord):
    near_zero: float
    medium: float
    high: float

    schema_version = "native-sampling-thresholds-v1"

    def __post_init__(self) -> None:
        for value in (self.near_zero, self.medium, self.high):
            finite(value)
        if not 0 <= self.near_zero < self.medium < self.high:
            raise ValueError("thresholds require 0 <= near_zero < medium < high")

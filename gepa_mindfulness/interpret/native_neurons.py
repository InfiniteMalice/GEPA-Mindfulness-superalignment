"""Experimental native inspection for explicitly supported, copied GPT-2/Llama models.

Hooks observe actual tensors, not gradients or causal effects. Token IDs are supplied by
the host; no tokenizer, model, external endpoint or executable content is downloaded.
"""

# Standard library
from __future__ import annotations

import copy
import json
import random
from hashlib import sha256
from typing import Any

# Local
from gepa_mindfulness.verification.cva_records import _record, texts

from .native_records import (
    HookKind,
    HookSpec,
    NativeBatch,
    NativeInput,
    NativeMeasurement,
    NativeModelIdentity,
    SamplingThresholds,
    indices,
)


def _enabled(enabled: bool) -> None:
    if enabled is not True:
        raise ValueError("native research requires enabled=True")


def _validate_structure(model: Any, family: str) -> None:
    """Validate the supported internal implementation, not just the outer model class."""
    import torch
    from transformers.activations import ACT2FN
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention, GPT2Block
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
    )
    from transformers.pytorch_utils import Conv1D

    if model.config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("unsupported internal attention implementation")
    blocks = model.transformer.h if family == "gpt2" else model.model.layers
    for block in blocks:
        if family == "gpt2":
            expected: tuple[tuple[Any, type], ...] = (
                (block, GPT2Block),
                (block.mlp, GPT2MLP),
                (block.attn, GPT2Attention),
                (block.ln_1, torch.nn.LayerNorm),
                (block.ln_2, torch.nn.LayerNorm),
                (block.mlp.c_fc, Conv1D),
                (block.mlp.c_proj, Conv1D),
                (block.mlp.act, type(ACT2FN[model.config.activation_function])),
            )
        else:
            expected = (
                (block, LlamaDecoderLayer),
                (block.mlp, LlamaMLP),
                (block.self_attn, LlamaAttention),
                (block.input_layernorm, LlamaRMSNorm),
                (block.post_attention_layernorm, LlamaRMSNorm),
                (block.mlp.gate_proj, torch.nn.Linear),
                (block.mlp.up_proj, torch.nn.Linear),
                (block.mlp.down_proj, torch.nn.Linear),
                (block.mlp.act_fn, type(ACT2FN[model.config.hidden_act])),
            )
        if any(type(module) is not kind for module, kind in expected):
            raise ValueError("unsupported internal module substitution")
    if any("forward" in vars(module) for module in model.modules()):
        raise ValueError("unsupported internal instance forward override")


def _json_attributes(module: Any) -> dict[str, Any]:
    """Capture ordinary forward-affecting scalar attributes of approved library modules."""
    result = {}
    for name, value in vars(module).items():
        if value is None or type(value) in (str, bool, int, float):
            result[name] = value
        elif type(value) in (tuple, list) and all(
            v is None or type(v) in (str, bool, int, float) for v in value
        ):
            result[name] = list(value)
    result["instance_forward_override"] = "forward" in vars(module)
    return result


def _state_digest(model: Any) -> str:
    """Bind dense state, buffers, module semantics and approved runtime settings.

    Transformers omits attention implementation from config serialization. Record it
    explicitly; class/ordinary attributes bind activation, normalization and dropout
    semantics that state_dict alone omits. Arbitrary Python/class monkeypatching is not
    a supported backend or a security sandbox.
    """
    import torch

    digest = sha256(model.config.to_json_string(use_diff=False).encode())
    semantics = {
        "attention_implementation": model.config._attn_implementation,
        "modules": [
            {
                "path": name,
                "class": f"{type(module).__module__}.{type(module).__qualname__}",
                "attributes": _json_attributes(module),
            }
            for name, module in model.named_modules()
        ],
    }
    digest.update(json.dumps(semantics, sort_keys=True, allow_nan=False).encode())
    tensors = [("state:" + name, t) for name, t in model.state_dict().items()]
    tensors += [("buffer:" + name, t) for name, t in model.named_buffers(remove_duplicate=False)]
    for name, tensor in sorted(tensors):
        if (
            tensor.layout is not torch.strided
            or tensor.is_quantized
            or tensor.device.type == "meta"
        ):
            raise ValueError("unsupported non-dense, quantized or meta model state")
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


class FrozenNativeModel:
    """A copied frozen inspection backend, not a training or model-modification interface.

    Defaults cap cloning at 50M parameters. CPU software probes qualify the hook paths;
    compatible single-device CUDA execution remains empirically unqualified here.
    Arbitrary hostile Python or user-defined forward code is outside the threat model.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        revision: str,
        max_parameters: int = 50_000_000,
        enabled: bool = False,
    ) -> None:
        _enabled(enabled)
        import torch
        import transformers
        from transformers import GPT2LMHeadModel, LlamaForCausalLM

        if type(model) is GPT2LMHeadModel:
            self.architecture = "gpt2"
        elif type(model) is LlamaForCausalLM:
            self.architecture = "llama"
        else:
            raise ValueError(
                "unsupported architecture; only exact GPT2LMHeadModel/LlamaForCausalLM"
            )
        _validate_structure(model, self.architecture)
        if type(max_parameters) is not int or not 1 <= max_parameters <= 1_000_000_000:
            raise ValueError("parameter budget requires integer in 1..1e9")
        count = sum(p.numel() for p in model.parameters())
        if count > max_parameters:
            raise ValueError("model exceeds declared parameter budget before copying")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("unsupported model device layout")
        if any(
            m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()
        ):
            raise ValueError("pre-existing model hooks are unsupported")
        self._model = copy.deepcopy(model).eval()
        self._model.requires_grad_(False)
        self._device = next(iter(devices))
        self.identity = NativeModelIdentity(
            model_id,
            revision,
            self.architecture,
            _state_digest(self._model),
            str(torch.__version__),
            transformers.__version__,
            count,
        )

    def _assert_frozen(self) -> None:
        if (
            any(m.training for m in self._model.modules())
            or any(
                m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                for m in self._model.modules()
            )
            or any(p.requires_grad for p in self._model.parameters())
            or _state_digest(self._model) != self.identity.state_digest
        ):
            raise ValueError("frozen model state changed; construct a separately identified copy")

    def hook(self, *, layer: int, kind: HookKind) -> HookSpec:
        """Describe an exact native tensor. No generic architecture/name guessing."""
        config = self._model.config
        layers = config.n_layer if self.architecture == "gpt2" else config.num_hidden_layers
        if type(layer) is not int or not 0 <= layer < layers:
            raise ValueError("layer outside model")
        if type(kind) is not HookKind:
            raise ValueError("unsupported hook kind")
        hidden = config.n_embd if self.architecture == "gpt2" else config.hidden_size
        heads = config.n_head if self.architecture == "gpt2" else config.num_attention_heads
        base = f"transformer.h.{layer}" if self.architecture == "gpt2" else f"model.layers.{layer}"
        mlp = self._model.get_submodule(base + ".mlp")
        attn = ".attn" if self.architecture == "gpt2" else ".self_attn"
        norm = base + (".ln_2" if self.architecture == "gpt2" else ".post_attention_layernorm")
        norm_module = self._model.get_submodule(norm)
        epsilon = getattr(norm_module, "eps", getattr(norm_module, "variance_epsilon", None))
        normalization = f"pre-MLP {type(norm_module).__name__} at {norm}, epsilon={epsilon}"
        if self.architecture == "gpt2":
            inner = mlp.c_fc.weight.shape[1]
            activation = type(mlp.act).__name__
            table = {
                HookKind.MLP_PRE: (".mlp.c_fc", "output", "linear preactivation", inner),
                HookKind.MLP_POST: (
                    ".mlp.c_proj",
                    "input",
                    f"{activation} postactivation; no gating",
                    inner,
                ),
                HookKind.ATTENTION_HEAD: (
                    attn + ".c_proj",
                    "input",
                    "concatenated head output",
                    hidden,
                ),
            }
            upstream: tuple[str, ...] = (norm, base + ".mlp.c_fc")
            downstream = (base + ".mlp.c_proj", base)
        else:
            inner = mlp.gate_proj.weight.shape[0]
            activation = type(mlp.act_fn).__name__
            head_dim = getattr(config, "head_dim", None) or hidden // heads
            table = {
                HookKind.MLP_PRE: (".mlp.gate_proj", "output", "gate linear preactivation", inner),
                HookKind.MLP_POST: (
                    ".mlp.act_fn",
                    "output",
                    f"{activation} gate postactivation, not product",
                    inner,
                ),
                HookKind.MLP_UP: (".mlp.up_proj", "output", "linear up branch, not product", inner),
                HookKind.MLP_PRODUCT: (
                    ".mlp.down_proj",
                    "input",
                    f"{activation}(gate) multiplied by up branch",
                    inner,
                ),
                HookKind.ATTENTION_HEAD: (
                    attn + ".o_proj",
                    "input",
                    "concatenated query-head output; GQA preserved",
                    heads * head_dim,
                ),
            }
            upstream = (norm, base + ".mlp.gate_proj", base + ".mlp.up_proj")
            downstream = (base + ".mlp.down_proj", base)
        table.update(
            {
                HookKind.MLP_RESIDUAL: (
                    ".mlp",
                    "output",
                    "MLP contribution before residual addition",
                    hidden,
                ),
                HookKind.ATTENTION_RESIDUAL: (
                    attn,
                    "output[0]",
                    "attention contribution before residual addition",
                    hidden,
                ),
                HookKind.RESIDUAL_STREAM: (
                    "",
                    "input",
                    "residual stream before layer normalization",
                    hidden,
                ),
            }
        )
        if kind not in table:
            raise ValueError("unsupported hook for this architecture")
        suffix, location, transformation, width = table[kind]
        path = base + suffix
        self._model.get_submodule(path)
        head_dim = width // heads if kind is HookKind.ATTENTION_HEAD else 0
        if kind in (HookKind.ATTENTION_HEAD, HookKind.ATTENTION_RESIDUAL):
            attention_norm = base + (".ln_1" if self.architecture == "gpt2" else ".input_layernorm")
            normalization = (
                f"pre-attention {type(self._model.get_submodule(attention_norm)).__name__}"
            )
            upstream = (attention_norm, base + attn)
            downstream = (
                base + attn + (".c_proj" if self.architecture == "gpt2" else ".o_proj"),
                base,
            )
        elif kind is HookKind.RESIDUAL_STREAM:
            normalization = "raw input before the layer's pre-normalization operations"
            upstream, downstream = (base,), (norm, base + attn)
        return HookSpec(
            self.architecture,
            layer,
            kind,
            path,
            location,
            transformation,
            normalization,
            int(width),
            int(heads) if head_dim else 0,
            int(head_dim),
            upstream,
            downstream,
        )

    def capture(
        self,
        hook: HookSpec,
        input_ids: Any,
        *,
        component_indices: tuple[int, ...],
        example_ids: tuple[str, ...],
        token_positions: tuple[int, ...],
        attention_mask: Any = None,
        enabled: bool = False,
    ) -> NativeBatch:
        _enabled(enabled)
        import torch

        selected = _record(hook, HookSpec)
        if selected != self.hook(layer=selected.layer, kind=selected.kind):
            raise ValueError("hook does not match verified architecture location")
        self._assert_frozen()
        if (
            type(input_ids) is not torch.Tensor
            or input_ids.dtype != torch.long
            or input_ids.device != self._device
            or input_ids.ndim != 2
        ):
            raise ValueError(
                "input_ids require exact LongTensor on model device with [batch,tokens]"
            )
        b, t = input_ids.shape
        if not 1 <= b <= 8 or not 1 <= t <= 128:
            raise ValueError("input exceeds 8 examples or 128 tokens")
        capacity = (
            self._model.config.n_positions
            if self.architecture == "gpt2"
            else self._model.config.max_position_embeddings
        )
        if t > capacity:
            raise ValueError("input exceeds declared architecture position capacity")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self._model.config.vocab_size:
            raise ValueError("input token outside vocabulary")
        positions = indices(token_positions, 128)
        components = indices(component_indices, 64)
        width = selected.heads if selected.kind is HookKind.ATTENTION_HEAD else selected.width
        if max(positions) >= t or max(components) >= width:
            raise ValueError("token position or component index outside captured geometry")
        texts(example_ids, required=True, maximum=8)
        if len(example_ids) != b or b * len(positions) * len(components) > 2048:
            raise ValueError("example identities or measurement count exceed bounds")
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if (
            type(mask) is not torch.Tensor
            or mask.device != self._device
            or mask.dtype != torch.long
            or mask.shape != input_ids.shape
            or not bool(((mask == 0) | (mask == 1)).all())
        ):
            raise ValueError("attention mask requires matching binary LongTensor")
        contexts = tuple(
            NativeInput(name, tuple(row), tuple(mask_row))
            for name, row, mask_row in zip(
                example_ids, input_ids.cpu().tolist(), mask.cpu().tolist()
            )
        )
        captured = []

        def save(value: Any) -> None:
            if type(value) is not torch.Tensor:
                raise ValueError("verified hook did not expose a tensor")
            if tuple(value.shape) != (b, t, selected.width) or not value.is_floating_point():
                raise ValueError("verified hook tensor shape/dtype mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite hook activation")
            captured.append(value.detach().clone())

        def before(module: Any, inputs: Any) -> None:
            save(inputs[0])

        def after(module: Any, inputs: Any, output: Any) -> None:
            if selected.location == "output[0]":
                if type(output) is not tuple or not output:
                    raise ValueError("verified hook requires tuple output")
                output = output[0]
            save(output)

        module = self._model.get_submodule(selected.module_path)
        handle = (
            module.register_forward_pre_hook(before)
            if selected.location == "input"
            else module.register_forward_hook(after)
        )
        try:
            with torch.no_grad():
                self._model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        finally:
            handle.remove()
        self._assert_frozen()
        if len(captured) != 1:
            raise ValueError("hook did not fire exactly once")
        values = captured[0]
        measurements = []
        model_digest, hook_digest = self.identity.digest, selected.digest
        for batch_index, context in enumerate(contexts):
            for position in positions:
                if not context.attention_mask[position]:
                    continue
                for component in components:
                    if selected.kind is HookKind.ATTENTION_HEAD:
                        start = component * selected.head_dim
                        activation = values[
                            batch_index, position, start : start + selected.head_dim
                        ]
                        vector = tuple(float(v) for v in activation.cpu().tolist())
                    else:
                        vector = (float(values[batch_index, position, component].item()),)
                    measurements.append(
                        NativeMeasurement(
                            model_digest,
                            hook_digest,
                            context.digest,
                            context.example_id,
                            position,
                            component,
                            vector,
                            context.token_ids,
                            context.attention_mask,
                        )
                    )
        return NativeBatch(
            self.identity,
            selected,
            contexts,
            components,
            positions,
            tuple(values.shape),
            str(values.dtype),
            str(values.device),
            tuple(measurements),
        )

    def component_weights(
        self,
        hook: HookSpec,
        *,
        component_index: int,
        enabled: bool = False,
    ) -> dict[str, Any]:
        """Inspect actual local writers/readers; structural weights are not causal evidence."""
        _enabled(enabled)
        selected = _record(hook, HookSpec)
        if selected != self.hook(layer=selected.layer, kind=selected.kind):
            raise ValueError("hook does not match verified architecture location")
        if selected.kind not in (
            HookKind.MLP_PRE,
            HookKind.MLP_POST,
            HookKind.MLP_UP,
            HookKind.MLP_PRODUCT,
        ):
            raise ValueError("unsupported component weight inspection hook")
        indices((component_index,), 1)
        if component_index >= selected.width:
            raise ValueError("component outside hook")
        self._assert_frozen()
        base = (
            f"transformer.h.{selected.layer}.mlp"
            if self.architecture == "gpt2"
            else f"model.layers.{selected.layer}.mlp"
        )
        mlp = self._model.get_submodule(base)
        upstream = []
        writers: tuple[tuple[str, Any, str], ...]
        if self.architecture == "gpt2":
            writers = (("c_fc", mlp.c_fc.weight[:, component_index], "column"),)
            readers = (("c_proj", mlp.c_proj.weight[component_index, :], "row"),)
        else:
            writers = tuple(
                (name, getattr(mlp, name).weight[component_index, :], "row")
                for name in ("gate_proj", "up_proj")
            )
            readers = (("down_proj", mlp.down_proj.weight[:, component_index], "column"),)
        for name, tensor, axis in writers:
            upstream.append(
                {
                    "tensor": f"{base}.{name}.weight",
                    "axis": axis,
                    "values": tensor.detach().cpu().tolist(),
                }
            )
        return {
            "schema_version": "native-local-weights-v1",
            "maturity": "experimental",
            "training_eligibility": "DEVELOPMENT",
            "optimizer_input": False,
            "evidence_status": "DISCOVERED_CORRELATION",
            "model_digest": self.identity.digest,
            "hook_digest": selected.digest,
            "component_index": component_index,
            "upstream": upstream,
            "downstream": [
                {
                    "tensor": f"{base}.{name}.weight",
                    "axis": axis,
                    "values": tensor.detach().cpu().tolist(),
                }
                for name, tensor, axis in readers
            ],
            "limitations": "Local weights only; normalization, gating and other paths matter.",
        }


def sample_signed_regimes(
    batch: NativeBatch,
    *,
    discovery_ids: tuple[str, ...],
    seed: int,
    per_regime: int,
    thresholds: SamplingThresholds,
    enabled: bool = False,
) -> dict[str, Any]:
    """Stratify declared discovery data only; missing regimes remain explicitly missing."""
    _enabled(enabled)
    snapshot = _record(batch, NativeBatch)
    limits = _record(thresholds, SamplingThresholds)
    if len(snapshot.component_indices) != 1:
        raise ValueError("regime sampling requires a single component")
    texts(discovery_ids, required=True)
    if set(discovery_ids) != {i.example_id for i in snapshot.inputs}:
        raise ValueError("batch must contain exactly the declared discovery examples")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed requires uint32")
    if type(per_regime) is not int or not 1 <= per_regime <= 32:
        raise ValueError("per_regime requires integer in 1..32")
    if any(len(m.activation) != 1 for m in snapshot.measurements):
        raise ValueError("vector observations require an explicitly specified scalar analysis")
    names = (
        "positive_high",
        "negative_high",
        "positive_medium",
        "negative_medium",
        "positive_weak",
        "negative_weak",
        "near_zero",
        "nonactivating",
    )
    bins: dict[str, list[NativeMeasurement]] = {name: [] for name in names}
    for measurement in snapshot.measurements:
        value = measurement.activation[0]
        magnitude = abs(value)
        if value == 0:
            regime = "nonactivating"
        elif magnitude <= limits.near_zero:
            regime = "near_zero"
        else:
            sign = "positive" if value > 0 else "negative"
            strength = (
                "high"
                if magnitude >= limits.high
                else ("medium" if magnitude >= limits.medium else "weak")
            )
            regime = f"{sign}_{strength}"
        bins[regime].append(measurement)
    randomizer = random.Random(seed)
    return {
        "schema_version": "native-regime-sampling-v1",
        "maturity": "experimental",
        "training_eligibility": "DEVELOPMENT",
        "optimizer_input": False,
        "evidence_status": "DISCOVERED_CORRELATION",
        "source_batch_digest": snapshot.digest,
        "seed": seed,
        "thresholds": limits.to_dict(),
        "discovery_ids": list(discovery_ids),
        "regimes": {
            name: [r.to_dict() for r in randomizer.sample(values, min(per_regime, len(values)))]
            for name, values in bins.items()
        },
        "available_counts": {name: len(values) for name, values in bins.items()},
        "missing_regimes": [name for name, values in bins.items() if not values],
    }


def measurement_graph(
    batch: NativeBatch,
    *,
    prompt: str,
    response: str,
    enabled: bool = False,
) -> Any:
    """Join the existing attribution graph without inventing edges, gradients or causality."""
    _enabled(enabled)
    snapshot = _record(batch, NativeBatch)
    if len(snapshot.inputs) != 1:
        raise ValueError("one prompt-response graph requires a single example")
    if any(len(m.activation) != 1 for m in snapshot.measurements):
        raise ValueError("vector observations require a declared reduction; none is inferred")
    from .attribution_graphs import AttributionGraph, AttributionNode

    nodes = [
        AttributionNode(
            snapshot.hook.layer,
            snapshot.hook.kind.value,
            m.component_index,
            m.token_position,
            m.activation[0],
            0.0,
        )
        for m in snapshot.measurements
    ]
    return AttributionGraph(
        prompt,
        response,
        nodes,
        [],
        "native_activation_observation",
        {
            "evidence_status": "DISCOVERED_CORRELATION",
            "source_batch_digest": snapshot.digest,
            "model": snapshot.model.to_dict(),
            "hook": snapshot.hook.to_dict(),
            "observations": [m.to_dict() for m in snapshot.measurements],
            "limitations": "No gradient scores, prediction tests or causal interventions were run.",
        },
    )

"""Attribution Graph extraction helpers for transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import networkx as nx
except ImportError:  # pragma: no cover - executed when networkx missing
    nx = None  # type: ignore[assignment]

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class AttributionNode:
    """Single node in an attribution graph."""

    layer: int
    component_type: str
    component_idx: Optional[int]
    token_position: int
    activation_value: float
    attribution_score: float


@dataclass
class AttributionEdge:
    """Directed edge showing causal influence between nodes."""

    source_node: AttributionNode
    target_node: AttributionNode
    attribution_weight: float


@dataclass
class AttributionGraph:
    """Complete attribution graph for a prompt-response pair."""

    prompt: str
    response: str
    nodes: List[AttributionNode]
    edges: List[AttributionEdge]
    method: str
    metadata: Dict[str, Any]

    def to_networkx(self) -> nx.DiGraph:
        """Convert the graph to a :class:`networkx.DiGraph`."""

        _require_dependency("networkx", nx is not None)
        graph = nx.DiGraph()
        for index, node in enumerate(self.nodes):
            graph.add_node(
                index,
                layer=node.layer,
                type=node.component_type,
                position=node.token_position,
                activation=node.activation_value,
                attribution=node.attribution_score,
            )

        id_to_index = {id(node): index for index, node in enumerate(self.nodes)}
        for edge in self.edges:
            src = id_to_index.get(id(edge.source_node))
            tgt = id_to_index.get(id(edge.target_node))
            if src is None or tgt is None:
                continue
            graph.add_edge(src, tgt, weight=edge.attribution_weight)
        return graph


class AttributionGraphExtractor:
    """Extract attribution graphs from HuggingFace transformer models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        method: str = "gradient_x_activation",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialise the extractor and register activation hooks."""

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.method = method
        self.device = torch.device(device)
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[RemovableHandle] = []
        self._register_hooks()

    def close(self) -> None:
        """Remove all registered hooks to release resources."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks for attention and MLP blocks."""

        if self._hook_handles:
            return

        def make_forward_hook(name: str):
            def hook(module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                self.activations[name] = tensor.detach()

            return hook

        def make_backward_hook(name: str):
            def hook(
                module: nn.Module,
                _grad_inputs: Tuple[torch.Tensor, ...],
                grad_outputs: Tuple[torch.Tensor, ...],
            ) -> None:
                tensor = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
                self.gradients[name] = tensor.detach()

            return hook

        for name, module in self.model.named_modules():
            lowered = name.lower()
            if "attn" in lowered or "mlp" in lowered or "feed_forward" in lowered:
                forward_handle = module.register_forward_hook(make_forward_hook(name))
                backward_handle = module.register_full_backward_hook(make_backward_hook(name))
                self._hook_handles.extend([forward_handle, backward_handle])

    def extract(
        self,
        prompt: str,
        response: str,
        layers: Optional[Iterable[int]] = None,
        threshold: float = 0.01,
    ) -> AttributionGraph:
        """Extract an attribution graph for ``prompt`` and ``response``."""

        total_layers = self.model.config.num_hidden_layers
        if layers is None:
            layers = range(total_layers)
        layer_set = {
            self._normalise_layer_index(layer=layer, total_layers=total_layers) for layer in layers
        }

        self.activations.clear()
        self.gradients.clear()
        full_text = prompt + response
        encoded = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        self.model.eval()
        with torch.enable_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            logits = outputs.logits[0]
            shift_logits = logits[:-1]
            shift_labels = encoded.input_ids[0, 1:]
            response_len = encoded.input_ids.shape[1] - prompt_len
            start_index = 0 if prompt_len == 0 else prompt_len - 1
            end_index = min(start_index + response_len, shift_logits.shape[0])

            if end_index <= start_index:
                loss = logits.sum() * 0.0
            else:
                target_logits = shift_logits[start_index:end_index]
                target_labels = shift_labels[start_index:end_index]
                log_probs = torch.log_softmax(target_logits, dim=-1)
                indices = torch.arange(target_labels.shape[0], device=self.device)
                loss = -log_probs[indices, target_labels].sum()

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        nodes, edges = self._dispatch_method(
            inputs=encoded,
            outputs=outputs,
            prompt_len=prompt_len,
            layers=layer_set,
            threshold=threshold,
        )
        self.model.zero_grad(set_to_none=True)

        metadata = {
            "layers_analyzed": sorted(layer_set),
            "threshold": threshold,
            "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
        }
        return AttributionGraph(
            prompt=prompt,
            response=response,
            nodes=nodes,
            edges=edges,
            method=self.method,
            metadata=metadata,
        )

    def _dispatch_method(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        outputs: Any,
        prompt_len: int,
        layers: Iterable[int],
        threshold: float,
    ) -> Tuple[List[AttributionNode], List[AttributionEdge]]:
        method = self.method.lower()
        if method == "gradient_x_activation":
            return self._gradient_x_activation(layers=layers, threshold=threshold)
        if method == "path_integrated_gradients":
            return self._path_integrated_gradients(layers=layers, threshold=threshold)
        if method == "activation_patching":
            return self._activation_patching(
                inputs=inputs,
                prompt_len=prompt_len,
                layers=layers,
                threshold=threshold,
            )
        raise ValueError(f"unknown attribution method: {self.method}")

    def _gradient_x_activation(
        self,
        *,
        layers: Iterable[int],
        threshold: float,
    ) -> Tuple[List[AttributionNode], List[AttributionEdge]]:
        target_layers = set(layers)
        nodes: List[AttributionNode] = []
        for name, activation in self.activations.items():
            gradient = self.gradients.get(name)
            if gradient is None:
                continue

            layer_num = self._parse_layer_number(name)
            if layer_num not in target_layers:
                continue

            if activation.dim() < 3 or gradient.dim() < 3:
                continue

            attribution = (activation * gradient).abs().mean(dim=-1)
            activation_mean = activation.mean(dim=-1)
            batch_size, seq_len = attribution.shape[:2]
            if batch_size == 0:
                continue

            for position in range(seq_len):
                score = attribution[0, position].item()
                if score <= threshold:
                    continue
                node = AttributionNode(
                    layer=layer_num,
                    component_type=self._parse_component_type(name),
                    component_idx=self._parse_component_idx(name),
                    token_position=position,
                    activation_value=activation_mean[0, position].item(),
                    attribution_score=score,
                )
                nodes.append(node)

        edges = self._build_edges(nodes)
        return nodes, edges

    def _path_integrated_gradients(
        self,
        *,
        layers: Iterable[int],
        threshold: float,
    ) -> Tuple[List[AttributionNode], List[AttributionEdge]]:
        return self._gradient_x_activation(layers=layers, threshold=threshold)

    def _activation_patching(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        prompt_len: int,
        layers: Iterable[int],
        threshold: float,
    ) -> Tuple[List[AttributionNode], List[AttributionEdge]]:
        return self._gradient_x_activation(layers=layers, threshold=threshold)

    def _build_edges(self, nodes: List[AttributionNode]) -> List[AttributionEdge]:
        nodes_by_layer: Dict[int, List[AttributionNode]] = {}
        for node in nodes:
            nodes_by_layer.setdefault(node.layer, []).append(node)

        edges: List[AttributionEdge] = []
        for layer in sorted(nodes_by_layer):
            next_layer = layer + 1
            if next_layer not in nodes_by_layer:
                continue
            for src in nodes_by_layer[layer]:
                for tgt in nodes_by_layer[next_layer]:
                    if src.token_position != tgt.token_position:
                        continue
                    weight = src.attribution_score * tgt.attribution_score
                    edges.append(
                        AttributionEdge(
                            source_node=src,
                            target_node=tgt,
                            attribution_weight=weight,
                        )
                    )
        return edges

    @staticmethod
    def _normalise_layer_index(*, layer: int, total_layers: int) -> int:
        if layer < 0:
            return max(total_layers + layer, 0)
        return layer

    @staticmethod
    def _parse_layer_number(name: str) -> int:
        parts = name.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
            if part.startswith("layer") and part[5:].isdigit():
                return int(part[5:])
        return -1

    @staticmethod
    def _parse_component_type(name: str) -> str:
        lowered = name.lower()
        if "attn" in lowered or "attention" in lowered:
            return "attention"
        if "mlp" in lowered or "feed_forward" in lowered:
            return "mlp"
        return "residual"

    @staticmethod
    def _parse_component_idx(_name: str) -> Optional[int]:
        return None


def extract_attribution_graph(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    method: str = "gradient_x_activation",
    layers: Optional[Iterable[int]] = None,
    threshold: float = 0.01,
) -> AttributionGraph:
    """Convenience wrapper returning a single attribution graph."""

    extractor = AttributionGraphExtractor(
        model=model,
        tokenizer=tokenizer,
        method=method,
        device=model.device,
    )
    try:
        return extractor.extract(
            prompt=prompt,
            response=response,
            layers=layers,
            threshold=threshold,
        )
    finally:
        extractor.close()


def _require_dependency(name: str, available: bool) -> None:
    """Raise a helpful error when optional dependencies are missing."""

    if not available:
        raise RuntimeError(
            f"{name} is required for attribution graph utilities. "
            "Install the optional analytics extras to enable this feature."
        )

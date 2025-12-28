"""Attribution Graph extraction helpers for transformer models."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

_NETWORKX_SPEC = importlib.util.find_spec("networkx")
if _NETWORKX_SPEC is not None:
    nx = importlib.import_module("networkx")
else:  # pragma: no cover - optional dependency missing
    nx = None  # type: ignore[assignment]

_TORCH_SPEC = importlib.util.find_spec("torch")
_TRANSFORMER_SPEC = importlib.util.find_spec("transformers")
if _TORCH_SPEC is not None and _TRANSFORMER_SPEC is not None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    transformers = importlib.import_module("transformers")
    PreTrainedModel = transformers.PreTrainedModel
    PreTrainedTokenizer = transformers.PreTrainedTokenizer
else:  # pragma: no cover - used when optional deps missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    PreTrainedModel = Any  # type: ignore[misc]
    PreTrainedTokenizer = Any  # type: ignore[misc]


def _require_torch() -> None:
    """Ensure optional torch dependencies are available."""

    if torch is None:
        raise ImportError(
            "AttributionGraphExtractor requires torch and transformers to be installed."
        )


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "AttributionGraph.to_networkx() requires networkx to be installed."
        )


@dataclass
class AttributionNode:
    """Single node in an attribution graph."""

    layer: int
    component_type: str
    component_idx: int | None
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
    nodes: list[AttributionNode]
    edges: list[AttributionEdge]
    method: str
    metadata: dict[str, Any]

    def to_networkx(self) -> nx.DiGraph:
        """Convert the graph to a :class:`networkx.DiGraph`."""

        _require_networkx()
        assert nx is not None
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
        device: Any = None,
    ) -> None:
        """Initialise the extractor and register activation hooks."""

        _require_torch()
        assert torch is not None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.method = method
        self.activations: dict[str, Any] = {}
        self.gradients: dict[str, Any] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks for attention and MLP blocks."""

        def make_forward_hook(name: str):
            def hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                self.activations[name] = tensor.detach()

            return hook

        def make_backward_hook(name: str):
            def hook(
                module: nn.Module,
                _grad_inputs: tuple[torch.Tensor, ...],
                grad_outputs: tuple[torch.Tensor, ...],
            ) -> None:
                tensor = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
                self.gradients[name] = tensor.detach()

            return hook

        for name, module in self.model.named_modules():
            lowered = name.lower()
            if "attn" in lowered or "mlp" in lowered or "feed_forward" in lowered:
                module.register_forward_hook(make_forward_hook(name))
                module.register_full_backward_hook(make_backward_hook(name))

    def extract(
        self,
        prompt: str,
        response: str,
        layers: Iterable[int] | None = None,
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
            model_inputs = {key: value for key, value in encoded.items()}
            model_inputs["output_hidden_states"] = True
            if getattr(self.model.config, "use_cache", None) is not None:
                model_inputs["use_cache"] = False
            outputs = self.model(**model_inputs)
            logits = outputs.logits[0]
            resp_logits = logits[prompt_len:]
            resp_token_ids = encoded.input_ids[0, prompt_len:]
            log_probs = torch.log_softmax(resp_logits, dim=-1)
            indices = torch.arange(resp_token_ids.shape[0], device=self.device)
            loss = -log_probs[indices, resp_token_ids].sum()

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
        inputs: dict[str, torch.Tensor],
        outputs: Any,
        prompt_len: int,
        layers: Iterable[int],
        threshold: float,
    ) -> tuple[list[AttributionNode], list[AttributionEdge]]:
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
    ) -> tuple[list[AttributionNode], list[AttributionEdge]]:
        target_layers = set(layers)
        nodes: list[AttributionNode] = []
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
    ) -> tuple[list[AttributionNode], list[AttributionEdge]]:
        return self._gradient_x_activation(layers=layers, threshold=threshold)

    def _activation_patching(
        self,
        *,
        inputs: dict[str, torch.Tensor],
        prompt_len: int,
        layers: Iterable[int],
        threshold: float,
    ) -> tuple[list[AttributionNode], list[AttributionEdge]]:
        return self._gradient_x_activation(layers=layers, threshold=threshold)

    def _build_edges(self, nodes: list[AttributionNode]) -> list[AttributionEdge]:
        nodes_by_layer: dict[int, list[AttributionNode]] = {}
        for node in nodes:
            nodes_by_layer.setdefault(node.layer, []).append(node)

        edges: list[AttributionEdge] = []
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
    def _parse_component_idx(_name: str) -> int | None:
        return None


def extract_attribution_graph(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
    method: str = "gradient_x_activation",
    layers: Iterable[int] | None = None,
    threshold: float = 0.01,
) -> AttributionGraph:
    """Convenience wrapper returning a single attribution graph."""

    extractor = AttributionGraphExtractor(
        model=model,
        tokenizer=tokenizer,
        method=method,
        device=model.device,
    )
    return extractor.extract(
        prompt=prompt,
        response=response,
        layers=layers,
        threshold=threshold,
    )

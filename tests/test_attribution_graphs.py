"""Tests for attribution graph extraction."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("torch")

import torch

from gepa_mindfulness.interpret.attribution_graphs import (
    AttributionGraphExtractor,
    extract_attribution_graph,
)


def _prompt_response() -> tuple[str, str]:
    return "hello world ", "response"


def test_gradient_activation_nodes(tiny_model, dummy_tokenizer) -> None:
    prompt, response = _prompt_response()
    extractor = AttributionGraphExtractor(
        model=tiny_model,
        tokenizer=dummy_tokenizer,
        method="gradient_x_activation",
        device=torch.device("cpu"),
    )
    graph = extractor.extract(prompt=prompt, response=response, layers=[0, 1])
    assert graph.nodes, "Expected nodes to be extracted"
    layers = {node.layer for node in graph.nodes}
    assert layers <= {0, 1}
    for edge in graph.edges:
        assert edge.target_node.layer == edge.source_node.layer + 1


@pytest.mark.parametrize("method", ["path_integrated_gradients", "activation_patching"])
def test_unsupported_attribution_methods_raise_clear_error(
    tiny_model, dummy_tokenizer, method: str
) -> None:
    prompt, response = _prompt_response()
    with pytest.raises(NotImplementedError, match=method):
        extract_attribution_graph(
            model=tiny_model,
            tokenizer=dummy_tokenizer,
            prompt=prompt,
            response=response,
            method=method,
            layers=[0, 1],
        )


def test_extraction_runtime_reasonable(tiny_model, dummy_tokenizer) -> None:
    prompt, response = _prompt_response()
    inputs = dummy_tokenizer(prompt + response)
    tiny_model.eval()
    start = time.time()
    with torch.no_grad():
        tiny_model(**inputs)
    inference_time = time.time() - start

    extractor = AttributionGraphExtractor(
        model=tiny_model,
        tokenizer=dummy_tokenizer,
        method="gradient_x_activation",
        device=torch.device("cpu"),
    )
    start = time.time()
    extractor.extract(prompt=prompt, response=response, layers=[0, 1])
    extraction_time = time.time() - start
    assert extraction_time <= inference_time * 4

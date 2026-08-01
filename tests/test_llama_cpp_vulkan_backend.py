"""Local-server contract tests for the inference-only llama.cpp backend."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

from gepa_mindfulness.training.backends.llama_cpp_vulkan import (
    LlamaCppServerClient,
    LlamaCppServerError,
    LlamaCppVulkanBackend,
)
from gepa_mindfulness.training.capability import Capability, CapabilityState
from gepa_mindfulness.training.contracts import RolloutBackend
from gepa_mindfulness.training.trajectory import RolloutRequest


@dataclass(frozen=True)
class _Reply:
    status: int = 200
    body: object = None
    delay_seconds: float = 0.0


class _MockLlamaServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object | None]] = []
        self._replies: defaultdict[tuple[str, str], deque[_Reply]] = defaultdict(deque)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._serve(None)

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._serve(payload)

            def _serve(self, payload: object | None) -> None:
                key = (self.command, self.path)
                owner.requests.append((self.command, self.path, payload))
                reply = owner._replies[key].popleft()
                if reply.delay_seconds:
                    time.sleep(reply.delay_seconds)
                body = (
                    reply.body
                    if isinstance(reply.body, bytes)
                    else json.dumps(reply.body).encode("utf-8")
                )
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = cast(tuple[str, int], self._server.server_address)
        self.endpoint = f"http://{host}:{port}"

    def enqueue(
        self,
        method: str,
        path: str,
        body: object,
        *,
        status: int = 200,
        delay_seconds: float = 0.0,
    ) -> None:
        self._replies[(method, path)].append(_Reply(status, body, delay_seconds))

    def prime_metadata(self) -> None:
        self.enqueue("GET", "/health", {"status": "ok", "version": "b4242"})
        self.enqueue(
            "GET",
            "/v1/models",
            {
                "object": "list",
                "data": [
                    {
                        "id": "tiny-model.gguf",
                        "object": "model",
                        "owned_by": "llama.cpp",
                        "meta": {"format": "gguf"},
                    }
                ],
            },
        )

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def mock_llama_server() -> Iterator[_MockLlamaServer]:
    server = _MockLlamaServer()
    try:
        yield server
    finally:
        server.close()


def _completion(
    content: str,
    *,
    tokens: list[int] | None = None,
    probabilities: list[float] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "content": content,
        "model": "tiny-model.gguf",
        "stop": True,
    }
    if tokens is not None:
        body["tokens"] = tokens
    if probabilities is not None:
        pieces = content.split("|")
        assert len(pieces) == len(probabilities)
        body["completion_probabilities"] = [
            {
                "content": piece,
                "probs": [
                    {
                        "prob": probability,
                        "tok_str": piece,
                    }
                ],
            }
            for piece, probability in zip(pieces, probabilities, strict=True)
        ]
        body["content"] = "".join(pieces)
    return body


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://127.0.0.1:8080",
        "http://user:secret@127.0.0.1:8080",
        "http://127.0.0.1:8080/#fragment",
        "http://127.0.0.1:8080/?query=yes",
        "http://example.com:8080",
        "not a URL",
    ],
)
def test_client_rejects_unsafe_endpoint_before_request(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        LlamaCppServerClient(endpoint)


def test_client_normalizes_loopback_endpoint_and_validates_health_models(
    mock_llama_server: _MockLlamaServer,
) -> None:
    mock_llama_server.prime_metadata()
    client = LlamaCppServerClient(mock_llama_server.endpoint + "/")

    health = client.health()
    models = client.models()

    assert client.endpoint == mock_llama_server.endpoint
    assert health == {"status": "ok", "version": "b4242"}
    assert models == (
        {
            "id": "tiny-model.gguf",
            "object": "model",
            "owned_by": "llama.cpp",
            "meta": {"format": "gguf"},
        },
    )
    assert mock_llama_server.requests == [
        ("GET", "/health", None),
        ("GET", "/v1/models", None),
    ]


@pytest.mark.parametrize("timeout", [True, 0, -1, 61, math.inf, "1"])
def test_client_rejects_invalid_or_unbounded_timeout(timeout: object) -> None:
    with pytest.raises((TypeError, ValueError), match="timeout"):
        LlamaCppServerClient("http://127.0.0.1:8080", timeout_seconds=timeout)  # type: ignore[arg-type]


def test_missing_server_probabilities_remain_null(
    mock_llama_server: _MockLlamaServer,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue("POST", "/completion", _completion("hello back"))
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    trajectory = backend.generate([RolloutRequest(prompt="hello")])[0]

    assert trajectory.response == "hello back"
    assert trajectory.old_log_probs is None
    assert trajectory.response_token_ids is None
    assert trajectory.reference_log_probs is None
    assert trajectory.value_predictions is None


def test_grouped_completions_preserve_order_and_forward_effective_sampling(
    mock_llama_server: _MockLlamaServer,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue(
        "POST",
        "/completion",
        _completion("first| sample", tokens=[11, 12], probabilities=[0.5, 0.25]),
    )
    mock_llama_server.enqueue(
        "POST",
        "/completion",
        _completion("second", tokens=[13], probabilities=[0.125]),
    )
    mock_llama_server.enqueue("POST", "/completion", _completion("third", tokens=[14]))
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint, max_new_tokens=32)
    requests = [
        RolloutRequest(
            prompt="grouped",
            case_id="case-a",
            num_samples=2,
            sampling_parameters={
                "max_new_tokens": 7,
                "temperature": 0.4,
                "top_p": 0.8,
                "do_sample": True,
            },
            policy_version="policy-3",
            seed=40,
        ),
        RolloutRequest(prompt="single", case_id="case-b", seed=90),
    ]

    trajectories = backend.generate(requests)

    assert [trajectory.response for trajectory in trajectories] == [
        "first sample",
        "second",
        "third",
    ]
    assert [trajectory.trajectory_id for trajectory in trajectories] == [
        "case-a-policy-3-0",
        "case-a-policy-3-1",
        "case-b-unversioned-0",
    ]
    assert [trajectory.seed for trajectory in trajectories] == [40, 41, 90]
    assert trajectories[0].response_token_ids == (11, 12)
    assert trajectories[0].old_log_probs == pytest.approx((math.log(0.5), math.log(0.25)))
    assert trajectories[2].response_token_ids == (14,)
    assert trajectories[2].old_log_probs is None
    assert trajectories[0].sampling_parameters == {
        "do_sample": True,
        "max_new_tokens": 7,
        "temperature": 0.4,
        "top_p": 0.8,
    }
    completion_payloads = [
        payload
        for method, path, payload in mock_llama_server.requests
        if method == "POST" and path == "/completion"
    ]
    assert completion_payloads == [
        {
            "n_predict": 7,
            "n_probs": 1,
            "prompt": "grouped",
            "seed": 40,
            "stream": False,
            "temperature": 0.4,
            "top_p": 0.8,
        },
        {
            "n_predict": 7,
            "n_probs": 1,
            "prompt": "grouped",
            "seed": 41,
            "stream": False,
            "temperature": 0.4,
            "top_p": 0.8,
        },
        {
            "n_predict": 32,
            "n_probs": 1,
            "prompt": "single",
            "seed": 90,
            "stream": False,
        },
    ]


def test_capabilities_use_only_observed_server_evidence(
    mock_llama_server: _MockLlamaServer,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue(
        "POST",
        "/completion",
        _completion("observed", tokens=[7], probabilities=[0.5]),
    )
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    before = backend.capabilities()
    backend.generate([RolloutRequest(prompt="hello")])
    after = backend.capabilities()

    assert isinstance(backend, RolloutBackend)
    assert before.state(Capability.SUPPORTS_GENERATION) is CapabilityState.UNKNOWN
    assert after.state(Capability.SUPPORTS_GENERATION) is CapabilityState.SUPPORTED
    assert after.state(Capability.SUPPORTS_GGUF) is CapabilityState.SUPPORTED
    assert after.state(Capability.SUPPORTS_TOKEN_LOG_PROBS) is CapabilityState.SUPPORTED
    assert after.state(Capability.SUPPORTS_VULKAN) is CapabilityState.UNKNOWN
    for capability in (
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_VALUE_HEAD,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
        Capability.SUPPORTS_LORA_TRAINING,
        Capability.SUPPORTS_MIXED_PRECISION,
        Capability.SUPPORTS_DISTRIBUTED_TRAINING,
        Capability.SUPPORTS_CUDA,
        Capability.SUPPORTS_REFERENCE_LOG_PROBS,
    ):
        assert after.state(capability) is CapabilityState.UNSUPPORTED


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"content": ["not", "text"]}, "content"),
        ({"content": "x", "tokens": [True]}, "tokens"),
        (
            {
                "content": "x",
                "tokens": [1, 2],
                "completion_probabilities": [
                    {"content": "x", "probs": [{"prob": 0.5, "tok_str": "x"}]}
                ],
            },
            "align",
        ),
        (
            {
                "content": "x",
                "completion_probabilities": [
                    {"content": "x", "probs": [{"prob": float("nan"), "tok_str": "x"}]}
                ],
            },
            "finite",
        ),
    ],
)
def test_completion_rejects_malformed_or_shape_inconsistent_payload(
    mock_llama_server: _MockLlamaServer,
    body: object,
    message: str,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue("POST", "/completion", body)
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    with pytest.raises(LlamaCppServerError, match=message):
        backend.generate([RolloutRequest(prompt="hello")])


def test_empty_probability_array_does_not_claim_token_probability_evidence(
    mock_llama_server: _MockLlamaServer,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue(
        "POST",
        "/completion",
        {"content": "x", "completion_probabilities": []},
    )
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    with pytest.raises(LlamaCppServerError, match="probabilities.*non-empty"):
        backend.generate([RolloutRequest(prompt="hello")])

    assert (
        backend.capabilities().state(Capability.SUPPORTS_TOKEN_LOG_PROBS) is CapabilityState.UNKNOWN
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"\xff\xfe", "UTF-8"),
        (b"not json", "JSON"),
        (b'{"content": NaN}', "non-finite"),
    ],
)
def test_completion_rejects_non_utf8_non_json_and_non_finite_json(
    mock_llama_server: _MockLlamaServer,
    body: bytes,
    message: str,
) -> None:
    mock_llama_server.prime_metadata()
    mock_llama_server.enqueue("POST", "/completion", body)
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    with pytest.raises(LlamaCppServerError, match=message):
        backend.generate([RolloutRequest(prompt="hello")])


def test_client_rejects_oversized_response(mock_llama_server: _MockLlamaServer) -> None:
    mock_llama_server.enqueue("GET", "/health", {"padding": "x" * 500})
    client = LlamaCppServerClient(mock_llama_server.endpoint, max_response_bytes=100)

    with pytest.raises(LlamaCppServerError, match="100 bytes"):
        client.health()


def test_client_translates_http_failure(mock_llama_server: _MockLlamaServer) -> None:
    mock_llama_server.enqueue("GET", "/health", {"error": "unavailable"}, status=503)
    client = LlamaCppServerClient(mock_llama_server.endpoint)

    with pytest.raises(LlamaCppServerError, match="HTTP 503.*health"):
        client.health()


def test_client_translates_timeout(mock_llama_server: _MockLlamaServer) -> None:
    mock_llama_server.enqueue("GET", "/health", {"status": "ok"}, delay_seconds=0.1)
    client = LlamaCppServerClient(mock_llama_server.endpoint, timeout_seconds=0.02)

    with pytest.raises(LlamaCppServerError, match="timed out.*health"):
        client.health()


def test_close_is_idempotent_and_prevents_more_requests(
    mock_llama_server: _MockLlamaServer,
) -> None:
    client = LlamaCppServerClient(mock_llama_server.endpoint)

    client.close()
    client.close()

    with pytest.raises(RuntimeError, match="closed"):
        client.health()
    assert mock_llama_server.requests == []


def test_backend_rejects_invalid_request_before_server_call(
    mock_llama_server: _MockLlamaServer,
) -> None:
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    with pytest.raises(ValueError, match="prompt"):
        backend.generate([RolloutRequest(prompt="   ")])

    assert mock_llama_server.requests == []


def test_backend_close_is_idempotent(mock_llama_server: _MockLlamaServer) -> None:
    backend = LlamaCppVulkanBackend(mock_llama_server.endpoint)

    backend.close()
    backend.close()

    with pytest.raises(RuntimeError, match="closed"):
        backend.generate([RolloutRequest(prompt="hello")])

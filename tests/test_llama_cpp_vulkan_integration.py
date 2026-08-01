"""Runtime-detection tests for optional local llama.cpp and Vulkan tools."""

from __future__ import annotations

import io
import json
import os
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

import gepa_mindfulness.training.backends.llama_cpp_vulkan as llama_backend
from gepa_mindfulness.training.backends.llama_cpp_vulkan import (
    detect_llama_cpp_runtime,
    detect_vulkan_evidence,
)
from gepa_mindfulness.training.capability import (
    Capability,
    CapabilityError,
    CapabilityState,
)


class _MetadataServer:
    def __init__(self, model: dict[str, object] | None = None) -> None:
        owner = self
        self.requests: list[str] = []
        self.model = model or {
            "id": "detected.gguf",
            "meta": {"format": "gguf"},
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                owner.requests.append(self.path)
                if self.path == "/health":
                    payload: dict[str, object] = {
                        "status": "ok",
                        "version": "server-b4242",
                    }
                elif self.path == "/v1/models":
                    payload = {"data": [owner.model]}
                else:
                    self.send_error(404)
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = cast(tuple[str, int], self._server.server_address)
        self.endpoint = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        running: bool = False,
    ) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.running = running
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return None if self.running else self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        if self.running:
            raise subprocess.TimeoutExpired("fake", 0)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.running = False
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.running = False
        self.returncode = -9


@pytest.fixture
def metadata_server() -> Iterator[_MetadataServer]:
    server = _MetadataServer()
    try:
        yield server
    finally:
        server.close()


def test_executable_probe_reports_sanitized_build_evidence_without_starting_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = os.path.abspath("tools/llama-server.exe")

    def fake_which(name: str) -> str | None:
        return executable if name == "llama-server" else None

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        if command != [executable, "--version"]:
            raise AssertionError("runtime detection may invoke only llama-server --version")
        if kwargs.get("shell") is not False:
            raise AssertionError("runtime probe must be shell-free")
        return _FakeProcess(
            stdout=b"llama.cpp build 4242\ncommit abc123\x00ignored",
            stderr=b"",
        )

    monkeypatch.setattr(llama_backend.shutil, "which", fake_which)
    monkeypatch.setattr(llama_backend.subprocess, "Popen", fake_popen)

    report = detect_llama_cpp_runtime()

    generation = report.capabilities[Capability.SUPPORTS_GENERATION]
    assert generation.state is CapabilityState.UNKNOWN
    assert "llama.cpp build 4242 commit abc123 ignored" in generation.evidence
    assert "\n" not in generation.evidence
    assert report.backend_version == "llama.cpp build 4242 commit abc123 ignored"
    assert report.state(Capability.SUPPORTS_VULKAN) is CapabilityState.UNKNOWN


def test_malformed_llama_version_output_is_not_reported_as_a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llama_backend.shutil,
        "which",
        lambda name: "llama-server" if name == "llama-server" else None,
    )
    monkeypatch.setattr(
        llama_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(stdout=b"unrelated output"),
    )

    report = detect_llama_cpp_runtime()

    assert report.backend_version == "unknown"
    assert (
        "usable llama.cpp build evidence"
        in report.capabilities[Capability.SUPPORTS_GENERATION].evidence
    )


def test_llama_build_with_explicit_vulkan_flag_is_positive_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llama_backend.shutil,
        "which",
        lambda name: "llama-server" if name == "llama-server" else None,
    )
    monkeypatch.setattr(
        llama_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(stdout=b"llama.cpp build 4242 GGML_VULKAN=1"),
    )

    report = detect_llama_cpp_runtime()

    assert report.state(Capability.SUPPORTS_VULKAN) is CapabilityState.SUPPORTED
    assert "GGML_VULKAN=1" in report.capabilities[Capability.SUPPORTS_VULKAN].evidence


def test_vulkaninfo_summary_is_sanitized_positive_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llama_backend.shutil,
        "which",
        lambda name: "vulkaninfo" if name == "vulkaninfo" else None,
    )
    monkeypatch.setattr(
        llama_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(
            stdout=b"Vulkan Instance Version: 1.3.280\nGPU0: Mock Device"
        ),
    )

    evidence = detect_vulkan_evidence()

    assert evidence.state is CapabilityState.SUPPORTED
    assert evidence.evidence == (
        "vulkaninfo --summary reported Vulkan Instance Version: 1.3.280 GPU0: Mock Device"
    )


@pytest.mark.parametrize("failure", ["missing", "timeout", "nonzero", "malformed"])
def test_vulkan_probe_failures_return_actionable_unknown_evidence(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "missing":
        monkeypatch.setattr(llama_backend.shutil, "which", lambda name: None)
    else:
        monkeypatch.setattr(llama_backend.shutil, "which", lambda name: "vulkaninfo")

        def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
            del command
            del kwargs
            if failure == "timeout":
                return _FakeProcess(running=True)
            if failure == "nonzero":
                return _FakeProcess(returncode=1, stderr=b"driver unavailable\n")
            return _FakeProcess(stdout=b"unrelated output")

        monkeypatch.setattr(llama_backend.subprocess, "Popen", fake_popen)

    evidence = detect_vulkan_evidence()

    assert evidence.state is CapabilityState.UNKNOWN
    assert any(
        term in evidence.evidence.lower()
        for term in ("install", "timed out", "failed", "positive vulkan")
    )


def test_reachable_endpoint_does_not_prove_vulkan(
    metadata_server: _MetadataServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llama_backend.shutil, "which", lambda name: None)

    report = detect_llama_cpp_runtime(endpoint=metadata_server.endpoint)

    assert report.state(Capability.SUPPORTS_GENERATION) is CapabilityState.SUPPORTED
    assert report.state(Capability.SUPPORTS_GGUF) is CapabilityState.SUPPORTED
    assert report.state(Capability.SUPPORTS_TOKEN_LOG_PROBS) is CapabilityState.UNKNOWN
    assert report.state(Capability.SUPPORTS_VULKAN) is CapabilityState.UNKNOWN
    assert metadata_server.requests == ["/health", "/v1/models"]


@pytest.mark.parametrize(
    "model",
    [
        {"id": "suffix-only.gguf"},
        {"id": "conflict.gguf", "meta": {"format": "safetensors"}},
    ],
)
def test_gguf_suffix_without_trusted_format_metadata_remains_unknown(
    model: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _MetadataServer(model)
    monkeypatch.setattr(llama_backend.shutil, "which", lambda name: None)
    try:
        report = detect_llama_cpp_runtime(endpoint=server.endpoint)
    finally:
        server.close()

    assert report.state(Capability.SUPPORTS_GGUF) is CapabilityState.UNKNOWN


def test_probe_caps_combined_output_and_cleans_up_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(
        stdout=b"x" * (llama_backend._MAX_PROBE_OUTPUT_BYTES + 1),
        running=True,
    )
    monkeypatch.setattr(llama_backend.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        llama_backend.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unbounded subprocess.run capture must not be used")
        ),
    )

    probe = llama_backend._run_read_only_probe(
        ["llama-server", "--version"],
        timeout_seconds=2.0,
        display_name="llama-server --version",
    )

    assert probe.output is None
    assert "output limit" in cast(str, probe.failure)
    assert process.terminated
    assert process.waited


def test_probe_timeout_terminates_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(running=True)
    monkeypatch.setattr(llama_backend.subprocess, "Popen", lambda *args, **kwargs: process)

    probe = llama_backend._run_read_only_probe(
        ["llama-server", "--version"],
        timeout_seconds=0.01,
        display_name="llama-server --version",
    )

    assert probe.output is None
    assert "timed out" in cast(str, probe.failure)
    assert process.terminated
    assert process.waited


def test_training_capabilities_remain_unsupported_before_training_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llama_backend.shutil, "which", lambda name: None)
    report = detect_llama_cpp_runtime()

    for capability in (
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_VALUE_HEAD,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
        Capability.SUPPORTS_LORA_TRAINING,
    ):
        assert report.state(capability) is CapabilityState.UNSUPPORTED
    with pytest.raises(CapabilityError, match="supports_backward"):
        report.require({Capability.SUPPORTS_BACKWARD})


def test_native_llama_cpp_endpoint_is_opt_in_and_never_starts_a_service() -> None:
    endpoint = os.environ.get("GEPA_LLAMA_CPP_ENDPOINT")
    if endpoint is None:
        pytest.skip("GEPA_LLAMA_CPP_ENDPOINT is not configured")

    report = detect_llama_cpp_runtime(endpoint=endpoint)

    assert report.state(Capability.SUPPORTS_GENERATION) is CapabilityState.SUPPORTED

"""Process-boundary tests for the Mojo actor coordinator protocol."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Mapping, Sequence, cast

import pytest
import torch
from torch import nn

from gepa_mindfulness.training import engine as rl_engine
from gepa_mindfulness.training import rl_cli
from gepa_mindfulness.training.adapter_publication import (
    AdapterCandidate,
    AdapterManifest,
    LocalAdapterPublisher,
)
from gepa_mindfulness.training.backends import TorchPolicyBackend
from gepa_mindfulness.training.backends.mojo_coordinator import (
    ActorHandshake,
    MojoCoordinatorBackend,
    MojoCoordinatorError,
    MojoProcessTransport,
)
from gepa_mindfulness.training.capability import Capability, CapabilityState
from gepa_mindfulness.training.contracts import ActorTransport, RolloutBackend
from gepa_mindfulness.training.engine import (
    RLTrainingEngine,
    _config_hash,
    _config_payload,
    build_hybrid_engine,
)
from gepa_mindfulness.training.policy_versions import PolicyVersion, StalenessPolicy
from gepa_mindfulness.training.runtime_config import (
    AlgorithmConfig,
    CheckpointConfig,
    DatasetConfig,
    HybridConfig,
    LoggingConfig,
    PolicyConfig,
    RLRunConfig,
    RuntimeConfig,
    load_rl_config,
)
from gepa_mindfulness.training.trajectory import RolloutRequest, Trajectory

_FAKE_COORDINATOR = r"""import json
import os
import sys
import time

mode = sys.argv[1]


def read_message():
    line = sys.stdin.buffer.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def send(kind, request_id, payload):
    message = {
        "protocol_version": "gepa-actor-v1",
        "type": kind,
        "request_id": request_id,
        "payload": payload,
    }
    sys.stdout.write(json.dumps(message, allow_nan=False) + "\n")
    sys.stdout.flush()


hello = read_message()
if mode == "timeout":
    time.sleep(10)
    raise SystemExit(0)
if mode == "crash_handshake":
    sys.stderr.write("private crash detail\n")
    raise SystemExit(7)
if mode == "invalid_utf8":
    sys.stdout.buffer.write(b"\xff\n")
    sys.stdout.buffer.flush()
    raise SystemExit(0)
if mode == "invalid_json":
    sys.stdout.write("not-json\n")
    sys.stdout.flush()
    raise SystemExit(0)
if mode == "partial":
    sys.stdout.write('{"protocol_version":"gepa-actor-v1"}')
    sys.stdout.flush()
    raise SystemExit(0)
if mode == "oversized_stdout":
    sys.stdout.write("x" * 2048 + "\n")
    sys.stdout.flush()
    raise SystemExit(0)
if mode == "oversized_stderr":
    sys.stderr.write("s" * 2048)
    sys.stderr.flush()
if mode == "bad_handshake":
    send("hello", hello["request_id"], {"backend_name": "fake", "extra": True})
    raise SystemExit(0)
if mode == "unsafe_handshake":
    send(
        "hello",
        hello["request_id"],
        {"backend_name": "fake\nname", "backend_version": "fake-1"},
    )
    raise SystemExit(0)
if mode == "duplicate_envelope_key":
    raw = (
        '{"protocol_version":"gepa-actor-v1","type":"hello","type":"hello",'
        '"request_id":"%s","payload":{"backend_name":"fake-mojo",'
        '"backend_version":"fake-1"}}\n' % hello["request_id"]
    )
    sys.stdout.write(raw)
    sys.stdout.flush()
    raise SystemExit(0)
if mode == "duplicate_nested_key":
    raw = (
        '{"protocol_version":"gepa-actor-v1","type":"hello",'
        '"request_id":"%s","payload":{"backend_name":"fake-mojo",'
        '"backend_name":"forged","backend_version":"fake-1"}}\n'
        % hello["request_id"]
    )
    sys.stdout.write(raw)
    sys.stdout.flush()
    raise SystemExit(0)
if mode in ("future_frame", "duplicate_frame"):
    first = {
        "protocol_version": "gepa-actor-v1",
        "type": "hello",
        "request_id": hello["request_id"],
        "payload": {"backend_name": "fake-mojo", "backend_version": "fake-1"},
    }
    second = dict(first)
    if mode == "future_frame":
        second["type"] = "close"
        second["request_id"] = "request-2"
        second["payload"] = {}
    sys.stdout.write(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    sys.stdout.flush()
    time.sleep(10)
    raise SystemExit(0)
send(
    "hello",
    hello["request_id"],
    {"backend_name": "fake-mojo", "backend_version": "fake-1"},
)
if mode == "stop_reading":
    time.sleep(10)
    raise SystemExit(0)


def trajectory(request, request_index, sample_index):
    metadata = request["metadata"]
    hybrid_echo = mode == "hybrid_echo"
    return {
        "trajectory_id": "fake-%d-%d" % (request_index, sample_index),
        "case_id": request["case_id"],
        "prompt": request["prompt"],
        "response": (
            ("chosen" if sample_index == 0 else "rejected")
            if hybrid_echo
            else "response-%d-%d" % (request_index, sample_index)
        ),
        "prompt_token_ids": [2, 3] if hybrid_echo else None,
        "response_token_ids": [4 if sample_index == 0 else 5] if hybrid_echo else None,
        "old_log_probs": [-1.0] if hybrid_echo else None,
        "reference_log_probs": [-1.0] if hybrid_echo else None,
        "value_predictions": None,
        "reward_total": None,
        "reward_components": {},
        "reward_component_evidence": {},
        "advantage": None,
        "return": None,
        "sampling_parameters": request["sampling_parameters"],
        "backend_name": "fake-mojo",
        "backend_version": "fake-1",
        "model_identifier": (
            metadata["model_identifier"] if hybrid_echo else "fake-model"
        ),
        "adapter_identifier": (
            metadata["adapter_identifier"] if hybrid_echo else None
        ),
        "adapter_sha256": metadata.get("adapter_sha256") if hybrid_echo else None,
        "policy_version": request["policy_version"],
        "seed": request["seed"],
        "trace_references": [],
    }


while True:
    message = read_message()
    if message["type"] == "close":
        send("close", message["request_id"], {})
        if mode == "ignore_close":
            time.sleep(10)
        raise SystemExit(0)
    if mode == "crash_generate":
        sys.stderr.write("secret crash during generate\n")
        raise SystemExit(9)
    if mode == "error_secret":
        prompt = message["payload"]["requests"][0]["prompt"]
        secret = os.environ.get("COORDINATOR_TEST_SECRET", "missing")
        send(
            "error",
            message["request_id"],
            {"code": "actor_unconfigured", "message": prompt + secret},
        )
        continue
    requests = message["payload"]["requests"]
    trajectories = []
    for request_index, request in enumerate(requests):
        for sample_index in range(request["num_samples"]):
            trajectories.append(trajectory(request, request_index, sample_index))
    if mode == "wrong_policy":
        trajectories[0]["policy_version"] = "999"
    if mode == "wrong_backend_name":
        trajectories[0]["backend_name"] = "forged"
    if mode == "wrong_backend_version":
        trajectories[0]["backend_version"] = "forged"
    if mode == "wrong_order":
        trajectories.reverse()
    if mode == "duplicate" and len(trajectories) > 1:
        trajectories[1]["trajectory_id"] = trajectories[0]["trajectory_id"]
    if mode == "extra_field":
        trajectories[0]["unexpected"] = True
    if mode == "legacy_no_hash":
        trajectories[0].pop("adapter_sha256")
    request_id = message["request_id"]
    if mode == "wrong_request_id":
        request_id = "request-999"
    counter, separator, nonce = request_id.partition(":")
    if mode in ("wrong_nonce", "guessed_nonce"):
        request_id = counter + ":" + "0" * 32
    if mode == "malformed_request_id":
        request_id = counter + ":not-canonical"
    if mode == "stale_request_id":
        request_id = "request-1:" + (nonce if separator else "0" * 32)
    if mode == "future_request_id":
        request_id = "request-999:" + (nonce if separator else "0" * 32)
    if mode == "counter_only_request_id":
        request_id = counter
    if mode == "missing_request_id":
        response = {
            "protocol_version": "gepa-actor-v1",
            "type": "generate",
            "payload": {"trajectories": trajectories},
        }
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()
        continue
    send("generate", request_id, {"trajectories": trajectories})
"""


@dataclass(frozen=True)
class FakeCoordinator:
    command: tuple[str, ...]


class InvalidHandshakeTransport:
    """Structurally valid transport that returns an invalid startup value."""

    def __init__(self) -> None:
        self.closed = False

    def start(self) -> object:
        return object()

    def generate(
        self,
        requests: Sequence[Mapping[str, object]],
    ) -> Sequence[Mapping[str, object]]:
        raise AssertionError("generate must not run after an invalid handshake")

    def close(self) -> None:
        self.closed = True


class BlockedWriteStream:
    """Test boundary that makes request-write ordering observable."""

    def __init__(self, wrapped: object) -> None:
        self.wrapped = wrapped
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, data: bytes) -> int:
        self.entered.set()
        if not self.release.wait(1.0):
            raise OSError("test write was not released")
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


class DelimiterBlockingStream:
    """Forward the JSON prefix but block its framing delimiter until process exit."""

    def __init__(self, wrapped: object, process: subprocess.Popen[bytes]) -> None:
        self.wrapped = wrapped
        self.process = process
        self.delimiter_entered = threading.Event()

    def write(self, data: bytes) -> int:
        if bytes(data) == b"\n":
            self.delimiter_entered.set()
            self.process.wait()
            raise BrokenPipeError("test delimiter pipe closed")
        write = getattr(self.wrapped, "write")
        result = write(data)
        assert isinstance(result, int)
        return result

    def flush(self) -> None:
        flush = getattr(self.wrapped, "flush")
        flush()

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


class ReleasableDelimiterStream:
    """Pause the final delimiter and forward it only after an explicit release."""

    def __init__(self, wrapped: object) -> None:
        self.wrapped = wrapped
        self.delimiter_entered = threading.Event()
        self.release = threading.Event()

    def write(self, data: bytes) -> int:
        if bytes(data) == b"\n":
            self.delimiter_entered.set()
            if not self.release.wait(1.0):
                raise OSError("test delimiter was not released")
        write = getattr(self.wrapped, "write")
        result = write(data)
        assert isinstance(result, int)
        return result

    def flush(self) -> None:
        flush = getattr(self.wrapped, "flush")
        flush()

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


class ShortWriteStream:
    """Forward at most three bytes per call and record the exact delivered frame."""

    def __init__(self, wrapped: object) -> None:
        self.wrapped = wrapped
        self.delivered = bytearray()

    def write(self, data: bytes) -> int:
        chunk = bytes(data)[:3]
        write = getattr(self.wrapped, "write")
        result = write(chunk)
        assert isinstance(result, int)
        self.delivered.extend(chunk[:result])
        return result

    def flush(self) -> None:
        flush = getattr(self.wrapped, "flush")
        flush()

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


class InvalidWriteProgressStream:
    """Return one impossible raw-pipe progress value without forwarding data."""

    def __init__(self, wrapped: object, progress: object) -> None:
        self.wrapped = wrapped
        self.progress = progress

    def write(self, data: bytes) -> object:
        if self.progress == "oversized":
            return len(data) + 1
        return self.progress

    def flush(self) -> None:
        return None

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


class CorrelationObservingStream:
    """Record which bytes become visible before and after correlation eligibility."""

    def __init__(self, wrapped: object, transport: MojoProcessTransport) -> None:
        self.wrapped = wrapped
        self.transport = transport
        self.before_armed = bytearray()
        self.after_armed = bytearray()

    def write(self, data: bytes) -> int:
        with self.transport._expectation_lock:
            destination = self.after_armed if self.transport._commit_pending else self.before_armed
        write = getattr(self.wrapped, "write")
        result = write(data)
        assert isinstance(result, int)
        destination.extend(bytes(data)[:result])
        return result

    def flush(self) -> None:
        flush = getattr(self.wrapped, "flush")
        flush()

    def close(self) -> None:
        close = getattr(self.wrapped, "close")
        close()


@pytest.fixture
def fake_coordinator_factory(tmp_path: Path) -> Iterator[object]:
    script = tmp_path / "fake_coordinator.py"
    script.write_text(_FAKE_COORDINATOR, encoding="utf-8")
    transports: list[MojoProcessTransport] = []

    def factory(mode: str = "happy", **kwargs: object) -> MojoProcessTransport:
        options: dict[str, object] = {
            "startup_timeout_seconds": 2.0,
            "request_timeout_seconds": 2.0,
            "shutdown_timeout_seconds": 0.2,
        }
        options.update(kwargs)
        transport = MojoProcessTransport(
            (sys.executable, "-u", str(script), mode),
            **options,  # type: ignore[arg-type]
        )
        transports.append(transport)
        return transport

    yield factory
    for transport in transports:
        transport.close()


@pytest.mark.parametrize(
    "command",
    [(), ("",), (sys.executable, 7), ("definitely-missing-mojo-command",)],
)
def test_transport_rejects_invalid_command(command: object) -> None:
    """The process boundary never coerces arguments or falls back to a shell."""
    with pytest.raises((TypeError, ValueError), match="command|executable"):
        MojoProcessTransport(command)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        ("wrong", "fake", "1"),
        ("gepa-actor-v1", "", "1"),
        ("gepa-actor-v1", "fake\nname", "1"),
        ("gepa-actor-v1", "x" * 129, "1"),
        ("gepa-actor-v1", "fake", ""),
    ],
)
def test_handshake_constructor_rejects_forged_or_unsafe_identity(
    values: tuple[str, str, str],
) -> None:
    """Handshake evidence cannot be constructed with forged protocol or unsafe identity."""
    with pytest.raises(ValueError):
        ActorHandshake(*values)


def test_transport_requires_typed_handshake_and_closes_idempotently(
    fake_coordinator_factory: object,
) -> None:
    """Startup observes the versioned hello response before requests are allowed."""
    transport = fake_coordinator_factory()  # type: ignore[operator]

    handshake = transport.start()
    transport.close()
    transport.close()

    assert handshake.protocol_version == "gepa-actor-v1"
    assert handshake.backend_name == "fake-mojo"
    assert handshake.backend_version == "fake-1"
    assert transport.closed is True
    assert isinstance(transport, ActorTransport)
    with pytest.raises(RuntimeError, match="closed"):
        transport.generate(())


def test_backend_restores_ordered_version_bound_trajectories(
    fake_coordinator_factory: object,
) -> None:
    """The backend binds every returned sample to its originating request evidence."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport)
    requests = (
        RolloutRequest(
            prompt="first",
            case_id="case-1",
            num_samples=2,
            sampling_parameters={"temperature": 0.5},
            policy_version="3",
            seed=7,
        ),
        RolloutRequest(prompt="second", case_id="case-2", policy_version="4"),
    )

    trajectories = backend.generate(requests)

    assert [item.trajectory_id for item in trajectories] == ["fake-0-0", "fake-0-1", "fake-1-0"]
    assert [item.prompt for item in trajectories] == ["first", "first", "second"]
    assert [item.case_id for item in trajectories] == ["case-1", "case-1", "case-2"]
    assert [item.policy_version for item in trajectories] == ["3", "3", "4"]


def test_backend_rejects_handshake_that_differs_from_configured_actor_identity(
    fake_coordinator_factory: object,
) -> None:
    transport = fake_coordinator_factory()  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport, expected_backend_name="expected-mojo")

    with pytest.raises(MojoCoordinatorError, match="configured actor backend identity"):
        backend.generate((RolloutRequest(prompt="prompt", policy_version="1"),))

    assert transport.closed is True


def test_protocol_v1_response_without_adapter_hash_remains_compatible(
    fake_coordinator_factory: object,
) -> None:
    transport = fake_coordinator_factory("legacy_no_hash")  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport, expected_backend_name="fake-mojo")

    trajectories = backend.generate((RolloutRequest(prompt="prompt", policy_version="1"),))

    assert len(trajectories) == 1
    assert trajectories[0].adapter_sha256 is None
    assert backend.capabilities().backend_name == "fake-mojo"
    assert isinstance(backend, RolloutBackend)
    report = backend.capabilities()
    assert report.state(Capability.SUPPORTS_GENERATION) is CapabilityState.SUPPORTED
    assert report.state(Capability.SUPPORTS_BACKWARD) is CapabilityState.UNSUPPORTED
    assert report.state(Capability.SUPPORTS_OPTIMIZER_STEP) is CapabilityState.UNSUPPORTED


@pytest.mark.parametrize("policy_version", [None, "", "01", "policy-1"])
def test_backend_rejects_missing_or_noncanonical_policy_before_start(
    fake_coordinator_factory: object,
    policy_version: str | None,
) -> None:
    """Every actor request carries one canonical policy version before process startup."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport)

    with pytest.raises(ValueError, match="policy_version"):
        backend.generate((RolloutRequest(prompt="secret", policy_version=policy_version),))

    assert transport.started is False


@pytest.mark.parametrize(
    ("mode", "message", "num_samples"),
    [
        ("wrong_request_id", "request ID", 1),
        ("wrong_policy", "policy_version", 1),
        ("wrong_order", "order|prompt|case", 1),
        ("duplicate", "duplicate", 2),
        ("extra_field", "schema", 1),
    ],
)
def test_backend_rejects_uncorrelated_or_malformed_trajectory_response(
    fake_coordinator_factory: object,
    mode: str,
    message: str,
    num_samples: int,
) -> None:
    """Correlation and common-schema mismatches fail before trajectory acceptance."""
    transport = fake_coordinator_factory(mode)  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport)
    request = RolloutRequest(
        prompt="private prompt",
        case_id="case-1",
        num_samples=num_samples,
        policy_version="3",
    )
    requests: tuple[RolloutRequest, ...] = (request,)
    if mode == "wrong_order":
        requests = (
            request,
            RolloutRequest(prompt="second prompt", case_id="case-2", policy_version="3"),
        )

    with pytest.raises(MojoCoordinatorError, match=message):
        backend.generate(requests)

    assert transport.closed is True
    assert transport.process_running is False
    with pytest.raises(RuntimeError, match="closed"):
        backend.generate(requests)


@pytest.mark.parametrize("mode", ["wrong_backend_name", "wrong_backend_version"])
def test_backend_binds_trajectory_identity_to_handshake(
    fake_coordinator_factory: object,
    mode: str,
) -> None:
    """A coordinator cannot forge trajectory backend provenance after its handshake."""
    transport = fake_coordinator_factory(mode)  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport)

    with pytest.raises(MojoCoordinatorError, match="backend.*identity|provenance"):
        backend.generate((RolloutRequest(prompt="prompt", policy_version="3"),))

    assert transport.closed is True


@pytest.mark.parametrize("mode", ["future_frame", "duplicate_frame"])
def test_transport_rejects_pres_sent_or_duplicate_response_frames(
    fake_coordinator_factory: object,
    mode: str,
) -> None:
    """Only one frame may satisfy the active expectation established before a write."""
    transport = fake_coordinator_factory(mode)  # type: ignore[operator]

    with pytest.raises(MojoCoordinatorError, match="unsolicited|duplicate|response"):
        transport.start()


def test_transport_rejects_response_observed_before_request_write_commits(
    fake_coordinator_factory: object,
) -> None:
    """A request-N frame observed during a blocked write cannot satisfy that request."""
    transport = fake_coordinator_factory("stop_reading")  # type: ignore[operator]
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    blocked = BlockedWriteStream(transport._process.stdin)
    transport._process.stdin = blocked  # type: ignore[assignment]
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            transport.generate((_raw_actor_request(),))
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=generate)
    caller.start()
    assert blocked.entered.wait(1.0)
    raw = (
        b'{"protocol_version":"gepa-actor-v1","type":"generate",'
        b'"request_id":"request-2","payload":{"trajectories":[]}}'
    )
    transport._put_stdout_frame(raw)
    blocked.release.set()
    caller.join(timeout=1.0)

    assert not caller.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], MojoCoordinatorError)
    assert "write" in str(errors[0]) or "response" in str(errors[0])


def test_transport_accepts_immediate_response_after_request_flush(
    fake_coordinator_factory: object,
) -> None:
    """A response emitted immediately after flush observes the awaiting state."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    transport.start()

    trajectories = transport.generate((_raw_actor_request(),))

    assert len(trajectories) == 1


def test_delimiter_block_does_not_hold_response_lock_or_escape_cleanup(
    fake_coordinator_factory: object,
) -> None:
    """A blocked delimiter leaves state coordination available for timeout cleanup."""
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "happy",
        request_timeout_seconds=0.1,
    )
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    blocked = DelimiterBlockingStream(transport._process.stdin, transport._process)
    transport._process.stdin = blocked  # type: ignore[assignment]
    errors: list[BaseException] = []
    caller_done = threading.Event()

    def generate() -> None:
        try:
            transport.generate((_raw_actor_request(),))
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    caller = threading.Thread(target=generate)
    caller.start()
    assert blocked.delimiter_entered.wait(1.0)
    lock_available = transport._expectation_lock.acquire(timeout=0.05)
    if lock_available:
        transport._expectation_lock.release()
    returned_within_budget = caller_done.wait(0.4)
    if not returned_within_budget and transport.process_running:
        transport._process.terminate()
    caller.join(timeout=1.0)

    assert lock_available is True
    assert returned_within_budget is True
    assert not caller.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], MojoCoordinatorError)
    _assert_transport_resources_stopped(transport, writer_expected=True)


def test_pres_sent_counter_response_cannot_become_eligible_after_commit(
    fake_coordinator_factory: object,
) -> None:
    """A predictable request-N frame seen during commit poisons after a successful release."""
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "stop_reading",
        request_timeout_seconds=0.5,
    )
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    blocked = ReleasableDelimiterStream(transport._process.stdin)
    transport._process.stdin = blocked  # type: ignore[assignment]
    results: list[tuple[Mapping[str, object], ...]] = []
    errors: list[BaseException] = []
    response_observed = threading.Event()

    def generate() -> None:
        try:
            results.append(transport.generate((_raw_actor_request(),)))
        except BaseException as exc:
            errors.append(exc)

    raw = (
        b'{"protocol_version":"gepa-actor-v1","type":"generate",'
        b'"request_id":"request-2","payload":{"trajectories":[]}}'
    )

    def inject_response() -> None:
        response_observed.set()
        transport._put_stdout_frame(raw)

    caller = threading.Thread(target=generate)
    caller.start()
    assert blocked.delimiter_entered.wait(1.0)
    responder = threading.Thread(target=inject_response)
    responder.start()
    assert response_observed.wait(1.0)
    blocked.release.set()
    caller.join(timeout=1.0)
    responder.join(timeout=1.0)

    assert not caller.is_alive()
    assert not responder.is_alive()
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], MojoCoordinatorError)
    assert transport.closed is True


def test_request_correlation_is_canonical_and_hidden_until_armed(
    fake_coordinator_factory: object,
) -> None:
    """The child sees the complete unguessable correlation only after eligibility is armed."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    observing = CorrelationObservingStream(transport._process.stdin, transport)
    transport._process.stdin = observing  # type: ignore[assignment]

    trajectories = transport.generate((_raw_actor_request(),))

    assert len(trajectories) == 1
    delivered = bytes(observing.before_armed + observing.after_armed)
    request_id = json.loads(delivered)["request_id"]
    assert re.fullmatch(r"request-2:[0-9a-f]{32}", request_id)
    nonce = request_id.partition(":")[2].encode("ascii")
    assert request_id.encode("ascii") not in observing.before_armed
    assert nonce not in observing.before_armed
    assert nonce in observing.after_armed


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_nonce",
        "guessed_nonce",
        "missing_request_id",
        "malformed_request_id",
        "stale_request_id",
        "future_request_id",
        "counter_only_request_id",
    ],
)
def test_transport_rejects_non_exact_correlation_without_nonce_leakage(
    fake_coordinator_factory: object,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Every missing, guessed, malformed, stale, future, or partial echo fails closed."""
    known_nonce = "f" * 32
    monkeypatch.setattr(secrets, "token_hex", lambda size: known_nonce)
    transport = fake_coordinator_factory(mode)  # type: ignore[operator]
    transport.start()

    with pytest.raises(MojoCoordinatorError) as captured:
        transport.generate((_raw_actor_request(),))

    error = str(captured.value)
    assert known_nonce not in error
    assert "0" * 32 not in error
    assert transport.closed is True


def test_short_pipe_writes_deliver_one_exact_complete_jsonl_frame(
    fake_coordinator_factory: object,
) -> None:
    """Short raw writes are advanced until the complete frame and delimiter are delivered."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    short = ShortWriteStream(transport._process.stdin)
    transport._process.stdin = short  # type: ignore[assignment]

    trajectories = transport.generate((_raw_actor_request(),))

    assert len(trajectories) == 1
    delivered = bytes(short.delivered)
    assert delivered.count(b"\n") == 1
    assert delivered.endswith(b"\n")
    decoded = json.loads(delivered)
    request_id = decoded.pop("request_id")
    assert re.fullmatch(r"request-2:[0-9a-f]{32}", request_id)
    assert decoded == {
        "protocol_version": "gepa-actor-v1",
        "type": "generate",
        "payload": {
            "requests": [
                {
                    "case_id": "case-1",
                    "metadata": {},
                    "num_samples": 1,
                    "policy_version": "3",
                    "prompt": "prompt",
                    "sampling_parameters": {},
                    "seed": None,
                }
            ]
        },
    }


@pytest.mark.parametrize("progress", [None, 0, -1, "oversized"])
def test_invalid_pipe_write_progress_fails_closed(
    fake_coordinator_factory: object,
    progress: object,
) -> None:
    """Impossible raw write progress poisons the transport without waiting for a response."""
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "happy",
        request_timeout_seconds=0.1,
    )
    transport.start()
    assert transport._process is not None
    assert transport._process.stdin is not None
    invalid = InvalidWriteProgressStream(transport._process.stdin, progress)
    transport._process.stdin = invalid  # type: ignore[assignment]

    with pytest.raises(MojoCoordinatorError, match="write progress"):
        transport.generate((_raw_actor_request(),))

    assert transport.closed is True
    _assert_transport_resources_stopped(transport, writer_expected=True)


@pytest.mark.parametrize("mode", ["duplicate_envelope_key", "duplicate_nested_key"])
def test_transport_rejects_duplicate_json_object_keys(
    fake_coordinator_factory: object,
    mode: str,
) -> None:
    """Duplicate envelope and nested payload keys cannot overwrite validated evidence."""
    transport = fake_coordinator_factory(mode)  # type: ignore[operator]

    with pytest.raises(MojoCoordinatorError, match="duplicate|JSON"):
        transport.start()


def _raw_actor_request(**changes: object) -> dict[str, object]:
    request: dict[str, object] = {
        "case_id": "case-1",
        "metadata": {},
        "num_samples": 1,
        "policy_version": "3",
        "prompt": "prompt",
        "sampling_parameters": {},
        "seed": None,
    }
    request.update(changes)
    return request


@pytest.mark.parametrize(
    "request_value",
    [
        {"prompt": "missing fields"},
        _raw_actor_request(extra=True),
        _raw_actor_request(prompt=""),
        _raw_actor_request(case_id=7),
        _raw_actor_request(num_samples=0),
        _raw_actor_request(policy_version="03"),
        _raw_actor_request(seed=True),
        _raw_actor_request(seed=2**32 - 1),
        _raw_actor_request(seed=2**32),
        _raw_actor_request(seed=2**32 - 2, num_samples=2),
        _raw_actor_request(metadata=[]),
        _raw_actor_request(sampling_parameters={"temperature": float("nan")}),
    ],
)
def test_direct_transport_rejects_invalid_request_without_writing(
    fake_coordinator_factory: object,
    request_value: dict[str, object],
) -> None:
    """Direct callers receive the same closed request validation as backend callers."""
    transport = fake_coordinator_factory()  # type: ignore[operator]
    transport.start()

    with pytest.raises((TypeError, ValueError), match="request|prompt|case|sample|policy|seed"):
        transport.generate((request_value,))

    assert transport._request_number == 1


def test_direct_transport_accepts_and_echoes_largest_portable_seed(
    fake_coordinator_factory: object,
) -> None:
    transport = fake_coordinator_factory()  # type: ignore[operator]
    transport.start()

    trajectories = transport.generate((_raw_actor_request(seed=2**32 - 2),))

    assert trajectories[0]["seed"] == 2**32 - 2


def test_mojo_backend_prepares_and_propagates_largest_portable_seed(
    fake_coordinator_factory: object,
) -> None:
    backend = MojoCoordinatorBackend(fake_coordinator_factory())  # type: ignore[operator]

    trajectory = backend.generate(
        (RolloutRequest(prompt="prompt", policy_version="3", seed=2**32 - 2),)
    )[0]

    assert trajectory.seed == 2**32 - 2


@pytest.mark.parametrize(
    ("mode", "message", "kwargs"),
    [
        ("timeout", "deadline", {"startup_timeout_seconds": 0.05}),
        ("crash_handshake", "exited", {}),
        ("invalid_utf8", "UTF-8", {}),
        ("invalid_json", "JSON", {}),
        ("partial", "newline|partial", {}),
        ("oversized_stdout", "frame", {"max_frame_bytes": 256}),
        ("oversized_stderr", "stderr", {"max_stderr_bytes": 64}),
        ("bad_handshake", "handshake", {}),
    ],
)
def test_transport_rejects_bounded_startup_protocol_failures(
    fake_coordinator_factory: object,
    mode: str,
    message: str,
    kwargs: dict[str, object],
) -> None:
    """Malformed, crashed, and unbounded startup behavior fails closed and cleans up."""
    transport = fake_coordinator_factory(mode, **kwargs)  # type: ignore[operator]

    with pytest.raises(MojoCoordinatorError, match=message):
        transport.start()

    assert transport.closed is True


def _assert_transport_resources_stopped(
    transport: MojoProcessTransport,
    *,
    writer_expected: bool,
) -> None:
    assert transport.process_running is False
    for reader in (transport._stdout_thread, transport._stderr_thread):
        assert reader is not None
        assert reader.is_alive() is False
    writer = getattr(transport, "_writer_thread", None)
    if writer_expected:
        assert writer is not None
        assert writer.is_alive() is False


def test_startup_timeout_returns_with_process_and_transport_threads_stopped(
    fake_coordinator_factory: object,
) -> None:
    """The startup budget reserves enough cleanup time to reap every owned resource."""
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "timeout",
        startup_timeout_seconds=0.1,
    )

    with pytest.raises(MojoCoordinatorError, match="deadline"):
        transport.start()

    _assert_transport_resources_stopped(transport, writer_expected=True)


def test_blocked_write_timeout_returns_with_process_and_transport_threads_stopped(
    fake_coordinator_factory: object,
) -> None:
    """The request budget reaps a blocked writer, process, and readers before returning."""
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "stop_reading",
        request_timeout_seconds=0.1,
        max_frame_bytes=1_000_000,
    )
    transport.start()

    with pytest.raises(MojoCoordinatorError, match="deadline|write"):
        transport.generate((_raw_actor_request(prompt="x" * 500_000),))

    _assert_transport_resources_stopped(transport, writer_expected=True)


def test_backend_wraps_unsafe_handshake_construction_and_fails_closed(
    fake_coordinator_factory: object,
) -> None:
    """Unsafe handshake construction uses the backend error contract and poisons startup."""
    transport = fake_coordinator_factory("unsafe_handshake")  # type: ignore[operator]
    backend = MojoCoordinatorBackend(transport)
    request = (RolloutRequest(prompt="prompt", policy_version="3"),)

    with pytest.raises(MojoCoordinatorError, match="handshake"):
        backend.generate(request)

    assert transport.closed is True
    _assert_transport_resources_stopped(transport, writer_expected=True)
    with pytest.raises(RuntimeError, match="closed"):
        backend.generate(request)


def test_backend_poisons_custom_transport_returning_untyped_handshake() -> None:
    """A non-ActorHandshake startup value closes the transport and later calls fail closed."""
    transport = InvalidHandshakeTransport()
    backend = MojoCoordinatorBackend(transport)
    request = (RolloutRequest(prompt="prompt", policy_version="3"),)

    with pytest.raises(MojoCoordinatorError, match="handshake"):
        backend.generate(request)

    assert transport.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        backend.generate(request)


def test_generate_error_and_crash_do_not_echo_prompt_stderr_or_environment_secret(
    fake_coordinator_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions expose bounded classifications rather than actor-controlled secrets."""
    monkeypatch.setenv("COORDINATOR_TEST_SECRET", "environment-secret")
    for mode in ("error_secret", "crash_generate"):
        transport = fake_coordinator_factory(mode)  # type: ignore[operator]
        backend = MojoCoordinatorBackend(transport)
        with pytest.raises(MojoCoordinatorError) as captured:
            backend.generate((RolloutRequest(prompt="private-prompt", policy_version="3"),))
        text = str(captured.value)
        assert "private-prompt" not in text
        assert "environment-secret" not in text
        assert "secret crash" not in text


def test_stderr_overflow_is_seen_when_reader_scheduling_is_delayed(
    fake_coordinator_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stdout frame cannot hide stderr already written before a delayed reader runs."""
    original = MojoProcessTransport._read_stderr

    def delayed_reader(transport: MojoProcessTransport, stream: object) -> None:
        time.sleep(0.2)
        original(transport, stream)  # type: ignore[arg-type]

    monkeypatch.setattr(MojoProcessTransport, "_read_stderr", delayed_reader)
    transport = fake_coordinator_factory(  # type: ignore[operator]
        "oversized_stderr",
        max_stderr_bytes=64,
    )

    with pytest.raises(MojoCoordinatorError, match="stderr"):
        transport.start()


def test_close_escalates_cleanup_when_coordinator_does_not_exit(
    fake_coordinator_factory: object,
) -> None:
    """A close acknowledgement cannot let a lingering child escape bounded cleanup."""
    transport = fake_coordinator_factory("ignore_close")  # type: ignore[operator]
    transport.start()

    transport.close()

    assert transport.closed is True
    assert transport.process_running is False


def test_operation_deadlines_bound_reader_start_blocked_write_and_shutdown(
    fake_coordinator_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader startup, blocked stdin, and shutdown cleanup share total operation budgets."""
    original = MojoProcessTransport._read_stderr

    def late_reader(transport: MojoProcessTransport, stream: object) -> None:
        time.sleep(0.2)
        original(transport, stream)  # type: ignore[arg-type]

    monkeypatch.setattr(MojoProcessTransport, "_read_stderr", late_reader)
    startup = fake_coordinator_factory(  # type: ignore[operator]
        "happy",
        startup_timeout_seconds=0.05,
    )
    started_at = time.monotonic()
    with pytest.raises(MojoCoordinatorError, match="deadline"):
        startup.start()
    assert time.monotonic() - started_at < 0.3
    monkeypatch.setattr(MojoProcessTransport, "_read_stderr", original)

    blocked = fake_coordinator_factory(  # type: ignore[operator]
        "stop_reading",
        request_timeout_seconds=0.05,
        max_frame_bytes=1_000_000,
    )
    blocked.start()
    large = _raw_actor_request(prompt="x" * 500_000)
    started_at = time.monotonic()
    with pytest.raises(MojoCoordinatorError, match="deadline|write"):
        blocked.generate((large,))
    assert time.monotonic() - started_at < 0.3

    shutdown = fake_coordinator_factory(  # type: ignore[operator]
        "ignore_close",
        shutdown_timeout_seconds=0.1,
    )
    shutdown.start()
    started_at = time.monotonic()
    shutdown.close()
    assert time.monotonic() - started_at < 0.3


@pytest.mark.skipif(shutil.which("mojo") is None, reason="Mojo toolchain is not installed")
def test_mojo_coordinator_source_runs_hello_and_close_protocol() -> None:
    """When Mojo is installed, validate close, unconfigured generate, and rejection paths."""
    source = Path("mojo/rl_coordinator/main.mojo")
    messages = (
        '{"protocol_version":"gepa-actor-v1","type":"hello",'
        '"request_id":"request-1:11111111111111111111111111111111","payload":{}}\n'
        '{"protocol_version":"gepa-actor-v1","type":"close",'
        '"request_id":"request-2:22222222222222222222222222222222","payload":{}}\n'
    )

    completed = subprocess.run(
        ["mojo", "run", str(source)],
        input=messages,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    frames = completed.stdout.splitlines()
    assert len(frames) == 2
    assert '"type":"hello"' in frames[0].replace(" ", "")
    assert '"type":"close"' in frames[1].replace(" ", "")

    representative_request = (
        '{"case_id":null,"metadata":{},"num_samples":1,"policy_version":"1",'
        '"prompt":"probe","sampling_parameters":{},"seed":null}'
    )
    generate_messages = (
        '{"protocol_version":"gepa-actor-v1","type":"hello",'
        '"request_id":"request-1:11111111111111111111111111111111","payload":{}}\n'
        '{"protocol_version":"gepa-actor-v1","type":"generate",'
        '"request_id":"request-2:22222222222222222222222222222222",'
        '"payload":{"requests":[' + representative_request + "]}}\n"
    )
    generated = subprocess.run(
        ["mojo", "run", str(source)],
        input=generate_messages,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    assert generated.returncode == 0, generated.stderr
    assert "actor_unconfigured" in generated.stdout

    invalid_messages = (
        messages.replace("request-1:", "request-9:", 1),
        generate_messages.replace('"payload":{"requests":', '"payload":{"extra":1,"requests":'),
        generate_messages.replace('"request_id":"request-2:', '"request_id":"request-3:'),
        generate_messages.replace('"type":"generate"', '"type":"close"'),
        generate_messages.replace("2" * 32, "not-canonical", 1),
        generate_messages.replace(":" + "2" * 32, "", 1),
    )
    for invalid in invalid_messages:
        malformed = subprocess.run(
            ["mojo", "run", str(source)],
            input=invalid,
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        assert malformed.returncode != 0


def test_mojo_source_declares_strict_unconfigured_protocol_contract() -> None:
    """The checked-in boundary names every accepted type and never emits trajectories."""
    source = Path("mojo/rl_coordinator/main.mojo").read_text(encoding="utf-8")

    assert "gepa-actor-v1" in source
    assert '\\"hello\\"' in source
    assert '\\"generate\\"' in source
    assert '\\"close\\"' in source
    assert "actor_unconfigured" in source
    assert "trajectories" not in source
    assert '\\"request_id\\":\\"request-1:' in source
    assert '\\"request_id\\":\\"request-2:' in source
    assert "is_canonical_nonce" in source
    assert '\\"case_id\\":null' in source
    assert '\\"metadata\\":{}' in source
    assert '\\"num_samples\\":1' in source
    assert '\\"policy_version\\":\\"1\\"' in source
    assert '\\"prompt\\":\\"probe\\"' in source
    assert '\\"sampling_parameters\\":{}' in source
    assert '\\"seed\\":null' in source
    assert "invalid gepa-actor-v1 hello frame" in source
    assert "invalid gepa-actor-v1 generate or close frame" in source


class _HybridTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    tokens = {"practice": 2, "slowly": 3, "chosen": 4, "rejected": 5}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.tokens[word] for word in text.split()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        words = {value: key for key, value in self.tokens.items()}
        return " ".join(words[token] for token in token_ids)


class _TinyLoraPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, _name_or_path="tiny-hybrid-model")
        self.embedding = nn.Embedding(6, 4)
        self.lm_head = nn.Linear(4, 6, bias=False)
        self.lora_logits = nn.Parameter(torch.zeros(6))
        self.embedding.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.peft_config = {"default": SimpleNamespace(peft_type="LORA")}

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden) + self.lora_logits
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        response = torch.full(
            (input_ids.shape[0], 1),
            4,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)


class _MockVersionedActor:
    def __init__(
        self,
        *,
        model_id: str,
        adapter_id: str,
        adapter_sha256: str | None = None,
        crash: bool = False,
    ) -> None:
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.adapter_sha256 = adapter_sha256
        self.crash = crash
        self.generate_calls = 0
        self.close_calls = 0

    def generate(self, requests: Sequence[RolloutRequest]) -> tuple[Trajectory, ...]:
        self.generate_calls += 1
        if self.crash:
            raise RuntimeError("mock actor crash")
        trajectories = []
        for request in requests:
            for sample in range(request.num_samples):
                response = "chosen" if sample == 0 else "rejected"
                trajectories.append(
                    Trajectory(
                        trajectory_id=f"actor-{self.generate_calls}-{sample}",
                        case_id=request.case_id,
                        prompt=request.prompt,
                        response=response,
                        prompt_token_ids=(2, 3),
                        response_token_ids=(4 if sample == 0 else 5,),
                        old_log_probs=(-1.0,),
                        reference_log_probs=(-1.0,),
                        value_predictions=None,
                        sampling_parameters=request.sampling_parameters,
                        backend_name="mock-mojo-vulkan",
                        backend_version="test-1",
                        model_identifier=self.model_id,
                        adapter_identifier=self.adapter_id,
                        adapter_sha256=(
                            self.adapter_sha256 or str(request.metadata.get("adapter_sha256", ""))
                        ),
                        policy_version=request.policy_version,
                        seed=request.seed,
                    )
                )
        return tuple(trajectories)

    def capabilities(self):  # type: ignore[no-untyped-def]
        from gepa_mindfulness.training.capability import BackendCapabilities, CapabilityEvidence

        return BackendCapabilities(
            backend_name="mock-mojo-vulkan",
            backend_version="test-1",
            capabilities={
                capability: CapabilityEvidence(
                    state=(
                        CapabilityState.SUPPORTED
                        if capability is Capability.SUPPORTS_GENERATION
                        else CapabilityState.UNSUPPORTED
                    ),
                    evidence="mock actor generation evidence",
                )
                for capability in Capability
            },
        )

    def close(self) -> None:
        self.close_calls += 1


class _InvalidIdentityActor(_MockVersionedActor):
    def __init__(self, *, change: Mapping[str, object]) -> None:
        super().__init__(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
        self.change = dict(change)

    def generate(self, requests: Sequence[RolloutRequest]) -> tuple[Trajectory, ...]:
        return tuple(
            replace(trajectory, **self.change) for trajectory in super().generate(requests)
        )


class _FailingPublisher:
    def __init__(self, publisher: LocalAdapterPublisher) -> None:
        self.publisher = publisher

    def current(self):  # type: ignore[no-untyped-def]
        return self.publisher.current()

    def current_artifact(self):  # type: ignore[no-untyped-def]
        return self.publisher.current_artifact()

    def publish(self, candidate: AdapterCandidate):  # type: ignore[no-untyped-def]
        del candidate
        raise RuntimeError("publisher failure")


def _write_hybrid_pair(path: Path) -> None:
    names = (
        "objective_fidelity",
        "feedback_integrity",
        "skill_transfer",
        "reality_contact",
        "exploit_disclosure",
        "long_horizon_agency",
        "benign_creativity",
        "repair_quality",
    )
    record = {
        "record_id": "hybrid:grounded_over_proxy",
        "source_case_id": "hybrid",
        "source_case_version": "1.0",
        "source_path": "authored/hybrid.jsonl",
        "source_line": 1,
        "source_sha256": "a" * 64,
        "pair_rule": "grounded_over_proxy",
        "prompt": "practice slowly",
        "chosen": "chosen",
        "rejected": "rejected",
        "chosen_class": "grounded_success",
        "rejected_class": "proxy_exploitation",
        "chosen_reward_components": {name: 0.5 for name in names},
        "rejected_reward_components": {name: -0.5 for name in names},
        "diagnostics": {"central": "authored", "supporting": []},
        "schema_version": "reward-integrity-rl-pairs-v1",
    }
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _hybrid_config(tmp_path: Path, *, staleness: StalenessPolicy) -> RLRunConfig:
    dataset = tmp_path / "hybrid-pairs.jsonl"
    _write_hybrid_pair(dataset)
    return RLRunConfig(
        runtime=RuntimeConfig(backend="mojo-vulkan-llamacpp"),
        policy=PolicyConfig(model_name=str(tmp_path / "learner-model"), max_new_tokens=1),
        algorithm=AlgorithmConfig(
            name="grpo",
            learning_rate=0.1,
            batch_size=1,
            max_steps=1,
            group_size=2,
            zero_variance_policy="skip",
        ),
        dataset=DatasetConfig(train_path=str(dataset)),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "checkpoints"), save_steps=1),
        logging=LoggingConfig(log_dir=str(tmp_path / "logs")),
        hybrid=HybridConfig(
            model_id="tiny-hybrid-model",
            expected_actor_backend="mock-mojo-vulkan",
            adapter_store=str(tmp_path / "adapters"),
            staleness_policy=staleness,
            max_policy_lag=0,
            downweight_decay=0.5 if staleness is StalenessPolicy.DOWN_WEIGHT else None,
        ),
        seed=42,
    )


def _bootstrap_adapter(publisher: LocalAdapterPublisher) -> AdapterManifest:
    source = TorchPolicyBackend(
        policy_model=_deterministic_hybrid_policy(),
        tokenizer=_HybridTokenizer(),
        device="cpu",
        learning_rate=0.1,
        max_new_tokens=1,
        training_mode="lora",
        model_identifier="tiny-hybrid-model",
        adapter_identifier="tiny-lora",
    )
    with torch.no_grad():
        source.policy_model.lora_logits.fill_(0.25)
    return publisher.publish(
        source.export_adapter(
            publisher.root.parent / "bootstrap.adapter",
            model_id="tiny-hybrid-model",
            policy_version=PolicyVersion(1),
            parent_policy_version=None,
        )
    )


def _deterministic_hybrid_policy() -> _TinyLoraPolicy:
    with torch.random.fork_rng():
        torch.manual_seed(314159)
        return _TinyLoraPolicy()


def _hybrid_learner(config: RLRunConfig) -> TorchPolicyBackend:
    return TorchPolicyBackend(
        policy_model=_deterministic_hybrid_policy(),
        tokenizer=_HybridTokenizer(),
        device="cpu",
        learning_rate=config.algorithm.learning_rate,
        max_new_tokens=1,
        training_mode="lora",
        model_identifier="tiny-hybrid-model",
        adapter_identifier="tiny-lora",
    )


def _invalid_bootstrap(
    publisher: LocalAdapterPublisher,
    failure: str,
) -> AdapterManifest:
    if failure == "independent-base":
        with torch.random.fork_rng():
            torch.manual_seed(271828)
            policy = _TinyLoraPolicy()
    else:
        policy = _deterministic_hybrid_policy()
    source = TorchPolicyBackend(
        policy_model=policy,
        tokenizer=_HybridTokenizer(),
        device="cpu",
        learning_rate=0.1,
        max_new_tokens=1,
        training_mode="lora",
        model_identifier="tiny-hybrid-model",
        adapter_identifier="tiny-lora",
    )
    candidate = source.export_adapter(
        publisher.root.parent / f"invalid-{failure}.adapter",
        model_id="tiny-hybrid-model",
        policy_version=PolicyVersion(1),
        parent_policy_version=None,
    )
    if failure == "wrong-source":
        candidate = replace(candidate, source_id="other-lora")
    elif failure == "wrong-format":
        candidate = replace(candidate, format_id="full-model-v1")
    elif failure == "wrong-model":
        candidate = replace(candidate, model_id="other-model")
    elif failure in {"full-model", "unsafe", "incompatible", "missing"}:
        payload = torch.load(candidate.artifact_path, map_location="cpu", weights_only=True)
        if failure == "full-model":
            payload["state_dict"]["embedding.weight"] = source.policy_model.embedding.weight
        elif failure == "unsafe":
            payload["unexpected"] = "untrusted"
        elif failure == "incompatible":
            payload["state_dict"]["lora_logits"] = torch.zeros(7)
        else:
            payload["state_dict"] = {}
        serialized = BytesIO()
        torch.save(payload, serialized)
        candidate.artifact_path.write_bytes(serialized.getvalue())
        candidate = replace(
            candidate,
            expected_sha256=hashlib.sha256(serialized.getvalue()).hexdigest(),
        )
    manifest = publisher.publish(candidate)
    if failure == "hash-mismatch":
        (publisher.root / manifest.artifact_path).write_bytes(b"corrupt-after-publication")
    return manifest


class _LearnerCapabilityProvider:
    def __init__(self, learner: TorchPolicyBackend) -> None:
        self.learner = learner

    def detect(self, config: RLRunConfig):  # type: ignore[no-untyped-def]
        del config
        return self.learner.capabilities()


def test_fresh_hybrid_loads_v1_before_actor_and_publishes_a_true_v2_child(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    v1 = _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    checksum_before_load = learner.policy_parameter_checksum()
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    checksum_at_actor_start = ""

    def actor_factory(selected: RLRunConfig, manifest: AdapterManifest) -> _MockVersionedActor:
        del selected
        nonlocal checksum_at_actor_start
        assert manifest == v1
        assert torch.allclose(learner.policy_model.lora_logits, torch.full((6,), 0.25))
        checksum_at_actor_start = learner.policy_parameter_checksum()
        return actor

    result = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=actor_factory,
        publisher_factory=lambda _: publisher,
    ).train(max_steps=1)

    assert checksum_at_actor_start != checksum_before_load
    assert result.policy_parameter_checksum_before == checksum_at_actor_start
    assert result.published_adapter is not None
    assert result.published_adapter.policy_version == PolicyVersion(2)
    assert result.published_adapter.parent_policy_version == PolicyVersion(1)
    v2, v2_payload = publisher.current_artifact()
    reader = _hybrid_learner(config)
    assert (
        reader.load_adapter_bytes(v2_payload, manifest=v2) == result.policy_parameter_checksum_after
    )
    assert not torch.allclose(reader.policy_model.lora_logits, torch.full((6,), 0.25))


@pytest.mark.parametrize(
    "failure",
    [
        "wrong-source",
        "wrong-format",
        "wrong-model",
        "hash-mismatch",
        "full-model",
        "unsafe",
        "incompatible",
        "missing",
        "independent-base",
    ],
)
def test_fresh_hybrid_rejects_invalid_lineage_before_logger_or_actor(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _invalid_bootstrap(publisher, failure)
    learner = _hybrid_learner(config)
    checksum_before = learner.policy_parameter_checksum()
    logger_calls = 0
    actor_calls = 0

    def logger_factory(selected: RLRunConfig) -> _NoOpPublicationLogger:
        del selected
        nonlocal logger_calls
        logger_calls += 1
        return _NoOpPublicationLogger()

    def actor_factory(selected: RLRunConfig, manifest: AdapterManifest) -> _MockVersionedActor:
        del selected, manifest
        nonlocal actor_calls
        actor_calls += 1
        return _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    with pytest.raises(ValueError, match="adapter|artifact|model|lineage|payload|tensor"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=actor_factory,
            publisher_factory=lambda _: publisher,
            logger_factory=logger_factory,
        ).train(max_steps=0)

    assert learner.policy_parameter_checksum() == checksum_before
    assert learner._step == 0
    assert logger_calls == 0
    assert actor_calls == 0


def test_fresh_hybrid_rejects_valid_later_publication_before_logger_or_actor(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    source = _hybrid_learner(config)
    publisher.publish(
        source.export_adapter(
            tmp_path / "later.adapter",
            model_id="tiny-hybrid-model",
            policy_version=PolicyVersion(2),
            parent_policy_version=PolicyVersion(1),
        )
    )
    learner = _hybrid_learner(config)
    logger_calls = 0
    actor_calls = 0

    def logger_factory(selected: RLRunConfig) -> _NoOpPublicationLogger:
        del selected
        nonlocal logger_calls
        logger_calls += 1
        return _NoOpPublicationLogger()

    def actor_factory(selected: RLRunConfig, manifest: AdapterManifest) -> _MockVersionedActor:
        del selected, manifest
        nonlocal actor_calls
        actor_calls += 1
        return _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    with pytest.raises(ValueError, match="bootstrap v1"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=actor_factory,
            publisher_factory=lambda _: publisher,
            logger_factory=logger_factory,
        ).train(max_steps=0)

    assert logger_calls == 0
    assert actor_calls == 0


def test_fresh_hybrid_rejects_bootstrap_with_checkpoint_ancestry_metadata(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    manifest = _bootstrap_adapter(publisher)
    _, payload = publisher.current_artifact()
    forged = replace(
        manifest,
        metadata={
            **dict(manifest.metadata),
            "checkpoint_id": "checkpoint-00000000",
            "global_step": 0,
        },
    )

    class ForgedPublisher:
        def current_artifact(self):  # type: ignore[no-untyped-def]
            return forged, payload

        def publish(self, candidate: object) -> object:
            del candidate
            raise AssertionError("fresh preflight must fail before publication")

    with pytest.raises(ValueError, match="bootstrap v1"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(_hybrid_learner(config)),
            backend_factory=lambda _: _hybrid_learner(config),
            publisher_factory=lambda _: ForgedPublisher(),  # type: ignore[arg-type]
            actor_factory=lambda *_: (_ for _ in ()).throw(AssertionError("actor must not start")),
            logger_factory=lambda _: (_ for _ in ()).throw(AssertionError("logger must not start")),
        ).train(max_steps=0)


def test_hybrid_step_updates_pytorch_and_publishes_exact_next_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    initial = _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    learner_close_calls = 0

    def close_learner() -> None:
        nonlocal learner_close_calls
        learner_close_calls += 1

    monkeypatch.setattr(learner, "close", close_learner)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    )

    result = engine.train(max_steps=1)

    assert result.policy_parameters_updated is True
    assert result.checkpoint is not None
    assert result.published_adapter is not None
    assert result.published_adapter.policy_version == PolicyVersion(2)
    assert result.published_adapter.parent_policy_version == PolicyVersion(1)
    assert publisher.current() == result.published_adapter
    assert result.published_adapter.format_id == "pytorch-lora-state-dict-v1"
    assert "gguf" not in json.dumps(result.to_dict()).lower()
    assert actor.close_calls == 1
    assert learner_close_calls == 1
    manifest = json.loads((result.log_directory / "run_manifest.json").read_text(encoding="utf-8"))
    trajectory_record = json.loads(
        (result.log_directory / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    actor_policy = manifest["device_capabilities"]["hybrid_actor_policy"]
    assert manifest["actor_backend"] == "mock-mojo-vulkan"
    assert manifest["learner_backend"] == "torch_portable"
    assert manifest["software_versions"] == {
        "actor_backend": "unobserved",
        "learner_backend": str(torch.__version__),
    }
    assert manifest["adapter"] == "tiny-lora"
    assert actor_policy["policy_version"] == "1"
    assert actor_policy["adapter_sha256"] == initial.artifact_sha256
    assert trajectory_record["actor_backend"] == manifest["actor_backend"]
    assert trajectory_record["learner_backend"] == manifest["learner_backend"]
    assert trajectory_record["policy_version"] == actor_policy["policy_version"]
    assert trajectory_record["trajectory"]["adapter_identifier"] == manifest["adapter"]


def test_hybrid_second_step_rejects_stale_actor_before_scoring(tmp_path: Path) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    )

    with pytest.raises(ValueError, match="stale actor policy"):
        engine.train(max_steps=2)

    assert publisher.current().policy_version == PolicyVersion(2)  # type: ignore[union-attr]
    assert actor.generate_calls == 2
    assert actor.close_calls == 1


def test_hybrid_downweights_stale_reward_once_and_publishes_next_version(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.DOWN_WEIGHT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    result = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    ).train(max_steps=2)

    assert result.published_adapter is not None
    assert result.published_adapter.policy_version == PolicyVersion(3)
    records = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stale = [
        record
        for record in records
        if "staleness_weight" in record["metrics"] and record["metrics"]["staleness_lag"] == 1.0
    ]
    assert len(stale) == 2
    assert {record["metrics"]["staleness_weight"] for record in stale} == {0.5}
    assert {record["metrics"]["staleness_accepted"] for record in stale} == {1.0}
    manifest = json.loads((result.log_directory / "run_manifest.json").read_text(encoding="utf-8"))
    trajectories = [
        json.loads(line)
        for line in (result.log_directory / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    publications = [
        json.loads(line)
        for line in (result.log_directory / "publications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["device_capabilities"]["hybrid_actor_policy"]["policy_version"] == "1"
    assert {record["policy_version"] for record in records} == {"1"}
    assert {record["policy_version"] for record in trajectories} == {"1"}
    assert [record["parent_policy_version"] for record in publications] == ["1", "2"]
    assert [record["policy_version"] for record in publications] == ["2", "3"]


def test_hybrid_actor_crash_closes_actor_and_learner_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    learner_closes = 0

    def close_learner() -> None:
        nonlocal learner_closes
        learner_closes += 1

    monkeypatch.setattr(learner, "close", close_learner)
    actor = _MockVersionedActor(
        model_id="tiny-hybrid-model",
        adapter_id="tiny-lora",
        crash=True,
    )
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    )

    with pytest.raises(RuntimeError, match="mock actor crash"):
        engine.train(max_steps=1)

    assert actor.close_calls == 1
    assert learner_closes == 1
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"policy_version": None}, "policy version"),
        ({"policy_version": "2"}, "policy version"),
        ({"model_identifier": "wrong-model"}, "model"),
        ({"adapter_identifier": "wrong-adapter"}, "adapter"),
    ],
)
def test_hybrid_rejects_missing_ahead_or_mismatched_actor_identity_before_update(
    tmp_path: Path,
    change: Mapping[str, object],
    message: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _InvalidIdentityActor(change=change)

    with pytest.raises(ValueError, match=message):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    assert learner._step == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]
    assert actor.close_calls == 1


def test_hybrid_publisher_failure_preserves_current_and_returns_no_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    failing = _FailingPublisher(publisher)
    learner = _hybrid_learner(config)
    learner_close_calls = 0

    def close_learner() -> None:
        nonlocal learner_close_calls
        learner_close_calls += 1

    monkeypatch.setattr(learner, "close", close_learner)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: failing,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="publisher failure"):
        engine.train(max_steps=1)

    assert learner._step == 1
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]
    assert actor.close_calls == 1
    assert learner_close_calls == 1


def test_hybrid_resume_requires_checkpoint_current_adapter_coherence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    first_actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    first_learner = _hybrid_learner(config)
    first = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(first_learner),
        backend_factory=lambda _: first_learner,
        actor_factory=lambda _, manifest: first_actor,
        publisher_factory=lambda _: publisher,
    ).train(max_steps=1)
    checkpoint_path = Path(config.checkpoint.output_dir) / first.checkpoint.checkpoint_id
    resumed_actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    resumed_learner = _hybrid_learner(config)

    def unexpected_adapter_load(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise AssertionError("resume must use the coherent checkpoint restoration path")

    monkeypatch.setattr(
        resumed_learner,
        "load_adapter_bytes",
        unexpected_adapter_load,
    )

    resumed = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(resumed_learner),
        backend_factory=lambda _: resumed_learner,
        actor_factory=lambda _, manifest: resumed_actor,
        publisher_factory=lambda _: publisher,
    ).resume(checkpoint_path, max_steps=0)

    assert resumed.global_step == 1
    assert resumed.trajectory_count == 0
    assert resumed.policy_parameters_updated is False
    assert resumed.published_adapter is None
    assert resumed_actor.generate_calls == 0
    assert resumed_actor.close_calls == 1


def test_hybrid_resume_rejects_same_metadata_with_different_adapter_state_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    first_learner = _hybrid_learner(config)
    first = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(first_learner),
        backend_factory=lambda _: first_learner,
        actor_factory=lambda _, manifest: _MockVersionedActor(
            model_id="tiny-hybrid-model", adapter_id="tiny-lora"
        ),
        publisher_factory=lambda _: publisher,
    ).train(max_steps=1)
    checkpoint_path = Path(config.checkpoint.output_dir) / first.checkpoint.checkpoint_id
    manifest, payload = publisher.current_artifact()
    decoded = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    decoded["state_dict"]["lora_logits"] = decoded["state_dict"]["lora_logits"] + 0.125
    changed = BytesIO()
    torch.save(decoded, changed)
    changed_payload = changed.getvalue()
    changed_manifest = replace(
        manifest,
        artifact_sha256=hashlib.sha256(changed_payload).hexdigest(),
        artifact_size=len(changed_payload),
    )
    root = Path(config.hybrid.adapter_store)
    (root / manifest.artifact_path).write_bytes(changed_payload)
    canonical = json.dumps(changed_manifest.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    (root / manifest.manifest_path).write_bytes(canonical.encode("utf-8"))
    (root / "current.json").write_bytes(canonical.encode("utf-8"))

    resumed_learner = _hybrid_learner(config)
    checksum_before = resumed_learner.policy_parameter_checksum()
    step_before = resumed_learner._step
    rng_before = torch.get_rng_state().clone()
    actor_calls = 0
    logger_calls = 0
    monkeypatch.setattr(rl_engine, "_seed_process", lambda selected: None)

    def actor_factory(selected: RLRunConfig, selected_manifest: AdapterManifest) -> object:
        del selected, selected_manifest
        nonlocal actor_calls
        actor_calls += 1
        return _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    def logger_factory(selected: RLRunConfig) -> object:
        del selected
        nonlocal logger_calls
        logger_calls += 1
        return _NoOpPublicationLogger()

    with pytest.raises(ValueError, match="incoherent"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(resumed_learner),
            backend_factory=lambda _: resumed_learner,
            actor_factory=actor_factory,
            publisher_factory=lambda _: publisher,
            logger_factory=logger_factory,
        ).resume(checkpoint_path, max_steps=0)

    assert resumed_learner.policy_parameter_checksum() == checksum_before
    assert resumed_learner._step == step_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert actor_calls == 0
    assert logger_calls == 0


@pytest.mark.parametrize(
    "failure",
    ["later-version", "wrong-source", "wrong-format", "unknown-payload-field"],
)
def test_hybrid_resume_rejects_forged_current_artifact_before_logger_or_actor(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    first_learner = _hybrid_learner(config)
    first = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(first_learner),
        backend_factory=lambda _: first_learner,
        actor_factory=lambda *_: _MockVersionedActor(
            model_id="tiny-hybrid-model", adapter_id="tiny-lora"
        ),
        publisher_factory=lambda _: publisher,
    ).train(max_steps=1)
    checkpoint_path = Path(config.checkpoint.output_dir) / first.checkpoint.checkpoint_id
    manifest, payload = publisher.current_artifact()
    decoded = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if failure == "later-version":
        decoded["policy_version"] = "3"
        manifest = replace(
            manifest,
            policy_version=PolicyVersion(3),
            parent_policy_version=PolicyVersion(2),
            artifact_path="versions/3/adapter.bin",
            manifest_path="versions/3/manifest.json",
        )
    elif failure == "wrong-source":
        decoded["adapter_identifier"] = "other-lora"
        manifest = replace(manifest, source_id="other-lora")
    elif failure == "wrong-format":
        decoded["format_id"] = "other-format"
        manifest = replace(manifest, format_id="other-format")
    else:
        decoded["unexpected"] = "closed-schema-violation"
    changed = BytesIO()
    torch.save(decoded, changed)
    changed_payload = changed.getvalue()
    manifest = replace(
        manifest,
        artifact_sha256=hashlib.sha256(changed_payload).hexdigest(),
        artifact_size=len(changed_payload),
    )

    class ForgedPublisher:
        calls = 0

        def current_artifact(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            return manifest, changed_payload

        def publish(self, candidate: object) -> object:
            del candidate
            raise AssertionError("resume preflight must fail before publication")

    forged = ForgedPublisher()
    actor_calls = 0
    logger_calls = 0

    def actor_factory(*args: object) -> object:
        del args
        nonlocal actor_calls
        actor_calls += 1
        raise AssertionError("actor must not start")

    def logger_factory(*args: object) -> object:
        del args
        nonlocal logger_calls
        logger_calls += 1
        raise AssertionError("logger must not start")

    with pytest.raises(ValueError, match="adapter|payload|incoherent|format|source"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(_hybrid_learner(config)),
            backend_factory=lambda _: _hybrid_learner(config),
            actor_factory=actor_factory,
            publisher_factory=lambda _: forged,  # type: ignore[arg-type]
            logger_factory=logger_factory,
        ).resume(checkpoint_path, max_steps=0)

    assert forged.calls == 1
    assert actor_calls == 0
    assert logger_calls == 0


@pytest.mark.parametrize("learner", [None, "mojo"])
def test_hybrid_cli_rejects_missing_or_mojo_learner_before_config_loading(
    monkeypatch: pytest.MonkeyPatch,
    learner: str | None,
) -> None:
    events: list[str] = []

    def load_config(path: str) -> RLRunConfig:
        events.append(f"config:{path}")
        raise AssertionError("config must not load")

    monkeypatch.setattr(rl_cli, "load_rl_run_config", load_config)
    result = rl_cli._handle_engine(  # noqa: SLF001 - direct fail-fast contract
        SimpleNamespace(
            backend="mojo-vulkan-llamacpp",
            learner=learner,
            config="should-not-load.yaml",
        )
    )

    assert result == 2
    assert events == []


def test_hybrid_config_is_strict_and_keeps_operator_context_out_of_template() -> None:
    with pytest.raises(ValueError, match="hybrid.lora contains unknown keys"):
        HybridConfig.from_mapping({"lora": {"unknown": 1}})

    template = Path("configs/rl/hybrid_vulkan_grpo.yaml").read_text(encoding="utf-8")
    assert "mojo-vulkan-llamacpp" in template
    assert "endpoint:" not in template
    assert "secret" not in template.casefold()
    assert "api_key" not in template.casefold()


def test_hybrid_lora_is_recursively_frozen_and_detached_from_caller_state() -> None:
    caller_lora: dict[str, object] = {
        "r": 4,
        "target_modules": ["q_proj", "v_proj"],
    }
    config = HybridConfig(lora=caller_lora)

    caller_lora["r"] = 99
    cast(list[str], caller_lora["target_modules"]).append("k_proj")

    assert config.lora == {"r": 4, "target_modules": ("q_proj", "v_proj")}
    with pytest.raises(TypeError):
        config.lora["r"] = 8  # type: ignore[index]
    with pytest.raises(AttributeError):
        cast(list[str], config.lora["target_modules"]).append("k_proj")


def test_hybrid_lora_canonical_serialization_is_deterministic() -> None:
    first = RLRunConfig(
        runtime=RuntimeConfig(backend="mojo-vulkan-llamacpp"),
        algorithm=AlgorithmConfig(name="grpo"),
        checkpoint=CheckpointConfig(save_steps=1),
        hybrid=HybridConfig(lora={"r": 4, "target_modules": ["q_proj", "v_proj"]}),
    )
    second = RLRunConfig(
        runtime=RuntimeConfig(backend="mojo-vulkan-llamacpp"),
        algorithm=AlgorithmConfig(name="grpo"),
        checkpoint=CheckpointConfig(save_steps=1),
        hybrid=HybridConfig(lora={"target_modules": ("q_proj", "v_proj"), "r": 4}),
    )

    assert _config_payload(first) == _config_payload(second)
    assert _config_hash(first) == _config_hash(second)


@pytest.mark.parametrize(
    "lora",
    [
        {"r": True},
        {"r": 0},
        {"r": 1.5},
        {"lora_alpha": True},
        {"lora_alpha": float("nan")},
        {"lora_alpha": 0},
        {"lora_dropout": True},
        {"lora_dropout": -0.1},
        {"lora_dropout": 1.0},
        {"bias": "all"},
        {"task_type": "SEQ_CLS"},
        {"target_modules": ["q_proj", "q_proj"]},
        {"target_modules": ["../q_proj"]},
        {"modules_to_save": ["head", "head"]},
    ],
)
def test_hybrid_lora_values_fail_strict_preflight(lora: Mapping[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match="hybrid.lora"):
        HybridConfig.from_mapping({"model_id": "safe-model", "lora": lora})


def test_hybrid_cli_rejects_invalid_lora_before_engine_or_actor_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "invalid-hybrid.json"
    config_path.write_text(
        json.dumps(
            {
                "runtime": {"backend": "mojo-vulkan-llamacpp"},
                "algorithm": {"name": "grpo", "group_size": 2},
                "checkpoint": {"save_steps": 1},
                "hybrid": {
                    "model_id": "safe-model",
                    "lora": {"r": True},
                },
            }
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    monkeypatch.setattr(
        rl_cli,
        "create_hybrid_engine",
        lambda config, command: events.append("engine"),
    )

    exit_code = rl_cli._handle_engine(  # noqa: SLF001 - direct fail-fast contract
        SimpleNamespace(
            backend="mojo-vulkan-llamacpp",
            learner="pytorch",
            coordinator_command=("coordinator",),
            actor_endpoint=None,
            endpoint=None,
            config=str(config_path),
            rl_command="train",
            dataset=None,
            output=None,
            max_steps=1,
        )
    )

    assert exit_code == 2
    assert events == []


@pytest.mark.parametrize("model_id", ["", ".", "..", "C:/models/learner", "model/id"])
def test_hybrid_model_id_is_safe_and_distinct_from_learner_locator(model_id: str) -> None:
    with pytest.raises(ValueError, match="hybrid.model_id"):
        HybridConfig(model_id=model_id)


@pytest.mark.parametrize("backend_id", ["", ".", "..", "actor/backend", "actor backend"])
def test_hybrid_expected_actor_backend_is_one_safe_identity(backend_id: str) -> None:
    with pytest.raises(ValueError, match="hybrid.expected_actor_backend"):
        HybridConfig(expected_actor_backend=backend_id)


@pytest.mark.parametrize(
    "adapter_sha256",
    [None, "A" * 64, "a" * 63, "b" * 64],
)
def test_hybrid_rejects_missing_malformed_or_mismatched_actor_adapter_hash(
    tmp_path: Path,
    adapter_sha256: str | None,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _InvalidIdentityActor(change={"adapter_sha256": adapter_sha256})

    with pytest.raises(ValueError, match="adapter.*SHA|adapter hash|adapter_sha256"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    assert learner._step == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]


@pytest.mark.parametrize("checksum_mode", ["missing", "unchanged"])
def test_hybrid_requires_per_step_policy_checksum_delta_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checksum_mode: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    if checksum_mode == "missing":
        monkeypatch.setattr(learner, "policy_parameter_checksum", None)
    else:
        monkeypatch.setattr(learner, "policy_parameter_checksum", lambda: "c" * 64)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    with pytest.raises(ValueError, match="policy checksum|policy parameters"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    checkpoints = Path(config.checkpoint.output_dir)
    assert not list(checkpoints.glob("checkpoint-*"))
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]


def test_hybrid_alias_rejection_closes_shared_backend_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    close_calls = 0

    def close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(learner, "close", close)
    with pytest.raises(ValueError, match="separate backends"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: learner,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    assert close_calls == 1


def test_hybrid_learner_failure_closes_actor_and_learner_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    learner_close_calls = 0

    def fail_evaluate(batch: object) -> None:
        del batch
        raise RuntimeError("learner evaluation failure")

    def close_learner() -> None:
        nonlocal learner_close_calls
        learner_close_calls += 1

    monkeypatch.setattr(learner, "evaluate", fail_evaluate)
    monkeypatch.setattr(learner, "close", close_learner)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    with pytest.raises(RuntimeError, match="learner evaluation failure"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    assert actor.close_calls == 1
    assert learner_close_calls == 1


def test_hybrid_logs_typed_publication_only_after_atomic_success(tmp_path: Path) -> None:
    from gepa_mindfulness.training.run_logging import JSONLLoggingSink, PublicationRecord

    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.DOWN_WEIGHT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    result = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    ).train(max_steps=2)

    payloads = [
        json.loads(line)
        for line in (result.log_directory / "publications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    records = tuple(PublicationRecord.from_mapping(payload) for payload in payloads)
    assert [record.parent_policy_version for record in records] == ["1", "2"]
    assert [record.policy_version for record in records] == ["2", "3"]
    assert [record.global_step for record in records] == [1, 2]
    assert all(record.adapter_identifier == "tiny-lora" for record in records)
    assert all(record.model_identifier == "tiny-hybrid-model" for record in records)
    assert all(record.actor_backend == "mock-mojo-vulkan" for record in records)
    assert all(record.learner_backend == "torch_portable" for record in records)
    assert payloads == [record.to_dict() for record in records]
    sink = JSONLLoggingSink(result.log_directory)
    with pytest.raises(ValueError, match="publication identity"):
        sink.log_publication(
            replace(
                records[-1],
                record_id="publication-cross-run-mismatch",
                model_identifier="different-safe-model",
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        {"adapter_sha256": "A" * 64},
        {"global_step": True},
        {"model_identifier": "../unsafe"},
        {"policy_version": "03"},
        {"extra": "unknown"},
    ],
)
def test_publication_record_rejects_noncanonical_or_open_payloads(
    change: Mapping[str, object],
) -> None:
    from gepa_mindfulness.training.run_logging import PublicationRecord

    payload: dict[str, object] = {
        "record_id": "publication-1",
        "run_id": "run-1",
        "timestamp": "2026-08-01T00:00:00Z",
        "global_step": 1,
        "backend": "mock-mojo-vulkan",
        "actor_backend": "mock-mojo-vulkan",
        "learner_backend": "torch_portable",
        "parent_policy_version": "1",
        "policy_version": "2",
        "adapter_identifier": "tiny-lora",
        "adapter_sha256": "a" * 64,
        "model_identifier": "tiny-hybrid-model",
        "checkpoint_id": "checkpoint-00000001",
        "schema_version": 1,
    }
    payload.update(change)

    with pytest.raises((TypeError, ValueError)):
        PublicationRecord.from_mapping(payload)


def test_hybrid_publication_failure_emits_no_success_record(tmp_path: Path) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: _FailingPublisher(publisher),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="publisher failure"):
        engine.train(max_steps=1)

    publications = tuple(Path(config.logging.log_dir).glob("rl-*/publications.jsonl"))
    assert len(publications) == 1
    assert publications[0].read_bytes() == b""


def test_hybrid_downweight_changes_observable_reward_exactly_once(tmp_path: Path) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.DOWN_WEIGHT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    result = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
    ).train(max_steps=2)
    metrics = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rewards = {
        step: [
            record["metrics"]["response_reward"]
            for record in metrics
            if record["global_step"] == step and "response_reward" in record["metrics"]
        ]
        for step in (0, 1)
    }
    assert len(rewards[0]) == len(rewards[1]) == 2
    assert rewards[1] == pytest.approx([value * 0.5 for value in rewards[0]])


class _LoggerWithoutPublication:
    def start(self, *args: object) -> None:
        del args

    def trajectories(self, *args: object) -> None:
        del args

    def metrics(self, *args: object) -> None:
        del args


class _FailingPublicationLogger(_LoggerWithoutPublication):
    def publication(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("publication audit hook failed")


class _NoOpPublicationLogger(_LoggerWithoutPublication):
    def publication(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _TrackingHybridLogger(_NoOpPublicationLogger):
    def __init__(self) -> None:
        self.trajectory_calls = 0
        self.metric_calls = 0
        self.staleness_calls = 0
        self.metric_values: tuple[object, ...] = ()

    def trajectories(self, *args: object) -> None:
        del args
        self.trajectory_calls += 1

    def metrics(self, rewards: tuple[object, ...], global_step: int) -> None:
        del global_step
        self.metric_calls += 1
        self.metric_values = rewards

    def staleness(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.staleness_calls += 1


class _CheckpointWithId:
    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id

    def load(self, path: Path) -> object:
        del path
        raise AssertionError("load is not used by a training run")

    def save(self, global_step: int, parent_checkpoint: str | None) -> object:
        del global_step, parent_checkpoint
        return SimpleNamespace(checkpoint_id=self.checkpoint_id)


class _CountingRewardProvider:
    def __init__(self, value: float = 1.0) -> None:
        self.score_calls = 0
        self.value = value

    def score(self, request: object) -> float:
        del request
        self.score_calls += 1
        return self.value


class _ChangingCheckpointResult:
    def __init__(self, first_id: str, later_id: str = "changed-checkpoint") -> None:
        self.first_id = first_id
        self.later_id = later_id
        self.checkpoint_id_reads = 0

    @property
    def checkpoint_id(self) -> str:
        self.checkpoint_id_reads += 1
        if self.checkpoint_id_reads == 1:
            return self.first_id
        return self.later_id


class _ChangingCheckpoint:
    def __init__(self, *, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.parents: list[str | None] = []
        self.results: list[_ChangingCheckpointResult] = []

    def load(self, path: Path) -> object:
        del path
        raise AssertionError("load is not used by a training run")

    def save(self, global_step: int, parent_checkpoint: str | None) -> object:
        self.parents.append(parent_checkpoint)
        canonical = f"checkpoint-{global_step:08d}"
        first_id = "invalid-first-read" if self.invalid_first else canonical
        result = _ChangingCheckpointResult(first_id)
        self.results.append(result)
        return result


@pytest.mark.parametrize(
    "checkpoint_id",
    ["custom-checkpoint", "checkpoint-00000002", "checkpoint-000000001"],
)
def test_hybrid_rejects_noncanonical_checkpoint_before_export_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_id: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    export_calls = 0
    export_adapter = learner.export_adapter

    def record_export(*args: object, **kwargs: object) -> AdapterCandidate:
        nonlocal export_calls
        export_calls += 1
        return export_adapter(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(learner, "export_adapter", record_export)
    with pytest.raises(ValueError, match="checkpoint"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
            checkpoint_factory=lambda _, backend: _CheckpointWithId(checkpoint_id),
        ).train(max_steps=1)

    assert export_calls == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]
    publications = tuple(Path(config.logging.log_dir).glob("rl-*/publications.jsonl"))
    assert len(publications) == 1
    assert publications[0].read_bytes() == b""
    assert actor.close_calls == 1


def test_hybrid_rejects_forged_trajectory_backend_before_scoring_with_custom_logger(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _InvalidIdentityActor(change={"backend_name": "forged-backend"})
    rewards = _CountingRewardProvider()

    with pytest.raises(ValueError, match="backend"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
            checkpoint_factory=lambda _, backend: _CheckpointWithId("checkpoint-00000001"),
            logger_factory=lambda _: _NoOpPublicationLogger(),
            reward_factory=lambda _: rewards,  # type: ignore[arg-type]
        ).train(max_steps=1)

    assert rewards.score_calls == 0
    assert learner._step == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]
    assert actor.close_calls == 1


@pytest.mark.parametrize("mode", ["collect", "evaluate"])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_identifier": "forged-model"}, "model"),
        ({"model_identifier": None}, "model"),
        ({"adapter_identifier": "forged-adapter"}, "adapter"),
        ({"adapter_identifier": None}, "adapter"),
        ({"adapter_sha256": "b" * 64}, "adapter.*SHA|adapter hash"),
        ({"adapter_sha256": None}, "adapter.*SHA|adapter hash"),
        ({"policy_version": "2"}, "policy version"),
        ({"policy_version": None}, "policy version"),
    ],
)
def test_hybrid_collect_and_evaluate_reject_forged_provenance_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    change: Mapping[str, object],
    message: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _InvalidIdentityActor(change=change)
    rewards = _CountingRewardProvider()
    logger = _TrackingHybridLogger()
    evaluate_calls = 0
    evaluate = learner.evaluate

    def record_evaluate(batch: object):  # type: ignore[no-untyped-def]
        nonlocal evaluate_calls
        evaluate_calls += 1
        return evaluate(batch)  # type: ignore[arg-type]

    monkeypatch.setattr(learner, "evaluate", record_evaluate)
    engine = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
        logger_factory=lambda _: logger,
        reward_factory=lambda _: rewards,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        getattr(engine, mode)()

    assert logger.trajectory_calls == 0
    assert logger.metric_calls == 0
    assert logger.staleness_calls == 0
    assert rewards.score_calls == 0
    assert evaluate_calls == 0
    assert actor.close_calls == 1


def test_hybrid_evaluate_rejects_staleness_before_reward_or_learner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    rewards = _CountingRewardProvider(2.0)
    logger = _TrackingHybridLogger()
    evaluate_calls = 0
    evaluate = learner.evaluate
    decide = rl_engine.evaluate_staleness

    def force_one_version_lag(
        learner_version: PolicyVersion,
        actor_version: PolicyVersion,
        **kwargs: object,
    ):
        del learner_version
        return decide(PolicyVersion(2), actor_version, **kwargs)  # type: ignore[arg-type]

    def record_evaluate(batch: object):  # type: ignore[no-untyped-def]
        nonlocal evaluate_calls
        evaluate_calls += 1
        return evaluate(batch)  # type: ignore[arg-type]

    monkeypatch.setattr(rl_engine, "evaluate_staleness", force_one_version_lag)
    monkeypatch.setattr(learner, "evaluate", record_evaluate)

    with pytest.raises(ValueError, match="stale actor policy"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
            logger_factory=lambda _: logger,
            reward_factory=lambda _: rewards,  # type: ignore[arg-type]
        ).evaluate()

    assert logger.staleness_calls == 1
    assert logger.trajectory_calls == 0
    assert logger.metric_calls == 0
    assert rewards.score_calls == 0
    assert evaluate_calls == 0


def test_hybrid_evaluate_applies_staleness_weight_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.DOWN_WEIGHT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    rewards = _CountingRewardProvider(2.0)
    logger = _TrackingHybridLogger()
    evaluate_calls = 0
    evaluate = learner.evaluate
    decide = rl_engine.evaluate_staleness

    def force_one_version_lag(
        learner_version: PolicyVersion,
        actor_version: PolicyVersion,
        **kwargs: object,
    ):
        del learner_version
        return decide(PolicyVersion(2), actor_version, **kwargs)  # type: ignore[arg-type]

    def record_evaluate(batch: object):  # type: ignore[no-untyped-def]
        nonlocal evaluate_calls
        evaluate_calls += 1
        return evaluate(batch)  # type: ignore[arg-type]

    monkeypatch.setattr(rl_engine, "evaluate_staleness", force_one_version_lag)
    monkeypatch.setattr(learner, "evaluate", record_evaluate)

    RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
        logger_factory=lambda _: logger,
        reward_factory=lambda _: rewards,  # type: ignore[arg-type]
    ).evaluate()

    assert logger.staleness_calls == 1
    assert logger.trajectory_calls == 1
    assert logger.metric_calls == 1
    assert logger.metric_values == pytest.approx((1.0, 1.0))
    assert rewards.score_calls == 2
    assert evaluate_calls == 1


def test_hybrid_checkpoint_identity_is_read_once_and_reused_downstream(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.DOWN_WEIGHT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    checkpoint = _ChangingCheckpoint()

    result = RLTrainingEngine(
        config,
        capability_provider=_LearnerCapabilityProvider(learner),
        backend_factory=lambda _: learner,
        actor_factory=lambda _, manifest: actor,
        publisher_factory=lambda _: publisher,
        checkpoint_factory=lambda _, backend: checkpoint,
    ).train(max_steps=2)

    assert [item.checkpoint_id_reads for item in checkpoint.results] == [1, 1]
    assert checkpoint.parents == [None, "checkpoint-00000001"]
    assert result.published_adapter is not None
    assert result.published_adapter.metadata["checkpoint_id"] == "checkpoint-00000002"
    publications = [
        json.loads(line)
        for line in (result.log_directory / "publications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["checkpoint_id"] for item in publications] == [
        "checkpoint-00000001",
        "checkpoint-00000002",
    ]


def test_hybrid_invalid_first_checkpoint_identity_is_not_reread_or_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")
    checkpoint = _ChangingCheckpoint(invalid_first=True)
    export_calls = 0
    export_adapter = learner.export_adapter

    def record_export(*args: object, **kwargs: object) -> AdapterCandidate:
        nonlocal export_calls
        export_calls += 1
        return export_adapter(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(learner, "export_adapter", record_export)

    with pytest.raises(ValueError, match="checkpoint"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
            checkpoint_factory=lambda _, backend: checkpoint,
        ).train(max_steps=1)

    assert checkpoint.results[0].checkpoint_id_reads == 1
    assert export_calls == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]
    publications = tuple(Path(config.logging.log_dir).glob("rl-*/publications.jsonl"))
    assert len(publications) == 1
    assert publications[0].read_bytes() == b""


def test_hybrid_rejects_logger_without_publication_hook_before_actor_factory(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor_factory_calls = 0

    def actor_factory(selected: RLRunConfig, manifest: object) -> _MockVersionedActor:
        del selected, manifest
        nonlocal actor_factory_calls
        actor_factory_calls += 1
        return _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    with pytest.raises(ValueError, match="publication.*logger hook"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=actor_factory,
            publisher_factory=lambda _: publisher,
            logger_factory=lambda _: _LoggerWithoutPublication(),  # type: ignore[arg-type]
        ).train(max_steps=1)

    assert actor_factory_calls == 0
    assert publisher.current().policy_version == PolicyVersion(1)  # type: ignore[union-attr]


def test_hybrid_validates_default_publication_hook_before_actor_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor_factory_calls = 0

    def actor_factory(selected: RLRunConfig, manifest: object) -> _MockVersionedActor:
        del selected, manifest
        nonlocal actor_factory_calls
        actor_factory_calls += 1
        return _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    monkeypatch.setattr(rl_engine._JSONLRunLogger, "publication", None)
    with pytest.raises(ValueError, match="publication.*logger hook"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=actor_factory,
            publisher_factory=lambda _: publisher,
        ).train(max_steps=1)

    assert actor_factory_calls == 0


def test_publication_audit_hook_failure_returns_no_success_and_keeps_advanced_current(
    tmp_path: Path,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    actor = _MockVersionedActor(model_id="tiny-hybrid-model", adapter_id="tiny-lora")

    with pytest.raises(RuntimeError, match="publication audit hook failed"):
        RLTrainingEngine(
            config,
            capability_provider=_LearnerCapabilityProvider(learner),
            backend_factory=lambda _: learner,
            actor_factory=lambda _, manifest: actor,
            publisher_factory=lambda _: publisher,
            logger_factory=lambda _: _FailingPublicationLogger(),
        ).train(max_steps=1)

    current = publisher.current()
    assert current is not None
    assert current.policy_version == PolicyVersion(2)
    assert current.parent_policy_version == PolicyVersion(1)
    assert current.metadata["global_step"] == 1
    assert actor.close_calls == 1


@pytest.mark.parametrize("mode", ["collect", "evaluate"])
def test_hybrid_collect_and_evaluate_send_actor_manifest_through_real_transport(
    tmp_path: Path,
    fake_coordinator_factory: object,
    mode: str,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    config = replace(
        config,
        hybrid=replace(config.hybrid, expected_actor_backend="fake-mojo"),
    )
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    unused_transport = fake_coordinator_factory("hybrid_echo")  # type: ignore[operator]
    command = unused_transport.command
    unused_transport.close()
    engine = build_hybrid_engine(config, command)
    engine.capability_provider = _LearnerCapabilityProvider(learner)
    engine.backend_factory = lambda _: learner

    result = getattr(engine, mode)()

    assert len(result.trajectories) == 2
    assert {item.backend_name for item in result.trajectories} == {"fake-mojo"}
    assert {item.model_identifier for item in result.trajectories} == {"tiny-hybrid-model"}
    assert {item.adapter_identifier for item in result.trajectories} == {"tiny-lora"}
    assert {item.adapter_sha256 for item in result.trajectories} == {
        publisher.current().artifact_sha256  # type: ignore[union-attr]
    }
    assert {item.policy_version for item in result.trajectories} == {"1"}


def test_build_hybrid_engine_real_process_echoes_identity_through_publication(
    tmp_path: Path,
    fake_coordinator_factory: object,
) -> None:
    config = _hybrid_config(tmp_path, staleness=StalenessPolicy.REJECT)
    config = replace(
        config,
        hybrid=replace(config.hybrid, expected_actor_backend="fake-mojo"),
    )
    publisher = LocalAdapterPublisher(config.hybrid.adapter_store)
    _bootstrap_adapter(publisher)
    learner = _hybrid_learner(config)
    unused_transport = fake_coordinator_factory("hybrid_echo")  # type: ignore[operator]
    command = unused_transport.command
    unused_transport.close()
    engine = build_hybrid_engine(config, command)
    engine.capability_provider = _LearnerCapabilityProvider(learner)
    engine.backend_factory = lambda _: learner

    result = engine.train(max_steps=1)

    manifest = json.loads((result.log_directory / "run_manifest.json").read_text(encoding="utf-8"))
    trajectories = [
        json.loads(line)
        for line in (result.log_directory / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    metrics = [
        json.loads(line)
        for line in (result.log_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    publications = [
        json.loads(line)
        for line in (result.log_directory / "publications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.global_step == 1
    assert result.published_adapter is not None
    assert manifest["actor_backend"] == "fake-mojo"
    assert manifest["software_versions"] == {
        "actor_backend": "unobserved",
        "learner_backend": str(torch.__version__),
    }
    assert {record["actor_backend"] for record in trajectories + metrics + publications} == {
        "fake-mojo"
    }
    assert {record["backend"] for record in trajectories + metrics + publications} == {"fake-mojo"}
    assert {record["learner_backend"] for record in trajectories + metrics + publications} == {
        "torch_portable"
    }
    actor_policy = manifest["device_capabilities"]["hybrid_actor_policy"]
    trajectory = trajectories[0]["trajectory"]
    assert trajectory["model_identifier"] == actor_policy["model_identifier"]
    assert trajectory["adapter_identifier"] == actor_policy["adapter_identifier"]
    assert trajectory["adapter_sha256"] == actor_policy["adapter_sha256"]
    assert trajectory["policy_version"] == actor_policy["policy_version"]
    assert len(publications) == 1
    publication = publications[0]
    assert publication["parent_policy_version"] == "1"
    assert publication["policy_version"] == "2"
    assert publication["global_step"] == 1
    assert publication["checkpoint_id"] == "checkpoint-00000001"
    assert publication["adapter_identifier"] == actor_policy["adapter_identifier"]
    assert publication["adapter_sha256"] == result.published_adapter.artifact_sha256
    assert publication["model_identifier"] == actor_policy["model_identifier"]


def test_shipped_hybrid_config_bootstraps_safe_current_manifest(tmp_path: Path) -> None:
    config = load_rl_config("configs/rl/hybrid_vulkan_grpo.yaml")
    publisher = LocalAdapterPublisher(tmp_path / "adapters")
    source = TorchPolicyBackend(
        policy_model=_deterministic_hybrid_policy(),
        tokenizer=_HybridTokenizer(),
        training_mode="lora",
        model_identifier=config.hybrid.model_id,
        adapter_identifier="peft-lora",
    )
    candidate = source.export_adapter(
        tmp_path / "bootstrap-v1.pt",
        model_id=config.hybrid.model_id,
        policy_version=PolicyVersion(1),
        parent_policy_version=None,
    )
    published = publisher.publish(candidate)
    verified, payload = publisher.current_artifact()
    reader = TorchPolicyBackend(
        policy_model=_deterministic_hybrid_policy(),
        tokenizer=_HybridTokenizer(),
        training_mode="lora",
        model_identifier=config.hybrid.model_id,
        adapter_identifier="peft-lora",
    )

    assert verified == published
    assert (
        reader.load_adapter_bytes(payload, manifest=verified) == source.policy_parameter_checksum()
    )
    assert published.model_id == config.hybrid.model_id
    assert config.policy.model_name.startswith("/absolute/path/")
    assert config.hybrid.expected_actor_backend == "MOJO_ACTOR_BACKEND_ID"
    readme = Path("docs/rl/README.md").read_text(encoding="utf-8")
    assert "source checkout" in readme
    assert "not included in the wheel" in readme
    assert "create_portable_backend" in readme
    assert "export_adapter" in readme
    assert "current_artifact" in readme
    assert "load_adapter_bytes" in readme
    assert "BOOTSTRAP_ADAPTER" not in readme
    assert "ADAPTER_SOURCE_ID" not in readme
    assert "LocalAdapterPublisher" in readme
    assert "publisher.current()" in readme
    for item in (
        "policy.model_name",
        "hybrid.model_id",
        "hybrid.expected_actor_backend",
        "--coordinator-command",
        "--actor-endpoint",
        "publisher.current()",
    ):
        assert item in readme
    assert "remains advanced" in readme
    assert "reconcile" in readme
    assert "adapter_sha256" in readme
    assert "schema v1" in readme

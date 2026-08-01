"""Process-boundary tests for the Mojo actor coordinator protocol."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from gepa_mindfulness.training.backends.mojo_coordinator import (
    ActorHandshake,
    MojoCoordinatorBackend,
    MojoCoordinatorError,
    MojoProcessTransport,
)
from gepa_mindfulness.training.capability import Capability, CapabilityState
from gepa_mindfulness.training.contracts import ActorTransport, RolloutBackend
from gepa_mindfulness.training.trajectory import RolloutRequest

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
    return {
        "trajectory_id": "fake-%d-%d" % (request_index, sample_index),
        "case_id": request["case_id"],
        "prompt": request["prompt"],
        "response": "response-%d-%d" % (request_index, sample_index),
        "prompt_token_ids": None,
        "response_token_ids": None,
        "old_log_probs": None,
        "reference_log_probs": None,
        "value_predictions": None,
        "reward_total": None,
        "reward_components": {},
        "reward_component_evidence": {},
        "advantage": None,
        "return": None,
        "sampling_parameters": request["sampling_parameters"],
        "backend_name": "fake-mojo",
        "backend_version": "fake-1",
        "model_identifier": "fake-model",
        "adapter_identifier": None,
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
    request_id = message["request_id"]
    if mode == "wrong_request_id":
        request_id = "request-999"
    send("generate", request_id, {"trajectories": trajectories})
"""


@dataclass(frozen=True)
class FakeCoordinator:
    command: tuple[str, ...]


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
        '"request_id":"request-1","payload":{}}\n'
        '{"protocol_version":"gepa-actor-v1","type":"close",'
        '"request_id":"request-2","payload":{}}\n'
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

    generate_messages = (
        '{"protocol_version":"gepa-actor-v1","type":"hello",'
        '"request_id":"request-1","payload":{}}\n'
        '{"protocol_version":"gepa-actor-v1","type":"generate",'
        '"request_id":"request-2","payload":{"requests":[]}}\n'
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

    malformed = subprocess.run(
        ["mojo", "run", str(source)],
        input=messages.replace("request-1", "request-9", 1),
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
    assert "invalid gepa-actor-v1 hello frame" in source
    assert "invalid gepa-actor-v1 generate or close frame" in source

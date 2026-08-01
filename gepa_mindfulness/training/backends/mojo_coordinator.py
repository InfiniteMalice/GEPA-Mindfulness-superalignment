"""Bounded JSONL transport for a versioned external Mojo actor coordinator."""

from __future__ import annotations

import json
import math
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

from ..capability import BackendCapabilities, Capability, CapabilityEvidence, CapabilityState
from ..contracts import ActorTransport
from ..policy_versions import PolicyVersion
from ..trajectory import RolloutRequest, Trajectory

_PROTOCOL_VERSION = "gepa-actor-v1"
_BACKEND_NAME = "mojo-coordinator"
_DEFAULT_MAX_FRAME_BYTES = 1_048_576
_DEFAULT_MAX_STDERR_BYTES = 65_536
_MAX_FRAME_BYTES = 16_777_216
_MAX_STDERR_BYTES = 1_048_576
_MAX_TIMEOUT_SECONDS = 60.0
_MAX_IDENTITY_CHARACTERS = 128
_READ_BYTES = 4096
_ENVELOPE_FIELDS = frozenset({"protocol_version", "type", "request_id", "payload"})
_HANDSHAKE_FIELDS = frozenset({"backend_name", "backend_version"})
_ERROR_FIELDS = frozenset({"code", "message"})
_GENERATE_FIELDS = frozenset({"trajectories"})
_REQUEST_FIELDS = frozenset(
    {
        "case_id",
        "metadata",
        "num_samples",
        "policy_version",
        "prompt",
        "sampling_parameters",
        "seed",
    }
)
_TRAJECTORY_FIELDS = frozenset(
    {
        "adapter_identifier",
        "advantage",
        "backend_name",
        "backend_version",
        "case_id",
        "model_identifier",
        "old_log_probs",
        "policy_version",
        "prompt",
        "prompt_token_ids",
        "reference_log_probs",
        "response",
        "response_token_ids",
        "return",
        "reward_component_evidence",
        "reward_components",
        "reward_total",
        "sampling_parameters",
        "seed",
        "trace_references",
        "trajectory_id",
        "value_predictions",
    }
)
_OPTIONAL_TRAJECTORY_FIELDS = frozenset({"evidence_references"})
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_UNSUPPORTED_TRAINING_CAPABILITIES = frozenset(
    {
        Capability.SUPPORTS_BACKWARD,
        Capability.SUPPORTS_CUDA,
        Capability.SUPPORTS_DISTRIBUTED_TRAINING,
        Capability.SUPPORTS_FULL_WEIGHT_TRAINING,
        Capability.SUPPORTS_LORA_TRAINING,
        Capability.SUPPORTS_MIXED_PRECISION,
        Capability.SUPPORTS_OPTIMIZER_STEP,
        Capability.SUPPORTS_REFERENCE_LOG_PROBS,
        Capability.SUPPORTS_VALUE_HEAD,
    }
)


class MojoCoordinatorError(RuntimeError):
    """A safe transport, protocol, or actor-boundary failure."""


@dataclass(frozen=True, slots=True)
class ActorHandshake:
    """Observed identity returned by a successful coordinator handshake."""

    protocol_version: str
    backend_name: str
    backend_version: str

    def __post_init__(self) -> None:
        if self.protocol_version != _PROTOCOL_VERSION:
            raise ValueError("protocol_version must equal gepa-actor-v1")
        for field_name in ("backend_name", "backend_version"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or not value.strip()
                or len(value) > _MAX_IDENTITY_CHARACTERS
                or not all(character.isprintable() for character in value)
            ):
                raise ValueError(
                    f"{field_name} must be a safe nonblank string of at most "
                    f"{_MAX_IDENTITY_CHARACTERS} characters"
                )


class MojoProcessTransport:
    """Exchange one in-flight request at a time with a JSONL coordinator process."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_seconds: float = 5.0,
        request_timeout_seconds: float = 10.0,
        shutdown_timeout_seconds: float = 2.0,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
        max_stderr_bytes: int = _DEFAULT_MAX_STDERR_BYTES,
    ) -> None:
        self.command = _validated_command(command)
        self.startup_timeout_seconds = _bounded_timeout(
            startup_timeout_seconds,
            "startup_timeout_seconds",
        )
        self.request_timeout_seconds = _bounded_timeout(
            request_timeout_seconds,
            "request_timeout_seconds",
        )
        self.shutdown_timeout_seconds = _bounded_timeout(
            shutdown_timeout_seconds,
            "shutdown_timeout_seconds",
        )
        self.max_frame_bytes = _bounded_size(
            max_frame_bytes,
            "max_frame_bytes",
            _MAX_FRAME_BYTES,
        )
        self.max_stderr_bytes = _bounded_size(
            max_stderr_bytes,
            "max_stderr_bytes",
            _MAX_STDERR_BYTES,
        )
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=4)
        self._stderr = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_overflow = threading.Event()
        self._stderr_reader_ready = threading.Event()
        self._expectation_lock = threading.Condition()
        self._expected_response: tuple[str, str] | None = None
        self._response_state = "idle"
        self._commit_pending = False
        self._response_seen = False
        self._protocol_error: str | None = None
        self._writer_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._exchange_lock = threading.RLock()
        self._request_number = 0
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> ActorHandshake:
        """Start the child and complete a bounded strict handshake."""
        operation_deadline = time.monotonic() + self.startup_timeout_seconds
        with self._exchange_lock:
            if self._closed:
                raise RuntimeError("Mojo coordinator transport is closed")
            if self._started:
                raise RuntimeError("Mojo coordinator transport is already started")
            try:
                self._spawn(_reserved_exchange_deadline(operation_deadline))
                self._started = True
                payload = self._exchange(
                    "hello",
                    {},
                    _reserved_exchange_deadline(operation_deadline),
                )
                if set(payload) != _HANDSHAKE_FIELDS:
                    raise MojoCoordinatorError("Mojo coordinator handshake schema is invalid")
                backend_name = payload.get("backend_name")
                backend_version = payload.get("backend_version")
                if not _nonblank_string(backend_name) or not _nonblank_string(backend_version):
                    raise MojoCoordinatorError("Mojo coordinator handshake identity is invalid")
                assert isinstance(backend_name, str)
                assert isinstance(backend_version, str)
                return ActorHandshake(
                    protocol_version=_PROTOCOL_VERSION,
                    backend_name=backend_name,
                    backend_version=backend_version,
                )
            except MojoCoordinatorError:
                self._cleanup(operation_deadline)
                raise
            except (TypeError, ValueError) as exc:
                self._cleanup(operation_deadline)
                raise MojoCoordinatorError(
                    "Mojo coordinator handshake identity is invalid"
                ) from exc
            except OSError as exc:
                self._cleanup(operation_deadline)
                raise MojoCoordinatorError("Mojo coordinator process could not start") from exc

    def generate(
        self,
        requests: Sequence[Mapping[str, object]],
    ) -> tuple[Mapping[str, object], ...]:
        """Exchange one generate batch after a successful handshake."""
        operation_deadline = time.monotonic() + self.request_timeout_seconds
        with self._exchange_lock:
            if self._closed:
                raise RuntimeError("Mojo coordinator transport is closed")
            if not self._started:
                raise RuntimeError("Mojo coordinator transport is not started")
            request_list = list(requests)
            if not all(isinstance(item, Mapping) for item in request_list):
                raise TypeError("actor requests must be JSON objects")
            for index, request_value in enumerate(request_list):
                _validate_actor_request(request_value, f"actor requests[{index}]")
            try:
                payload = self._exchange(
                    "generate",
                    {"requests": request_list},
                    _reserved_exchange_deadline(operation_deadline),
                )
                if set(payload) != _GENERATE_FIELDS:
                    raise MojoCoordinatorError("Mojo generate response schema is invalid")
                trajectories = payload.get("trajectories")
                if not isinstance(trajectories, list) or not all(
                    isinstance(item, Mapping) for item in trajectories
                ):
                    raise MojoCoordinatorError("Mojo generate trajectories schema is invalid")
                return tuple(dict(item) for item in trajectories)
            except MojoCoordinatorError:
                self._cleanup(operation_deadline)
                raise

    def close(self) -> None:
        """Request shutdown once, then terminate or kill within bounded cleanup."""
        operation_deadline = time.monotonic() + self.shutdown_timeout_seconds
        with self._exchange_lock:
            if self._closed:
                return
            close_error: MojoCoordinatorError | None = None
            if self._started and self.process_running:
                try:
                    payload = self._exchange(
                        "close",
                        {},
                        _reserved_exchange_deadline(operation_deadline, reserve_fraction=0.5),
                    )
                    if payload:
                        raise MojoCoordinatorError("Mojo close response schema is invalid")
                except MojoCoordinatorError as exc:
                    close_error = exc
            self._cleanup(operation_deadline)
            if close_error is not None:
                raise close_error

    def _spawn(self, deadline: float) -> None:
        self._process = subprocess.Popen(
            list(self.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        if not self._stderr_reader_ready.wait(_remaining(deadline)):
            raise MojoCoordinatorError("Mojo coordinator stderr reader startup deadline expired")

    def _exchange(
        self,
        message_type: str,
        payload: Mapping[str, object],
        deadline: float,
    ) -> Mapping[str, object]:
        process = self._process
        if process is None or process.stdin is None:
            raise MojoCoordinatorError("Mojo coordinator process is unavailable")
        stdin = process.stdin
        if process.poll() is not None:
            raise self._exited_error(process.returncode)
        self._request_number += 1
        request_id = f"request-{self._request_number}"
        envelope: dict[str, object] = {
            "protocol_version": _PROTOCOL_VERSION,
            "type": message_type,
            "request_id": request_id,
            "payload": dict(payload),
        }
        try:
            encoded = json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise MojoCoordinatorError("Mojo request is not finite JSON") from exc
        if len(encoded) > self.max_frame_bytes:
            raise MojoCoordinatorError("Mojo request frame exceeds the configured limit")
        with self._expectation_lock:
            if self._expected_response is not None:
                raise MojoCoordinatorError("Mojo coordinator already has an active response")
            if self._protocol_error is not None:
                raise MojoCoordinatorError(self._protocol_error)
            self._expected_response = (message_type, request_id)
            self._response_state = "writing"
            self._commit_pending = False
            self._response_seen = False
        write_done = threading.Event()
        write_errors: list[BaseException] = []

        def write_frame() -> None:
            try:
                _write_all(stdin, encoded, deadline)
                with self._expectation_lock:
                    if self._expected_response != (message_type, request_id):
                        raise MojoCoordinatorError("Mojo coordinator write was cancelled")
                    self._commit_pending = True
                _write_all(stdin, b"\n", deadline)
                stdin.flush()
                with self._expectation_lock:
                    if self._expected_response == (message_type, request_id):
                        self._response_state = "awaiting"
                        self._commit_pending = False
                    self._expectation_lock.notify_all()
            except (BrokenPipeError, MojoCoordinatorError, OSError, TypeError, ValueError) as exc:
                write_errors.append(exc)
                with self._expectation_lock:
                    if self._expected_response == (message_type, request_id):
                        self._response_state = "write_failed"
                        self._commit_pending = False
                    self._expectation_lock.notify_all()
            finally:
                write_done.set()

        self._writer_thread = threading.Thread(
            target=write_frame,
            name="mojo-coordinator-writer",
            daemon=True,
        )
        self._writer_thread.start()
        try:
            if not write_done.wait(_remaining(deadline)):
                raise MojoCoordinatorError("Mojo coordinator write deadline expired")
            if write_errors:
                if isinstance(write_errors[0], MojoCoordinatorError):
                    raise write_errors[0]
                raise self._exited_error(process.poll()) from write_errors[0]
            return self._receive(message_type, request_id, deadline)
        finally:
            with self._expectation_lock:
                self._expected_response = None
                self._response_state = "idle"
                self._commit_pending = False
                self._expectation_lock.notify_all()

    def _receive(
        self,
        message_type: str,
        request_id: str,
        deadline: float,
    ) -> Mapping[str, object]:
        while True:
            with self._expectation_lock:
                protocol_error = self._protocol_error
            if protocol_error is not None:
                raise MojoCoordinatorError(protocol_error)
            if self._stderr_overflow.is_set():
                raise MojoCoordinatorError("Mojo coordinator stderr exceeded its byte limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise MojoCoordinatorError("Mojo coordinator total deadline expired")
            try:
                event, value = self._stdout_events.get(timeout=remaining)
            except queue.Empty as exc:
                raise MojoCoordinatorError("Mojo coordinator total deadline expired") from exc
            if event == "error":
                raise MojoCoordinatorError(str(value))
            if event == "eof":
                process = self._process
                returncode = process.poll() if process is not None else None
                raise self._exited_error(returncode)
            assert event == "frame" and isinstance(value, bytes)
            remaining = max(0.0, deadline - time.monotonic())
            # Separate OS pipes have no cross-stream ordering. Give the already-ready stderr
            # reader one bounded scheduling turn before accepting a stdout response frame.
            self._stderr_overflow.wait(min(remaining, 0.01))
            if self._stderr_overflow.is_set():
                raise MojoCoordinatorError("Mojo coordinator stderr exceeded its byte limit")
            with self._expectation_lock:
                protocol_error = self._protocol_error
            if protocol_error is not None:
                raise MojoCoordinatorError(protocol_error)
            return self._decode_envelope(value, message_type, request_id)

    def _decode_envelope(
        self,
        raw: bytes,
        message_type: str,
        request_id: str,
    ) -> Mapping[str, object]:
        if not raw:
            raise MojoCoordinatorError("Mojo coordinator emitted a blank stdout frame")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MojoCoordinatorError("Mojo coordinator frame is not valid UTF-8") from exc
        try:
            decoded = json.loads(
                text,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_object_keys,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise MojoCoordinatorError("Mojo coordinator frame is not valid finite JSON") from exc
        _validate_json_value(decoded, "Mojo coordinator frame")
        if not isinstance(decoded, Mapping) or set(decoded) != _ENVELOPE_FIELDS:
            raise MojoCoordinatorError("Mojo coordinator envelope schema is invalid")
        if decoded.get("protocol_version") != _PROTOCOL_VERSION:
            raise MojoCoordinatorError("Mojo coordinator protocol version is invalid")
        if decoded.get("request_id") != request_id:
            raise MojoCoordinatorError("Mojo coordinator response request ID is invalid")
        response_type = decoded.get("type")
        payload = decoded.get("payload")
        if not isinstance(payload, Mapping):
            raise MojoCoordinatorError("Mojo coordinator response payload is invalid")
        if response_type == "error":
            self._raise_actor_error(payload)
        if response_type != message_type:
            raise MojoCoordinatorError("Mojo coordinator response type is invalid")
        return dict(payload)

    def _raise_actor_error(self, payload: Mapping[str, object]) -> None:
        if set(payload) != _ERROR_FIELDS:
            raise MojoCoordinatorError("Mojo coordinator error schema is invalid")
        code = payload.get("code")
        message = payload.get("message")
        if (
            not isinstance(code, str)
            or _ERROR_CODE.fullmatch(code) is None
            or not _nonblank_string(message)
        ):
            raise MojoCoordinatorError("Mojo coordinator error schema is invalid")
        raise MojoCoordinatorError(f"Mojo coordinator returned error code {code!r}")

    def _read_stdout(self, stream: BinaryIO) -> None:
        pending = bytearray()
        try:
            while True:
                chunk = stream.read(_READ_BYTES)
                if not chunk:
                    if pending:
                        self._put_stdout_event(
                            "error",
                            "Mojo coordinator stdout ended with a partial frame without newline",
                        )
                    else:
                        self._put_stdout_event("eof", None)
                    return
                pending.extend(chunk)
                while b"\n" in pending:
                    raw, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    if len(raw) > self.max_frame_bytes:
                        self._put_stdout_event(
                            "error",
                            "Mojo coordinator stdout frame exceeded its byte limit",
                        )
                        return
                    self._put_stdout_frame(bytes(raw))
                if len(pending) > self.max_frame_bytes:
                    self._put_stdout_event(
                        "error",
                        "Mojo coordinator stdout frame exceeded its byte limit",
                    )
                    return
        except (OSError, ValueError):
            self._put_stdout_event("eof", None)

    def _put_stdout_event(self, event: str, value: object) -> None:
        try:
            self._stdout_events.put_nowait((event, value))
        except queue.Full:
            try:
                self._stdout_events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._stdout_events.put_nowait(
                    ("error", "Mojo coordinator emitted unsolicited stdout frames")
                )
            except queue.Full:
                pass

    def _put_stdout_frame(self, raw: bytes) -> None:
        with self._expectation_lock:
            if self._expected_response is None:
                self._protocol_error = "Mojo coordinator emitted an unsolicited response frame"
                return
            if self._response_state == "writing":
                if not self._commit_pending:
                    self._protocol_error = (
                        "Mojo coordinator emitted a response before its request write committed"
                    )
                    return
                self._expectation_lock.wait_for(
                    lambda: self._response_state != "writing",
                )
                if self._expected_response is None:
                    return
            if self._response_state != "awaiting":
                self._protocol_error = "Mojo coordinator response state is invalid"
                return
            if self._response_seen:
                self._protocol_error = "Mojo coordinator emitted a duplicate response frame"
                return
            self._response_seen = True
        self._put_stdout_event("frame", raw)

    def _read_stderr(self, stream: BinaryIO) -> None:
        self._stderr_reader_ready.set()
        try:
            while True:
                chunk = stream.read(_READ_BYTES)
                if not chunk:
                    return
                with self._stderr_lock:
                    remaining = self.max_stderr_bytes - len(self._stderr)
                    if len(chunk) > remaining:
                        if remaining > 0:
                            self._stderr.extend(chunk[:remaining])
                        self._stderr_overflow.set()
                    else:
                        self._stderr.extend(chunk)
        except (OSError, ValueError):
            return

    def _exited_error(self, returncode: int | None) -> MojoCoordinatorError:
        with self._stderr_lock:
            captured = len(self._stderr)
            overflow = self._stderr_overflow.is_set()
        suffix = " (stderr limit exceeded)" if overflow else ""
        return MojoCoordinatorError(
            f"Mojo coordinator exited with status {returncode}; captured {captured} stderr bytes"
            f"{suffix}"
        )

    def _cleanup(self, deadline: float | None = None) -> None:
        process = self._process
        self._closed = True
        if process is None:
            return
        if deadline is None:
            deadline = time.monotonic() + self.shutdown_timeout_seconds
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=_remaining(deadline) / 3.0)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=_remaining(deadline) / 2.0)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=_remaining(deadline))
            except subprocess.TimeoutExpired:
                pass
        writer = self._writer_thread
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=_remaining(deadline))
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                _close_stream(stream)
        for worker in (self._stdout_thread, self._stderr_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=_remaining(deadline))


class MojoCoordinatorBackend:
    """Restore strictly correlated trajectories from a Mojo actor transport."""

    def __init__(self, transport: ActorTransport) -> None:
        if not isinstance(transport, ActorTransport):
            raise TypeError("transport must implement ActorTransport")
        self.transport = transport
        self._handshake: ActorHandshake | None = None
        self._generation_observed = False
        self._closed = False

    def generate(self, requests: Sequence[RolloutRequest]) -> tuple[Trajectory, ...]:
        """Generate and validate one ordered trajectory for every requested sample."""
        if self._closed:
            raise RuntimeError("Mojo coordinator backend is closed")
        prepared, expected = _prepare_requests(requests)
        if not prepared:
            return ()
        try:
            if self._handshake is None:
                try:
                    handshake = self.transport.start()
                except (TypeError, ValueError) as exc:
                    raise MojoCoordinatorError(
                        "Actor transport returned an invalid handshake"
                    ) from exc
                if not isinstance(handshake, ActorHandshake):
                    raise MojoCoordinatorError("Actor transport returned an invalid handshake")
                self._handshake = handshake
            raw_trajectories = self.transport.generate(prepared)
            if len(raw_trajectories) != len(expected):
                raise MojoCoordinatorError("Mojo trajectory count does not match actor requests")
            trajectories: list[Trajectory] = []
            identifiers: set[str] = set()
            for index, (raw, request) in enumerate(zip(raw_trajectories, expected, strict=True)):
                keys = set(raw)
                if (
                    keys != _TRAJECTORY_FIELDS
                    and keys != _TRAJECTORY_FIELDS | _OPTIONAL_TRAJECTORY_FIELDS
                ):
                    raise MojoCoordinatorError("Mojo trajectory common schema is invalid")
                try:
                    _validate_json_value(raw, f"Mojo trajectory {index}")
                    trajectory = Trajectory.from_dict(raw)
                except (TypeError, ValueError, RecursionError) as exc:
                    raise MojoCoordinatorError("Mojo trajectory common schema is invalid") from exc
                if trajectory.trajectory_id in identifiers:
                    raise MojoCoordinatorError("Mojo trajectory IDs must not contain duplicates")
                identifiers.add(trajectory.trajectory_id)
                if trajectory.prompt != request.prompt:
                    raise MojoCoordinatorError(
                        "Mojo trajectory order or prompt correlation is invalid"
                    )
                if trajectory.case_id != request.case_id:
                    raise MojoCoordinatorError(
                        "Mojo trajectory order or case correlation is invalid"
                    )
                if trajectory.policy_version != request.policy_version:
                    raise MojoCoordinatorError(
                        "Mojo trajectory policy_version correlation is invalid"
                    )
                assert self._handshake is not None
                if (
                    trajectory.backend_name != self._handshake.backend_name
                    or trajectory.backend_version != self._handshake.backend_version
                ):
                    raise MojoCoordinatorError(
                        "Mojo trajectory backend identity provenance is invalid"
                    )
                trajectories.append(trajectory)
        except (MojoCoordinatorError, TypeError, ValueError):
            self._poison()
            raise
        self._generation_observed = True
        return tuple(trajectories)

    def capabilities(self) -> BackendCapabilities:
        """Report only generation observed through a valid coordinator exchange."""
        capabilities: dict[Capability, CapabilityEvidence] = {}
        for capability in Capability:
            if capability in _UNSUPPORTED_TRAINING_CAPABILITIES:
                capabilities[capability] = CapabilityEvidence(
                    state=CapabilityState.UNSUPPORTED,
                    evidence=(
                        "The Mojo coordinator boundary is actor-only and exposes no "
                        f"{capability.value} operation."
                    ),
                )
            else:
                capabilities[capability] = CapabilityEvidence(
                    state=CapabilityState.UNKNOWN,
                    evidence=f"No positive {capability.value} evidence has been observed.",
                )
        if self._generation_observed:
            capabilities[Capability.SUPPORTS_GENERATION] = CapabilityEvidence(
                state=CapabilityState.SUPPORTED,
                evidence="A correlated versioned generate response was validated.",
            )
        version = self._handshake.backend_version if self._handshake else "unknown"
        return BackendCapabilities(
            backend_name=_BACKEND_NAME,
            backend_version=version,
            capabilities=capabilities,
        )

    def close(self) -> None:
        """Close the transport idempotently."""
        if not self._closed:
            self.transport.close()
            self._closed = True

    def _poison(self) -> None:
        try:
            self.transport.close()
        except (MojoCoordinatorError, RuntimeError):
            pass
        self._closed = True


def _prepare_requests(
    requests: Sequence[RolloutRequest],
) -> tuple[tuple[Mapping[str, object], ...], tuple[RolloutRequest, ...]]:
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("requests must be a sequence of RolloutRequest values")
    prepared: list[Mapping[str, object]] = []
    expected: list[RolloutRequest] = []
    for request in requests:
        if not isinstance(request, RolloutRequest):
            raise TypeError("requests must contain RolloutRequest values")
        if not _nonblank_string(request.prompt):
            raise ValueError("prompt must be a nonblank string")
        if request.case_id is not None and not isinstance(request.case_id, str):
            raise ValueError("case_id must be a string or null")
        if type(request.num_samples) is not int or request.num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        try:
            version = PolicyVersion.from_json(request.policy_version)
        except ValueError as exc:
            raise ValueError("policy_version must be a canonical policy version") from exc
        if request.seed is not None and (type(request.seed) is not int or request.seed < 0):
            raise ValueError("seed must be a non-negative integer or null")
        payload: dict[str, object] = {
            "case_id": request.case_id,
            "metadata": dict(request.metadata),
            "num_samples": request.num_samples,
            "policy_version": version.to_json(),
            "prompt": request.prompt,
            "sampling_parameters": dict(request.sampling_parameters),
            "seed": request.seed,
        }
        if set(payload) != _REQUEST_FIELDS:
            raise AssertionError("internal actor request schema is invalid")
        _validate_json_value(payload, "actor request")
        prepared.append(payload)
        expected.extend(request for _ in range(request.num_samples))
    return tuple(prepared), tuple(expected)


def _validate_actor_request(value: Mapping[str, object], field_name: str) -> None:
    if set(value) != _REQUEST_FIELDS:
        raise ValueError(f"{field_name} request schema is invalid")
    prompt = value.get("prompt")
    if not _nonblank_string(prompt):
        raise ValueError(f"{field_name} prompt must be a nonblank string")
    case_id = value.get("case_id")
    if case_id is not None and type(case_id) is not str:
        raise ValueError(f"{field_name} case_id must be a string or null")
    num_samples = value.get("num_samples")
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError(f"{field_name} num_samples must be a positive integer")
    try:
        PolicyVersion.from_json(value.get("policy_version"))
    except ValueError as exc:
        raise ValueError(f"{field_name} policy_version must be canonical") from exc
    seed = value.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError(f"{field_name} seed must be a non-negative integer or null")
    for mapping_name in ("metadata", "sampling_parameters"):
        mapping_value = value.get(mapping_name)
        if not isinstance(mapping_value, Mapping) or not all(
            isinstance(key, str) for key in mapping_value
        ):
            raise ValueError(f"{field_name} {mapping_name} must be a string-keyed object")
    _validate_json_value(value, field_name)


def _validated_command(command: object) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise TypeError("command must be a nonempty sequence of strings")
    if not all(
        type(argument) is str and argument and "\x00" not in argument for argument in command
    ):
        raise TypeError("command arguments must be nonempty strings without NUL bytes")
    arguments = tuple(command)
    executable = shutil.which(arguments[0])
    if executable is None:
        candidate = Path(arguments[0])
        if not candidate.is_file():
            raise ValueError("command executable could not be selected")
        executable = str(candidate.resolve())
    return (executable, *arguments[1:])


def _bounded_timeout(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 < value <= _MAX_TIMEOUT_SECONDS
    ):
        raise ValueError(f"{field_name} must be finite and in (0, {_MAX_TIMEOUT_SECONDS:g}]")
    return float(value)


def _bounded_size(value: object, field_name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{field_name} must be an integer in [1, {maximum}]")
    return value


def _nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _write_all(stream: object, data: bytes, deadline: float) -> None:
    write = getattr(stream, "write", None)
    if not callable(write):
        raise MojoCoordinatorError("Mojo coordinator pipe write is unavailable")
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        if _remaining(deadline) <= 0.0:
            raise MojoCoordinatorError("Mojo coordinator write deadline expired")
        try:
            progress = write(view[offset:])
        except (BrokenPipeError, OSError, TypeError, ValueError) as exc:
            raise MojoCoordinatorError("Mojo coordinator pipe write failed") from exc
        remaining_bytes = len(view) - offset
        if type(progress) is not int or not 0 < progress <= remaining_bytes:
            raise MojoCoordinatorError("Mojo coordinator pipe write progress is invalid")
        offset += progress
    if _remaining(deadline) <= 0.0:
        raise MojoCoordinatorError("Mojo coordinator write deadline expired")


def _reserved_exchange_deadline(
    operation_deadline: float,
    *,
    reserve_fraction: float = 0.5,
) -> float:
    remaining = _remaining(operation_deadline)
    return time.monotonic() + remaining * (1.0 - reserve_fraction)


def _close_stream(stream: object) -> None:
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        close()
    except (OSError, ValueError):
        pass


def _validate_json_value(value: object, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{field_name} contains a non-string object key")
        for key, item in value.items():
            _validate_json_value(item, f"{field_name}.{key}")
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


__all__ = [
    "ActorHandshake",
    "MojoCoordinatorBackend",
    "MojoCoordinatorError",
    "MojoProcessTransport",
]

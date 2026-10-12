"""A single bounded HTTPS operation, isolated so DNS cannot evade its deadline."""

# Standard library
from __future__ import annotations

import base64
import json
import multiprocessing
import re
import ssl
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

# Third-party
# Local

MAX_BODY = 1024 * 1024
MAX_IPC = 2 * MAX_BODY
ENDPOINTS = ("responses", "responses/input_tokens")


@dataclass(frozen=True)
class TransportResult:
    """Bounded, credential-redacted evidence; content remains untrusted."""

    status: int | None
    request_id: str | None
    body: bytes
    error_code: str | None
    redacted: bool
    incomplete: bool
    started_at: str
    ended_at: str
    elapsed_seconds: float


class Transport(Protocol):
    def post(self, endpoint: str, payload: dict[str, Any]) -> TransportResult: ...


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result(
    *, body: bytes = b"", status: int | None = None, error: str | None = None
) -> TransportResult:
    stamp = _stamp()
    return TransportResult(status, None, body, error, False, False, stamp, stamp, 0.0)


class NoRedirect(HTTPRedirectHandler):
    """Never forward a bearer credential or resend a request to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _redact(data: bytes, key: str) -> bytes:
    # JSON may escape credential characters; sanitize either representation.
    pattern = b"".join(
        b"(?:" + re.escape(c.encode("ascii")) + b"|\\\\u00" + f"{ord(c):02x}".encode() + b")"
        for c in key
    )
    return re.sub(pattern, b"[REDACTED]", data, flags=re.IGNORECASE)


def _perform_http(endpoint: str, payload: dict[str, Any], key: str) -> TransportResult:
    """Worker-side operation; all errors become controlled codes before IPC."""
    if endpoint not in ENDPOINTS:
        raise ValueError("unsupported endpoint")
    start, stamp = time.monotonic(), _stamp()
    result = _result(error="transport_error")
    try:
        request = Request(
            "https://api.openai.com/v1/" + endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        opener = build_opener(
            ProxyHandler({}), NoRedirect(), HTTPSHandler(context=ssl.create_default_context())
        )
        try:
            response = opener.open(request, timeout=45)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(MAX_BODY + 1)
            body = _redact(raw, key)
            request_id = (
                _redact(str(response.headers.get("x-request-id", "")).encode("utf-8"), key).decode(
                    "utf-8"
                )[:256]
                or None
            )
            status = response.status
            oversized = len(raw) > MAX_BODY
            result = TransportResult(
                status,
                request_id,
                body[:MAX_BODY],
                "response_too_large" if oversized else (None if status == 200 else "http_error"),
                body != raw or request_id != (response.headers.get("x-request-id") or None),
                oversized,
                stamp,
                _stamp(),
                time.monotonic() - start,
            )
    except Exception:
        # An exception may embed the request or secret; never stringify it.
        pass
    return replace(
        result, started_at=stamp, ended_at=_stamp(), elapsed_seconds=time.monotonic() - start
    )


def _encode_result(result: TransportResult) -> bytes:
    value = asdict(result)
    value["body"] = base64.b64encode(result.body).decode("ascii")
    return json.dumps(value, allow_nan=False).encode("utf-8")


def _http_worker(connection: Any, endpoint: str, payload: dict[str, Any], key: str) -> None:
    try:
        connection.send_bytes(_encode_result(_perform_http(endpoint, payload, key)))
    finally:
        connection.close()


class LiveTransport:
    """Spawn one operation with a parent-enforced wall-clock deadline and no retry."""

    def __init__(
        self, key: str, *, deadline_seconds: float = 45, worker: Callable[..., None] = _http_worker
    ) -> None:
        self._key = key
        self._deadline = min(45.0, max(0.01, deadline_seconds))
        self._worker = worker

    def post(self, endpoint: str, payload: dict[str, Any]) -> TransportResult:
        if endpoint not in ENDPOINTS:
            raise ValueError("unsupported endpoint")
        start, stamp = time.monotonic(), _stamp()
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=self._worker, args=(sender, endpoint, payload, self._key))
        received: list[bytes] = []
        ready = threading.Event()

        def receive() -> None:
            try:
                received.append(receiver.recv_bytes(MAX_IPC))
            except (EOFError, OSError):
                pass
            finally:
                ready.set()

        reader = threading.Thread(target=receive, daemon=True)
        result = _result(error="worker_error")
        try:
            process.start()
            sender.close()
            reader.start()
            remaining = max(0.0, self._deadline - (time.monotonic() - start))
            if not ready.wait(remaining):
                result = _result(error="deadline_exceeded")
            elif received and time.monotonic() - start < self._deadline:
                value = json.loads(received[0])
                value["body"] = base64.b64decode(value["body"], validate=True)
                if len(value["body"]) > MAX_BODY:
                    raise ValueError("worker body exceeds limit")
                result = TransportResult(**value)
        except Exception:
            result = _result(error="worker_error")
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join(2)
                if not process.is_alive():
                    process.close()
            sender.close()
            receiver.close()
            if reader.ident is not None:
                reader.join(1)
        return replace(
            result, started_at=stamp, ended_at=_stamp(), elapsed_seconds=time.monotonic() - start
        )


def make_live_transport(credential_file: Path) -> Transport:
    """Read only the explicitly selected credential after caller validation/ownership."""
    with credential_file.open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("invalid credential")
    try:
        key = raw.decode("ascii").strip()
    except UnicodeError:
        raise ValueError("invalid credential") from None
    if not re.fullmatch(r"sk-[A-Za-z0-9_-]{8,512}", key):
        raise ValueError("invalid credential")
    return LiveTransport(key)

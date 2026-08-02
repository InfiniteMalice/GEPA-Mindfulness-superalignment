"""Deterministic tests for the shared Windows byte-lock implementation."""

from __future__ import annotations

import errno
import importlib
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gepa_mindfulness.training.adapter_publication as publication_module
import gepa_mindfulness.training.run_logging as logging_module


class _FakeMsvcrt:
    LK_NBLCK = 1
    LK_UNLCK = 2
    LK_LOCK = 3

    def __init__(
        self,
        failures: list[OSError],
        *,
        unlock_error: OSError | None = None,
    ) -> None:
        self.failures = failures
        self.unlock_error = unlock_error
        self.calls: list[tuple[int, int, int]] = []
        self.positions: list[int] = []

    def locking(self, descriptor: int, mode: int, size: int) -> None:
        self.calls.append((descriptor, mode, size))
        self.positions.append(os.lseek(descriptor, 0, os.SEEK_CUR))
        if mode == self.LK_NBLCK and self.failures:
            raise self.failures.pop(0)
        if mode == self.LK_UNLCK and self.unlock_error is not None:
            raise self.unlock_error


def _windows_lock_module() -> Any:
    return importlib.import_module("gepa_mindfulness.training._windows_file_lock")


def test_windows_byte_lock_retries_two_contention_errors_without_blocking(
    tmp_path: Path,
) -> None:
    lock_module = _windows_lock_module()
    fake_msvcrt = _FakeMsvcrt(
        [
            OSError(errno.EACCES, "first contention"),
            OSError(errno.EACCES, "second contention"),
        ]
    )
    sleeps: list[float] = []
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with lock_module.windows_byte_lock(
            descriptor,
            timeout=30.0,
            retry_interval=0.01,
            msvcrt_module=fake_msvcrt,
            monotonic=lambda: 0.0,
            sleeper=sleeps.append,
        ):
            pass
    finally:
        os.close(descriptor)

    modes = [call[1] for call in fake_msvcrt.calls]
    assert modes == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert fake_msvcrt.LK_LOCK not in modes
    assert sleeps == [0.01, 0.01]


def test_windows_byte_lock_retries_native_lock_violation(tmp_path: Path) -> None:
    lock_module = _windows_lock_module()
    contention = OSError(errno.EPERM, "native lock violation")
    contention.winerror = 33
    fake_msvcrt = _FakeMsvcrt([contention])
    sleeps: list[float] = []
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with lock_module.windows_byte_lock(
            descriptor,
            msvcrt_module=fake_msvcrt,
            monotonic=lambda: 0.0,
            sleeper=sleeps.append,
        ):
            pass
    finally:
        os.close(descriptor)

    assert [call[1] for call in fake_msvcrt.calls] == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert sleeps == [0.05]


def test_windows_byte_lock_times_out_at_monotonic_deadline(tmp_path: Path) -> None:
    lock_module = _windows_lock_module()
    fake_msvcrt = _FakeMsvcrt([OSError(errno.EACCES, "contention") for _ in range(10)])
    now = 100.0
    sleeps: list[float] = []

    def sleep(duration: float) -> None:
        nonlocal now
        sleeps.append(duration)
        now += duration

    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(TimeoutError, match="lock contention.*timed out"):
            with lock_module.windows_byte_lock(
                descriptor,
                timeout=0.025,
                retry_interval=0.01,
                msvcrt_module=fake_msvcrt,
                monotonic=lambda: now,
                sleeper=sleep,
            ):
                pytest.fail("the lock body must not run after deadline exhaustion")
    finally:
        os.close(descriptor)

    assert sum(sleeps) == pytest.approx(0.025)
    assert all(0.0 < duration <= 0.01 for duration in sleeps)
    assert fake_msvcrt.calls[-1][1] == fake_msvcrt.LK_NBLCK


def test_windows_byte_lock_timeout_zero_attempts_exactly_once(tmp_path: Path) -> None:
    lock_module = _windows_lock_module()
    fake_msvcrt = _FakeMsvcrt([OSError(errno.EACCES, "contention")])
    sleeps: list[float] = []
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(TimeoutError, match="lock contention.*timed out"):
            with lock_module.windows_byte_lock(
                descriptor,
                timeout=0.0,
                msvcrt_module=fake_msvcrt,
                monotonic=lambda: 10.0,
                sleeper=sleeps.append,
            ):
                pytest.fail("the lock body must not run after deadline exhaustion")
    finally:
        os.close(descriptor)

    assert [call[1] for call in fake_msvcrt.calls] == [fake_msvcrt.LK_NBLCK]
    assert sleeps == []


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        ("4", TypeError),
        (1.5, TypeError),
        (-1, ValueError),
    ],
)
def test_windows_byte_lock_rejects_invalid_descriptor(
    value: object,
    error_type: type[Exception],
) -> None:
    lock_module = _windows_lock_module()

    with pytest.raises(error_type, match="descriptor.*non-negative integer"):
        with lock_module.windows_byte_lock(value, msvcrt_module=_FakeMsvcrt([])):
            pass


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("timeout", True, "timeout.*finite number"),
        ("timeout", "30", "timeout.*finite number"),
        ("timeout", math.inf, "timeout.*finite number"),
        ("timeout", -0.01, "timeout.*non-negative"),
        ("retry_interval", False, "retry interval.*finite number"),
        ("retry_interval", None, "retry interval.*finite number"),
        ("retry_interval", math.nan, "retry interval.*finite number"),
        ("retry_interval", 0.0, "retry interval.*positive"),
    ],
)
def test_windows_byte_lock_rejects_invalid_timing_argument(
    argument: str,
    value: object,
    message: str,
) -> None:
    lock_module = _windows_lock_module()
    kwargs = {argument: value}

    with pytest.raises((TypeError, ValueError), match=message):
        with lock_module.windows_byte_lock(0, msvcrt_module=_FakeMsvcrt([]), **kwargs):
            pass


@pytest.mark.parametrize("error_number", [errno.EAGAIN, errno.EPERM])
def test_windows_byte_lock_reraises_non_contention_error_unchanged(
    tmp_path: Path,
    error_number: int,
) -> None:
    lock_module = _windows_lock_module()
    original = OSError(error_number, "not lock contention", "publication.lock")
    fake_msvcrt = _FakeMsvcrt([original])
    sleeps: list[float] = []
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(OSError) as raised:
            with lock_module.windows_byte_lock(
                descriptor,
                msvcrt_module=fake_msvcrt,
                monotonic=lambda: 0.0,
                sleeper=sleeps.append,
            ):
                pytest.fail("the lock body must not run after a non-contention error")
    finally:
        os.close(descriptor)

    assert raised.value is original
    assert sleeps == []


def test_windows_byte_lock_reraises_acquisition_seek_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_module = _windows_lock_module()
    original = OSError(errno.EBADF, "seek failed", "publication.lock")
    fake_msvcrt = _FakeMsvcrt([])

    def fail_seek(descriptor: int, position: int, whence: int) -> int:
        del descriptor, position, whence
        raise original

    monkeypatch.setattr(lock_module.os, "lseek", fail_seek)

    with pytest.raises(OSError) as raised:
        with lock_module.windows_byte_lock(7, msvcrt_module=fake_msvcrt):
            pytest.fail("the lock body must not run after a seek error")

    assert raised.value is original
    assert fake_msvcrt.calls == []


def test_windows_byte_lock_seeks_to_zero_before_unlocking_after_body_error(
    tmp_path: Path,
) -> None:
    lock_module = _windows_lock_module()
    fake_msvcrt = _FakeMsvcrt([])
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        os.lseek(descriptor, 1, os.SEEK_SET)
        with pytest.raises(RuntimeError, match="body failed"):
            with lock_module.windows_byte_lock(
                descriptor,
                msvcrt_module=fake_msvcrt,
            ):
                os.lseek(descriptor, 1, os.SEEK_SET)
                raise RuntimeError("body failed")
    finally:
        os.close(descriptor)

    assert [call[1] for call in fake_msvcrt.calls] == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert fake_msvcrt.positions == [0, 0]


def test_windows_byte_lock_reraises_unlock_error_unchanged(tmp_path: Path) -> None:
    lock_module = _windows_lock_module()
    original = OSError(errno.EIO, "unlock failed", "publication.lock")
    fake_msvcrt = _FakeMsvcrt([], unlock_error=original)
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(OSError) as raised:
            with lock_module.windows_byte_lock(descriptor, msvcrt_module=fake_msvcrt):
                pass
    finally:
        os.close(descriptor)

    assert raised.value is original


def test_windows_byte_lock_unlock_error_preserves_body_error_as_context(
    tmp_path: Path,
) -> None:
    lock_module = _windows_lock_module()
    body_error = RuntimeError("body failed")
    unlock_error = OSError(errno.EIO, "unlock failed", "publication.lock")
    fake_msvcrt = _FakeMsvcrt([], unlock_error=unlock_error)
    lock_path = tmp_path / "lock"
    lock_path.write_bytes(b"0")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(OSError) as raised:
            with lock_module.windows_byte_lock(descriptor, msvcrt_module=fake_msvcrt):
                raise body_error
    finally:
        os.close(descriptor)

    assert raised.value is unlock_error
    assert raised.value.__context__ is body_error


def test_adapter_publication_windows_path_uses_shared_byte_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors: list[int] = []

    @contextmanager
    def recording_lock(descriptor: int) -> Iterator[None]:
        descriptors.append(descriptor)
        yield

    monkeypatch.setattr(publication_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(publication_module, "windows_byte_lock", recording_lock)

    with publication_module._descriptor_lock(41):
        pass

    assert descriptors == [41]


def test_jsonl_logging_windows_path_uses_shared_byte_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors: list[int] = []

    @contextmanager
    def recording_lock(descriptor: int) -> Iterator[None]:
        descriptors.append(descriptor)
        yield

    stream = SimpleNamespace(fileno=lambda: 43)
    monkeypatch.setattr(logging_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(logging_module, "windows_byte_lock", recording_lock)

    with logging_module._exclusive_stream_lock(stream):
        pass

    assert descriptors == [43]

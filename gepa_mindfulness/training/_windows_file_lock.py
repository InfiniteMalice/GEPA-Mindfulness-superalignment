"""Bounded Windows byte locking shared by training file writers."""

from __future__ import annotations

import errno
import importlib
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from numbers import Real
from typing import Protocol, cast

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_RETRY_INTERVAL_SECONDS = 0.05
_LOCK_CONTENTION_ERRNO = errno.EACCES
_ERROR_LOCK_VIOLATION = 33


class _MsvcrtLike(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, descriptor: int, mode: int, size: int) -> None: ...


def _is_lock_contention(error: OSError) -> bool:
    return error.errno == _LOCK_CONTENTION_ERRNO or (
        getattr(error, "winerror", None) == _ERROR_LOCK_VIOLATION
    )


def _validate_timing(value: object, field_name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise TypeError(f"Windows byte-lock {field_name} must be a finite number")
    converted = float(value)
    if converted < 0.0 or not allow_zero and converted == 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Windows byte-lock {field_name} must be {qualifier}")
    return converted


@contextmanager
def windows_byte_lock(
    descriptor: int,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    retry_interval: float = _DEFAULT_RETRY_INTERVAL_SECONDS,
    msvcrt_module: _MsvcrtLike | None = None,
    monotonic: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> Iterator[None]:
    """Lock byte zero with bounded nonblocking retries.

    The 30-second default permits critical sections longer than the implicit retry window of
    ``LK_LOCK`` while ensuring that a permanently held lock cannot block a writer forever.
    """
    if isinstance(descriptor, bool) or not isinstance(descriptor, int):
        raise TypeError("Windows byte-lock descriptor must be a non-negative integer")
    if descriptor < 0:
        raise ValueError("Windows byte-lock descriptor must be a non-negative integer")
    timeout = _validate_timing(timeout, "timeout", allow_zero=True)
    retry_interval = _validate_timing(
        retry_interval,
        "retry interval",
        allow_zero=False,
    )

    if msvcrt_module is None:
        msvcrt_module = cast(_MsvcrtLike, importlib.import_module("msvcrt"))
    clock = time.monotonic if monotonic is None else monotonic
    sleep = time.sleep if sleeper is None else sleeper
    deadline = clock() + timeout

    while True:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt_module.locking(descriptor, msvcrt_module.LK_NBLCK, 1)
            break
        except OSError as error:
            if not _is_lock_contention(error):
                raise
            remaining = deadline - clock()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"Windows byte-lock contention timed out after {timeout:.3f} seconds"
                ) from error
            sleep(min(retry_interval, remaining))

    try:
        yield
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt_module.locking(descriptor, msvcrt_module.LK_UNLCK, 1)

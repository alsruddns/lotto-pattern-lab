"""Bounded retries and diagnostics for transient SQLite lock contention."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")
_LOGGER = logging.getLogger("lotto_pattern_lab.sqlite")


def is_database_locked(error: BaseException) -> bool:
    """Return whether an SQLite error represents transient database lock contention."""
    if not isinstance(error, sqlite3.OperationalError):
        return False
    message = str(error).casefold()
    return (
        "database is locked" in message
        or "database is busy" in message
        or "table is locked" in message
    )


def log_database_failure(operation: str, database: str | Path, error: BaseException) -> None:
    """Write actionable operation, path, PID, and exception details to the app log."""
    _LOGGER.error(
        "SQLite operation failed: operation=%s database=%s pid=%s error=%s",
        operation,
        database,
        os.getpid(),
        error,
        exc_info=(type(error), error, error.__traceback__),
    )


def run_with_busy_retry(
    operation: str,
    database: str | Path,
    callback: Callable[[], _T],
    *,
    attempts: int = 3,
    base_delay_seconds: float = 0.1,
) -> _T:
    """Retry an idempotent, transaction-scoped operation a bounded number of times."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except sqlite3.OperationalError as error:
            if not is_database_locked(error):
                raise
            elapsed = time.monotonic() - started
            details = {
                "operation": operation,
                "database": str(database),
                "attempt": attempt,
                "attempt_limit": attempts,
                "elapsed_seconds": round(elapsed, 3),
                "pid": os.getpid(),
            }
            if attempt == attempts:
                _LOGGER.error("SQLite remained locked after bounded retries: %s", details)
                raise sqlite3.OperationalError(
                    f"{error}; operation={operation}; database={database}; "
                    f"attempts={attempts}; elapsed={elapsed:.2f}s"
                ) from error
            delay = base_delay_seconds * (2 ** (attempt - 1))
            _LOGGER.warning(
                "SQLite lock encountered; retrying after %.2fs: %s",
                delay,
                details,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def retry_sqlite_operation(operation: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorate an atomic store method with bounded busy/locked retries."""

    def decorate(function: Callable[..., _T]) -> Callable[..., _T]:
        @wraps(function)
        def wrapped(store: Any, *args: Any, **kwargs: Any) -> _T:
            return run_with_busy_retry(
                operation,
                store.path,
                lambda: function(store, *args, **kwargs),
            )

        return wrapped

    return decorate

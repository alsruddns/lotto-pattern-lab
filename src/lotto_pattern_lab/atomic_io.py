"""Durable, serialized file replacement for research artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class AtomicWriteError(OSError):
    """An atomic write failed while preserving its complete temporary payload."""

    def __init__(self, target: Path, temporary: Path, attempts: int, cause: OSError) -> None:
        self.target = target
        self.temporary = temporary
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"Could not replace {target} after {attempts} attempts; "
            f"recoverable payload remains at {temporary}: {cause}"
        )


class LockTimeoutError(TimeoutError):
    """A file or research worker lock could not be acquired in time."""


_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}


def _canonical_key(key: str | Path) -> str:
    if isinstance(key, Path):
        return str(key.resolve())
    return key


@contextmanager
def interprocess_lock(
    key: str | Path, *, timeout: float = 60.0, scope: str = "write"
) -> Iterator[None]:
    """Lock a key in both threads and processes, with a bounded wait."""
    canonical = _canonical_key(key)
    with _thread_locks_guard:
        thread_lock = _thread_locks.setdefault(canonical, threading.RLock())
    if not thread_lock.acquire(timeout=max(0.0, timeout)):
        raise LockTimeoutError(f"Timed out waiting for {scope} lock: {canonical}")

    lock_file = None
    try:
        lock_root = Path(tempfile.gettempdir()) / "lotto-pattern-lab-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_name = hashlib.sha256(f"{scope}:{canonical}".encode()).hexdigest() + ".lock"
        lock_path = lock_root / lock_name
        lock_file = lock_path.open("a+b")
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            deadline = time.monotonic() + max(0.0, timeout)
            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise LockTimeoutError(
                            f"Timed out waiting for {scope} lock: {canonical}"
                        ) from error
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            deadline = time.monotonic() + max(0.0, timeout)
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if time.monotonic() >= deadline:
                        raise LockTimeoutError(
                            f"Timed out waiting for {scope} lock: {canonical}"
                        ) from error
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        if lock_file is not None:
            lock_file.close()
        thread_lock.release()


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    replace_attempts: int = 7,
    initial_delay: float = 0.05,
    max_delay: float = 0.8,
) -> Path:
    """Write beside the target, close and flush it, then retry atomic replacement.

    A complete temporary payload is intentionally preserved if Windows or another
    process keeps the destination locked after all replacement attempts.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if replace_attempts < 1:
        raise ValueError("replace_attempts must be at least one")

    with interprocess_lock(target, scope="write"):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{target.stem}_", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            delay = initial_delay
            for attempt in range(1, replace_attempts + 1):
                try:
                    os.replace(temporary, target)
                    return target
                except PermissionError as error:
                    if attempt == replace_attempts:
                        raise AtomicWriteError(target, temporary, attempt, error) from error
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                except OSError as error:
                    if getattr(error, "winerror", None) not in {5, 32, 33}:
                        raise
                    if attempt == replace_attempts:
                        raise AtomicWriteError(target, temporary, attempt, error) from error
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        except AtomicWriteError:
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    raise AssertionError("unreachable")

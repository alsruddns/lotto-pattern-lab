from __future__ import annotations

import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lotto_pattern_lab.atomic_io import (
    AtomicWriteError,
    LockTimeoutError,
    atomic_write_bytes,
    interprocess_lock,
)


def _hold_interprocess_lock(key: str, ready, release) -> None:
    with interprocess_lock(key, timeout=5, scope="test-process-worker"):
        ready.set()
        release.wait(timeout=5)


def test_atomic_write_retries_windows_permission_error_then_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "checkpoint.json"
    target.write_text('{"old": true}', encoding="utf-8")
    import lotto_pattern_lab.atomic_io as atomic_io

    original_replace = atomic_io.os.replace
    attempts = 0

    def sharing_violation(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "access denied")
        original_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", sharing_violation)
    atomic_write_bytes(target, b'{"new": true}', initial_delay=0, max_delay=0)

    assert attempts == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_preserves_previous_file_and_recoverable_payload_on_final_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "experiment.json"
    target.write_text('{"status": "old"}', encoding="utf-8")
    import lotto_pattern_lab.atomic_io as atomic_io

    monkeypatch.setattr(
        atomic_io.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(PermissionError(5, "locked")),
    )
    with pytest.raises(AtomicWriteError) as error:
        atomic_write_bytes(target, b'{"status": "new"}', replace_attempts=2, initial_delay=0)

    assert target.read_text(encoding="utf-8") == '{"status": "old"}'
    assert error.value.attempts == 2
    assert error.value.temporary.read_bytes() == b'{"status": "new"}'


def test_atomic_writes_to_same_target_are_serialized(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"

    def save(index: int) -> None:
        payload = json.dumps({"writer": index, "data": "x" * 100_000}).encode("utf-8")
        atomic_write_bytes(target, payload)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(save, range(12)))

    assert json.loads(target.read_text(encoding="utf-8"))["writer"] in range(12)
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_rejects_another_thread_when_worker_lock_is_held(tmp_path: Path) -> None:
    key = tmp_path / "research.json"
    ready = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with interprocess_lock(key, timeout=1, scope="test-worker"):
            ready.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert ready.wait(timeout=1)
    try:
        with pytest.raises(LockTimeoutError):
            with interprocess_lock(key, timeout=0, scope="test-worker"):
                pass
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_worker_lock_rejects_a_second_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    key = str(tmp_path / "research.json")
    process = context.Process(target=_hold_interprocess_lock, args=(key, ready, release))
    process.start()
    assert ready.wait(timeout=5)
    try:
        with pytest.raises(LockTimeoutError):
            with interprocess_lock(key, timeout=0, scope="test-process-worker"):
                pass
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_atomic_replace_wait_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lotto_pattern_lab.atomic_io as atomic_io

    monkeypatch.setattr(
        atomic_io.os,
        "replace",
        lambda *_: (_ for _ in ()).throw(PermissionError(32, "sharing violation")),
    )
    started = time.monotonic()
    with pytest.raises(AtomicWriteError):
        atomic_write_bytes(
            tmp_path / "out.json",
            b"payload",
            replace_attempts=3,
            initial_delay=0.001,
            max_delay=0.002,
        )
    assert time.monotonic() - started < 1

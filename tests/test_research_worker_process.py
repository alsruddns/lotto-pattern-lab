from __future__ import annotations

import subprocess

import pytest

from lotto_pattern_lab import research_worker_process as worker_process


def test_detached_worker_launcher_hides_console_logs_and_blocks_duplicate_request(
    tmp_path, monkeypatch
) -> None:
    class Process:
        pid = 4321

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(worker_process, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(worker_process.subprocess, "Popen", fake_popen)
    checkpoint = tmp_path / "worker.json"
    log = tmp_path / "worker.log"

    process = worker_process.launch_worker(["python", "-m", "worker"], checkpoint, log)

    assert process.pid == 4321
    assert calls[0][1]["stdout"].closed
    assert calls[0][1]["stderr"] == subprocess.STDOUT
    if __import__("os").name == "nt":
        assert calls[0][1]["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert calls[0][1]["creationflags"] & subprocess.DETACHED_PROCESS
    with pytest.raises(RuntimeError, match="시작 요청이 이미 처리 중"):
        worker_process.launch_worker(["python", "-m", "worker"], checkpoint, log)
    assert len(calls) == 1

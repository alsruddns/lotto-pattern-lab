from __future__ import annotations

import json
from datetime import UTC, datetime

from lotto_pattern_lab.optimization_process_state import launch_optimization_worker


class FakePopen:
    pid = 48123

    def __init__(self) -> None:
        self.unserializable_runtime_handle = object()


def test_optuna_worker_start_persists_only_explicit_json_safe_fields(tmp_path) -> None:
    output = tmp_path / "study.state.json.process.json"
    command = ["python", "-m", "lotto_pattern_lab", "optimize", "--study-name", "study-a"]
    process = FakePopen()
    calls = []

    def launcher(command_arg, checkpoint_path, log_path):
        calls.append((command_arg, checkpoint_path, log_path))
        return process

    state = launch_optimization_worker(
        launcher,  # type: ignore[arg-type]
        command,
        tmp_path / "study.state.json",
        tmp_path / "study.log",
        output,
        study_name="study-a",
        run_id="run-a",
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert state.pid == 48123
    assert loaded["study_name"] == "study-a"
    assert loaded["run_id"] == "run-a"
    assert loaded["pid"] == 48123
    assert loaded["command"] == command
    assert datetime.fromisoformat(loaded["started_at"]).tzinfo == UTC
    assert calls == [(command, tmp_path / "study.state.json", tmp_path / "study.log")]
    assert "process" not in loaded
    json.dumps(loaded)


def test_optimization_worker_start_failure_does_not_write_running_process_record(tmp_path) -> None:
    output = tmp_path / "study.state.json.process.json"

    def failed_launcher(command, checkpoint_path, log_path):
        raise OSError("mock worker launch failure")

    try:
        launch_optimization_worker(
            failed_launcher,  # type: ignore[arg-type]
            ["python", "-m", "lotto_pattern_lab", "optimize"],
            tmp_path / "study.state.json",
            tmp_path / "study.log",
            output,
            study_name="study-a",
        )
    except OSError as error:
        assert str(error) == "mock worker launch failure"
    else:
        raise AssertionError("worker start failure was swallowed")

    assert not output.exists()

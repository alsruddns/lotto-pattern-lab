from __future__ import annotations

import json
from pathlib import Path

import pytest

from lotto_pattern_lab.queue_scheduler import _research_command, scheduler_step
from lotto_pattern_lab.research_queue import ResearchQueueStore
from lotto_pattern_lab.research_rounds import ResearchRoundStore


def _add(store: ResearchQueueStore, queue: str, seconds: int = 30) -> str:
    return store.add_item(
        queue,
        engine="intermittent_gp",
        feature_profile="legacy",
        duration_seconds=seconds,
        config={"seed": 12345, "population_size": 8},
    )


def test_worker_command_uses_snapshotted_gp_and_pysr_settings(tmp_path: Path) -> None:
    item = {
        "engine": "comparison",
        "feature_profile": "calendar_round",
        "duration_seconds": 7200,
        "config": {
            "seed": 7,
            "workers": 3,
            "population_size": 64,
            "generations": 20,
            "pysr": {"target": "main_positions", "niterations": 50, "workers": 1},
        },
    }

    command = _research_command(item, tmp_path / "research.json", "research-id", resume=True)

    assert command.count("--engine") == 2
    assert command[command.index("--duration-minutes") + 1] == "120.0"
    assert command[command.index("--population") + 1] == "64"
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--pysr-iterations") + 1] == "50"
    assert "--resume" in command


def test_queue_snapshot_edit_cancel_and_reorder_are_durable(tmp_path: Path) -> None:
    database = tmp_path / "queue.sqlite"
    store = ResearchQueueStore(database)
    queue = store.create_queue()
    first, second, third = (_add(store, queue) for _ in range(3))
    store.move(third, "first")
    assert [item["item_id"] for item in store.list_items(queue)] == [third, first, second]
    store.edit_waiting(
        second,
        engine="pysr_symbolic_regression",
        feature_profile="calendar_round",
        duration_seconds=3600,
        config={"seed": 7},
    )
    store.cancel_waiting(first, "user request")
    reopened = ResearchQueueStore(database)
    by_id = {item["item_id"]: item for item in reopened.list_items(queue)}
    assert by_id[second]["duration_seconds"] == 3600
    assert by_id[second]["config"] == {"seed": 7}
    assert by_id[first]["status"] == "cancelled"
    with pytest.raises(ValueError):
        reopened.edit_waiting(
            first,
            engine="intermittent_gp",
            feature_profile="legacy",
            duration_seconds=30,
            config={},
        )


def test_lease_allows_one_scheduler_owner(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    assert store.acquire_lease("owner-a", queue)
    assert not store.acquire_lease("owner-b", queue)
    assert store.heartbeat("owner-a")
    store.release_lease("owner-a")
    assert store.acquire_lease("owner-b", queue)


def test_mock_scheduler_runs_independent_items_in_order(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    first, second, third = (_add(store, queue, seconds) for seconds in (30, 60, 120))
    store.set_queue_status(queue, "running")
    rounds = ResearchRoundStore(tmp_path / "patterns.sqlite")
    launched: list[tuple[str, str, Path]] = []
    states: dict[str, str] = {}

    def launch(item, checkpoint, research_id):
        launched.append((item["item_id"], research_id, checkpoint))
        states[item["item_id"]] = "running"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"status": "running"}), encoding="utf-8")
        return 1000 + len(launched)

    def inspect(item):
        state = states[item["item_id"]]
        return state == "running", "completed" if state == "completed" else None

    for expected_id in (first, second):
        assert (
            scheduler_step(
                store,
                queue,
                launch=launch,
                inspect=inspect,
                workspace_root=tmp_path,
                round_store=rounds,
            )
            == "worker_started"
        )
        assert launched[-1][0] == expected_id
        states[expected_id] = "completed"
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    assert launched[-1][0] == third
    states[third] = "completed"
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "completed"
    )
    items = store.list_items(queue)
    assert [item["status"] for item in items] == ["completed"] * 3
    assert len({item["research_id"] for item in items}) == 3
    assert len({item["research_round"] for item in items}) == 3
    assert all(Path(item["checkpoint_path"]).is_relative_to(tmp_path) for item in items)


def test_pause_waits_for_boundary_and_resume_keeps_same_research(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    item_id = _add(store, queue)
    store.set_queue_status(queue, "running")
    rounds = ResearchRoundStore(tmp_path / "patterns.sqlite")
    alive = True
    launch_calls: list[tuple[str, bool]] = []

    def launch(item, checkpoint, research_id):
        launch_calls.append((research_id, bool(item.get("resume"))))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{}", encoding="utf-8")
        return 4321

    def inspect(item):
        return alive, "paused" if not alive else None

    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    before = store.list_items(queue)[0]
    store.set_queue_status(queue, "pause_requested")
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_running"
    )
    alive = False
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "queue_paused"
    )
    paused = store.list_items(queue)[0]
    assert paused["status"] == "paused"
    assert paused["research_id"] == before["research_id"]
    assert paused["research_round"] == before["research_round"]
    store.set_queue_status(queue, "running")
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=inspect,
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_resumed"
    )
    resumed = store.list_items(queue)[0]
    assert resumed["item_id"] == item_id
    assert resumed["research_id"] == before["research_id"]
    assert launch_calls[0][0] == launch_calls[1][0]
    assert launch_calls[1][1] is True


def test_failure_stops_queue_and_skip_requires_explicit_action(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue(error_policy="stop")
    failed_id = _add(store, queue)
    waiting_id = _add(store, queue)
    store.set_queue_status(queue, "running")
    rounds = ResearchRoundStore(tmp_path / "patterns.sqlite")

    def launch(item, checkpoint, research_id):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{}", encoding="utf-8")
        return 77

    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (False, "engine error"),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (False, "engine error"),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "queue_failed"
    )
    assert store.queue(queue)["status"] == "failed"
    assert store.list_items(queue)[1]["status"] == "waiting"
    store.skip_failed(failed_id)
    assert store.queue(queue)["status"] == "paused"
    store.set_queue_status(queue, "running")
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (True, None),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    assert store.list_items(queue)[1]["item_id"] == waiting_id


def test_continue_policy_only_skips_recoverable_experiment_errors(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue(error_policy="continue")
    first, second = _add(store, queue), _add(store, queue)
    store.set_queue_status(queue, "running")
    rounds = ResearchRoundStore(tmp_path / "patterns.sqlite")

    def launch(item, checkpoint, research_id):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("{}", encoding="utf-8")
        return 88

    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (True, None),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (False, "failed:experiment:temporary model error"),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "worker_started"
    )
    assert store.list_items(queue)[0]["item_id"] == first
    assert store.list_items(queue)[1]["item_id"] == second

    # A storage integrity failure stops even when the queue explicitly continues on
    # ordinary experiment errors.
    assert (
        scheduler_step(
            store,
            queue,
            launch=launch,
            inspect=lambda _: (False, "failed:storage:disk write error"),
            workspace_root=tmp_path,
            round_store=rounds,
        )
        == "queue_failed"
    )
    assert store.queue(queue)["status"] == "failed"

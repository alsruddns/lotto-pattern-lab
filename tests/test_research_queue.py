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


def test_scheduled_items_keep_timezone_and_conflict_detection(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    start = (datetime.now(UTC) + timedelta(hours=2)).replace(microsecond=0)
    first = store.add_item(
        queue,
        engine="intermittent_gp",
        feature_profile="legacy",
        duration_seconds=3600,
        config={"seed": 1},
        schedule_mode="at_time",
        scheduled_at=start.isoformat(),
    )
    conflicts = store.schedule_conflicts((start + timedelta(minutes=30)).isoformat(), 3600)
    assert [item["item_id"] for item in conflicts] == [first]
    assert store.schedule_conflicts((start + timedelta(hours=2)).isoformat(), 300) == []
    with pytest.raises(ValueError, match="future and timezone-aware"):
        store.add_item(
            queue,
            engine="intermittent_gp",
            feature_profile="legacy",
            duration_seconds=60,
            config={},
            schedule_mode="at_time",
            scheduled_at=datetime.now(UTC).isoformat(),
        )


def test_future_scheduled_item_stays_waiting_and_queue_live(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    store.add_item(
        queue,
        engine="intermittent_gp",
        feature_profile="legacy",
        duration_seconds=300,
        config={"seed": 8},
        schedule_mode="at_time",
        scheduled_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    store.set_queue_status(queue, "running")
    assert store.claim_next(queue) is None
    assert store.queue(queue)["status"] == "running"
    assert store.list_items(queue)[0]["status"] == "waiting"


def test_legacy_queue_schema_is_backed_up_before_schedule_migration(tmp_path: Path) -> None:
    import sqlite3

    database = tmp_path / "queue.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE research_queues (
                queue_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                error_policy TEXT NOT NULL DEFAULT 'stop', created_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT, last_error TEXT
            );
            CREATE TABLE research_queue_items (
                item_id TEXT PRIMARY KEY, queue_id TEXT NOT NULL, position INTEGER NOT NULL,
                status TEXT NOT NULL, engine TEXT NOT NULL, feature_profile TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL, config_json TEXT NOT NULL,
                research_id TEXT, research_round INTEGER, checkpoint_path TEXT, worker_pid INTEGER,
                started_at TEXT, completed_at TEXT, cancelled_at TEXT, last_error TEXT,
                UNIQUE(queue_id, position)
            );
            CREATE TABLE research_queue_lease (
                singleton INTEGER PRIMARY KEY, owner TEXT NOT NULL, queue_id TEXT NOT NULL,
                heartbeat_at REAL NOT NULL
            );
            INSERT INTO research_queues(queue_id,name,status,created_at)
            VALUES('legacy','legacy','waiting','2026-01-01T00:00:00+00:00');
            """
        )
    store = ResearchQueueStore(database)
    assert store.queue("legacy")["status"] == "waiting"
    backups = list((tmp_path / "backups").glob("queue_pre_schedule_migration_*.sqlite"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(research_queue_items)")}
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert "scheduled_at" not in columns
    assert "created_at" not in columns


def test_request_key_makes_reservation_insertion_idempotent(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    options = {
        "engine": "intermittent_gp",
        "feature_profile": "legacy",
        "duration_seconds": 300,
        "config": {"seed": 11},
        "request_key": "same-ui-submit",
    }
    first = store.add_item(queue, **options)
    duplicate = store.add_item(queue, **options)
    assert duplicate == first
    assert len(store.list_items(queue)) == 1


def test_run_waiting_now_keeps_item_and_uses_scheduler_lock_path(tmp_path: Path) -> None:
    from datetime import datetime

    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    item_id = _add(store, queue)
    selected_queue = store.run_waiting_now(item_id)
    item = store.list_items(selected_queue)[0]
    assert selected_queue == queue
    assert item["status"] == "waiting"
    assert item["schedule_mode"] == "at_time"
    assert item["created_at"] is not None
    assert item["updated_at"] is not None
    assert datetime.fromisoformat(item["scheduled_at"]).tzinfo is not None


def test_scheduler_waits_for_global_worker_before_claiming(tmp_path: Path, monkeypatch) -> None:
    import lotto_pattern_lab.queue_scheduler as scheduler

    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    item = _add(store, queue)
    store.set_queue_status(queue, "running")
    monkeypatch.setattr(scheduler, "research_worker_active", lambda: True)
    launched: list[str] = []
    result = scheduler.scheduler_step(
        store,
        queue,
        launch=lambda row, checkpoint, research_id: launched.append(row["item_id"]) or 1,
        inspect=lambda row: (False, "interrupted"),
        workspace_root=tmp_path,
        round_store=ResearchRoundStore(tmp_path / "patterns.sqlite"),
    )
    assert result == "waiting_for_global_worker"
    assert launched == []
    assert store.list_items(queue)[0]["item_id"] == item
    assert store.list_items(queue)[0]["status"] == "waiting"


def test_schedule_conflicts_include_running_fixed_reservation_estimate(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue()
    item_id = _add(store, queue, seconds=3600)
    store.run_waiting_now(item_id)
    store.set_queue_status(queue, "running")
    claimed = store.claim_next(queue)
    assert claimed is not None
    proposed = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
    assert [item["item_id"] for item in store.schedule_conflicts(proposed, 600)] == [item_id]


def test_reservation_insert_and_queue_activation_are_atomic(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue = store.create_queue(error_policy="stop")
    item_id = store.add_item(
        queue,
        engine="intermittent_gp",
        feature_profile="legacy",
        duration_seconds=300,
        config={"seed": 42},
        activate_queue=True,
        error_policy="continue",
    )
    assert store.queue(queue)["status"] == "running"
    assert store.queue(queue)["error_policy"] == "continue"
    assert store.list_items(queue)[0]["item_id"] == item_id

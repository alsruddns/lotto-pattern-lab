"""Durable SQLite primitives for ordered, independent research reservations."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT

QUEUE_DB = PROJECT_ROOT / "data" / "database" / "research_queue.sqlite"
QUEUE_STATES = {
    "waiting",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
    "skipped",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ResearchQueueStore:
    """Transaction-safe queue API. A custom path keeps tests fully isolated."""

    def __init__(self, path: str | Path = QUEUE_DB) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_queues (
                    queue_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    error_policy TEXT NOT NULL DEFAULT 'stop', created_at TEXT NOT NULL,
                    started_at TEXT, completed_at TEXT, last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS research_queue_items (
                    item_id TEXT PRIMARY KEY,
                    queue_id TEXT NOT NULL REFERENCES research_queues(queue_id),
                    position INTEGER NOT NULL, status TEXT NOT NULL, engine TEXT NOT NULL,
                    feature_profile TEXT NOT NULL, duration_seconds INTEGER NOT NULL,
                    config_json TEXT NOT NULL, research_id TEXT, research_round INTEGER,
                    checkpoint_path TEXT, worker_pid INTEGER, started_at TEXT, completed_at TEXT,
                    cancelled_at TEXT, last_error TEXT, UNIQUE(queue_id, position)
                );
                CREATE INDEX IF NOT EXISTS queue_items_status_idx
                    ON research_queue_items(queue_id, status, position);
                CREATE TABLE IF NOT EXISTS research_queue_lease (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT NOT NULL,
                    queue_id TEXT NOT NULL, heartbeat_at REAL NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def create_queue(self, name: str = "연구 예약", error_policy: str = "stop") -> str:
        if error_policy not in {"stop", "continue"}:
            raise ValueError("error_policy must be stop or continue")
        queue_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                """INSERT INTO research_queues
                (queue_id,name,status,error_policy,created_at) VALUES(?,?,?,?,?)""",
                (queue_id, name, "waiting", error_policy, _now()),
            )
        return queue_id

    def add_item(
        self,
        queue_id: str,
        *,
        engine: str,
        feature_profile: str,
        duration_seconds: int,
        config: dict[str, Any],
    ) -> str:
        if engine not in {"intermittent_gp", "pysr_symbolic_regression", "comparison"}:
            raise ValueError("Unsupported research engine.")
        if feature_profile not in {"legacy", "calendar_round", "bonus_lag_v1"}:
            raise ValueError("Unsupported feature profile.")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        item_id = str(uuid.uuid4())
        snapshot = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            queue = db.execute(
                "SELECT status FROM research_queues WHERE queue_id=?", (queue_id,)
            ).fetchone()
            if not queue or queue["status"] not in {"waiting", "paused"}:
                raise ValueError("Reservations can only be added to a waiting or paused queue.")
            position = db.execute(
                "SELECT COALESCE(MAX(position),0)+1 FROM research_queue_items WHERE queue_id=?",
                (queue_id,),
            ).fetchone()[0]
            db.execute(
                """INSERT INTO research_queue_items
                (item_id,queue_id,position,status,engine,feature_profile,duration_seconds,config_json)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    item_id,
                    queue_id,
                    position,
                    "waiting",
                    engine,
                    feature_profile,
                    duration_seconds,
                    snapshot,
                ),
            )
        return item_id

    def list_items(self, queue_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM research_queue_items WHERE queue_id=? ORDER BY position",
                (queue_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["config"] = json.loads(item.pop("config_json"))
        return result

    def list_queues(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM research_queues ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def queue(self, queue_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM research_queues WHERE queue_id=?", (queue_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Queue not found.")
        return dict(row)

    def current_item(self, queue_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM research_queue_items WHERE queue_id=?
                AND status IN ('running','paused','interrupted') ORDER BY position LIMIT 1""",
                (queue_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def acquire_lease(self, owner: str, queue_id: str, ttl_seconds: int = 45) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT owner,heartbeat_at FROM research_queue_lease WHERE singleton=1"
            ).fetchone()
            now = time.time()
            if row and row["owner"] != owner and now - float(row["heartbeat_at"]) <= ttl_seconds:
                return False
            db.execute(
                """INSERT INTO research_queue_lease
                (singleton,owner,queue_id,heartbeat_at) VALUES(1,?,?,?)
                ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner,
                queue_id=excluded.queue_id,heartbeat_at=excluded.heartbeat_at""",
                (owner, queue_id, now),
            )
            return True

    def heartbeat(self, owner: str) -> bool:
        with self._connect() as db:
            result = db.execute(
                "UPDATE research_queue_lease SET heartbeat_at=? WHERE singleton=1 AND owner=?",
                (time.time(), owner),
            )
            return result.rowcount == 1

    def lease_is_live(self, queue_id: str, ttl_seconds: int = 45) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT heartbeat_at FROM research_queue_lease WHERE singleton=1 AND queue_id=?",
                (queue_id,),
            ).fetchone()
        return bool(row and time.time() - float(row["heartbeat_at"]) <= ttl_seconds)

    def release_lease(self, owner: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM research_queue_lease WHERE singleton=1 AND owner=?", (owner,))

    def edit_waiting(
        self,
        item_id: str,
        *,
        engine: str,
        feature_profile: str,
        duration_seconds: int,
        config: dict[str, Any],
    ) -> None:
        if engine not in {
            "intermittent_gp",
            "pysr_symbolic_regression",
            "comparison",
        }:
            raise ValueError("Unsupported research engine.")
        if feature_profile not in {"legacy", "calendar_round", "bonus_lag_v1"}:
            raise ValueError("Unsupported feature profile.")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        snapshot = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                """UPDATE research_queue_items SET engine=?,feature_profile=?,
                duration_seconds=?,config_json=? WHERE item_id=? AND status='waiting'""",
                (engine, feature_profile, duration_seconds, snapshot, item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only a waiting reservation can be edited.")

    def cancel_waiting(self, item_id: str, reason: str | None = None) -> None:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE research_queue_items SET status='cancelled',cancelled_at=?,
                last_error=? WHERE item_id=? AND status='waiting'""",
                (_now(), reason, item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only a waiting reservation can be cancelled.")

    def move(self, item_id: str, direction: str) -> None:
        if direction not in {"first", "up", "down", "last"}:
            raise ValueError("Invalid queue movement.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute(
                "SELECT * FROM research_queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not item or item["status"] != "waiting":
                raise ValueError("Only a waiting reservation can be reordered.")
            rows = db.execute(
                """SELECT item_id,status FROM research_queue_items
                WHERE queue_id=? ORDER BY position""",
                (item["queue_id"],),
            ).fetchall()
            waiting_ids = [row["item_id"] for row in rows if row["status"] == "waiting"]
            waiting_slots = [index for index, row in enumerate(rows) if row["status"] == "waiting"]
            old = waiting_ids.index(item_id)
            target = {
                "first": 0,
                "up": max(0, old - 1),
                "down": min(len(waiting_ids) - 1, old + 1),
                "last": len(waiting_ids) - 1,
            }[direction]
            waiting_ids.insert(target, waiting_ids.pop(old))
            ids = [row["item_id"] for row in rows]
            for slot, identifier in zip(waiting_slots, waiting_ids, strict=True):
                ids[slot] = identifier
            # Free the UNIQUE position constraint before assigning the compact order atomically.
            db.execute(
                "UPDATE research_queue_items SET position=position+1000000 WHERE queue_id=?",
                (item["queue_id"],),
            )
            for position, identifier in enumerate(ids, 1):
                db.execute(
                    "UPDATE research_queue_items SET position=? WHERE item_id=?",
                    (position, identifier),
                )

    def set_queue_status(self, queue_id: str, status: str) -> None:
        if status not in {
            "waiting",
            "running",
            "pause_requested",
            "paused",
            "completed",
            "failed",
            "interrupted",
        }:
            raise ValueError("Unsupported queue state.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM research_queues WHERE queue_id=?", (queue_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Queue not found.")
            db.execute(
                """UPDATE research_queues SET status=?,
                started_at=CASE WHEN ?='running' THEN COALESCE(started_at,?) ELSE started_at END,
                completed_at=CASE WHEN ? IN ('completed','failed') THEN ? ELSE completed_at END
                WHERE queue_id=?""",
                (status, status, _now(), status, _now(), queue_id),
            )

    def set_queue_error(self, queue_id: str, message: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE research_queues SET last_error=? WHERE queue_id=?",
                (message, queue_id),
            )

    def set_error_policy(self, queue_id: str, policy: str) -> None:
        if policy not in {"stop", "continue"}:
            raise ValueError("error_policy must be stop or continue")
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE research_queues SET error_policy=? WHERE queue_id=?
                AND status IN ('waiting','paused')""",
                (policy, queue_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Error policy can only change before or between runs.")

    def claim_next(self, queue_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            queue = db.execute(
                "SELECT status FROM research_queues WHERE queue_id=?", (queue_id,)
            ).fetchone()
            if not queue or queue["status"] != "running":
                return None
            row = db.execute(
                """SELECT item_id FROM research_queue_items WHERE queue_id=?
                AND status='waiting' ORDER BY position LIMIT 1""",
                (queue_id,),
            ).fetchone()
            if row is None:
                db.execute(
                    "UPDATE research_queues SET status='completed',completed_at=? WHERE queue_id=?",
                    (_now(), queue_id),
                )
                return None
            item_id = row["item_id"]
            db.execute(
                """UPDATE research_queue_items SET status='running',started_at=?
                WHERE item_id=? AND status='waiting'""",
                (_now(), item_id),
            )
            claimed = db.execute(
                "SELECT * FROM research_queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
        item = dict(claimed)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def finish_item(self, item_id: str, status: str, *, error: str | None = None) -> None:
        if status not in {"paused", "completed", "failed", "interrupted", "skipped"}:
            raise ValueError("Invalid terminal reservation status.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT queue_id FROM research_queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            db.execute(
                """UPDATE research_queue_items SET status=?,completed_at=?,last_error=?
                WHERE item_id=? AND status='running'""",
                (status, _now(), error, item_id),
            )
            if row and status == "failed":
                db.execute(
                    "UPDATE research_queues SET last_error=? WHERE queue_id=?",
                    (error, row["queue_id"]),
                )

    def skip_failed(self, item_id: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute(
                "SELECT queue_id FROM research_queue_items WHERE item_id=? AND status='failed'",
                (item_id,),
            ).fetchone()
            if item is None:
                raise ValueError("Only a failed reservation can be skipped.")
            db.execute(
                "UPDATE research_queue_items SET status='skipped',completed_at=? WHERE item_id=?",
                (_now(), item_id),
            )
            db.execute(
                "UPDATE research_queues SET status='paused',last_error=NULL WHERE queue_id=?",
                (item["queue_id"],),
            )

    def retry_failed(self, item_id: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute(
                "SELECT queue_id FROM research_queue_items WHERE item_id=? AND status='failed'",
                (item_id,),
            ).fetchone()
            if item is None:
                raise ValueError("Only a failed reservation can be recovered.")
            db.execute(
                "UPDATE research_queue_items SET status='paused',completed_at=NULL WHERE item_id=?",
                (item_id,),
            )
            db.execute(
                "UPDATE research_queues SET status='paused' WHERE queue_id=?",
                (item["queue_id"],),
            )

    def mark_interrupted(self, item_id: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                "UPDATE research_queue_items SET status='interrupted',last_error=? "
                "WHERE item_id=? AND status='running'",
                (
                    "워커 중단. 체크포인트를 확인한 뒤 수동 복구하세요.",
                    item_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only a running reservation can be marked interrupted.")
            row = db.execute(
                "SELECT queue_id FROM research_queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            db.execute(
                "UPDATE research_queues SET status='interrupted' WHERE queue_id=?",
                (row["queue_id"],),
            )

    def claim_resume(self, item_id: str) -> dict[str, Any]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                """UPDATE research_queue_items SET status='running',completed_at=NULL
                WHERE item_id=? AND status IN ('paused','interrupted')""",
                (item_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only a paused or interrupted reservation can be resumed.")
            row = db.execute(
                "SELECT * FROM research_queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["resume"] = True
        return item

    def set_item_research(
        self,
        item_id: str,
        research_id: str,
        round_number: int,
        checkpoint: str,
        pid: int | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE research_queue_items SET research_id=?,research_round=?,
                checkpoint_path=?,worker_pid=? WHERE item_id=?""",
                (research_id, round_number, checkpoint, pid, item_id),
            )

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.research_rounds import ResearchRoundStore
from lotto_pattern_lab.sqlite_support import run_with_busy_retry


def _legacy_pattern_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE patterns (
                pattern_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO patterns VALUES "
            "('legacy-1', 'fingerprint-1', '2026-01-01', ?, 'new_discovery')",
            (json.dumps({"formulas": ["add(x0, 0)"], "outputs_per_event": 6}),),
        )


def test_pattern_schema_migration_runs_once_without_rewriting_existing_rows(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    _legacy_pattern_db(database)
    library = PatternLibrary(database)
    assert library.get_pattern("legacy-1")["formulas"] == ["add(x0, 0)"]

    with library._connect() as connection:
        connection.execute("CREATE TABLE update_audit (updates INTEGER NOT NULL)")
        connection.execute("INSERT INTO update_audit VALUES (0)")
        connection.execute(
            """CREATE TRIGGER count_pattern_updates AFTER UPDATE ON patterns
               BEGIN UPDATE update_audit SET updates=updates+1; END"""
        )

    PatternLibrary(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT updates FROM update_audit").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM patterns").fetchone()[0] == 1
        version = connection.execute(
            "SELECT MAX(version) FROM pattern_schema_versions"
        ).fetchone()[0]
        assert version == 2


def test_legacy_round_migration_is_idempotent_and_does_not_update_on_repeat(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    checkpoint = {
        "research_id": "research-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "completed_experiments": [],
    }
    (checkpoints / "research_one.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    store = ResearchRoundStore(database)
    store.migrate_legacy(checkpoints, experiments)
    with store._connect() as connection:
        connection.execute("CREATE TABLE update_audit (updates INTEGER NOT NULL)")
        connection.execute("INSERT INTO update_audit VALUES (0)")
        connection.execute(
            """CREATE TRIGGER count_round_updates AFTER UPDATE ON research_rounds
               BEGIN UPDATE update_audit SET updates=updates+1; END"""
        )
        applied_at = connection.execute(
            "SELECT applied_at FROM research_round_migrations WHERE migration_key='legacy_v1'"
        ).fetchone()[0]

    store.migrate_legacy(checkpoints, experiments)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT updates FROM update_audit").fetchone()[0] == 0
        assert connection.execute(
            "SELECT applied_at FROM research_round_migrations WHERE migration_key='legacy_v1'"
        ).fetchone()[0] == applied_at
        assert connection.execute("SELECT COUNT(*) FROM research_rounds").fetchone()[0] == 1


def test_simultaneous_round_creates_are_serialized_and_preserved(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    store = ResearchRoundStore(database)
    store.migrate_legacy(checkpoints, experiments)

    def create(index: int) -> int:
        return ResearchRoundStore(database).ensure_round(
            f"research-{index}",
            {"created_at": f"2026-01-{index + 1:02d}T00:00:00+00:00", "status": "queued"},
            checkpoints,
            experiments,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        round_numbers = list(executor.map(create, range(8)))
    assert len(set(round_numbers)) == 8
    assert len(store.list()) == 8
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_readers_can_query_while_wal_writer_has_uncommitted_change(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    store = ResearchRoundStore(database)
    store.ensure_round(
        "research-1",
        {"status": "queued"},
        tmp_path / "checkpoints",
        tmp_path / "experiments",
    )
    with store._connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")

    writer = sqlite3.connect(database, timeout=1, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "UPDATE research_rounds SET metadata_json=? WHERE research_id=?",
        (json.dumps({"status": "running"}), "research-1"),
    )
    try:
        assert store.get("research-1")["status"] == "queued"
    finally:
        writer.rollback()
        writer.close()


def test_busy_retry_succeeds_after_short_lock_and_is_bounded_after_timeout(tmp_path) -> None:
    database = tmp_path / "locked.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
    holder = sqlite3.connect(
        database, timeout=1, isolation_level=None, check_same_thread=False
    )
    holder.execute("BEGIN IMMEDIATE")
    release = threading.Timer(0.18, holder.commit)
    release.start()

    def insert_after_contention() -> None:
        with sqlite3.connect(database, timeout=0.03, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO sample VALUES (1)")

    run_with_busy_retry(
        "test insert", database, insert_after_contention, attempts=5, base_delay_seconds=0.05
    )
    release.join(timeout=2)
    holder.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 1

    calls = 0

    def remain_locked() -> None:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="attempts=2"):
        run_with_busy_retry(
            "test exhausted retry",
            database,
            remain_locked,
            attempts=2,
            base_delay_seconds=0,
        )
    assert calls == 2


def test_sqlite_connection_closes_and_rolls_back_on_exception(tmp_path) -> None:
    store = ResearchRoundStore(tmp_path / "patterns.sqlite")
    captured = None
    with pytest.raises(RuntimeError, match="rollback"):
        with store._connect() as connection:
            captured = connection
            connection.execute(
                "INSERT INTO research_rounds "
                "(research_id, created_at, metadata_json) VALUES (?, ?, ?)",
                ("not-committed", "2026-01-01", "{}"),
            )
            raise RuntimeError("rollback")
    assert captured is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        captured.execute("SELECT 1")
    assert store.get("not-committed") is None


def test_research_and_dashboard_pages_render_and_tab_selection_survives_rerun() -> None:
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    pages = root / "src/lotto_pattern_lab/pages"
    research_page = next(pages.glob("3_*.py"))
    dashboard_page = next(pages.glob("5_*.py"))
    app = AppTest.from_file(research_page).run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]

    optimization_label = next(tab.label for tab in app.tabs if "최적화" in tab.label)
    app.session_state["research_workspace_tab"] = optimization_label
    app.run(timeout=60)
    assert app.session_state["research_workspace_tab"] == optimization_label
    app.run(timeout=60)
    assert app.session_state["research_workspace_tab"] == optimization_label

    dashboard = AppTest.from_file(dashboard_page).run(timeout=60)
    assert not dashboard.exception, [exception.message for exception in dashboard.exception]

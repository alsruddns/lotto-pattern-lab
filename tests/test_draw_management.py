from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from lotto_pattern_lab.dataset import DrawRecord, load_csv
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.draw_management import (
    confirm_manual_official_comparison,
    create_draw_correction_request,
    draw_registration_status,
    register_draw,
    verify_draw_against_official,
)


@pytest.fixture(autouse=True)
def _isolate_research_guard(tmp_path, monkeypatch):
    import lotto_pattern_lab.draw_management as manager

    monkeypatch.setattr(manager, "_research_lock_path", lambda: tmp_path / "research_global")
    monkeypatch.setattr(
        manager, "_dataset_update_lock_path", lambda: tmp_path / "dataset_update_gate"
    )
    monkeypatch.setattr(
        manager, "_research_queue_database_path", lambda: tmp_path / "research_queue.sqlite"
    )


def _write_sources(tmp_path):
    raw = tmp_path / "draws.csv"
    raw.write_text(
        "draw_no,n1,n2,n3,n4,n5,n6,bonus\n"
        "753,2,17,19,24,37,41,3\n"
        "754,1,5,13,24,29,33,9\n",
        encoding="utf-8",
    )
    official = tmp_path / "official.csv"
    official.write_text(
        "round,date,n1,n2,n3,n4,n5,n6\n"
        "753,2017-05-06,2,17,19,24,37,41\n"
        "754,2017-05-13,1,5,13,24,29,33\n",
        encoding="utf-8",
    )
    return raw, official


def _register(raw, official, **overrides):
    values = {
        "round_number": 755,
        "draw_date": date(2017, 5, 20),
        "main_numbers": [4, 12, 18, 27, 36, 44],
        "bonus_number": 7,
        "official_dates_path": official,
        "backup_directory": raw.parent / "backups",
    }
    values.update(overrides)
    return register_draw(raw, **values)


def test_registration_backups_appends_and_preserves_fixed_range(tmp_path):
    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()

    result = _register(raw, official)

    assert result["status"] == "registered"
    assert result["fixed_range_before_sha256"] == result["fixed_range_after_sha256"]
    assert raw.read_bytes().startswith(original)
    assert result["backup_path"]
    assert Path(result["backup_path"]).read_bytes() == original
    loaded = load_csv(raw)
    assert loaded.records[-1].round_number == 755
    assert loaded.records[-1].draw_date == date(2017, 5, 20)
    assert loaded.records[-1].verification_status == "manual_unverified"
    with raw.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[-1]["bonus"] == "7"
    assert json.loads((tmp_path / "draws.csv.metadata.json").read_text())[
        "verification_status"
    ] == "manual_unverified"


def _write_official_provenance_sources(tmp_path):
    raw, official = _write_sources(tmp_path)
    raw.write_text(
        "round,date,n1,n2,n3,n4,n5,n6,bonus\n"
        "753,2017-05-06,2,17,19,24,37,41,3\n"
        "754,2017-05-13,1,5,13,24,29,33,9\n",
        encoding="utf-8",
        newline="",
    )
    date_sidecar = tmp_path / "draws_dates.csv"
    date_sidecar.write_text(
        "round,date\n753,2017-05-06\n754,2017-05-13\n", encoding="utf-8", newline=""
    )
    manifest_path = Path(f"{raw}.manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source_kind": "official_verified",
                "verification_status": "official_verified",
                "source_description": "Test official source",
                "source_url": "https://example.invalid/results",
                "verified_at": "2026-10-01T00:00:00+00:00",
                "start_round": 753,
                "end_round": 754,
                "draw_count": 2,
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "date_sidecar": date_sidecar.name,
                "date_sidecar_sha256": hashlib.sha256(date_sidecar.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return raw, official, manifest_path, date_sidecar


def test_official_manifest_is_advanced_with_manual_draw_and_keeps_official_prefix(
    tmp_path,
):
    raw, official, manifest_path, date_sidecar = _write_official_provenance_sources(tmp_path)
    old_csv = raw.read_bytes()
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = _register(raw, official)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = load_csv(raw)
    assert manifest["source_kind"] == "manual_unverified"
    assert manifest["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["official_source"]["sha256"] == old_manifest["sha256"]
    assert manifest["official_source"]["byte_length"] == len(old_csv)
    assert hashlib.sha256(raw.read_bytes()[: len(old_csv)]).hexdigest() == old_manifest["sha256"]
    assert date_sidecar.is_file()
    assert loaded.records[0].verification_status == "official_verified"
    assert loaded.records[1].verification_status == "official_verified"
    assert loaded.records[-1].verification_status == "manual_unverified"
    assert loaded.source_kind == "manual_unverified"
    assert result["manifest_backup_path"]
    assert Path(result["manifest_backup_path"]).read_bytes() == json.dumps(
        old_manifest
    ).encode("utf-8")


def test_official_manifest_checksum_mismatch_is_not_rewritten_or_backed_up(tmp_path):
    raw, official, manifest_path, _ = _write_official_provenance_sources(tmp_path)
    raw.write_bytes(raw.read_bytes().replace(b"753,2017-05-06,2,17", b"753,2017-05-06,2,16"))
    original_csv = raw.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="CSV content does not match"):
        _register(raw, official)

    assert raw.read_bytes() == original_csv
    assert manifest_path.read_bytes() == original_manifest
    assert not (tmp_path / "backups").exists()


def test_registration_is_blocked_by_live_reservation_worker_without_changing_csv(
    tmp_path, monkeypatch
):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()
    queue_db = tmp_path / "research_queue.sqlite"
    with sqlite3.connect(queue_db) as connection:
        connection.executescript(
            """
            CREATE TABLE research_queues (status TEXT NOT NULL);
            CREATE TABLE research_queue_items (
                status TEXT NOT NULL, worker_pid INTEGER, checkpoint_path TEXT
            );
            INSERT INTO research_queues VALUES ('running');
            INSERT INTO research_queue_items VALUES ('running', NULL, NULL);
            """
        )
    monkeypatch.setattr(manager, "_research_lock_path", lambda: tmp_path / "global-lock")
    monkeypatch.setattr(manager, "_research_queue_database_path", lambda: queue_db)

    reason = manager.draw_registration_block_reason()
    assert reason and "상태 확인" in reason
    with pytest.raises(RuntimeError):
        _register(raw, official)

    assert raw.read_bytes() == original
    assert not (tmp_path / "draws.csv.metadata.json").exists()
    assert not (tmp_path / "backups").exists()


def test_waiting_reservations_do_not_block_registration(tmp_path, monkeypatch):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    queue_db = tmp_path / "research_queue.sqlite"
    with sqlite3.connect(queue_db) as connection:
        connection.executescript(
            """
            CREATE TABLE research_queues (status TEXT NOT NULL);
            CREATE TABLE research_queue_items (
                status TEXT NOT NULL, worker_pid INTEGER, checkpoint_path TEXT,
                scheduled_at TEXT
            );
            INSERT INTO research_queues VALUES ('running');
            INSERT INTO research_queue_items VALUES ('waiting', NULL, NULL, NULL);
            INSERT INTO research_queue_items VALUES ('waiting', NULL, NULL, NULL);
            INSERT INTO research_queue_items VALUES (
                'waiting', NULL, NULL, '2099-01-01T00:00:00+00:00'
            );
            """
        )
    monkeypatch.setattr(manager, "_research_queue_database_path", lambda: queue_db)

    assert draw_registration_status() == "등록 가능 — 대기 중인 예약 3건"
    result = _register(raw, official)

    assert result["status"] == "registered"
    with sqlite3.connect(queue_db) as connection:
        assert connection.execute(
            "SELECT status,count(*) FROM research_queue_items GROUP BY status"
        ).fetchall() == [("waiting", 3)]


def test_stale_queue_running_status_with_dead_pid_and_completed_checkpoint_allows_write(
    tmp_path, monkeypatch
):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    queue_db = tmp_path / "research_queue.sqlite"
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"status":"completed"}', encoding="utf-8")
    with sqlite3.connect(queue_db) as connection:
        connection.executescript(
            """
            CREATE TABLE research_queue_items (
                status TEXT NOT NULL, worker_pid INTEGER, checkpoint_path TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO research_queue_items VALUES ('running', ?, ?)",
            (2_147_483_647, str(checkpoint)),
        )
    monkeypatch.setattr(manager, "_research_queue_database_path", lambda: queue_db)

    assert draw_registration_status() == "등록 가능 — 실행 중인 연구 없음"
    assert _register(raw, official)["status"] == "registered"


def test_live_scheduler_lease_and_pid_show_reservation_starting(tmp_path, monkeypatch):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    queue_db = tmp_path / "research_queue.sqlite"
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"status":"starting"}', encoding="utf-8")
    with sqlite3.connect(queue_db) as connection:
        connection.executescript(
            """
            CREATE TABLE research_queue_items (
                status TEXT NOT NULL, worker_pid INTEGER, checkpoint_path TEXT
            );
            CREATE TABLE research_queue_lease (
                singleton INTEGER PRIMARY KEY, owner TEXT, queue_id TEXT, heartbeat_at REAL
            );
            """
        )
        connection.execute(
            "INSERT INTO research_queue_items VALUES ('running', ?, ?)",
            (os.getpid(), str(checkpoint)),
        )
        connection.execute(
            "INSERT INTO research_queue_lease VALUES (1, 'owner', 'queue', ?)",
            (time.time(),),
        )
    monkeypatch.setattr(manager, "_research_queue_database_path", lambda: queue_db)

    assert draw_registration_status() == "등록 불가 — 예약 연구가 실행을 시작 중입니다."
    with pytest.raises(RuntimeError):
        _register(raw, official)
    assert not Path(f"{raw}.metadata.json").exists()


def test_live_research_lock_blocks_registration_without_db_running_status(
    tmp_path, monkeypatch
):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()
    lock_path = tmp_path / "research_global"
    monkeypatch.setattr(manager, "_research_lock_path", lambda: lock_path)
    real_lock = manager.interprocess_lock

    def active_worker_lock(key, *, timeout=60.0, scope="write"):
        if Path(key) == lock_path and scope == "research-global":
            raise TimeoutError("simulated active research process")
        return real_lock(key, timeout=timeout, scope=scope)

    monkeypatch.setattr(manager, "interprocess_lock", active_worker_lock)
    assert "등록 대기" in manager.draw_registration_block_reason()
    with pytest.raises(RuntimeError, match="잠금"):
        _register(raw, official)
    assert raw.read_bytes() == original


def test_duplicate_round_and_invalid_draw_values_are_rejected(tmp_path):
    raw, official = _write_sources(tmp_path)
    _register(raw, official)
    duplicate = _register(raw, official)
    assert duplicate["status"] == "duplicate"
    with pytest.raises(ValueError, match="이미 존재"):
        _register(raw, official, main_numbers=[3, 4, 12, 18, 27, 36])
    with pytest.raises(ValueError, match="중복"):
        register_draw(
            raw,
            round_number=756,
            draw_date=date(2017, 5, 27),
            main_numbers=[1, 1, 3, 4, 5, 6],
            bonus_number=7,
            official_dates_path=official,
        )
    with pytest.raises(ValueError, match="겹치면 안"):
        register_draw(
            raw,
            round_number=756,
            draw_date=date(2017, 5, 27),
            main_numbers=[1, 2, 3, 4, 5, 6],
            bonus_number=1,
            official_dates_path=official,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"round_number": 0}, "1 이상의 정수"),
        ({"round_number": 755.5}, "1 이상의 정수"),
        ({"main_numbers": [1, 2, 3, 4, 5]}, "정확히 6개"),
        ({"main_numbers": [0, 2, 3, 4, 5, 6]}, "1~45 범위"),
        ({"bonus_number": 46}, "1~45"),
        ({"draw_date": "2017-05-20"}, "추첨일을 입력"),
        ({"draw_date": date.today() + timedelta(days=1)}, "미래 날짜"),
        ({"draw_date": date(2017, 5, 13)}, "직전 회차의 추첨일"),
        ({"round_number": 756}, "다음 등록 가능 회차는 755"),
    ],
)
def test_registration_rejects_invalid_round_number_date_or_sequence(
    tmp_path, overrides, message
):
    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()

    with pytest.raises(ValueError, match=message):
        _register(raw, official, **overrides)

    assert raw.read_bytes() == original
    assert not (tmp_path / "backups").exists()


def test_metadata_write_failure_restores_original_csv(tmp_path, monkeypatch):
    import lotto_pattern_lab.draw_management as manager

    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()
    real_write = manager.atomic_write_bytes
    metadata_path = manager._metadata_path(raw.resolve())

    def fail_metadata(path, content, **kwargs):
        if path == metadata_path:
            raise OSError("simulated metadata failure")
        return real_write(path, content, **kwargs)

    monkeypatch.setattr(manager, "atomic_write_bytes", fail_metadata)
    with pytest.raises(OSError, match="simulated"):
        _register(raw, official)
    assert raw.read_bytes() == original
    assert not metadata_path.exists()
    assert not (tmp_path / "draws.csv.history.jsonl").exists()


def test_manifest_write_failure_restores_csv_and_original_manifest(tmp_path, monkeypatch):
    import lotto_pattern_lab.draw_management as manager

    raw, official, manifest_path, _ = _write_official_provenance_sources(tmp_path)
    original_csv = raw.read_bytes()
    original_manifest = manifest_path.read_bytes()
    real_write = manager.atomic_write_bytes
    failed = False

    def fail_manifest(path, content, **kwargs):
        nonlocal failed
        if path == manifest_path and not failed:
            failed = True
            raise OSError("simulated manifest failure")
        return real_write(path, content, **kwargs)

    monkeypatch.setattr(manager, "atomic_write_bytes", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        _register(raw, official)

    assert raw.read_bytes() == original_csv
    assert manifest_path.read_bytes() == original_manifest
    assert not Path(f"{raw}.metadata.json").exists()
    assert not Path(f"{raw}.history.jsonl").exists()


def test_official_main_number_mismatch_stops_before_backup_or_write(tmp_path):
    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()
    official.write_text(
        "round,date,n1,n2,n3,n4,n5,n6\n"
        "753,2017-05-06,2,17,19,24,37,41\n"
        "754,2017-05-13,1,5,13,24,29,34\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="메인 번호가 공식 데이터와 달라"):
        _register(raw, official)

    assert raw.read_bytes() == original
    assert not (tmp_path / "backups").exists()


def test_manual_and_automated_official_verification_remain_distinct(tmp_path):
    raw, official = _write_sources(tmp_path)
    _register(raw, official)

    manual = confirm_manual_official_comparison(
        raw, 755, confirmation_note="공식 사이트에서 직접 대조"
    )
    assert manual["status"] == "manual_unverified"
    assert manual["manual_user_confirmation"]["confirmed"]

    verified = verify_draw_against_official(
        raw,
        755,
        official_record=DrawRecord(
            755,
            Draw.from_numbers((4, 12, 18, 27, 36, 44), 7),
            date(2017, 5, 20),
        ),
    )
    assert verified["status"] == "official_verified"
    assert all(verified["field_matches"].values())


def test_correction_request_does_not_modify_csv(tmp_path):
    raw, official = _write_sources(tmp_path)
    original = raw.read_bytes()
    request = create_draw_correction_request(
        raw,
        round_number=753,
        requested_record={"main_numbers": [1, 2, 3, 4, 5, 6], "bonus_number": 7},
        reason="확인 필요",
        directory=tmp_path / "requests",
    )
    assert request.is_file()
    assert raw.read_bytes() == original

"""Safe, versioned management for one manually registered draw at a time."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import time
import uuid
from contextlib import ExitStack
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .data_audit import dataset_hashes, dataset_range_hashes
from .dataset import DrawDataset, DrawRecord, load_csv
from .paths import project_path


def _metadata_path(csv_path: Path) -> Path:
    return Path(f"{csv_path}.metadata.json")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _round_range_hash(dataset: DrawDataset, start_round: int, end_round: int) -> str:
    rows = [
        {
            "round": record.round_number,
            "numbers": list(record.draw.numbers),
            "date": record.draw_date.isoformat() if record.draw_date else None,
            "bonus": record.draw.bonus,
        }
        for record in dataset.records
        if start_round <= record.round_number <= end_round
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _official_dates(path: str | Path) -> tuple[dict[str, str], dict[int, DrawRecord]]:
    official_path = project_path(path)
    if not official_path.is_file():
        return {}, {}
    official = load_csv(official_path)
    dates = {
        str(record.round_number): record.draw_date.isoformat()
        for record in official.records
        if record.draw_date is not None
    }
    return dates, {record.round_number: record for record in official.records}


def _read_raw_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("기준 CSV에 헤더가 없습니다.") from error
        rows = list(reader)
    if not header or any(len(row) != len(header) for row in rows):
        raise ValueError("기준 CSV의 컬럼 수가 일정하지 않습니다.")
    round_column = "round" if "round" in header else "draw_no" if "draw_no" in header else None
    if round_column is None or not all(f"n{i}" in header for i in range(1, 7)):
        raise ValueError("기준 CSV는 round 또는 draw_no와 n1~n6 컬럼이 필요합니다.")
    if not any(name in header for name in ("bonus", "bonus_no", "bnsWnNo")):
        raise ValueError("보너스 번호 컬럼이 없는 CSV에는 신규 회차를 등록할 수 없습니다.")
    return header, rows


def _backup_original(csv_path: Path, content: bytes, backup_directory: str | Path | None) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = project_path(
        backup_directory or "data/backups/manual_draws"
    )
    backup_path = backup_root / f"{csv_path.name}.{stamp}.{digest[:12]}.bak"
    atomic_write_bytes(backup_path, content)
    if hashlib.sha256(backup_path.read_bytes()).hexdigest() != digest:
        backup_path.unlink(missing_ok=True)
        raise OSError("원본 CSV 백업의 SHA-256 검증에 실패했습니다.")
    return backup_path


def _research_lock_path() -> Path:
    return project_path("checkpoints/research_global")


def _dataset_update_lock_path() -> Path:
    return project_path("checkpoints/dataset_update_gate")


def _research_queue_database_path() -> Path:
    return project_path("data/database/research_queue.sqlite")


def _pid_is_alive(pid: int | None) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _queue_registration_state(
    queue_database_path: str | Path | None = None,
) -> tuple[int, str | None]:
    """Return waiting reservation count and a reason if a real worker may be active."""
    queue_path = (
        Path(queue_database_path).resolve()
        if queue_database_path is not None
        else _research_queue_database_path()
    )
    if not queue_path.is_file():
        return 0, None

    connection = None
    try:
        connection = sqlite3.connect(f"{queue_path.as_uri()}?mode=ro", uri=True, timeout=1)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "research_queue_items" not in tables:
            raise RuntimeError("research_queue_items table is missing")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(research_queue_items)")}
        required = {"status", "worker_pid", "checkpoint_path"}
        if not required.issubset(columns):
            raise RuntimeError("research queue worker columns are missing")
        pending_count = int(
            connection.execute(
                "SELECT count(*) FROM research_queue_items WHERE status='waiting'"
            ).fetchone()[0]
        )
        active_rows = connection.execute(
            "SELECT worker_pid,checkpoint_path FROM research_queue_items WHERE status='running'"
        ).fetchall()
        lease_live = False
        if "research_queue_lease" in tables:
            lease = connection.execute(
                "SELECT heartbeat_at FROM research_queue_lease WHERE singleton=1"
            ).fetchone()
            lease_live = bool(lease and time.time() - float(lease[0]) <= 45)
        for worker_pid, checkpoint_path in active_rows:
            pid = int(worker_pid) if worker_pid is not None else None
            checkpoint = Path(checkpoint_path) if checkpoint_path else None
            if checkpoint is None:
                return pending_count, "unknown_worker_state"
            try:
                with interprocess_lock(checkpoint, timeout=0, scope="research-worker"):
                    pass
            except TimeoutError:
                return pending_count, "queue_worker"
            if pid is None:
                return pending_count, "unknown_worker_state"
            if _pid_is_alive(pid):
                # A PID without its worker lock is ambiguous; the scheduler heartbeat
                # distinguishes a launch transition from a stale/reused PID.
                return pending_count, "queue_starting" if lease_live else "unknown_worker_state"
            # A live scheduler lease is allowed for waiting work; it cannot claim work
            # during registration because scheduler_step shares the update gate.
            if lease_live:
                continue
    except sqlite3.Error as error:
        raise RuntimeError(
            f"예약 연구 상태를 안전하게 확인할 수 없어 등록을 보류했습니다: {error}"
        ) from error
    except RuntimeError:
        raise
    finally:
        if connection is not None:
            connection.close()
    return pending_count, None


def draw_registration_block_reason() -> str | None:
    """Return a safe, read-only UI reason when draw writes could affect research."""
    try:
        with interprocess_lock(_dataset_update_lock_path(), timeout=0, scope="dataset-update"):
            with interprocess_lock(
                _research_lock_path(), timeout=0, scope="research-global"
            ):
                pass
    except TimeoutError:
        return "등록 대기 — 다른 작업이 연구 데이터 잠금을 사용 중입니다."
    try:
        _, state = _queue_registration_state()
        if state == "queue_worker":
            return "등록 불가 — 예약 연구 워커가 실행 중입니다."
        if state == "queue_starting":
            return "등록 불가 — 예약 연구가 실행을 시작 중입니다."
        if state == "unknown_worker_state":
            return "등록 불가 — 연구 실행 상태 확인이 필요합니다."
    except (OSError, RuntimeError, sqlite3.Error) as error:
        return f"등록 불가 — 연구 실행 상태 확인이 필요합니다: {error}"
    return None


def draw_registration_status() -> str:
    """Describe whether registration is safe, including queued reservations."""
    try:
        with interprocess_lock(_dataset_update_lock_path(), timeout=0, scope="dataset-update"):
            with interprocess_lock(
                _research_lock_path(), timeout=0, scope="research-global"
            ):
                pass
        pending_count, state = _queue_registration_state()
    except TimeoutError:
        return "등록 대기 — 다른 작업이 연구 데이터 잠금을 사용 중입니다."
    except (OSError, RuntimeError, sqlite3.Error) as error:
        return f"등록 불가 — 연구 실행 상태 확인이 필요합니다: {error}"
    if state == "queue_worker":
        return "등록 불가 — 예약 연구 워커가 실행 중입니다."
    if state == "queue_starting":
        return "등록 불가 — 예약 연구가 실행을 시작 중입니다."
    if state:
        return "등록 불가 — 연구 실행 상태 확인이 필요합니다."
    if pending_count:
        return f"등록 가능 — 대기 중인 예약 {pending_count}건"
    return "등록 가능 — 실행 중인 연구 없음"


def register_draw(
    csv_path: str | Path,
    *,
    round_number: int,
    draw_date: date,
    main_numbers: tuple[int, ...] | list[int],
    bonus_number: int,
    official_dates_path: str | Path = "data/official/draws.csv",
    backup_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize data writes against active workers and scheduler launch transitions."""
    with ExitStack() as stack:
        try:
            stack.enter_context(
                interprocess_lock(
                    _dataset_update_lock_path(), timeout=0, scope="dataset-update"
                )
            )
            stack.enter_context(
                interprocess_lock(
                    _research_lock_path(), timeout=0, scope="research-global"
                )
            )
        except TimeoutError as error:
            raise RuntimeError(
                "연구 데이터 또는 실행 잠금이 사용 중이어서 등록을 보류했습니다."
            ) from error
        _, state = _queue_registration_state()
        if state in {"queue_worker", "queue_starting"}:
            raise RuntimeError("예약 연구 워커가 실행 중이어서 등록을 보류했습니다.")
        if state == "unknown_worker_state":
            raise RuntimeError("연구 실행 상태를 확인할 수 없어 등록을 보류했습니다.")
        return _register_draw_under_research_lock(
            csv_path,
            round_number=round_number,
            draw_date=draw_date,
            main_numbers=main_numbers,
            bonus_number=bonus_number,
            official_dates_path=official_dates_path,
            backup_directory=backup_directory,
        )


def _register_draw_under_research_lock(
    csv_path: str | Path,
    *,
    round_number: int,
    draw_date: date,
    main_numbers: tuple[int, ...] | list[int],
    bonus_number: int,
    official_dates_path: str | Path = "data/official/draws.csv",
    backup_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, back up, append, re-read, atomically replace and version a new draw."""
    path = project_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"기준 당첨번호 CSV가 없습니다: {path}")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("회차는 1 이상의 정수여야 합니다.")
    if len(main_numbers) != 6:
        raise ValueError("메인 당첨번호는 정확히 6개여야 합니다.")
    if any(isinstance(number, bool) or not isinstance(number, int) for number in main_numbers):
        raise ValueError("메인 당첨번호는 정수여야 합니다.")
    if any(number < 1 or number > 45 for number in main_numbers):
        raise ValueError("메인 당첨번호는 1~45 범위여야 합니다.")
    if len(set(main_numbers)) != 6:
        raise ValueError("메인 당첨번호에 중복이 있습니다.")
    if isinstance(bonus_number, bool) or not isinstance(bonus_number, int):
        raise ValueError("보너스 번호는 정수여야 합니다.")
    if bonus_number < 1 or bonus_number > 45 or bonus_number in main_numbers:
        raise ValueError("보너스 번호는 1~45이며 메인 번호와 겹치면 안 됩니다.")
    if not isinstance(draw_date, date) or isinstance(draw_date, datetime):
        raise ValueError("추첨일을 입력해야 합니다.")
    if draw_date > date.today():
        raise ValueError("미래 날짜의 추첨 결과는 등록할 수 없습니다.")

    metadata_path = _metadata_path(path)
    manifest_path = Path(f"{path}.manifest.json")
    with interprocess_lock(path, scope="manual-draw-write"):
        original = path.read_bytes()
        old_hash = hashlib.sha256(original).hexdigest()
        old_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
        old_manifest_value = (
            json.loads(old_manifest.decode("utf-8")) if old_manifest is not None else None
        )
        header, rows = _read_raw_rows(path)
        index = {name: header.index(name) for name in header}
        round_column = "round" if "round" in index else "draw_no"
        bonus_column = next(name for name in ("bonus", "bonus_no", "bnsWnNo") if name in index)
        records = load_csv(path)
        current_metadata = None
        if metadata_path.exists():
            current_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if current_metadata.get("csv_sha256") != old_hash:
                raise ValueError("CSV와 데이터 버전 기록이 달라 쓰기를 중단했습니다.")
            dates = dict(current_metadata.get("draw_dates", {}))
            statuses = dict(current_metadata.get("round_verification", {}))
        else:
            dates = {
                str(record.round_number): record.draw_date.isoformat()
                for record in records.records
                if record.draw_date is not None
            }
            dates.update(_official_dates(official_dates_path)[0])
            statuses = {
                str(record.round_number): {
                    "status": record.verification_status or "manual_unverified",
                    "method": (
                        "verified by original official provenance manifest"
                        if record.verification_status == "official_verified"
                        else "legacy source row has not been fully reconciled"
                    ),
                }
                for record in records.records
            }

        existing = next(
            (record for record in records.records if record.round_number == round_number), None
        )
        if existing is not None:
            same = (
                tuple(main_numbers) == existing.draw.numbers
                and bonus_number == existing.draw.bonus
                and draw_date.isoformat() == dates.get(str(round_number), "")
            )
            if same:
                return {"status": "duplicate", "round_number": round_number, "csv_sha256": old_hash}
            raise ValueError(
                "해당 회차는 이미 존재하며 값이 다릅니다. 수정 요청 절차를 사용하세요."
            )
        latest = records.records[-1].round_number
        present_rounds = {record.round_number for record in records.records}
        missing_rounds = [
            value
            for value in range(records.records[0].round_number, latest + 1)
            if value not in present_rounds
        ]
        if missing_rounds:
            raise ValueError(
                f"기준 데이터에 기존 회차 누락이 있어 등록을 중단했습니다: {missing_rounds[:20]}"
            )
        if round_number != latest + 1:
            raise ValueError(
                f"다음 등록 가능 회차는 {latest + 1}회입니다. 회차 누락을 먼저 확인하세요."
            )

        official_date_map, official_records = _official_dates(official_dates_path)
        mismatched_official_rows = [
            record.round_number
            for record in records.records
            if record.round_number in official_records
            and record.draw.numbers != official_records[record.round_number].draw.numbers
        ]
        if mismatched_official_rows:
            raise ValueError(
                "기준 CSV의 메인 번호가 공식 데이터와 달라 등록을 중단했습니다. "
                f"불일치 회차: {mismatched_official_rows[:20]}"
            )
        last_date_text = dates.get(str(latest)) or official_date_map.get(str(latest))
        if last_date_text and draw_date <= date.fromisoformat(last_date_text):
            raise ValueError("추첨일은 직전 회차의 추첨일보다 늦어야 합니다.")
        for existing_round, existing_date in dates.items():
            if date.fromisoformat(existing_date) == draw_date:
                raise ValueError(f"추첨일이 이미 {existing_round}회와 겹칩니다.")

        values: dict[str, str] = {
            round_column: str(round_number),
            **{f"n{i}": str(number) for i, number in enumerate(sorted(main_numbers), 1)},
            bonus_column: str(bonus_number),
        }
        for column in ("date", "draw_date"):
            if column in index:
                values[column] = draw_date.isoformat()
        new_row = [values.get(column, "") for column in header]
        line_ending = "\r\n" if b"\r\n" in original else "\n"
        row_buffer = io.StringIO(newline="")
        csv.writer(row_buffer, lineterminator=line_ending).writerow(new_row)
        encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
        row_bytes = row_buffer.getvalue().encode("utf-8")
        separator = b"" if original.endswith((b"\n", b"\r")) else line_ending.encode()
        updated_bytes = original + separator + row_bytes

        temporary = path.with_name(f"{path.stem}_{uuid.uuid4().hex}.tmp")
        atomic_write_bytes(temporary, updated_bytes)
        try:
            new_dataset = load_csv(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        if new_dataset.records[-1].round_number != round_number:
            raise ValueError("임시 CSV 재검증에서 신규 회차를 확인하지 못했습니다.")
        new_draw = new_dataset.records[-1].draw
        if new_draw.numbers != tuple(sorted(main_numbers)) or new_draw.bonus != bonus_number:
            raise ValueError("임시 CSV 재검증 결과가 입력값과 일치하지 않습니다.")
        dates[str(round_number)] = draw_date.isoformat()
        statuses[str(round_number)] = {
            "status": "manual_unverified",
            "method": "user_entered_pending_official_comparison",
            "registered_at": datetime.now(UTC).isoformat(),
        }
        records_with_dates = [
            DrawRecord(
                record.round_number,
                record.draw,
                date.fromisoformat(dates[str(record.round_number)])
                if str(record.round_number) in dates
                else record.draw_date,
                statuses.get(str(record.round_number), {}).get("status"),
            )
            for record in new_dataset.records
        ]
        dated_dataset = DrawDataset(
            tuple(records_with_dates), "manual_unverified", "User-managed draw version."
        )
        old_dated_records = [
            DrawRecord(
                record.round_number,
                record.draw,
                date.fromisoformat(dates[str(record.round_number)])
                if str(record.round_number) in dates
                else record.draw_date,
                statuses.get(str(record.round_number), {}).get("status"),
            )
            for record in records.records
        ]
        old_dataset_for_hash = DrawDataset(
            tuple(old_dated_records), "manual_unverified", "Previous manual data version."
        )
        fixed_range_before = dataset_range_hashes(
            old_dataset_for_hash,
            records.records[0].round_number,
            latest,
        )
        fixed_range_after = dataset_range_hashes(
            dated_dataset,
            records.records[0].round_number,
            latest,
        )
        backup_path = _backup_original(path, original, backup_directory)
        new_hash = hashlib.sha256(updated_bytes).hexdigest()
        metadata = {
            "schema_version": 1,
            "version_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "csv_path": str(path),
            "csv_sha256": new_hash,
            "previous_csv_sha256": old_hash,
            "backup_path": str(backup_path),
            "round_range": [dated_dataset.records[0].round_number, round_number],
            "draw_count": len(dated_dataset.records),
            "draw_dates": dates,
            "round_verification": statuses,
            "hash_algorithm_version": "lotto-dataset-hashes-v2",
            "hashes": dataset_hashes(dated_dataset),
            "fixed_range_before_hashes": fixed_range_before,
            "fixed_range_after_hashes": fixed_range_after,
            "fixed_range_before_sha256": _round_range_hash(
                old_dataset_for_hash,
                records.records[0].round_number,
                latest,
            ),
            "fixed_range_after_sha256": _round_range_hash(
                dated_dataset,
                records.records[0].round_number,
                latest,
            ),
            "verification_status": "manual_unverified",
            "format_encoding": encoding,
            "format_columns": header,
        }
        official_source = None
        if old_manifest_value is not None:
            if old_manifest_value.get("source_kind") == "official_verified":
                official_source = dict(old_manifest_value)
                official_source["byte_length"] = len(original)
            elif old_manifest_value.get("source_kind") == "manual_unverified":
                official_source = old_manifest_value.get("official_source")
                if not isinstance(official_source, dict):
                    raise ValueError("기존 Manifest에 공식 원본 출처 정보가 없습니다.")
            else:
                raise ValueError("지원하지 않는 기존 CSV provenance Manifest입니다.")
        new_manifest = None
        if official_source is not None:
            new_manifest = {
                "schema_version": 2,
                "source_kind": "manual_unverified",
                "verification_status": "manual_unverified",
                "source_description": "Official verified history with manually registered draws.",
                "sha256": new_hash,
                "start_round": dated_dataset.records[0].round_number,
                "end_round": round_number,
                "draw_count": len(dated_dataset.records),
                "official_source": official_source,
            }
        old_metadata = metadata_path.read_bytes() if metadata_path.exists() else None
        history_path = Path(f"{path}.history.jsonl")
        old_history = history_path.read_bytes() if history_path.exists() else b""
        history_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "draw_registered",
            "recorded_at": datetime.now(UTC).isoformat(),
            "round_number": round_number,
            "old_csv_sha256": old_hash,
            "new_csv_sha256": new_hash,
            "backup_path": str(backup_path),
            "version_id": metadata["version_id"],
            "fixed_range_before_sha256": metadata["fixed_range_before_sha256"],
            "fixed_range_after_sha256": metadata["fixed_range_after_sha256"],
        }
        manifest_backup_path = None
        if old_manifest is not None:
            manifest_backup_path = backup_path.with_name(f"{backup_path.name}.manifest.json")
            atomic_write_bytes(manifest_backup_path, old_manifest)
            if hashlib.sha256(manifest_backup_path.read_bytes()).hexdigest() != hashlib.sha256(
                old_manifest
            ).hexdigest():
                raise OSError("Provenance Manifest 백업 SHA-256 검증에 실패했습니다.")
        try:
            atomic_write_bytes(path, updated_bytes)
            atomic_write_bytes(metadata_path, _json_bytes(metadata))
            if new_manifest is not None:
                atomic_write_bytes(manifest_path, _json_bytes(new_manifest))
            atomic_write_bytes(
                history_path,
                old_history + json.dumps(history_event, ensure_ascii=False).encode("utf-8") + b"\n",
            )
            saved_dataset = load_csv(path)
            if (
                saved_dataset.records[-1].round_number != round_number
                or hashlib.sha256(path.read_bytes()).hexdigest() != new_hash
            ):
                raise ValueError("저장 파일을 다시 읽은 결과가 저장 예정 데이터와 다릅니다.")
        except Exception as write_error:
            rollback_errors = []
            restore_bytes = [(path, original)]
            if old_manifest is not None:
                restore_bytes.append((manifest_path, old_manifest))
            if old_metadata is not None:
                restore_bytes.append((metadata_path, old_metadata))
            if old_history:
                restore_bytes.append((history_path, old_history))
            for target, content in restore_bytes:
                try:
                    atomic_write_bytes(target, content)
                except Exception as rollback_error:  # preserve every possible artifact
                    rollback_errors.append(f"{target}: {rollback_error}")
            for target, existed in (
                (metadata_path, old_metadata is not None),
                (history_path, bool(old_history)),
            ):
                if existed:
                    continue
                try:
                    target.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(f"{target}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "저장 실패 후 일부 파일 복구를 완료하지 못했습니다. "
                    f"원본 백업은 {backup_path} 및 {manifest_backup_path}에 보존됩니다. "
                    + "; ".join(rollback_errors)
                ) from write_error
            raise
    return {
        "status": "registered",
        "round_number": round_number,
        "csv_path": str(path),
        "backup_path": str(backup_path),
        "manifest_backup_path": (
            str(manifest_backup_path) if manifest_backup_path is not None else None
        ),
        "old_csv_sha256": old_hash,
        "new_csv_sha256": new_hash,
        "fixed_range_before_sha256": metadata["fixed_range_before_sha256"],
        "fixed_range_after_sha256": metadata["fixed_range_after_sha256"],
        "metadata_path": str(metadata_path),
        "version_id": metadata["version_id"],
        "draw_count": len(dated_dataset.records),
        "verification_status": "manual_unverified",
        "source_used_by_future_research": True,
    }


def create_draw_correction_request(
    csv_path: str | Path,
    *,
    round_number: int,
    requested_record: dict[str, Any],
    reason: str,
    directory: str | Path = "data/manual_draw_requests",
) -> Path:
    """Keep an immutable correction proposal; never edit or delete an existing draw."""
    path = project_path(csv_path)
    dataset = load_csv(path)
    existing = next(
        (record for record in dataset.records if record.round_number == round_number), None
    )
    if existing is None:
        raise ValueError("수정 요청 대상 회차를 찾을 수 없습니다.")
    if not reason.strip():
        raise ValueError("수정 사유를 입력해야 합니다.")
    requested_numbers = requested_record.get("main_numbers")
    requested_bonus = requested_record.get("bonus_number")
    if (
        not isinstance(requested_numbers, (list, tuple))
        or len(requested_numbers) != 6
        or any(type(number) is not int or number < 1 or number > 45 for number in requested_numbers)
        or len(set(requested_numbers)) != 6
        or type(requested_bonus) is not int
        or requested_bonus < 1
        or requested_bonus > 45
        or requested_bonus in requested_numbers
    ):
        raise ValueError("수정 요청 번호는 중복 없는 6개 번호와 올바른 보너스 번호여야 합니다.")
    request = {
        "request_id": str(uuid.uuid4()),
        "requested_at": datetime.now(UTC).isoformat(),
        "round_number": round_number,
        "existing_record": {
            "numbers": list(existing.draw.numbers),
            "bonus": existing.draw.bonus,
            "date": existing.draw_date.isoformat() if existing.draw_date else None,
        },
        "requested_record": requested_record,
        "reason": reason.strip(),
        "current_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "pending_review",
    }
    output_dir = project_path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{request['request_id']}.json"
    atomic_write_bytes(output, _json_bytes(request))
    return output


def verify_draw_against_official(
    csv_path: str | Path,
    round_number: int,
    *,
    official_record: DrawRecord,
) -> dict[str, Any]:
    """Update verification metadata only after a caller fetched official source data."""
    path = project_path(csv_path)
    meta_path = _metadata_path(path)
    with interprocess_lock(path, scope="manual-draw-write"):
        dataset = load_csv(path)
        actual = next((row for row in dataset.records if row.round_number == round_number), None)
        if actual is None or official_record.round_number != round_number:
            raise ValueError("공식 대조 대상 회차가 일치하지 않습니다.")
        matches = (
            actual.draw.numbers == official_record.draw.numbers
            and actual.draw.bonus == official_record.draw.bonus
            and actual.draw_date == official_record.draw_date
        )
        if not meta_path.exists():
            raise ValueError("수동 데이터 버전 메타데이터가 없습니다.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        previous_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        verification = {
            "status": "official_verified" if matches else "verification_failed",
            "method": "automated_official_endpoint_comparison",
            "verified_at": datetime.now(UTC).isoformat(),
            "official_date": official_record.draw_date.isoformat()
            if official_record.draw_date
            else None,
            "field_matches": {
                "main_numbers": actual.draw.numbers == official_record.draw.numbers,
                "bonus_number": actual.draw.bonus == official_record.draw.bonus,
                "draw_date": actual.draw_date == official_record.draw_date,
            },
        }
        metadata["round_verification"][str(round_number)] = verification
        metadata["verification_status"] = "mixed_manual_and_official"
        metadata["previous_metadata_sha256"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        metadata["csv_sha256"] = previous_hash
        old_metadata = meta_path.read_bytes()
        history_path = Path(f"{path}.history.jsonl")
        old_history = history_path.read_bytes() if history_path.exists() else b""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "draw_verification",
            "recorded_at": verification["verified_at"],
            "round_number": round_number,
            "status": verification["status"],
            "field_matches": verification["field_matches"],
            "csv_sha256": previous_hash,
            "metadata_sha256_before": metadata["previous_metadata_sha256"],
        }
        try:
            atomic_write_bytes(meta_path, _json_bytes(metadata))
            atomic_write_bytes(
                history_path,
                old_history + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n",
            )
        except Exception:
            atomic_write_bytes(meta_path, old_metadata)
            if old_history:
                atomic_write_bytes(history_path, old_history)
            else:
                history_path.unlink(missing_ok=True)
            raise
    return {"round_number": round_number, **verification}


def confirm_manual_official_comparison(
    csv_path: str | Path,
    round_number: int,
    *,
    confirmation_note: str,
) -> dict[str, Any]:
    """Record a user's manual check without representing it as an automated verification."""
    if not confirmation_note.strip():
        raise ValueError("공식 결과를 확인한 방법을 간단히 기록해야 합니다.")
    path = project_path(csv_path)
    meta_path = _metadata_path(path)
    with interprocess_lock(path, scope="manual-draw-write"):
        dataset = load_csv(path)
        if not any(row.round_number == round_number for row in dataset.records):
            raise ValueError("대상 회차가 없습니다.")
        if not meta_path.is_file():
            raise ValueError("수동 관리 버전의 메타데이터가 없습니다.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get("csv_sha256") != csv_hash:
            raise ValueError("CSV와 데이터 버전 기록이 달라 확인을 기록하지 못했습니다.")
        timestamp = datetime.now(UTC).isoformat()
        round_status = metadata.setdefault("round_verification", {}).setdefault(
            str(round_number), {"status": "manual_unverified"}
        )
        round_status["manual_user_confirmation"] = {
            "confirmed": True,
            "method": "user_checked_official_result_page",
            "confirmed_at": timestamp,
            "note": confirmation_note.strip(),
        }
        old_metadata = meta_path.read_bytes()
        history_path = Path(f"{path}.history.jsonl")
        old_history = history_path.read_bytes() if history_path.exists() else b""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "manual_official_comparison_confirmation",
            "recorded_at": timestamp,
            "round_number": round_number,
            "verification_status": round_status.get("status", "manual_unverified"),
            "confirmation": round_status["manual_user_confirmation"],
            "csv_sha256": csv_hash,
        }
        try:
            atomic_write_bytes(meta_path, _json_bytes(metadata))
            atomic_write_bytes(
                history_path,
                old_history + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n",
            )
        except Exception:
            atomic_write_bytes(meta_path, old_metadata)
            if old_history:
                atomic_write_bytes(history_path, old_history)
            else:
                history_path.unlink(missing_ok=True)
            raise
    return {"round_number": round_number, **round_status}

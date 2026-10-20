from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from lotto_pattern_lab.dataset import DrawRecord, load_csv
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.draw_management import (
    confirm_manual_official_comparison,
    create_draw_correction_request,
    register_draw,
    verify_draw_against_official,
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

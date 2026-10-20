from __future__ import annotations

import json

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.future_validation import (
    audit_future_ledger,
    freeze_formula,
    load_frozen_formula,
    record_future_comparison,
    save_frozen_prediction,
)


def _experiment() -> dict:
    candidate = {
        "activation_condition": "add(x0, 1)",
        "formulas": ["x0"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
        "validation_score": 1.0,
        "test_score": 1.0,
    }
    return {
        "experiment_id": "freeze-test-experiment",
        "research_id": "freeze-test-research",
        "experiment_type": "intermittent",
        "status": "completed",
        "config": {"max_lag": 10, "feature_profile": "legacy"},
        "splits": {"train_rounds": [753, 1000]},
        "six_number_set_model": {
            "candidates": [candidate],
            "selected": candidate,
            "train_rounds": list(range(753, 1001)),
        },
    }


def test_frozen_formula_is_immutable_and_hash_checked(tmp_path) -> None:
    frozen = freeze_formula(
        _experiment(),
        "six_number_set",
        1,
        latest_confirmed_round=1246,
        dataset_sha256="data-hash",
        code_version="test",
        directory=tmp_path,
    )

    loaded = load_frozen_formula(frozen["freeze_id"], tmp_path)
    assert loaded["freeze_record_sha256"] == frozen["freeze_record_sha256"]
    target = tmp_path / f"{frozen['freeze_id']}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["frozen_experiment"]["config"]["max_lag"] = 1
    target.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_frozen_formula(frozen["freeze_id"], tmp_path)
    except ValueError as error:
        assert "변경" in str(error)
    else:
        raise AssertionError("modified freeze was accepted")


def test_future_prediction_is_persisted_and_result_is_append_only(tmp_path) -> None:
    synthetic = make_random_dataset(10, seed=99, first_round=1237)
    history = DrawDataset(
        synthetic.records,
        "official_verified",
        "test official",
        verified_at="test",
    )
    frozen = freeze_formula(
        _experiment(),
        "six_number_set",
        1,
        latest_confirmed_round=1246,
        dataset_sha256="before-draw-hash",
        code_version="test",
        directory=tmp_path,
    )
    prediction = save_frozen_prediction(
        frozen["freeze_id"],
        history,
        1247,
        data_file_sha256="before-draw-hash",
        official_latest_round=1246,
        random_seed=17,
        directory=tmp_path,
    )
    assert prediction["recommendation_status"] == "pre_draw_prediction"
    assert (tmp_path / f"{prediction['recommendation_id']}.json").is_file()

    actual = DrawRecord(1247, history.records[-1].draw)
    after_draw = DrawDataset(
        (*history.records, actual), "csv_unverified", "manual result not reconciled"
    )
    result = record_future_comparison(prediction, after_draw, directory=tmp_path)
    assert result is not None
    assert result["result_status"] == "manual_unverified"
    assert record_future_comparison(prediction, after_draw, directory=tmp_path) == result
    assert audit_future_ledger(tmp_path)["valid"]


def test_future_prediction_rejects_unverified_latest_draw(tmp_path) -> None:
    history = make_random_dataset(10, seed=99, first_round=1237)
    frozen = freeze_formula(
        _experiment(),
        "six_number_set",
        1,
        latest_confirmed_round=1246,
        dataset_sha256="before-draw-hash",
        code_version="test",
        directory=tmp_path,
    )

    try:
        save_frozen_prediction(
            frozen["freeze_id"],
            history,
            1247,
            data_file_sha256="before-draw-hash",
            official_latest_round=1246,
            random_seed=17,
            directory=tmp_path,
        )
    except ValueError as error:
        assert "공식 대조" in str(error)
    else:
        raise AssertionError("unverified draw data was accepted for a prospective forecast")


def test_future_ledger_detects_edit_and_blocks_future_appends(tmp_path) -> None:
    from lotto_pattern_lab.future_validation import append_future_event

    append_future_event("test", {"value": 1}, tmp_path)
    ledger = tmp_path / "events.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8").replace('"value":1', '"value":2'))
    assert not audit_future_ledger(tmp_path)["valid"]
    try:
        append_future_event("test", {"value": 2}, tmp_path)
    except ValueError as error:
        assert "무결성" in str(error)
    else:
        raise AssertionError("damaged ledger accepted a new event")

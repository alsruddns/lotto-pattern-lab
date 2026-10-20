"""Immutable formula freezes and append-only prospective prediction records."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .dataset import DrawDataset
from .paths import project_path
from .prediction import compare_recommendation, create_recommendation, save_recommendation

LEDGER_VERSION = 1


def _require_official_latest_draw(dataset: DrawDataset) -> None:
    latest = dataset.records[-1]
    status = latest.verification_status or dataset.verification_status
    if dataset.source_kind != "official_verified" and status != "official_verified":
        raise ValueError("미래 예측에는 공식 대조가 완료된 최신 회차 데이터가 필요합니다.")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _ledger_path(directory: str | Path | None = None) -> Path:
    root = project_path(directory or "predictions/future_validation")
    return root / "events.jsonl"


def audit_future_ledger(directory: str | Path | None = None) -> dict[str, Any]:
    """Verify the local hash chain; it detects edits but is not externally signed."""
    path = _ledger_path(directory)
    if not path.exists():
        return {"valid": True, "event_count": 0, "last_hash": "0" * 64, "error": None}
    previous = "0" * 64
    count = 0
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            event = json.loads(line)
            recorded_hash = event.pop("event_hash")
            if event.get("previous_hash") != previous:
                raise ValueError(f"{line_number}번째 기록의 이전 해시가 맞지 않습니다.")
            calculated = _hash(event)
            if recorded_hash != calculated:
                raise ValueError(f"{line_number}번째 기록의 해시가 맞지 않습니다.")
            previous = recorded_hash
            count += 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {"valid": False, "event_count": count, "last_hash": previous, "error": str(error)}
    return {"valid": True, "event_count": count, "last_hash": previous, "error": None}


def append_future_event(
    event_type: str, payload: dict[str, Any], directory: str | Path | None = None
) -> dict[str, Any]:
    """Append one event under an interprocess lock, refusing a damaged ledger."""
    path = _ledger_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_lock(path, scope="future-ledger"):
        audit = audit_future_ledger(directory)
        if not audit["valid"]:
            raise ValueError(f"미래 검증 이력 무결성 오류: {audit['error']}")
        event = {
            "ledger_version": LEDGER_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "recorded_at": datetime.now(UTC).isoformat(),
            "previous_hash": audit["last_hash"],
            "payload": payload,
        }
        event_hash = _hash(event)
        encoded = json.dumps(
            {**event, "event_hash": event_hash}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        with path.open("ab") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        return {**event, "event_hash": event_hash}


def _freeze_directory(directory: str | Path | None = None) -> Path:
    path = project_path(directory or "predictions/future_validation/freezes")
    path.mkdir(parents=True, exist_ok=True)
    return path


def freeze_formula(
    experiment: dict[str, Any],
    model_type: str,
    candidate_rank: int,
    *,
    latest_confirmed_round: int,
    dataset_sha256: str,
    code_version: str,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Save a new, content-addressed frozen snapshot without changing its experiment."""
    if (
        experiment.get("experiment_type") != "intermittent"
        or experiment.get("status") != "completed"
    ):
        raise ValueError("완료된 GP 수식 실험만 동결할 수 있습니다.")
    if model_type not in {"one_number", "six_number_set"}:
        raise ValueError("지원하지 않는 수식 모델입니다.")
    key = "one_number_model" if model_type == "one_number" else "six_number_set_model"
    model = experiment.get(key)
    if not isinstance(model, dict):
        raise ValueError("선택한 실험에 해당 수식 모델이 없습니다.")
    candidates = model.get("candidates") or [model.get("selected", {})]
    if candidate_rank < 1 or candidate_rank > len(candidates):
        raise ValueError("선택한 공식 후보를 찾을 수 없습니다.")
    candidate = candidates[candidate_rank - 1]
    if not candidate.get("activation_condition") or not candidate.get("formulas"):
        raise ValueError("후보에 활성화 조건 또는 수식이 없습니다.")
    selected_model = {**model, "selected": candidate}
    frozen_experiment = {
        "experiment_id": experiment["experiment_id"],
        "research_id": experiment.get("research_id"),
        "experiment_type": experiment["experiment_type"],
        "status": experiment["status"],
        "engine": experiment.get("engine", experiment.get("engine_version", "unknown")),
        "engine_version": experiment.get("engine_version"),
        "config": experiment.get("config", {}),
        "splits": experiment.get("splits", {}),
        "selected_candidate_rank": candidate_rank,
        key: selected_model,
    }
    frozen_at = datetime.now(UTC).isoformat()
    payload = {
        "freeze_id": str(uuid.uuid4()),
        "frozen_at": frozen_at,
        "latest_confirmed_round": int(latest_confirmed_round),
        "data_snapshot_sha256": dataset_sha256,
        "code_version": code_version,
        "model_type": model_type,
        "candidate_rank": candidate_rank,
        "origin_experiment_id": experiment["experiment_id"],
        "origin_research_id": experiment.get("research_id"),
        "feature_profile": str(experiment.get("config", {}).get("feature_profile", "legacy")),
        "postprocessing": "prediction.create_recommendation duplicate increment with wrap 45→1",
        "frozen_experiment": frozen_experiment,
    }
    payload["formula_content_sha256"] = _hash(
        {
            "model_type": model_type,
            "feature_profile": payload["feature_profile"],
            "activation_condition": candidate["activation_condition"],
            "formulas": candidate["formulas"],
            "postprocessing": payload["postprocessing"],
        }
    )
    payload["freeze_record_sha256"] = _hash(payload)
    folder = _freeze_directory(directory)
    target = folder / f"{payload['freeze_id']}.json"
    with interprocess_lock(target, scope="formula-freeze"):
        if target.exists():
            raise FileExistsError(target)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        atomic_write_bytes(target, encoded + b"\n")
    append_future_event(
        "formula_frozen",
        {
            "freeze_id": payload["freeze_id"],
            "freeze_record_sha256": payload["freeze_record_sha256"],
            "formula_content_sha256": payload["formula_content_sha256"],
            "origin_experiment_id": payload["origin_experiment_id"],
            "origin_research_id": payload["origin_research_id"],
            "latest_confirmed_round": payload["latest_confirmed_round"],
        },
        directory,
    )
    return payload


def load_frozen_formula(
    freeze_id: str, directory: str | Path | None = None
) -> dict[str, Any]:
    path = _freeze_directory(directory) / f"{freeze_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("freeze_record_sha256", None)
    if expected != _hash(payload):
        raise ValueError("동결 공식 기록이 변경되었거나 손상되었습니다.")
    payload["freeze_record_sha256"] = expected
    formula_model_key = (
        "one_number_model" if payload["model_type"] == "one_number" else "six_number_set_model"
    )
    selected = payload["frozen_experiment"][formula_model_key]["selected"]
    actual_formula_hash = _hash(
        {
            "model_type": payload["model_type"],
            "feature_profile": payload["feature_profile"],
            "activation_condition": selected["activation_condition"],
            "formulas": selected["formulas"],
            "postprocessing": payload["postprocessing"],
        }
    )
    if actual_formula_hash != payload["formula_content_sha256"]:
        raise ValueError("동결 수식 콘텐츠 해시가 일치하지 않습니다.")
    return payload


def save_frozen_prediction(
    freeze_id: str,
    dataset: DrawDataset,
    target_round: int,
    *,
    data_file_sha256: str,
    official_latest_round: int,
    target_date: Any = None,
    random_seed: int | None = None,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Create a prospectively timed recommendation from a verified frozen snapshot."""
    _require_official_latest_draw(dataset)
    frozen = load_frozen_formula(freeze_id, directory)
    if official_latest_round < frozen["latest_confirmed_round"]:
        raise ValueError("공식 최신 회차가 수식 동결 시점보다 이전입니다.")
    if target_round != official_latest_round + 1:
        raise ValueError("예측 대상은 최신 확정 회차의 바로 다음 회차여야 합니다.")
    experiment = frozen["frozen_experiment"]
    recommendation = create_recommendation(
        experiment,
        dataset,
        target_round,
        frozen["model_type"],
        data_file_sha256=data_file_sha256,
        random_seed=random_seed,
        official_latest_round=official_latest_round,
        target_date=target_date,
    )
    if recommendation["recommendation_status"] != "pre_draw_prediction":
        raise ValueError("대상 회차가 이미 데이터에 포함되어 있어 사전 예측을 저장할 수 없습니다.")
    recommendation["freeze_id"] = freeze_id
    recommendation["freeze_record_sha256"] = frozen["freeze_record_sha256"]
    recommendation["formula_content_sha256"] = frozen["formula_content_sha256"]
    saved_path = save_recommendation(recommendation, directory or "predictions")
    append_future_event(
        "prediction_registered",
        {
            "freeze_id": freeze_id,
            "recommendation_id": recommendation["recommendation_id"],
            "recommendation_sha256": _hash(recommendation),
            "target_round": target_round,
            "created_at": recommendation["created_at"],
            "recommended_numbers": recommendation["recommended_numbers"],
            "source_file_sha256": data_file_sha256,
            "official_latest_round": official_latest_round,
            "recommendation_status": recommendation["recommendation_status"],
            "prediction_path": str(saved_path),
        },
        directory,
    )
    return recommendation


def record_future_comparison(
    recommendation: dict[str, Any],
    dataset: DrawDataset,
    *,
    directory: str | Path | None = None,
) -> dict[str, Any] | None:
    """Append a single result event only when the prediction predates result registration."""
    if recommendation.get("recommendation_status") != "pre_draw_prediction":
        return None
    if not recommendation.get("freeze_id"):
        return None
    ledger = _ledger_path(directory)
    integrity = audit_future_ledger(directory)
    if not integrity["valid"]:
        raise ValueError(f"미래 검증 이력 무결성 오류: {integrity['error']}")
    prediction_event = None
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if (
                event.get("event_type") == "prediction_registered"
                and event.get("payload", {}).get("recommendation_id")
                == recommendation["recommendation_id"]
            ):
                prediction_event = event
                break
    if prediction_event is None:
        raise ValueError("사전 등록된 해시 체인 기록을 찾을 수 없습니다.")
    if prediction_event["payload"].get("recommendation_sha256") != _hash(recommendation):
        raise ValueError("예측 JSON과 append-only 이력의 내용이 일치하지 않습니다.")
    actual = next(
        (row for row in dataset.records if row.round_number == recommendation["target_round"]),
        None,
    )
    if actual is None:
        return None
    existing = []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if (
                event.get("event_type") == "future_result_registered"
                and event.get("payload", {}).get("recommendation_id")
                == recommendation["recommendation_id"]
            ):
                existing.append(event)
    if existing:
        return existing[0]["payload"]
    comparison = compare_recommendation(
        recommendation, actual.draw.numbers, actual_bonus=actual.draw.bonus
    )
    result = {
        **comparison,
        "freeze_id": recommendation["freeze_id"],
        "formula_content_sha256": recommendation.get("formula_content_sha256"),
        "prediction_created_at": recommendation["created_at"],
        "draw_date": actual.draw_date.isoformat() if actual.draw_date else None,
        "verification_status": actual.verification_status or dataset.verification_status,
        "result_status": {
            "official_verified": "official_verified",
            "verification_failed": "verification_failed",
        }.get(actual.verification_status or dataset.verification_status, "manual_unverified"),
    }
    append_future_event(
        "future_result_registered",
        {"recommendation_id": recommendation["recommendation_id"], **result},
        directory,
    )
    return result

"""Versioned, field-separated hashes for reproducible dataset comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .dataset import DrawDataset
from .experiment import dataset_metadata
from .features import build_examples

HASH_ALGORITHM_VERSION = "lotto-dataset-hashes-v2"
LEGACY_HASH_ALGORITHM_VERSION = "dataset_metadata-json-sha256-v1"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_hashes(dataset: DrawDataset, *, max_lag: int = 10) -> dict[str, str]:
    """Return canonical hashes with explicit fields and an algorithm version."""
    numbers = [
        {"round": row.round_number, "numbers": list(row.draw.numbers)} for row in dataset.records
    ]
    dates = [
        {"round": row.round_number, "date": row.draw_date.isoformat() if row.draw_date else None}
        for row in dataset.records
    ]
    combined = [
        {
            "round": row.round_number,
            "numbers": list(row.draw.numbers),
            "date": row.draw_date.isoformat() if row.draw_date else None,
        }
        for row in dataset.records
    ]
    with_bonus = [
        {**item, "bonus": row.draw.bonus}
        for item, row in zip(combined, dataset.records, strict=True)
    ]
    feature_rows = [
        {"round": item.round_number, "features": list(item.features)}
        for item in build_examples(dataset, max_lag)
    ]
    return {
        "algorithm_version": HASH_ALGORITHM_VERSION,
        "main_numbers_sha256": _digest(numbers),
        "main_numbers_dates_sha256": _digest(combined),
        "dates_sha256": _digest(dates),
        "bonus_inclusive_sha256": _digest(with_bonus),
        "features_legacy_lag_sha256": _digest(feature_rows),
    }


def legacy_dataset_hash(dataset: DrawDataset) -> str:
    """Reproduce the original aggregate JSON hash (numbers, date, bonus)."""
    return str(dataset_metadata(dataset)["dataset_sha256"])


def compare_datasets(
    previous: DrawDataset | None,
    current: DrawDataset,
    *,
    previous_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare actual row snapshots; metadata-only legacy hashes cannot prove equality."""
    current_hashes = dataset_hashes(current)
    changes: list[dict[str, Any]] = []
    same_rows: bool | None = None
    previous_hashes = dataset_hashes(previous) if previous is not None else None
    if previous is not None:
        old = {row.round_number: row for row in previous.records}
        new = {row.round_number: row for row in current.records}
        for round_number in sorted(old.keys() | new.keys()):
            before, after = old.get(round_number), new.get(round_number)
            if before is None or after is None:
                changes.append(
                    {
                        "round_number": round_number,
                        "field": "회차",
                        "previous": bool(before),
                        "current": bool(after),
                    }
                )
                continue
            fields = {
                "main_numbers": (list(before.draw.numbers), list(after.draw.numbers)),
                "draw_date": (
                    before.draw_date.isoformat() if before.draw_date else None,
                    after.draw_date.isoformat() if after.draw_date else None,
                ),
                "bonus": (before.draw.bonus, after.draw.bonus),
            }
            for field, (left, right) in fields.items():
                if left != right:
                    changes.append(
                        {
                            "round_number": round_number,
                            "field": field,
                            "previous": left,
                            "current": right,
                        }
                    )
        same_rows = not changes
    stored_hash = (previous_metadata or {}).get("dataset_sha256")
    return {
        "algorithm_version": HASH_ALGORITHM_VERSION,
        "previous_hashes": previous_hashes,
        "current_hashes": current_hashes,
        "previous_stored_aggregate_sha256": stored_hash,
        "previous_stored_hash_algorithm": (previous_metadata or {}).get(
            "dataset_hash_algorithm_version", LEGACY_HASH_ALGORITHM_VERSION
        )
        if stored_hash
        else None,
        "same_rows": same_rows,
        "same_aggregate_hash": bool(stored_hash and stored_hash == legacy_dataset_hash(current)),
        "comparison_possible": previous is not None,
        "comparison_note": (
            "과거 행 스냅샷이 없어 저장된 단일 집계 해시만으로 변경 필드를 역산할 수 없습니다."
            if previous is None and stored_hash
            else None
        ),
        "changes": changes,
    }


def split_hash(rounds: Iterable[int]) -> str:
    """Hash an ordered list of evaluation rounds."""
    return _digest(list(rounds))


def settings_hash(settings: dict[str, Any]) -> str:
    """Hash stable experiment settings independently of dataset content."""
    return _digest(settings)

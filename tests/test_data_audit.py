from __future__ import annotations

from lotto_pattern_lab.data_audit import (
    HASH_ALGORITHM_VERSION,
    compare_datasets,
    dataset_hashes,
    legacy_dataset_hash,
)
from lotto_pattern_lab.dataset import DrawDataset, DrawRecord
from lotto_pattern_lab.draw import Draw


def _dataset(*, bonus: int | None = None) -> DrawDataset:
    return DrawDataset(
        (
            DrawRecord(753, Draw.from_numbers((1, 2, 3, 4, 5, 6), bonus)),
            DrawRecord(754, Draw.from_numbers((2, 3, 4, 5, 6, 7), bonus)),
        ),
        "official_verified",
        "fixture",
    )


def test_versioned_hashes_split_main_and_bonus_fields() -> None:
    without_bonus = dataset_hashes(_dataset())
    with_bonus = dataset_hashes(_dataset(bonus=8))

    assert without_bonus["algorithm_version"] == HASH_ALGORITHM_VERSION
    assert without_bonus["main_numbers_sha256"] == with_bonus["main_numbers_sha256"]
    assert without_bonus["bonus_inclusive_sha256"] != with_bonus["bonus_inclusive_sha256"]


def test_legacy_hash_and_historical_metadata_limit() -> None:
    current = _dataset()
    assert len(legacy_dataset_hash(current)) == 64
    report = compare_datasets(
        None,
        current,
        previous_metadata={"dataset_sha256": "old-aggregate"},
    )

    assert report["comparison_possible"] is False
    assert report["same_rows"] is None
    assert "역산" in report["comparison_note"]

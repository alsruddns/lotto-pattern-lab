"""Historical and explicitly synthetic Lotto datasets."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from .draw import Draw


@dataclass(frozen=True, slots=True)
class DrawRecord:
    round_number: int
    draw: Draw


@dataclass(frozen=True, slots=True)
class DrawDataset:
    records: tuple[DrawRecord, ...]
    source_kind: str
    source_description: str
    source_url: str | None = None

    def __post_init__(self) -> None:
        if len(self.records) < 2:
            raise ValueError("A dataset must contain at least two draws.")
        rounds = [record.round_number for record in self.records]
        if any(number < 1 for number in rounds):
            raise ValueError("Round numbers must be positive integers.")
        if rounds != sorted(set(rounds)):
            raise ValueError("Rounds must be unique and in ascending order.")
        if self.source_kind not in {"csv_unverified", "official_verified", "synthetic"}:
            raise ValueError("Unsupported dataset source kind.")


def load_csv(path: str | Path, source_url: str | None = None) -> DrawDataset:
    """Load round,n1,...,n6 CSV; provenance is recorded but not independently authenticated."""
    records: list[DrawRecord] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"round", *(f"n{i}" for i in range(1, 7))}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV requires columns: round,n1,n2,n3,n4,n5,n6 (bonus is optional).")
        for line_number, row in enumerate(reader, start=2):
            try:
                round_number = int(row["round"])
                numbers = tuple(int(row[f"n{i}"]) for i in range(1, 7))
                bonus_value = row.get("bonus", "")
                bonus = int(bonus_value) if bonus_value else None
                records.append(DrawRecord(round_number, Draw.from_numbers(numbers, bonus)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid data on CSV line {line_number}: {error}") from error
    records.sort(key=lambda record: record.round_number)
    return DrawDataset(
        tuple(records),
        "csv_unverified",
        "User supplied CSV; source has not been independently verified.",
        source_url,
    )


def make_random_dataset(length: int, seed: int, first_round: int = 1) -> DrawDataset:
    """Generate independent six-number sets; this is synthetic control data."""
    if length < 2:
        raise ValueError("Synthetic dataset length must be at least two.")
    rng = random.Random(seed)
    records = tuple(
        DrawRecord(first_round + offset, Draw.from_numbers(rng.sample(range(1, 46), 6)))
        for offset in range(length)
    )
    return DrawDataset(records, "synthetic", f"Independent random 6/45 draws (seed={seed}).")

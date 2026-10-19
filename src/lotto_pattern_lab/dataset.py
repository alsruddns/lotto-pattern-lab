"""Historical and explicitly synthetic Lotto datasets."""

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .draw import Draw


@dataclass(frozen=True, slots=True)
class DrawRecord:
    round_number: int
    draw: Draw
    draw_date: date | None = None


@dataclass(frozen=True, slots=True)
class DrawDataset:
    records: tuple[DrawRecord, ...]
    source_kind: str
    source_description: str
    source_url: str | None = None
    verification_status: str | None = None
    verified_at: str | None = None

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
        expected_status = {
            "csv_unverified": "unverified",
            "official_verified": "official_verified",
            "synthetic": "synthetic",
        }[self.source_kind]
        if self.verification_status is None:
            object.__setattr__(self, "verification_status", expected_status)
        elif self.verification_status != expected_status:
            raise ValueError("Verification status does not match dataset source kind.")


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
                date_value = row.get("date") or row.get("draw_date") or ""
                draw_date = date.fromisoformat(date_value) if date_value else None
                records.append(
                    DrawRecord(round_number, Draw.from_numbers(numbers, bonus), draw_date)
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid data on CSV line {line_number}: {error}") from error
    records.sort(key=lambda record: record.round_number)
    manifest_path = Path(f"{path}.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if manifest.get("sha256") != digest:
            raise ValueError("CSV content does not match its provenance manifest checksum.")
        if manifest.get("source_kind") != "official_verified":
            raise ValueError("CSV provenance manifest has an unsupported source kind.")
        return DrawDataset(
            tuple(records),
            "official_verified",
            str(manifest.get("source_description", "Official source CSV.")),
            str(manifest.get("source_url", "")),
            verified_at=str(manifest.get("verified_at", "")),
        )
    return DrawDataset(
        tuple(records),
        "csv_unverified",
        "User supplied CSV; source has not been independently verified.",
        source_url,
    )


def save_official_csv(dataset: DrawDataset, path: str | Path) -> tuple[Path, Path]:
    """Save verified official records plus a checksum and provenance sidecar."""
    if dataset.source_kind != "official_verified" or not dataset.verified_at:
        raise ValueError("Only a freshly fetched official dataset can be saved as verified.")
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("round", "date", "n1", "n2", "n3", "n4", "n5", "n6"))
        for record in dataset.records:
            if record.draw_date is None:
                raise ValueError("Official verified draw is missing its draw date.")
            writer.writerow(
                (record.round_number, record.draw_date.isoformat(), *record.draw.numbers)
            )
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = Path(f"{csv_path}.manifest.json")
    manifest = {
        "source_kind": dataset.source_kind,
        "verification_status": dataset.verification_status,
        "source_description": dataset.source_description,
        "source_url": dataset.source_url,
        "verified_at": dataset.verified_at,
        "start_round": dataset.records[0].round_number,
        "end_round": dataset.records[-1].round_number,
        "start_date": dataset.records[0].draw_date.isoformat(),
        "end_date": dataset.records[-1].draw_date.isoformat(),
        "draw_count": len(dataset.records),
        "sha256": digest,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path, manifest_path


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


def filter_dataset(
    dataset: DrawDataset,
    start_date: date | None = None,
    end_date: date | None = None,
    start_round: int | None = None,
    end_round: int | None = None,
) -> DrawDataset:
    """Select a requested range without importing earlier rounds as hidden GP history."""
    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must not be after end date.")
    if start_round is not None and end_round is not None and start_round > end_round:
        raise ValueError("Start round must not be after end round.")
    selected = tuple(
        record
        for record in dataset.records
        if (start_date is None or (record.draw_date is not None and record.draw_date >= start_date))
        and (end_date is None or (record.draw_date is not None and record.draw_date <= end_date))
        and (start_round is None or record.round_number >= start_round)
        and (end_round is None or record.round_number <= end_round)
    )
    if len(selected) < 2:
        raise ValueError(
            "The selected range contains fewer than two draws with the requested filters."
        )
    return DrawDataset(
        selected,
        dataset.source_kind,
        dataset.source_description,
        dataset.source_url,
        dataset.verification_status,
        dataset.verified_at,
    )

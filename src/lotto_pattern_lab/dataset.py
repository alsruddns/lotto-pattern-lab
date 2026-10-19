"""Historical and explicitly synthetic Lotto datasets."""

import csv
import hashlib
import json
import os
import random
import tempfile
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
        dated = [record.draw_date for record in self.records if record.draw_date is not None]
        if any(left >= right for left, right in zip(dated, dated[1:], strict=False)):
            raise ValueError("Draw dates must be strictly increasing when present.")
        if self.source_kind not in {
            "csv_unverified",
            "csv_locally_validated",
            "official_verified",
            "synthetic",
        }:
            raise ValueError("Unsupported dataset source kind.")
        expected_status = {
            "csv_unverified": "unverified",
            "csv_locally_validated": "locally_validated_unreconciled",
            "official_verified": "official_verified",
            "synthetic": "synthetic",
        }[self.source_kind]
        if self.verification_status is None:
            object.__setattr__(self, "verification_status", expected_status)
        elif self.verification_status != expected_status:
            raise ValueError("Verification status does not match dataset source kind.")


def load_csv(path: str | Path, source_url: str | None = None) -> DrawDataset:
    """Load six-number CSVs, with optional bonus and date columns for compatibility."""
    records: list[DrawRecord] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing.")
        round_column = "round" if "round" in reader.fieldnames else "draw_no"
        required = {round_column, *(f"n{i}" for i in range(1, 7))}
        if not required.issubset(reader.fieldnames):
            raise ValueError("CSV requires round or draw_no plus n1,n2,n3,n4,n5,n6 columns.")
        bonus_column = next(
            (name for name in ("bonus", "bonus_no", "bnsWnNo") if name in reader.fieldnames),
            None,
        )
        for line_number, row in enumerate(reader, start=2):
            try:
                round_number = int(row[round_column])
                numbers = tuple(int(row[f"n{i}"]) for i in range(1, 7))
                bonus_value = row.get(bonus_column, "") if bonus_column else ""
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
        date_file_name = manifest.get("date_sidecar")
        if date_file_name:
            date_path = Path(path).parent / str(date_file_name)
            date_digest = hashlib.sha256(date_path.read_bytes()).hexdigest()
            if date_digest != manifest.get("date_sidecar_sha256"):
                raise ValueError("Official date sidecar does not match its provenance checksum.")
            with date_path.open(encoding="utf-8-sig", newline="") as date_file:
                date_reader = csv.DictReader(date_file)
                dates = {int(row["round"]): date.fromisoformat(row["date"]) for row in date_reader}
            if set(dates) != {record.round_number for record in records}:
                raise ValueError("Official date sidecar does not match the CSV round range.")
            records = [
                DrawRecord(record.round_number, record.draw, dates[record.round_number])
                for record in records
            ]
        return DrawDataset(
            tuple(records),
            "official_verified",
            str(manifest.get("source_description", "Official source CSV.")),
            str(manifest.get("source_url", "")),
            verified_at=str(manifest.get("verified_at", "")),
        )
    rounds = [record.round_number for record in records]
    is_complete_study_range = len(rounds) == 494 and rounds == list(range(753, 1247))
    source_kind = "csv_locally_validated" if is_complete_study_range else "csv_unverified"
    description = (
        "Local checks passed for draws 753-1246; full row-by-row official reconciliation pending."
        if is_complete_study_range
        else "User supplied CSV; source has not been independently verified."
    )
    return DrawDataset(tuple(records), source_kind, description, source_url)


def dataset_integrity_report(dataset: DrawDataset) -> dict[str, object]:
    """Report round continuity, date coverage, and current source verification state."""
    rounds = [record.round_number for record in dataset.records]
    known_rounds = set(rounds)
    missing = [number for number in range(rounds[0], rounds[-1] + 1) if number not in known_rounds]
    dated_count = sum(record.draw_date is not None for record in dataset.records)
    bonus_count = sum(record.draw.bonus is not None for record in dataset.records)
    return {
        "round_range": [rounds[0], rounds[-1]],
        "draw_count": len(rounds),
        "duplicate_rounds": [],
        "missing_rounds": missing,
        "rounds_contiguous": not missing,
        "dated_draw_count": dated_count,
        "all_draws_dated": dated_count == len(rounds),
        "bonus_draw_count": bonus_count,
        "all_draws_have_bonus": bonus_count == len(rounds),
        "source_kind": dataset.source_kind,
        "verification_status": dataset.verification_status,
    }


def save_official_csv(dataset: DrawDataset, path: str | Path) -> tuple[Path, Path]:
    """Save verified official records plus a checksum and provenance sidecar."""
    if dataset.source_kind != "official_verified" or not dataset.verified_at:
        raise ValueError("Only a freshly fetched official dataset can be saved as verified.")
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    old_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest() if csv_path.exists() else None
    old_range = None
    old_manifest_path = Path(f"{csv_path}.manifest.json")
    if old_manifest_path.exists():
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        old_range = [old_manifest.get("start_round"), old_manifest.get("end_round")]
    descriptor, temporary_csv = tempfile.mkstemp(
        prefix=f"{csv_path.stem}_", suffix=".tmp", dir=csv_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus")
            )
            for record in dataset.records:
                if record.draw_date is None:
                    raise ValueError("Official verified draw is missing its draw date.")
                writer.writerow(
                    (
                        record.round_number,
                        record.draw_date.isoformat(),
                        *record.draw.numbers,
                        record.draw.bonus,
                    )
                )
        digest = hashlib.sha256(Path(temporary_csv).read_bytes()).hexdigest()
        os.replace(temporary_csv, csv_path)
    finally:
        if os.path.exists(temporary_csv):
            os.unlink(temporary_csv)
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
    descriptor, temporary_manifest = tempfile.mkstemp(
        prefix="draw_manifest_", suffix=".tmp", dir=manifest_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_manifest, manifest_path)
    finally:
        if os.path.exists(temporary_manifest):
            os.unlink(temporary_manifest)
    history_path = Path(f"{csv_path}.history.jsonl")
    previous_history = history_path.read_text(encoding="utf-8") if history_path.exists() else ""
    history_entry = {
        "updated_at": dataset.verified_at,
        "source_url": dataset.source_url,
        "old_sha256": old_digest,
        "new_sha256": digest,
        "old_round_range": old_range,
        "new_round_range": [dataset.records[0].round_number, dataset.records[-1].round_number],
        "draw_count": len(dataset.records),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix="draw_history_", suffix=".tmp", dir=history_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(previous_history)
            output.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
        os.replace(temporary, history_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return csv_path, manifest_path


def make_random_dataset(
    length: int, seed: int, first_round: int = 1, include_bonus: bool = False
) -> DrawDataset:
    """Generate independent six-number sets; this is synthetic control data."""
    if length < 2:
        raise ValueError("Synthetic dataset length must be at least two.")
    rng = random.Random(seed)
    records_list = []
    for offset in range(length):
        numbers = rng.sample(range(1, 46), 6)
        bonus = rng.choice([number for number in range(1, 46) if number not in numbers])
        records_list.append(
            DrawRecord(
                first_round + offset,
                Draw.from_numbers(numbers, bonus if include_bonus else None),
            )
        )
    records = tuple(records_list)
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

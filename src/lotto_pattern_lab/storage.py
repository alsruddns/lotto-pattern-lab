"""Atomic experiment persistence with lossless compressed detail sidecars."""

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import project_path

DETAIL_KEYS = {
    "activation_events",
    "active_gaps",
    "rule_active_rounds",
    "rule_active_gaps",
    "state_rows",
    "online_change_points",
    "progress",
    "train_rounds",
    "validation_rounds",
    "test_rounds",
}
DETAIL_MIN_ITEMS = 32


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _atomic_bytes(path, content)


def _detail_pointer(path: tuple[str | int, ...]) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _split_details(value: Any, path: tuple[str | int, ...], details: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _split_details(item, (*path, key), details) for key, item in value.items()}
    if isinstance(value, list):
        if path and str(path[-1]) in DETAIL_KEYS and len(value) >= DETAIL_MIN_ITEMS:
            pointer = _detail_pointer(path)
            details[pointer] = value
            return {"$detail_ref": pointer, "item_count": len(value)}
        return [_split_details(item, (*path, index), details) for index, item in enumerate(value)]
    return value


def _join_details(summary: Any, details: dict[str, Any]) -> Any:
    if isinstance(summary, dict):
        reference = summary.get("$detail_ref")
        if reference is not None:
            if reference not in details:
                raise ValueError(f"Experiment detail archive is missing {reference}.")
            return details[reference]
        return {key: _join_details(value, details) for key, value in summary.items()}
    if isinstance(summary, list):
        return [_join_details(value, details) for value in summary]
    return summary


def _serialize_result(result: dict[str, Any], minimum_size: int) -> tuple[bytes, bytes | None]:
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if result.get("status") == "running" or len(raw) < minimum_size:
        return raw, None
    details: dict[str, Any] = {}
    summary = _split_details(result, (), details)
    if not details:
        return raw, None
    detail_payload = json.dumps(
        {"schema_version": 1, "experiment_id": result["experiment_id"], "details": details},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    compressed = gzip.compress(detail_payload, compresslevel=6, mtime=0)
    dataset = result.get("dataset", {})
    dataset_hash = dataset.get("dataset_sha256")
    dataset_ref = (
        dataset_hash
        or "metadata-ref:"
        + hashlib.sha256(
            json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    summary["storage"] = {
        "format": "summary+gzip-details-v1",
        "details_file": f"{result['experiment_id']}.details.json.gz",
        "details_sha256": hashlib.sha256(compressed).hexdigest(),
        "details_uncompressed_bytes": len(detail_payload),
        "detail_collection_counts": {pointer: len(value) for pointer, value in details.items()},
        "dataset_ref": dataset_ref,
        "dataset_ref_type": "content_sha256" if dataset_hash else "metadata_reference",
        "statistics": _summary_statistics(result),
    }
    summary_raw = (
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    return summary_raw, compressed


def _summary_statistics(result: dict[str, Any]) -> dict[str, int]:
    exact_matches = 0
    archived_candidates = 0
    for model_key in ("one_number_model", "six_number_set_model"):
        model = result.get(model_key, {})
        for candidate in model.get("candidates", []):
            archived_candidates += 1
            exact_matches += sum(
                int(candidate.get(key, 0) or 0)
                for key in (
                    "training_full_set_matches",
                    "validation_full_set_matches",
                    "test_full_set_matches",
                )
            )
    return {
        "candidate_count": archived_candidates,
        "full_set_match_count": exact_matches,
        "progress_event_count": int(
            result.get("progress_total_count", len(result.get("progress", [])))
        ),
        "random_control_replicates": int(
            result.get("random_control_replicates", len(result.get("random_controls", [])))
        ),
        "failed_run_count": len(result.get("failed_runs", [])),
    }


class ExperimentStore:
    """Persist compact summaries and preserve full detail in verified gzip sidecars."""

    def __init__(
        self, directory: str | Path = "experiments", *, minimum_compress_bytes: int = 1_000_000
    ) -> None:
        self.directory = project_path(directory)
        self.minimum_compress_bytes = minimum_compress_bytes
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: dict[str, Any]) -> Path:
        experiment_id = str(result["experiment_id"])
        if Path(experiment_id).name != experiment_id:
            raise ValueError("Experiment ID must be a plain file name stem.")
        path = self.directory / f"{experiment_id}.json"
        snapshot = result
        if result.get("status") == "running" and isinstance(result.get("progress"), list):
            progress = result["progress"]
            snapshot = {
                **result,
                "progress": progress[-25:],
                "progress_total_count": len(progress),
                "progress_tail_only": len(progress) > 25,
            }
        summary, compressed = _serialize_result(snapshot, self.minimum_compress_bytes)
        if compressed is not None:
            detail_path = self.directory / f"{experiment_id}.details.json.gz"
            _atomic_bytes(detail_path, compressed)
            with gzip.open(detail_path, "rb") as archive:
                archived = json.loads(archive.read().decode("utf-8"))
            if archived.get("experiment_id") != experiment_id:
                raise ValueError("Compressed experiment detail verification failed.")
        _atomic_bytes(path, summary)
        return path

    def load(self, experiment_id: str, *, expand_details: bool = True) -> dict[str, Any]:
        if Path(experiment_id).name != experiment_id:
            raise ValueError("Experiment ID must be a plain file name stem.")
        path = self.directory / f"{experiment_id}.json"
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
        storage = result.get("storage", {})
        if not expand_details or storage.get("format") != "summary+gzip-details-v1":
            return result
        detail_name = str(storage["details_file"])
        if Path(detail_name).name != detail_name:
            raise ValueError("Experiment detail archive reference must be a file name.")
        detail_path = self.directory / detail_name
        compressed = detail_path.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != storage.get("details_sha256"):
            raise ValueError(f"Experiment detail archive checksum failed: {detail_path}")
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
        if payload.get("experiment_id") != experiment_id:
            raise ValueError("Experiment detail archive ID does not match its summary.")
        expanded = _join_details(result, payload["details"])
        expanded.pop("storage", None)
        return expanded

    def list(self) -> list[dict[str, Any]]:
        results = []
        for path in sorted(
            self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
        ):
            with path.open(encoding="utf-8") as source:
                results.append(json.load(source))
        return results

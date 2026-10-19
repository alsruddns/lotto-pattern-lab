"""Atomic JSON persistence for experiment snapshots and completed results."""

import json
import os
import tempfile
from pathlib import Path


class ExperimentStore:
    def __init__(self, directory: str | Path = "experiments") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: dict[str, object]) -> Path:
        experiment_id = str(result["experiment_id"])
        path = self.directory / f"{experiment_id}.json"
        handle, temporary_name = tempfile.mkstemp(
            prefix=f"{experiment_id}_", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2)
                output.write("\n")
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path

    def load(self, experiment_id: str) -> dict[str, object]:
        if Path(experiment_id).name != experiment_id:
            raise ValueError("Experiment ID must be a plain file name stem.")
        path = self.directory / f"{experiment_id}.json"
        with path.open(encoding="utf-8") as source:
            return json.load(source)

    def list(self) -> list[dict[str, object]]:
        results = []
        for path in sorted(
            self.directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
        ):
            with path.open(encoding="utf-8") as source:
                results.append(json.load(source))
        return results

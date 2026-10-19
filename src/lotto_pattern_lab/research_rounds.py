"""Research-round numbering, compact summaries, comparisons, and report exports."""

from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import project_path


def _read_top_level_fields(path: Path, wanted: set[str]) -> dict[str, Any]:
    """Read compact metadata fields from pretty-printed experiment JSON without loading details."""
    found: dict[str, Any] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.startswith("  ") or line.startswith("    ") or ":" not in line:
                continue
            try:
                key, raw = line.strip().split(":", 1)
                name = json.loads(key)
                if name in wanted:
                    value = raw.strip().rstrip(",")
                    found[name] = json.loads(value)
            except (ValueError, json.JSONDecodeError):
                continue
            if wanted <= found.keys():
                break
    return found


def legacy_research_records(
    checkpoint_directory: str | Path, experiment_directory: str | Path
) -> list[dict[str, Any]]:
    """Collect research identities and creation times without modifying source records."""
    records: dict[str, dict[str, Any]] = {}
    checkpoint_root = project_path(checkpoint_directory)
    experiment_root = project_path(experiment_directory)
    if checkpoint_root.is_dir():
        for path in checkpoint_root.glob("research*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            research_id = data.get("research_id")
            if not research_id:
                continue
            records[str(research_id)] = {
                **data,
                "research_id": str(research_id),
                "checkpoint_path": str(path.resolve()),
                "created_at": data.get("created_at") or data.get("started_at"),
                "experiment_ids": [
                    str(item["experiment_id"])
                    for item in data.get("completed_experiments", [])
                    if item.get("experiment_id")
                ],
                "legacy_source": "checkpoint",
            }
    if experiment_root.is_dir():
        wanted = {"research_id", "experiment_id", "started_at", "finished_at", "status"}
        for path in experiment_root.glob("*.json"):
            try:
                fields = _read_top_level_fields(path, wanted)
            except OSError:
                continue
            research_id = fields.get("research_id")
            if not research_id:
                continue
            entry = records.setdefault(
                str(research_id),
                {
                    "research_id": str(research_id),
                    "created_at": fields.get("started_at") or fields.get("finished_at"),
                    "status": fields.get("status", "legacy"),
                    "experiment_ids": [],
                    "legacy_source": "experiment_json",
                },
            )
            experiment_id = fields.get("experiment_id") or path.stem
            if experiment_id not in entry["experiment_ids"]:
                entry["experiment_ids"].append(str(experiment_id))
            if not entry.get("created_at"):
                entry["created_at"] = fields.get("started_at") or fields.get("finished_at")
    return sorted(
        records.values(),
        key=lambda item: (str(item.get("created_at") or "9999"), item["research_id"]),
    )


class ResearchRoundStore:
    """Append-only round identifiers stored in the existing pattern-library SQLite file."""

    def __init__(self, database: str | Path = "data/database/patterns.sqlite") -> None:
        self.path = project_path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS research_rounds (
                    round_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    research_id TEXT NOT NULL UNIQUE,
                    created_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS research_rounds_created_idx "
                "ON research_rounds(created_at, round_number)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS research_round_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL REFERENCES research_rounds(research_id),
                    status TEXT NOT NULL DEFAULT 'completed'
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS research_round_experiments_research_idx "
                "ON research_round_experiments(research_id)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS research_round_migrations (
                    migration_key TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _sync_experiments(
        connection: sqlite3.Connection, research_id: str, experiment_ids: list[str]
    ) -> None:
        for experiment_id in experiment_ids:
            existing = connection.execute(
                "SELECT research_id FROM research_round_experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if existing and existing["research_id"] != research_id:
                raise ValueError(
                    f"Experiment {experiment_id} is already linked to research "
                    f"{existing['research_id']}, not {research_id}."
                )
            if not existing:
                connection.execute(
                    "INSERT INTO research_round_experiments(experiment_id, research_id) "
                    "VALUES (?, ?)",
                    (experiment_id, research_id),
                )

    @staticmethod
    def _insert(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
        research_id = str(record["research_id"])
        allowed = {
            "status",
            "started_at",
            "last_updated_at",
            "total_elapsed_seconds",
            "elapsed_seconds",
            "checkpoint_path",
            "dataset_sha256",
            "dataset_source_kind",
            "dataset_verification_status",
            "dataset_round_range",
            "config",
            "engines",
            "completed_runs",
            "completed_experiment_ids",
            "experiment_ids",
            "next_seed",
            "dashboard_summary",
            "failed_run_summary",
            "failed_runs",
            "discovered_exact_matches",
        }
        metadata = {key: record[key] for key in allowed if key in record}
        if "completed_experiment_ids" not in metadata:
            metadata["completed_experiment_ids"] = record.get("experiment_ids", []) or [
                item.get("experiment_id")
                for item in record.get("completed_experiments", [])
                if item.get("experiment_id")
            ]
        if isinstance(metadata.get("failed_runs"), list):
            metadata["failed_run_summary"] = [
                {key: item.get(key) for key in ("seed", "error_type", "error", "failed_at")}
                for item in metadata.pop("failed_runs")[-20:]
            ]
            metadata["failed_runs"] = len(record.get("failed_runs", []))
        elite = record.get("elite_seeding")
        if isinstance(elite, dict):
            metadata["elite_seeding"] = {
                key: value
                for key, value in elite.items()
                if key not in {"selected_patterns", "skipped"}
            }
        connection.execute(
            "INSERT INTO research_rounds(research_id, created_at, metadata_json) VALUES (?, ?, ?)",
            (
                research_id,
                record.get("created_at"),
                json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            ),
        )

    def migrate_legacy(
        self, checkpoint_directory: str | Path, experiment_directory: str | Path
    ) -> None:
        candidates = legacy_research_records(checkpoint_directory, experiment_directory)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            migrated = connection.execute(
                "SELECT 1 FROM research_round_migrations WHERE migration_key='legacy_v1'"
            ).fetchone()
            if migrated:
                return
            for record in candidates:
                exists = connection.execute(
                    "SELECT 1 FROM research_rounds WHERE research_id=?",
                    (record["research_id"],),
                ).fetchone()
                if not exists:
                    self._insert(connection, record)
                self._sync_experiments(
                    connection,
                    str(record["research_id"]),
                    list(record.get("completed_experiment_ids", record.get("experiment_ids", []))),
                )
            connection.execute(
                "INSERT INTO research_round_migrations VALUES ('legacy_v1', ?)",
                (datetime.now(UTC).isoformat(),),
            )

    def ensure_round(
        self,
        research_id: str,
        metadata: dict[str, Any],
        checkpoint_directory: str | Path,
        experiment_directory: str | Path,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            migrated = connection.execute(
                "SELECT 1 FROM research_round_migrations WHERE migration_key='legacy_v1'"
            ).fetchone()
            if not migrated:
                candidates = legacy_research_records(checkpoint_directory, experiment_directory)
                for record in candidates:
                    exists = connection.execute(
                        "SELECT 1 FROM research_rounds WHERE research_id=?",
                        (record["research_id"],),
                    ).fetchone()
                    if not exists:
                        self._insert(connection, record)
                    self._sync_experiments(
                        connection,
                        str(record["research_id"]),
                        list(
                            record.get("completed_experiment_ids", record.get("experiment_ids", []))
                        ),
                    )
                connection.execute(
                    "INSERT INTO research_round_migrations VALUES ('legacy_v1', ?)", (now,)
                )
            row = connection.execute(
                "SELECT round_number FROM research_rounds WHERE research_id=?", (research_id,)
            ).fetchone()
            if row:
                self._sync_experiments(
                    connection,
                    research_id,
                    list(metadata.get("completed_experiment_ids", [])),
                )
                return int(row["round_number"])
            self._insert(
                connection,
                {
                    **metadata,
                    "research_id": research_id,
                    "created_at": metadata.get("created_at") or now,
                },
            )
            row = connection.execute(
                "SELECT round_number FROM research_rounds WHERE research_id=?", (research_id,)
            ).fetchone()
            return int(row["round_number"])

    def update(self, research_id: str, metadata: dict[str, Any]) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM research_rounds WHERE research_id=?", (research_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Research round is not registered: {research_id}")
            merged = json.loads(row["metadata_json"])
            merged.update(metadata)
            connection.execute(
                "UPDATE research_rounds SET metadata_json=? WHERE research_id=?",
                (json.dumps(merged, ensure_ascii=False, allow_nan=False), research_id),
            )
            self._sync_experiments(
                connection,
                research_id,
                list(merged.get("completed_experiment_ids", [])),
            )

    def get(self, research_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_rounds WHERE research_id=?", (research_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM research_rounds ORDER BY round_number DESC"
            ).fetchall()
        return [self._decode(row) for row in rows]

    def experiment_ids(self, research_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id FROM research_round_experiments "
                "WHERE research_id=? ORDER BY experiment_id",
                (research_id,),
            ).fetchall()
        return [str(row["experiment_id"]) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            **json.loads(row["metadata_json"]),
            "research_id": row["research_id"],
            "research_round": int(row["round_number"]),
            "created_at": row["created_at"],
        }


def empty_dashboard_summary() -> dict[str, Any]:
    return {
        "summary_schema_version": 1,
        "completed_experiment_count": 0,
        "failed_experiment_count": 0,
        "candidate_count": 0,
        "new_pattern_count": 0,
        "duplicate_pattern_count": 0,
        "inherited_pattern_count": 0,
        "child_pattern_count": 0,
        "exact_match_count": 0,
        "model_metrics": {"one_number": [], "six_number_set": []},
        "paired_metrics": {"one_number": [], "six_number_set": []},
        "hit_totals": {},
        "experiment_summaries": [],
        "registration_metrics_available": True,
    }


def summarize_experiment(
    result: dict[str, Any],
    registered: list[dict[str, Any]] | None = None,
    exact_matches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build bounded per-experiment aggregates without retaining activation histories."""
    registration_metrics_available = (
        registered is not None or "pattern_library_candidates" in result
    )
    registered = (
        registered if registered is not None else result.get("pattern_library_candidates", [])
    )
    exact_matches = (
        exact_matches if exact_matches is not None else result.get("full_set_match_discoveries", [])
    )
    config = result.get("config", {})
    output: dict[str, Any] = {
        "experiment_id": result.get("experiment_id"),
        "seed": result.get("seed", config.get("seed")),
        "runtime_seconds": result.get("runtime_seconds"),
        "candidate_count": _progress_count(result) * int(config.get("population_size", 0)),
        "new_pattern_count": (
            sum(bool(item.get("created")) for item in registered)
            if registration_metrics_available
            else None
        ),
        "duplicate_pattern_count": (
            sum(not bool(item.get("created")) for item in registered)
            if registration_metrics_available
            else None
        ),
        "inherited_pattern_count": sum(
            int(result.get("elite_seeding", {}).get(model, {}).get("seeded_count", 0) or 0)
            for model in ("one_number_search", "six_number_set_search")
        ),
        "child_pattern_count": sum(
            bool(item.get("parent_pattern_ids")) for item in registered if item.get("created")
        ),
        "exact_match_count": len(exact_matches),
        "registration_metrics_available": registration_metrics_available,
        "models": {},
        "paired": {},
        "hits": {},
    }
    for model_key, short_key in (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
    ):
        model = result.get(model_key, {})
        selected = model.get("selected", {})
        model_summary = {
            "training_score": selected.get("training_score"),
            "validation_score": selected.get("validation_score"),
            "independent_test_score": selected.get("test_score"),
            "training_activation_count": sum(
                bool(event.get("active"))
                for event in _events(selected)
                if _event_phase(event, result) == "training"
            ),
            "validation_activation_count": sum(
                bool(event.get("active"))
                for event in _events(selected)
                if _event_phase(event, result) == "validation"
            ),
            "independent_test_activation_count": sum(
                bool(event.get("active"))
                for event in _events(selected)
                if _event_phase(event, result) == "independent_test"
            ),
            "training_hit_count": selected.get("training_hit_count"),
            "validation_hit_count": selected.get("validation_hit_count"),
            "independent_test_hit_count": selected.get("test_hit_count"),
            "activation_rate": selected.get("test_activation_rate"),
            "match_distribution": selected.get("match_distribution_by_split", {}),
            "at_least_3_hits": selected.get("at_least_3_hits_by_split", {}),
            "at_least_4_hits": selected.get("at_least_4_hits_by_split", {}),
            "at_least_5_hits": selected.get("at_least_5_hits_by_split", {}),
            "full_matches": {
                "training": selected.get("training_full_set_matches"),
                "validation": selected.get("validation_full_set_matches"),
                "independent_test": selected.get("test_full_set_matches"),
            },
        }
        output["models"][short_key] = model_summary
        for split, distribution in model_summary["match_distribution"].items():
            key = f"{short_key}:{split}"
            output["hits"][key] = {
                "3_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 3),
                "4_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 4),
                "5_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 5),
                "6_exact": int(distribution.get("6", 0) or 0),
            }
    paired = result.get("paired_unseeded_comparison") or {}
    for model_key, short_key in (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
    ):
        selected = paired.get(model_key, {}).get("selected", {})
        if selected:
            output["paired"][short_key] = {
                "validation_score": selected.get("validation_score"),
                "independent_test_score": selected.get("test_score"),
                "same_dataset_and_splits": paired.get("same_dataset_and_splits"),
            }
    output["p_values"] = result.get("random_control_empirical_p_value", {})
    output["control_replicates"] = result.get("random_control_replicates")
    output["elite_seeding"] = result.get("elite_seeding", {})
    output["top_candidates"] = _top_candidates(result)
    return output


def _events(selected: dict[str, Any]) -> list[dict[str, Any]]:
    events = selected.get("activation_events", [])
    return events if isinstance(events, list) else []


def _progress_count(result: dict[str, Any]) -> int:
    progress = result.get("progress")
    if isinstance(progress, list):
        return len(progress)
    return int(result.get("storage", {}).get("statistics", {}).get("progress_event_count", 0))


def _event_phase(event: dict[str, Any], result: dict[str, Any]) -> str | None:
    round_number = int(event.get("round_number", -1))
    splits = result.get("splits", {})
    for name, key in (
        ("training", "train_rounds"),
        ("validation", "validation_rounds"),
        ("independent_test", "test_rounds"),
    ):
        bounds = splits.get(key)
        if bounds and int(bounds[0]) <= round_number <= int(bounds[1]):
            return name
    return None


def _top_candidates(result: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    candidates = []
    for model_key, label in (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
    ):
        for rank, item in enumerate(result.get(model_key, {}).get("candidates", []), 1):
            candidates.append(
                {
                    "model": label,
                    "rank": rank,
                    "activation_condition": item.get("activation_condition"),
                    "formulas": item.get("formulas"),
                    "training_score": item.get("training_score"),
                    "validation_score": item.get("validation_score"),
                    "independent_test_score": item.get("test_score"),
                    "training_activation_rate": item.get("training_activation_rate"),
                    "validation_activation_rate": item.get("validation_activation_rate"),
                    "independent_test_activation_rate": item.get("test_activation_rate"),
                    "complexity": item.get("complexity"),
                    "validation_full_set_matches": item.get("validation_full_set_matches"),
                    "independent_test_full_set_matches": item.get("test_full_set_matches"),
                    "selection_basis": "validation score, train-validation stability, complexity",
                }
            )
    candidates.sort(
        key=lambda item: (
            item.get("validation_score") is not None,
            float(item.get("validation_score") or -1),
            -abs(float(item.get("training_score") or 0) - float(item.get("validation_score") or 0)),
            -int(item.get("complexity") or 0),
        ),
        reverse=True,
    )
    return candidates[:limit]


def rebuild_dashboard_summary(
    experiments: list[dict[str, Any]], *, failed_experiment_count: int = 0
) -> dict[str, Any]:
    summary = empty_dashboard_summary()
    summary["failed_experiment_count"] = failed_experiment_count
    all_registration_data = True
    for experiment in experiments:
        if experiment.get("status") != "completed":
            summary["failed_experiment_count"] += 1
            continue
        summary["completed_experiment_count"] += 1
        compact = summarize_experiment(experiment)
        all_registration_data = (
            all_registration_data and compact["registration_metrics_available"]
        )
        summary["candidate_count"] += compact["candidate_count"]
        if compact["new_pattern_count"] is not None:
            summary["new_pattern_count"] += compact["new_pattern_count"]
        if compact["duplicate_pattern_count"] is not None:
            summary["duplicate_pattern_count"] += compact["duplicate_pattern_count"]
        summary["inherited_pattern_count"] += compact["inherited_pattern_count"]
        if compact["registration_metrics_available"]:
            summary["child_pattern_count"] += compact["child_pattern_count"]
        summary["exact_match_count"] += compact["exact_match_count"]
        summary["model_metrics"]["one_number"].append(compact["models"]["one_number"])
        summary["model_metrics"]["six_number_set"].append(compact["models"]["six_number_set"])
        summary["experiment_summaries"].append(compact)
        for model, values in compact["paired"].items():
            summary["paired_metrics"].setdefault(model, []).append(values)
        for key, values in compact["hits"].items():
            target = summary["hit_totals"].setdefault(
                key, {"3_plus": 0, "4_plus": 0, "5_plus": 0, "6_exact": 0}
            )
            for metric, value in values.items():
                target[metric] += int(value)
    summary["registration_metrics_available"] = all_registration_data
    if not all_registration_data:
        summary["new_pattern_count"] = None
        summary["duplicate_pattern_count"] = None
        summary["child_pattern_count"] = None
    return summary


def load_round_experiments(
    record: dict[str, Any], experiment_directory: str | Path
) -> list[dict[str, Any]]:
    """Load compact experiment summaries only; detail sidecars remain on disk and unopened."""
    root = project_path(experiment_directory)
    experiment_ids = record.get("completed_experiment_ids") or record.get("experiment_ids") or []
    results = []
    for experiment_id in experiment_ids:
        if not experiment_id or Path(str(experiment_id)).name != str(experiment_id):
            continue
        path = root / f"{experiment_id}.json"
        try:
            with path.open(encoding="utf-8") as source:
                results.append(json.load(source))
        except (OSError, json.JSONDecodeError):
            continue
    return results


def collect_prediction_candidates(
    experiment_ids: list[str], prediction_directory: str | Path = "predictions"
) -> list[dict[str, Any]]:
    root = project_path(prediction_directory)
    wanted = set(experiment_ids)
    candidates = []
    if not root.is_dir():
        return candidates
    for path in root.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as source:
                item = json.load(source)
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("experiment_id") not in wanted:
            continue
        candidates.append(
            {
                "recommendation_id": item.get("recommendation_id"),
                "target_round": item.get("target_round"),
                "created_at": item.get("created_at"),
                "recommendation_status": item.get("recommendation_status"),
                "result_status": item.get("result_status"),
                "model_type": item.get("model_type"),
                "recommended_numbers": item.get("recommended_numbers"),
                "activation_condition": item.get("activation_condition"),
                "used_formulas": item.get("used_formulas"),
                "pattern_context": item.get("pattern_context"),
                "model_validation": item.get("model_validation"),
                "data_provenance": item.get("data_provenance"),
            }
        )
    return sorted(candidates, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _stats(values: list[Any]) -> dict[str, float | None]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "mean": statistics.fmean(numbers) if numbers else None,
        "median": statistics.median(numbers) if numbers else None,
        "best": max(numbers) if numbers else None,
        "count": len(numbers),
    }


def model_metric_stats(summary: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for model, experiments in summary.get("model_metrics", {}).items():
        output[model] = {
            metric: _stats([item.get(metric) for item in experiments])
            for metric in ("training_score", "validation_score", "independent_test_score")
        }
    return output


def compare_research_rounds(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_config = left.get("config", {})
    right_config = right.get("config", {})
    left_hash = left.get("dataset_sha256")
    right_hash = right.get("dataset_sha256")
    same_dataset = bool(left_hash and right_hash and left_hash == right_hash)
    comparable_fields = (
        "feature_profile",
        "train_ratio",
        "validation_ratio",
        "max_lag",
        "population_size",
        "generations",
        "crossover_probability",
        "mutation_probability",
        "max_depth",
        "workers",
        "control_runs",
        "min_activation_rate",
        "hidden_states",
    )
    same_evaluation = left.get("engines") == right.get("engines") and all(
        left_config.get(key) == right_config.get(key) for key in comparable_fields
    )
    return {
        "directly_comparable": bool(same_dataset and same_evaluation),
        "same_dataset": same_dataset,
        "same_evaluation_settings": same_evaluation,
        "differences": {
            key: {"left": left_config.get(key), "right": right_config.get(key)}
            for key in comparable_fields
            if left_config.get(key) != right_config.get(key)
        },
        "left_metrics": model_metric_stats(left.get("dashboard_summary", {})),
        "right_metrics": model_metric_stats(right.get("dashboard_summary", {})),
    }


def _display(value: Any) -> str:
    return "미수집" if value is None else str(value)


def build_chatgpt_report(round_record: dict[str, Any], *, detailed: bool = False) -> str:
    summary = round_record.get("dashboard_summary", {})
    config = round_record.get("config", {})
    metrics = model_metric_stats(summary)
    models = summary.get("model_metrics", {})
    test_ratio = None
    if config.get("train_ratio") is not None and config.get("validation_ratio") is not None:
        test_ratio = 1 - config["train_ratio"] - config["validation_ratio"]
    lines = [
        "# Lotto Pattern Lab 연구 결과",
        "",
        f"- 연구 회차: {round_record.get('research_round', '미수집')}회차",
        f"- 연구 ID: {round_record.get('research_id', '미수집')}",
        f"- 상태: {round_record.get('status', '미수집')}",
        f"- 연구 시간(초): {_display(round_record.get('total_elapsed_seconds'))}",
        "",
        "## 데이터와 설정",
        f"- 데이터 해시: {_display(round_record.get('dataset_sha256'))}",
        f"- 로또 회차 범위: {_display(round_record.get('dataset_round_range'))}",
        f"- 데이터 검증: {_display(round_record.get('dataset_verification_status'))}",
        f"- 엔진: {_display(round_record.get('engines'))}",
        f"- 특징 프로파일: {_display(config.get('feature_profile'))}",
        f"- 개체군/세대: {_display(config.get('population_size'))} / "
        f"{_display(config.get('generations'))}",
        f"- 교차/돌연변이/깊이: {_display(config.get('crossover_probability'))} / "
        f"{_display(config.get('mutation_probability'))} / {_display(config.get('max_depth'))}",
        f"- 학습/검증/독립 테스트 비율: {_display(config.get('train_ratio'))} / "
        f"{_display(config.get('validation_ratio'))} / {_display(test_ratio)}",
        f"- CPU 워커/무작위 대조 반복: {_display(config.get('workers'))} / "
        f"{_display(config.get('control_runs'))}",
        f"- Elite Seeding/계승 비율: {_display(config.get('elite_seeding_enabled'))} / "
        f"{_display(config.get('elite_seed_ratio'))}",
        f"- 계승 원본 패턴 ID: "
        f"{_display(round_record.get('elite_seeding', {}).get('source_pattern_ids'))}",
        "",
        "## 연구 집계",
        f"- 완료/실패 실험: {_display(summary.get('completed_experiment_count'))} / "
        f"{_display(summary.get('failed_experiment_count'))}",
        f"- 검사 수식 후보: {_display(summary.get('candidate_count'))}",
        f"- 신규/중복 패턴 등록: {_display(summary.get('new_pattern_count'))} / "
        f"{_display(summary.get('duplicate_pattern_count'))}",
        f"- 실제 계승/자식 패턴: {_display(summary.get('inherited_pattern_count'))} / "
        f"{_display(summary.get('child_pattern_count'))}",
        f"- 과거 데이터 완전 일치 후보: {_display(summary.get('exact_match_count'))} "
        "(회고적 적합이며 사전 예측으로 간주하지 않음)",
        "",
        "## 모델별 성능",
    ]
    for key, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
        lines.append(f"### {label}")
        for metric, name in (
            ("training_score", "학습"),
            ("validation_score", "검증"),
            ("independent_test_score", "독립 테스트"),
        ):
            values = metrics.get(key, {}).get(metric, {})
            lines.append(
                f"- {name} 점수 평균/중앙값/최고: "
                f"{_display(values.get('mean'))} / {_display(values.get('median'))} / "
                f"{_display(values.get('best'))} "
                f"(실험 수 {_display(values.get('count'))})"
            )
        model_rows = models.get(key, [])
        for split, split_label in (
            ("training", "학습"),
            ("validation", "검증"),
            ("independent_test", "독립 테스트"),
        ):
            active = [row.get(f"{split}_activation_count") for row in model_rows]
            hits = [row.get(f"{split}_hit_count") for row in model_rows]
            lines.append(
                f"- {split_label} 활성화/적중 번호 합계: "
                f"{_display(sum(x for x in active if isinstance(x, int)) if active else None)} / "
                f"{_display(sum(x for x in hits if isinstance(x, int)) if hits else None)}"
            )
    lines.extend(["", "## 무작위 대조 및 초기화 비교"])
    lines.append(f"- 대조 반복 수: {_display(round_record.get('config', {}).get('control_runs'))}")
    pvalues = [
        item.get("p_values", {}).get(model)
        for item in summary.get("experiment_summaries", [])
        for model in ("one_number", "six_number_set")
        if item.get("p_values", {}).get(model) is not None
    ]
    lines.append(f"- 경험적 p값 관측치: {_display(pvalues if pvalues else None)}")
    lines.append(
        "- Elite GP와 무작위 초기화 GP는 동일 데이터 분할 paired 비교가 저장된 실험에 한해 "
        "비교합니다."
    )
    lines.extend(["", "## 적중 현황 (선택 모델의 학습/검증/독립 테스트 집계)"])
    for key, values in summary.get("hit_totals", {}).items():
        lines.append(
            f"- {key}: 3+ {values['3_plus']}, 4+ {values['4_plus']}, "
            f"5+ {values['5_plus']}, 6개 {values['6_exact']}"
        )
    lines.extend(["", "## 상위 후보 (검증 기반 정렬)"])
    candidates = [
        candidate
        for experiment in summary.get("experiment_summaries", [])
        for candidate in experiment.get("top_candidates", [])
    ]
    candidates.sort(
        key=lambda item: (
            item.get("validation_score") is not None,
            float(item.get("validation_score") or -1),
            -int(item.get("complexity") or 0),
        ),
        reverse=True,
    )
    for index, candidate in enumerate(candidates[: 10 if detailed else 5], start=1):
        lines.append(
            f"{index}. {candidate.get('model')} rank {candidate.get('rank')} — "
            f"검증 {_display(candidate.get('validation_score'))}, 독립 테스트 "
            f"{_display(candidate.get('independent_test_score'))}, "
            f"활성률 {_display(candidate.get('validation_activation_rate'))}, "
            f"복잡도 {_display(candidate.get('complexity'))}"
        )
        lines.append(f"   - 조건: `{_display(candidate.get('activation_condition'))}`")
        lines.append(f"   - 수식: `{'; '.join(candidate.get('formulas') or [])}`")
        lines.append("   - 독립 테스트 점수는 후보 선정에 사용되지 않았습니다.")
    if not candidates:
        lines.append("- 후보 상세: 미수집")
    lines.extend(["", "## 완전 일치 후보 및 검증"])
    matches = round_record.get("discovered_exact_matches", [])
    if matches:
        for match in matches[: 50 if detailed else 10]:
            lines.append(
                f"- 회차 {match.get('round_number', '미수집')}: "
                f"패턴 `{match.get('pattern_id', '미수집')}`, "
                f"번호 `{match.get('winning_numbers', '미수집')}`, 상태 회고적 적합, "
                f"사전 예측 여부 {match.get('prediction_created_before_draw', False)}"
            )
            if match.get("activation_condition"):
                lines.append(
                    f"  - 조건 `{match['activation_condition']}` / 수식 "
                    f"`{'; '.join(match.get('formulas') or [])}`"
                )
    else:
        lines.append("- 해당 없음 또는 미수집")
    lines.extend(["", "## 다음 연구 준비"])
    predictions = round_record.get("prediction_candidates", [])
    lines.append("### 다음 회차 예측 후보")
    if predictions:
        for prediction in predictions[: 20 if detailed else 5]:
            lines.append(
                f"- 대상 로또 {prediction.get('target_round', '미수집')}회 / "
                f"상태 {prediction.get('recommendation_status', '미수집')} / "
                f"{prediction.get('model_type', '미수집')} / "
                f"번호 `{prediction.get('recommended_numbers', '미수집')}` / "
                f"패턴 {prediction.get('pattern_context', {}).get('pattern_id', '미수집')}"
            )
    else:
        lines.append("- 미수집")
    lines.append(f"- 다음 난수 시드: {_display(round_record.get('next_seed'))}")
    lines.append(
        f"- 다음 연구 비교 설정: 프로파일 {_display(config.get('feature_profile'))}, "
        f"개체군 {_display(config.get('population_size'))}, "
        f"세대 {_display(config.get('generations'))}"
    )
    failures = round_record.get("failed_run_summary") or round_record.get("failed_runs", [])
    failure_summary = [
        {
            "seed": item.get("seed"),
            "error_type": item.get("error_type"),
            "error": item.get("error"),
        }
        for item in failures
        if isinstance(item, dict)
    ]
    lines.append(f"- 오류 요약: {_display(failure_summary if failure_summary else None)}")
    lines.append(
        "- 분석 시 검증 구간과 독립 테스트를 분리하고, 회고적 완전 일치를 "
        "미래 예측 성능으로 해석하지 마세요."
    )
    return "\n".join(lines)

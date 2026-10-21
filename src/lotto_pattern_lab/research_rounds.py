"""Research-round numbering, compact summaries, comparisons, and report exports."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import project_path
from .presentation import (
    format_count,
    format_decimal,
    format_duration,
    format_percent,
    format_probability,
    pretty_formula,
)
from .sqlite_support import retry_sqlite_operation, run_with_busy_retry
from .storage import ExperimentStore

SUMMARY_SCHEMA_VERSION = 2


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
        if run_with_busy_retry(
            "check research round schema", self.path, self._schema_needs_initialization
        ):
            run_with_busy_retry(
                "research round schema initialization", self.path, self._initialize_schema
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=2000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _schema_needs_initialization(self) -> bool:
        with self._connect() as connection:
            tables = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            indexes = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
            return not {
                "research_rounds",
                "research_round_experiments",
                "research_round_migrations",
            } <= tables or not {
                "research_rounds_created_idx",
                "research_round_experiments_research_idx",
            } <= indexes

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
            "dataset_snapshot_sha256",
            "dataset_hash_algorithm",
            "research_mode",
            "research_strategy",
            "target_duration_seconds",
            "dataset_source_kind",
            "dataset_verification_status",
            "dataset_round_range",
            "config",
            "engines",
            "pysr_config",
            "completed_runs",
            "completed_experiment_ids",
            "experiment_ids",
            "next_seed",
            "dashboard_summary",
            "optimization_origin",
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
        if run_with_busy_retry(
            "check legacy research-round migration", self.path, self._legacy_migration_complete
        ):
            return
        candidates = legacy_research_records(checkpoint_directory, experiment_directory)

        def migrate() -> None:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                migrated = connection.execute(
                    "SELECT 1 FROM research_round_migrations WHERE migration_key='legacy_v1'"
                ).fetchone()
                if migrated:
                    return
                self._migrate_records(connection, candidates)
                connection.execute(
                    "INSERT INTO research_round_migrations VALUES ('legacy_v1', ?)",
                    (datetime.now(UTC).isoformat(),),
                )

        run_with_busy_retry("legacy research-round migration", self.path, migrate)

    def _legacy_migration_complete(self) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM research_round_migrations WHERE migration_key='legacy_v1'"
                ).fetchone()
                is not None
            )

    def _migrate_records(
        self, connection: sqlite3.Connection, candidates: list[dict[str, Any]]
    ) -> None:
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

    def ensure_round(
        self,
        research_id: str,
        metadata: dict[str, Any],
        checkpoint_directory: str | Path,
        experiment_directory: str | Path,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        migration_done = run_with_busy_retry(
            "check legacy research-round migration", self.path, self._legacy_migration_complete
        )
        candidates = (
            legacy_research_records(checkpoint_directory, experiment_directory)
            if not migration_done
            else []
        )

        def ensure() -> int:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                migrated = connection.execute(
                    "SELECT 1 FROM research_round_migrations WHERE migration_key='legacy_v1'"
                ).fetchone()
                if not migrated:
                    self._migrate_records(connection, candidates)
                    connection.execute(
                        "INSERT INTO research_round_migrations VALUES ('legacy_v1', ?)",
                        (now,),
                    )
                row = connection.execute(
                    "SELECT round_number FROM research_rounds WHERE research_id=?",
                    (research_id,),
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
                    "SELECT round_number FROM research_rounds WHERE research_id=?",
                    (research_id,),
                ).fetchone()
                return int(row["round_number"])

        return run_with_busy_retry("ensure research round", self.path, ensure)

    @retry_sqlite_operation("update research round metadata")
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
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
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
        "unique_hit_round_sets": {},
        "unique_hit_totals": {},
        "p_value_distributions": {"one_number": [], "six_number_set": []},
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
        "dashboard_compact": True,
        "experiment_id": result.get("experiment_id"),
        "status": result.get("status"),
        "seed": result.get("seed", config.get("seed")),
        "runtime_seconds": result.get("runtime_seconds"),
        "candidate_count": int(
            result.get("candidate_count")
            or (_progress_count(result) * int(config.get("population_size", 0)))
        ),
        "engine": result.get("engine", "intermittent_gp"),
        "engine_metrics": result.get("metrics"),
        "numeric_random_controls": result.get("numeric_random_controls"),
        "six_number_set_metrics": result.get("six_number_set_metrics"),
        "evaluation_cache": result.get("evaluation_cache"),
        "quality_classification_counts": {
            status: sum(
                candidate.get("classification") == status
                for candidate in result.get("candidates", [])
            )
            for status in ("PROMISING", "NEEDS_REVIEW", "REDUNDANT", "LOW_VALUE", "INVALID")
        },
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
        distribution = selected.get("match_distribution_by_split", {})
        events = _events(selected)
        activation_counts = {}
        for phase in ("training", "validation", "independent_test"):
            phase_events = [
                event for event in events if _event_phase(event, result) == phase
            ]
            if events:
                activation_counts[phase] = sum(bool(event.get("active")) for event in phase_events)
            else:
                # Compact experiment JSON stores split-level active match distributions while
                # activation-event arrays live in a compressed sidecar.
                activation_counts[phase] = sum(
                    int(value) for value in distribution.get(phase, {}).values()
                )
        model_summary = {
            "training_score": selected.get("training_score"),
            "validation_score": selected.get("validation_score"),
            "independent_test_score": selected.get("test_score"),
            "complexity": selected.get("complexity"),
            "training_activation_count": activation_counts["training"],
            "validation_activation_count": activation_counts["validation"],
            "independent_test_activation_count": activation_counts["independent_test"],
            "training_target_round_count": _phase_round_count(result, "train_rounds"),
            "validation_target_round_count": _phase_round_count(result, "validation_rounds"),
            "independent_test_target_round_count": _phase_round_count(result, "test_rounds"),
            "training_hit_count": selected.get("training_hit_count"),
            "validation_hit_count": selected.get("validation_hit_count"),
            "independent_test_hit_count": selected.get("test_hit_count"),
            "activation_rate": selected.get("test_activation_rate"),
            "match_distribution": distribution,
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
        output.setdefault("unique_hit_rounds", {})[short_key] = {
            phase: {
                str(threshold): sorted(
                    {
                        int(event["round_number"])
                        for event in events
                        if event.get("active")
                        and _event_phase(event, result) == phase
                        and (
                            bool(event.get("full_set_match"))
                            if threshold == 6
                            else int(event.get("hit_count") or 0) >= threshold
                        )
                    }
                )
                for threshold in (3, 4, 5, 6)
            }
            for phase in ("training", "validation", "independent_test")
        }
        for split, distribution in model_summary["match_distribution"].items():
            key = f"{short_key}:{split}"
            output["hits"][key] = {
                "3_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 3),
                "4_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 4),
                "5_plus": sum(int(v) for k, v in distribution.items() if int(k) >= 5),
                "6_exact": int(distribution.get("6", 0) or 0),
            }
    paired = result.get("paired_unseeded_comparison") or {}
    progress = result.get("progress", [])
    actual_progress_span = _progress_span(progress, "actual_")
    paired_progress_span = _progress_span(progress, "paired_unseeded_")
    actual_runtime = result.get("actual_gp_runtime_seconds")
    paired_runtime = paired.get("runtime_seconds")
    runtime_source = "measured"
    if actual_runtime is None:
        actual_runtime = actual_progress_span
        paired_runtime = paired_progress_span or paired_runtime
        runtime_source = "progress_timestamp_span_estimate"
    for model_key, short_key in (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
    ):
        selected = paired.get(model_key, {}).get("selected", {})
        if selected:
            output["paired"][short_key] = {
                "actual_validation_score": result.get(model_key, {}).get("selected", {}).get(
                    "validation_score"
                ),
                "actual_independent_test_score": result.get(model_key, {}).get(
                    "selected", {}
                ).get("test_score"),
                "validation_score": selected.get("validation_score"),
                "independent_test_score": selected.get("test_score"),
                "same_dataset_and_splits": paired.get("same_dataset_and_splits"),
                "actual_runtime_seconds": actual_runtime,
                "paired_runtime_seconds": paired_runtime,
                "runtime_source": runtime_source,
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


def _progress_span(progress: Any, prefix: str) -> float | None:
    if not isinstance(progress, list):
        return None
    timestamps = []
    for event in progress:
        if not str(event.get("run", "")).startswith(prefix):
            continue
        try:
            timestamps.append(datetime.fromisoformat(event["timestamp"]).timestamp())
        except (KeyError, TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return None
    return max(timestamps) - min(timestamps)


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


def _phase_round_count(result: dict[str, Any], split_key: str) -> int | None:
    bounds = result.get("splits", {}).get(split_key)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    return max(0, int(bounds[1]) - int(bounds[0]) + 1)


def merge_unique_hit_rounds(
    summary: dict[str, Any], additions: dict[str, Any]
) -> None:
    """Merge distinct target rounds for 3+, 4+, 5+, and exact six hits."""
    round_sets = summary.setdefault("unique_hit_round_sets", {})
    for model, phases in additions.items():
        model_target = round_sets.setdefault(model, {})
        for phase, thresholds in phases.items():
            phase_target = model_target.setdefault(phase, {})
            for threshold, rounds in thresholds.items():
                phase_target[threshold] = sorted(
                    set(phase_target.get(threshold, ())) | {int(value) for value in rounds}
                )
    summary["unique_hit_totals"] = {
        model: {
            phase: {
                f"{threshold}_plus" if int(threshold) < 6 else "6_exact": len(rounds)
                for threshold, rounds in thresholds.items()
            }
            for phase, thresholds in phases.items()
        }
        for model, phases in round_sets.items()
    }


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
        compact = (
            experiment
            if experiment.get("dashboard_compact")
            else summarize_experiment(experiment)
        )
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
        for model, phases in compact.get("unique_hit_rounds", {}).items():
            model_target = summary["unique_hit_round_sets"].setdefault(model, {})
            for phase, thresholds in phases.items():
                phase_target = model_target.setdefault(phase, {})
                for threshold, rounds in thresholds.items():
                    phase_target.setdefault(threshold, set()).update(rounds)
        for model, value in compact.get("p_values", {}).items():
            if value is not None and model in summary["p_value_distributions"]:
                summary["p_value_distributions"][model].append(value)
    summary["registration_metrics_available"] = all_registration_data
    if not all_registration_data:
        summary["new_pattern_count"] = None
        summary["duplicate_pattern_count"] = None
        summary["child_pattern_count"] = None
    summary["unique_hit_totals"] = {
        model: {
            phase: {
                    f"{threshold}_plus" if int(threshold) < 6 else "6_exact": len(rounds)
                for threshold, rounds in thresholds.items()
            }
            for phase, thresholds in phases.items()
        }
        for model, phases in summary["unique_hit_round_sets"].items()
    }
    summary["unique_hit_round_sets"] = {
        model: {
            phase: {threshold: sorted(rounds) for threshold, rounds in thresholds.items()}
            for phase, thresholds in phases.items()
        }
        for model, phases in summary["unique_hit_round_sets"].items()
    }
    return summary


def rebuild_dashboard_summary_from_store(
    research_record: dict[str, Any], experiment_directory: str | Path
) -> dict[str, Any]:
    """Rebuild compact aggregates from one expanded experiment at a time."""
    root = project_path(experiment_directory)
    experiment_ids = research_record.get("completed_experiment_ids") or []
    compact_experiments = []
    store = ExperimentStore(root)
    for experiment_id in experiment_ids:
        if not experiment_id or Path(str(experiment_id)).name != str(experiment_id):
            continue
        try:
            experiment = store.load(str(experiment_id), expand_details=True)
        except (OSError, ValueError):
            continue
        compact_experiments.append(summarize_experiment(experiment))
    failures = research_record.get("failed_runs", [])
    failed_count = (
        int(failures)
        if isinstance(failures, int)
        else len(failures)
        if isinstance(failures, list)
        else int(research_record.get("failed_run_count", 0) or 0)
    )
    return rebuild_dashboard_summary(
        compact_experiments,
        failed_experiment_count=failed_count,
    )


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


def paired_comparison_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Summarize paired seeded-versus-new-initialization comparisons by model."""
    output = {}
    for model, experiments in summary.get("paired_metrics", {}).items():
        row: dict[str, Any] = {}
        for metric, label in (
            ("validation_score", "validation"),
            ("independent_test_score", "test"),
        ):
            deltas = [
                float(item[f"actual_{metric}"]) - float(item[metric])
                for item in experiments
                if item.get(f"actual_{metric}") is not None and item.get(metric) is not None
            ]
            row[f"{label}_mean_delta"] = statistics.fmean(deltas) if deltas else None
            row[f"{label}_wins"] = sum(value > 1e-12 for value in deltas)
            row[f"{label}_losses"] = sum(value < -1e-12 for value in deltas)
            row[f"{label}_ties"] = sum(abs(value) <= 1e-12 for value in deltas)
            row[f"{label}_count"] = len(deltas)
        actual_times = [
            float(item["actual_runtime_seconds"])
            for item in experiments
            if item.get("actual_runtime_seconds") is not None
        ]
        paired_times = [
            float(item["paired_runtime_seconds"])
            for item in experiments
            if item.get("paired_runtime_seconds") is not None
        ]
        row["actual_runtime_mean"] = (
            statistics.fmean(actual_times) if actual_times else None
        )
        row["paired_runtime_mean"] = (
            statistics.fmean(paired_times) if paired_times else None
        )
        row["runtime_difference_mean"] = (
            row["paired_runtime_mean"] - row["actual_runtime_mean"]
            if actual_times and paired_times and len(actual_times) == len(paired_times)
            else None
        )
        row["runtime_source"] = next(
            (item.get("runtime_source") for item in experiments if item.get("runtime_source")),
            None,
        )
        output[model] = row
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
        "# 로또 패턴 연구 결과 보고서",
        "",
        "이 보고서는 과거 데이터에 대한 탐색 결과입니다. "
        "평가 점수는 번호 적중률과 다른 지표이며, 어떤 결과도 다음 추첨을 보장하지 않습니다.",
        "",
        f"- 연구 회차: {round_record.get('research_round', '미수집')}회차",
        f"- 연구 ID: {round_record.get('research_id', '미수집')}",
        f"- 상태: {round_record.get('status', '미수집')}",
        f"- 연구 시간: {format_duration(round_record.get('total_elapsed_seconds'))}",
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
        "- 검사 후보 수는 세대별 개체 평가 횟수이며 같은 수식의 반복 평가를 포함합니다.",
        f"- 신규/중복 패턴 등록: {_display(summary.get('new_pattern_count'))} / "
        f"{_display(summary.get('duplicate_pattern_count'))}",
        f"- 패턴 등록 시도: {_display(summary.get('registration_attempt_count'))}",
        f"- 계승 개체/자식 패턴: {_display(summary.get('inherited_pattern_count'))} / "
        f"{_display(summary.get('child_pattern_count'))}",
        f"- 신규 수정 자식/신규 전체 자식: "
        f"{_display(summary.get('new_modified_child_pattern_count'))} / "
        f"{_display(summary.get('new_child_pattern_count'))}",
        f"- 계보 연결 부모 패턴: {_display(summary.get('parent_pattern_count'))}",
        f"- 과거 데이터 완전 일치 후보: {_display(summary.get('exact_match_count'))} "
        "(회고적 적합이며 사전 예측으로 간주하지 않음)",
        "",
        "## 모델별 성능",
        "평가 점수는 학습·검증·독립 테스트 구간별 모델 점수입니다. "
        "적중 현황은 활성 예측의 맞힌 번호 수를 따로 집계합니다.",
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
                f"{format_decimal(values.get('mean'), suffix='점')} / "
                f"{format_decimal(values.get('median'), suffix='점')} / "
                f"{format_decimal(values.get('best'), suffix='점')} "
                f"(실험 수 {format_count(values.get('count'))})"
            )
        model_rows = models.get(key, [])
        for split, split_label in (
            ("training", "학습"),
            ("validation", "검증"),
            ("independent_test", "독립 테스트"),
        ):
            active = [row.get(f"{split}_activation_count") for row in model_rows]
            targets = [row.get(f"{split}_target_round_count") for row in model_rows]
            hits = [row.get(f"{split}_hit_count") for row in model_rows]
            active_count = format_count(
                sum(x for x in active if isinstance(x, int)) if active else None
            )
            target_count = format_count(
                sum(x for x in targets if isinstance(x, int)) if targets else None
            )
            hit_count = format_count(sum(x for x in hits if isinstance(x, int)) if hits else None)
            lines.append(
                f"- {split_label} 활성/전체 대상 회차 및 적중 번호 합계: "
                f"{active_count} / {target_count} / {hit_count}"
            )
    lines.extend(["", "## 무작위 대조 및 초기화 비교"])
    control_runs = config.get("control_runs")
    lines.append(f"- 대조 반복 수/실험: {_display(control_runs)}")
    for model, label in (("one_number", "한 번호"), ("six_number_set", "6개 집합")):
        values = summary.get("p_value_distributions", {}).get(model, [])
        stats = _stats(values)
        resolution = 1 / (1 + control_runs) if isinstance(control_runs, int) else None
        lines.append(
            f"- {label} 경험적 p값 평균/중앙값/범위: "
            f"{format_probability(stats['mean'])} / "
            f"{format_probability(stats['median'])} / "
            f"{_display((min(values), max(values)) if values else None)} "
            f"(계산 실험 {format_count(stats['count'])}, "
            f"반복별 해상도 {format_probability(resolution)})"
        )
    paired_summary = paired_comparison_summary(summary)
    paired_count = paired_summary.get("one_number", {}).get("validation_count")
    runtime_is_measured = all(
        paired_summary.get(model, {}).get("runtime_source") == "measured"
        for model in ("one_number", "six_number_set")
    )
    runtime_note = (
        "시간은 탐색 구간 stopwatch 측정입니다. "
        if runtime_is_measured
        else "시간은 진행 로그 첫/마지막 이벤트 간 추정치이며 초기화·마무리 비용을 제외합니다. "
    )
    lines.append("- Elite GP 대 신규 무작위 초기화 GP paired 결과:")
    for model, label in (("one_number", "한 번호"), ("six_number_set", "6개 집합")):
        stats = paired_summary.get(model, {})
        lines.append(
            f"  - {label}: 검증 평균 차 { _display(stats.get('validation_mean_delta')) }, "
            f"우세/열세/동률 { _display(stats.get('validation_wins')) }/"
            f"{ _display(stats.get('validation_losses')) }/"
            f"{ _display(stats.get('validation_ties')) }; "
            f"독립 테스트 평균 차 { _display(stats.get('test_mean_delta')) }, "
            f"우세/열세/동률 { _display(stats.get('test_wins')) }/"
            f"{ _display(stats.get('test_losses')) }/{ _display(stats.get('test_ties'))}; "
            f"진행 로그 구간 시간 평균 실제/paired {_display(stats.get('actual_runtime_mean'))}/"
            f"{_display(stats.get('paired_runtime_mean'))}초, 차 "
            f"{_display(stats.get('runtime_difference_mean'))}초"
        )
    lines.append(
        f"- 같은 데이터 분할에서 {_display(paired_count)}회 paired 비교했습니다. "
        f"난수 시드 결과는 완전히 독립이 아닙니다. {runtime_note}"
        "무작위 데이터 대조는 누적 탐색 이력을 재현하지 않아 검증에 제한이 있습니다."
    )
    lines.append(
        f"- p값은 실험당 대조 반복이 적어 분해능이 거칩니다. {_display(paired_count)}개 관측을 "
        "독립 표본처럼 평균내어 통계적 유의성으로 해석하지 마세요."
    )
    lines.extend(["", "## 적중 현황 (선택 모델의 학습/검증/독립 테스트 집계)"])
    for key, values in summary.get("hit_totals", {}).items():
        lines.append(
            f"- {key}: 3+ {format_count(values['3_plus'])}, "
            f"4+ {format_count(values['4_plus'])}, "
            f"5+ {format_count(values['5_plus'])}, 6개 {format_count(values['6_exact'])}"
        )
    lines.append("- 위 적중 횟수는 모델·실험·평가 구간별 활성 예측을 모두 합한 반복 평가 수입니다.")
    for model, phases in summary.get("unique_hit_totals", {}).items():
        for phase, counts in phases.items():
            lines.append(
                f"- 고유 회차 {model}/{phase}: 3+ {format_count(counts.get('3_plus', 0))}, "
                f"4+ {format_count(counts.get('4_plus', 0))}, "
                f"5+ {format_count(counts.get('5_plus', 0))}, "
                f"6개 {format_count(counts.get('6_exact', 0))}"
            )
    lines.append(
        "- 완전 일치는 활성 6개 집합 예측의 번호 집합이 당첨 6개 번호와 모두 같은 경우로 셉니다. "
        "3+/4+/5+/6 기준은 활성 평가 회차를 분모로 합니다. 반복 평가와 고유 로또 회차는 구분합니다."
    )
    pysr_trials = [
        item
        for item in summary.get("experiment_summaries", [])
        if item.get("engine") == "pysr_symbolic_regression"
    ]
    if pysr_trials:
        lines.extend(["", "## PySR 기호 회귀 결과"])
        lines.append(f"- PySR 실험 수: {len(pysr_trials)}")
        cache_reused = sum(
            (item.get("evaluation_cache") or {}).get("reused", 0) for item in pysr_trials
        )
        cache_computed = sum(
            (item.get("evaluation_cache") or {}).get("computed", 0) for item in pysr_trials
        )
        lines.append(
            f"- 평가 캐시 재사용 / 신규 계산: {cache_reused} / {cache_computed}"
        )
        lines.append(
            "- 캐시는 PySR 탐색 이후 동일 조건의 수식 점수 계산을 재사용합니다. "
            "PySR 내부 개체군의 적합도 계산은 생략하지 않습니다."
        )
        for trial in pysr_trials[: 20 if detailed else 5]:
            lines.append(
                f"- 실험 `{trial.get('experiment_id')}` 검사 후보 "
                f"{_display(trial.get('candidate_count'))}; 품질 분류 "
                f"{_display(trial.get('quality_classification_counts'))}"
            )
            for target, phases in (trial.get("engine_metrics") or {}).items():
                validation = phases.get("validation", {})
                test = phases.get("independent_test", {})
                lines.append(
                    f"  - {target}: 검증 MAE {_display(validation.get('mae'))}, "
                    f"독립 테스트 MAE {_display(test.get('mae'))}"
                )
                controls = trial.get("numeric_random_controls") or {}
                if target in controls:
                    control_validation = controls["validation"]
                    lines.append(
                        f"    - 검증 무작위 대조 적중률 "
                        f"{_display(control_validation.get('random_control_hit_rate'))}, "
                        f"반복 {control_validation.get('random_control_replicates')}, "
                        f"경험적 p { _display(control_validation.get('empirical_p_value')) }"
                    )
            set_metrics = trial.get("six_number_set_metrics") or {}
            for phase, phase_label in (("validation", "검증"), ("independent_test", "독립 테스트")):
                counts = set_metrics.get(phase)
                if counts:
                    lines.append(
                        f"  - 6개 집합 {phase_label}: 평균 적중 "
                        f"{_display(counts.get('mean_hits'))}, "
                        f"3+/4+/5+/6개 {counts.get('three_plus')}/"
                        f"{counts.get('four_plus')}/{counts.get('five_plus')}/"
                        f"{counts.get('six_exact')} (분모 {counts.get('target_round_count')})"
                    )
                    lines.append(
                        f"    - 무작위 대조 평균 적중 "
                        f"{_display(counts.get('random_control_mean_hits'))}, "
                        f"반복 {counts.get('random_control_replicates')}, "
                        f"경험적 p { _display(counts.get('empirical_p_value')) }"
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
        model_label = {
            "one_number": "한 번호 모델",
            "six_number_set": "6개 집합 모델",
        }.get(candidate.get("model"), str(candidate.get("model", "모델 미수집")))
        lines.append(
            f"{index}. {model_label} · 후보 {candidate.get('rank')} — "
            f"검증 {format_decimal(candidate.get('validation_score'), suffix='점')}, "
            f"독립 테스트 {format_decimal(candidate.get('independent_test_score'), suffix='점')}, "
            f"활성률 {format_percent(candidate.get('validation_activation_rate'))}, "
            f"복잡도 {format_count(candidate.get('complexity'))}"
        )
        lines.append(f"   - 조건: `{_display(candidate.get('activation_condition'))}`")
        formulae = candidate.get("formulas") or []
        readable_formulas = "; ".join(pretty_formula(item) for item in formulae)
        lines.append(f"   - 읽기 쉬운 수식: `{readable_formulas}`")
        lines.append(f"   - 원본 수식: `{'; '.join(formulae)}`")
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
    failures = round_record.get("failed_run_summary")
    if not isinstance(failures, list):
        failures = round_record.get("failed_runs", [])
    if not isinstance(failures, list):
        failures = []
    failure_summary = [
        {
            "seed": item.get("seed"),
            "error_type": item.get("error_type"),
            "error": item.get("error"),
            "worker_exit_code": item.get("worker_exit_code"),
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

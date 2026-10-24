"""Local SQLite registry for immutable discovered patterns and append-only evaluations."""

import ast
import copy
import json
import math
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import DrawDataset
from .feature_schema import (
    CODE_VERSION,
    FEATURE_SCHEMA_VERSION,
    MAX_SUPPORTED_FEATURE_SCHEMA_VERSION,
    PRIMITIVE_SET_VERSION,
    feature_mapping,
    remap_formula_features,
)
from .paths import project_path
from .sqlite_support import retry_sqlite_operation, run_with_busy_retry

_COMMUTATIVE = {"add", "mul", "min2", "max2", "bit_xor", "bit_and", "bit_or"}
PATTERN_STATUSES = {
    "new_discovery",
    "pending_validation",
    "under_validation",
    "tentative_pass",
    "refuted",
    "insufficient_evidence",
}
PATTERN_SCHEMA_VERSION = 2
_PATTERN_COLUMNS: dict[str, str] = {
    "model_type": "TEXT",
    "feature_profile": "TEXT",
    "feature_schema_version": "INTEGER",
    "primitive_set_version": "TEXT",
    "origin_research_id": "TEXT",
    "last_used_research_id": "TEXT",
    "discovered_at": "TEXT",
    "inheritance_count": "INTEGER NOT NULL DEFAULT 0",
    "child_count": "INTEGER NOT NULL DEFAULT 0",
    "formula": "TEXT",
    "activation_condition": "TEXT",
    "training_score": "REAL",
    "validation_score": "REAL",
    "independent_test_score": "REAL",
    "complexity": "INTEGER",
    "activation_rate": "REAL",
    "code_version": "TEXT",
}


def _canonical_node(node: ast.AST) -> str:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_canonical_node(argument) for argument in node.args]
        if node.func.id in _COMMUTATIVE:
            arguments.sort()
        return f"{node.func.id}({','.join(arguments)})"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return repr(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"neg({_canonical_node(node.operand)})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _canonical_node(node.operand)
    raise ValueError("Formula contains syntax outside the registered GP expression grammar.")


def canonical_formula(formula: str) -> str:
    """Normalize whitespace and commutative child ordering for duplicate detection."""
    try:
        return _canonical_node(ast.parse(formula.strip(), mode="eval").body)
    except SyntaxError as error:
        raise ValueError(f"Invalid formula syntax: {error.msg}") from error


def pattern_fingerprint(activation_condition: str, formulas: list[str] | tuple[str, ...]) -> str:
    payload = {
        "activation_condition": canonical_formula(activation_condition),
        "formulas": [canonical_formula(formula) for formula in formulas],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class PatternLibrary:
    """Store patterns once, then append validation and exact-match observations."""

    def __init__(self, path: str | Path = "data/database/patterns.sqlite") -> None:
        self.path = project_path(path)
        self._read_only = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if run_with_busy_retry(
            "check pattern library schema", self.path, self._schema_needs_migration
        ):
            run_with_busy_retry(
                "pattern library schema migration", self.path, self._initialize_schema
            )

    @classmethod
    def read_only(cls, path: str | Path = "data/database/patterns.sqlite") -> "PatternLibrary":
        """Open the existing library without creating directories or migrating schema."""
        instance = cls.__new__(cls)
        instance.path = project_path(path)
        instance._read_only = True
        return instance

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._read_only:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro", uri=True, timeout=2.0
            )
        else:
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

    def _schema_needs_migration(self) -> bool:
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
            required_tables = {
                "patterns",
                "observations",
                "pattern_events",
                "pattern_lineage",
                "pattern_schema_versions",
            }
            if not required_tables <= tables:
                return True
            if not {
                "observations_pattern_idx",
                "pattern_lineage_parent_idx",
                "pattern_lineage_child_idx",
            } <= indexes:
                return True
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(patterns)")
            }
            if not set(_PATTERN_COLUMNS) <= columns:
                return True
            version = connection.execute(
                "SELECT MAX(version) FROM pattern_schema_versions"
            ).fetchone()[0]
            if int(version or 0) > PATTERN_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Pattern database schema {version} is newer than supported "
                    f"schema {PATTERN_SCHEMA_VERSION}."
                )
            return int(version or 0) < PATTERN_SCHEMA_VERSION

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._migrate_schema(connection)

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        """Add lineage and searchable metadata while retaining every existing row."""
        connection.execute(
            """CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                round_number INTEGER NOT NULL,
                phase TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(pattern_id, round_number, phase)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS observations_pattern_idx "
            "ON observations(pattern_id, round_number)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS pattern_events (
                event_id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(patterns)").fetchall()
        }
        missing_columns = set(_PATTERN_COLUMNS) - columns
        for name, sql_type in _PATTERN_COLUMNS.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE patterns ADD COLUMN {name} {sql_type}")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS pattern_lineage (
                lineage_id TEXT PRIMARY KEY,
                parent_pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                child_pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                research_id TEXT,
                generation INTEGER NOT NULL DEFAULT 0,
                method TEXT NOT NULL,
                operators TEXT NOT NULL,
                parent_formula TEXT,
                child_formula TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(parent_pattern_id, child_pattern_id, research_id, method)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS pattern_lineage_parent_idx "
            "ON pattern_lineage(parent_pattern_id, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS pattern_lineage_child_idx "
            "ON pattern_lineage(child_pattern_id, created_at)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS pattern_schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        version_rows = connection.execute(
            "SELECT MAX(version) FROM pattern_schema_versions"
        ).fetchone()
        current_version = int(version_rows[0] or 0)
        if current_version > PATTERN_SCHEMA_VERSION:
            raise RuntimeError(
                f"Pattern database schema {current_version} is newer than supported "
                f"schema {PATTERN_SCHEMA_VERSION}."
            )
        if current_version < PATTERN_SCHEMA_VERSION or missing_columns:
            rows = connection.execute("SELECT * FROM patterns").fetchall()
            for row in rows:
                payload = json.loads(row["payload"])
                config = payload.get("experiment_config", {})
                profile = payload.get("feature_profile") or config.get("feature_profile", "legacy")
                outputs = int(payload.get("outputs_per_event", 0))
                model_type = payload.get("model_type") or (
                    "one_number" if outputs == 1 else "six_number_set" if outputs == 6 else None
                )
                performance = payload.get("performance", {})
                values = {
                    "model_type": model_type,
                    "feature_profile": profile,
                    "feature_schema_version": int(
                        payload.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
                    ),
                    "primitive_set_version": payload.get(
                        "primitive_set_version", PRIMITIVE_SET_VERSION
                    ),
                    "origin_research_id": payload.get("origin_research_id")
                    or payload.get("research_id"),
                    "last_used_research_id": payload.get("last_used_research_id"),
                    "discovered_at": payload.get("discovered_at", row["created_at"]),
                    "inheritance_count": int(payload.get("inheritance_count", 0)),
                    "child_count": int(payload.get("child_count", 0)),
                    "formula": json.dumps(payload.get("formulas", []), ensure_ascii=False),
                    "activation_condition": payload.get("activation_condition"),
                    "training_score": performance.get("training_score"),
                    "validation_score": performance.get("validation_score"),
                    "independent_test_score": performance.get("test_score"),
                    "complexity": payload.get("complexity"),
                    "activation_rate": performance.get("validation_activation_rate"),
                    "code_version": payload.get("code_version", CODE_VERSION),
                }
                payload.setdefault("model_type", model_type)
                payload.setdefault("feature_profile", profile)
                payload.setdefault("feature_schema_version", FEATURE_SCHEMA_VERSION)
                payload.setdefault("primitive_set_version", PRIMITIVE_SET_VERSION)
                payload.setdefault("origin_research_id", values["origin_research_id"])
                payload.setdefault("discovered_at", row["created_at"])
                payload.setdefault("inheritance_count", values["inheritance_count"])
                payload.setdefault("child_count", values["child_count"])
                payload.setdefault("code_version", CODE_VERSION)
                serialized_payload = json.dumps(payload, ensure_ascii=False, allow_nan=False)
                if all(row[key] == value for key, value in values.items()) and row[
                    "payload"
                ] == serialized_payload:
                    continue
                assignments = ", ".join(f"{column}=?" for column in values)
                connection.execute(
                    f"UPDATE patterns SET {assignments}, payload=? WHERE pattern_id=?",
                    (*values.values(), serialized_payload, row["pattern_id"]),
                )
        connection.execute(
            "INSERT OR IGNORE INTO pattern_schema_versions VALUES (?, ?)",
            (PATTERN_SCHEMA_VERSION, datetime.now(UTC).isoformat()),
        )

    @retry_sqlite_operation("register discovered pattern")
    def register_pattern(self, payload: dict[str, Any]) -> tuple[str, bool]:
        activation = str(payload["activation_condition"])
        formulas = tuple(str(formula) for formula in payload["formulas"])
        fingerprint = pattern_fingerprint(activation, formulas)
        initial_status = str(payload.get("status", "new_discovery"))
        if initial_status not in PATTERN_STATUSES:
            raise ValueError(f"Unknown pattern status: {initial_status}")
        now = datetime.now(UTC).isoformat()
        pattern_id = str(uuid.uuid4())
        schema = payload.get("feature_schema", {})
        performance = payload.get("performance", {})
        output_count = int(payload.get("outputs_per_event", 0))
        model_type = payload.get("model_type") or (
            "one_number" if output_count == 1 else "six_number_set" if output_count == 6 else None
        )
        stored_payload = {
            **payload,
            "pattern_id": pattern_id,
            "created_at": now,
            "discovered_at": payload.get("discovered_at", now),
            "origin_research_id": payload.get("origin_research_id") or payload.get("research_id"),
            "last_used_research_id": payload.get("last_used_research_id"),
            "inheritance_count": int(payload.get("inheritance_count", 0)),
            "child_count": int(payload.get("child_count", 0)),
            "model_type": model_type,
            "feature_profile": payload.get(
                "feature_profile", schema.get("feature_profile", "legacy")
            ),
            "feature_schema_version": int(
                payload.get(
                    "feature_schema_version",
                    schema.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
                )
            ),
            "primitive_set_version": payload.get(
                "primitive_set_version", schema.get("primitive_set_version", PRIMITIVE_SET_VERSION)
            ),
            "code_version": payload.get("code_version", schema.get("code_version", CODE_VERSION)),
            "canonical_fingerprint": fingerprint,
            "status": initial_status,
        }
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO patterns (
                       pattern_id, fingerprint, created_at, payload, status,
                       model_type, feature_profile, feature_schema_version,
                       primitive_set_version, origin_research_id, last_used_research_id,
                       discovered_at, inheritance_count, child_count, formula,
                       activation_condition, training_score, validation_score,
                       independent_test_score, complexity, activation_rate, code_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern_id,
                    fingerprint,
                    now,
                    json.dumps(stored_payload, ensure_ascii=False, allow_nan=False),
                    initial_status,
                    model_type,
                    stored_payload["feature_profile"],
                    stored_payload["feature_schema_version"],
                    stored_payload["primitive_set_version"],
                    stored_payload["origin_research_id"],
                    stored_payload["last_used_research_id"],
                    stored_payload["discovered_at"],
                    stored_payload["inheritance_count"],
                    stored_payload["child_count"],
                    json.dumps(formulas, ensure_ascii=False),
                    activation,
                    performance.get("training_score"),
                    performance.get("validation_score"),
                    performance.get("test_score"),
                    payload.get("complexity"),
                    performance.get("validation_activation_rate"),
                    stored_payload["code_version"],
                ),
            )
            if cursor.rowcount:
                return pattern_id, True
            row = connection.execute(
                "SELECT pattern_id FROM patterns WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            assert row is not None
            connection.execute(
                "INSERT INTO pattern_events VALUES (?, ?, ?, 'duplicate_discovery', ?)",
                (
                    str(uuid.uuid4()),
                    str(row["pattern_id"]),
                    now,
                    json.dumps(
                        {
                            "experiment_id": payload.get("experiment_id"),
                            "model": payload.get("model"),
                            "candidate_rank": payload.get("candidate_rank"),
                            "research_id": payload.get("research_id"),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            return str(row["pattern_id"]), False

    def select_elite_candidates(
        self,
        *,
        feature_profile: str,
        max_lag: int,
        max_candidates: int,
        max_per_origin: int = 2,
        target_model_type: str = "main",
        max_data_round: int | None = None,
        require_temporal_provenance: bool = False,
        experiment_directory: str | Path = "experiments",
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Rank eligible patterns from train/validation evidence only, preserving diversity."""
        if target_model_type not in {"main", "bonus_number"}:
            raise ValueError("Target model type must be main or bonus_number.")
        from .storage import ExperimentStore

        target_mapping = feature_mapping(feature_profile, max_lag)
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()
        experiment_cache: dict[str, dict[str, Any] | None] = {}
        scan_limit = min(500, max(100, max_candidates * 10))
        for pattern in self.list_elite_patterns(
            limit=scan_limit, target_model_type=target_model_type
        ):
            pattern_id = str(pattern["pattern_id"])
            pattern_type = pattern.get("model_type")
            if target_model_type == "main" and pattern_type == "bonus_number":
                skipped.append(
                    {
                        "pattern_id": pattern_id,
                        "reason": "bonus target is isolated from main-number Elite Seeding",
                    }
                )
                continue
            if target_model_type == "bonus_number" and pattern_type != "bonus_number":
                skipped.append(
                    {
                        "pattern_id": pattern_id,
                        "reason": "main-number patterns cannot seed the bonus-number target",
                    }
                )
                continue
            performance = pattern.get("performance", {})
            experiment_id = str(pattern.get("experiment_id", ""))
            if require_temporal_provenance or performance.get("validation_score") is None:
                if experiment_id not in experiment_cache:
                    try:
                        experiment_cache[experiment_id] = ExperimentStore(
                            experiment_directory
                        ).load(
                            experiment_id, expand_details=False
                        )
                    except (OSError, ValueError, KeyError):
                        experiment_cache[experiment_id] = None
                experiment = experiment_cache.get(experiment_id)
                if require_temporal_provenance:
                    if not experiment or experiment.get("status") != "completed":
                        skipped.append(
                            {"pattern_id": pattern_id, "reason": "source experiment is unavailable"}
                        )
                        continue
                    split_data = experiment.get("splits") or {}
                    if (
                        experiment.get("independent_test_used_for_selection") is True
                        or split_data.get("test_set_used_for_selection") is True
                    ):
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": "source independent test was used for selection",
                            }
                        )
                        continue
                    train_rounds = split_data.get("train_rounds") or []
                    validation_rounds = split_data.get("validation_rounds") or []
                    test_rounds = split_data.get("test_rounds") or []
                    if not train_rounds or not validation_rounds or not test_rounds:
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": "source time-ordered split provenance is missing",
                            }
                        )
                        continue
                    if not (
                        max(train_rounds) < min(validation_rounds)
                        and max(validation_rounds) < min(test_rounds)
                    ):
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": "source splits are not chronological",
                            }
                        )
                        continue
                    model_key = str(pattern.get("model", ""))
                    model_record = experiment.get(model_key, {})
                    candidates = model_record.get("candidates") or [
                        model_record.get("selected", {})
                    ]
                    candidate_rank = int(pattern.get("candidate_rank", 1))
                    if not 1 <= candidate_rank <= len(candidates):
                        skipped.append(
                            {"pattern_id": pattern_id, "reason": "source candidate rank is missing"}
                        )
                        continue
                    source_candidate = candidates[candidate_rank - 1]
                    outputs = int(pattern.get("outputs_per_event", 0))
                    rule_index = int(pattern.get("rule_index", 1))
                    source_formulas = source_candidate.get("formulas", [])
                    formula_start = (rule_index - 1) * outputs
                    if (
                        outputs not in {1, 6}
                        or rule_index not in {1, 2}
                        or len(source_formulas) < formula_start + outputs
                        or str(pattern.get("activation_condition"))
                        != str(source_candidate.get("activation_condition"))
                        or list(map(str, pattern.get("formulas", ())))
                        != list(map(str, source_formulas[formula_start : formula_start + outputs]))
                    ):
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": "formula does not match its recorded source candidate",
                            }
                        )
                        continue
                    pattern["performance"] = {
                        **pattern.get("performance", {}),
                        **{
                            key: source_candidate.get(key)
                            for key in (
                                "training_score",
                                "validation_score",
                                "training_hit_rate",
                                "validation_hit_rate",
                                "training_activation_rate",
                                "validation_activation_rate",
                            )
                        },
                    }
                    control_model_key = (
                        "one_number"
                        if model_key == "one_number_model"
                        else "six_number_set"
                    )
                    control_validation_scores = [
                        float(control[control_model_key]["selected"]["validation_score"])
                        for control in experiment.get("random_controls", [])
                        if control.get(control_model_key, {}).get("selected", {}).get(
                            "validation_score"
                        )
                        is not None
                    ]
                    selected_validation = pattern["performance"].get("validation_score")
                    if selected_validation is not None and control_validation_scores:
                        pattern["performance"]["random_control_validation_p_value"] = (
                            1
                            + sum(
                                score >= float(selected_validation)
                                for score in control_validation_scores
                            )
                        ) / (1 + len(control_validation_scores))
                    if max_data_round is not None and max(test_rounds) > max_data_round:
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": (
                                    "source experiment used rounds beyond the current data snapshot"
                                ),
                            }
                        )
                        continue
                    stored_range = pattern.get("input_range")
                    if (
                        stored_range
                        and max_data_round is not None
                        and int(stored_range[-1]) > max_data_round
                    ):
                        skipped.append(
                            {
                                "pattern_id": pattern_id,
                                "reason": "pattern provenance exceeds current data",
                            }
                        )
                        continue
                if experiment:
                    model_key = str(pattern.get("model", ""))
                    model = experiment.get(model_key, {})
                    candidates = model.get("candidates") or [model.get("selected", {})]
                    rank = int(pattern.get("candidate_rank", 1))
                    if 1 <= rank <= len(candidates):
                        candidate = candidates[rank - 1]
                        performance = {
                            **performance,
                            **{
                                key: candidate.get(key)
                                for key in (
                                    "training_score",
                                    "validation_score",
                                    "training_hit_rate",
                                    "validation_hit_rate",
                                    "training_activation_rate",
                                    "validation_activation_rate",
                                    "complexity",
                                    "height",
                                )
                            },
                        }
                        pattern["performance"] = performance
                        pattern["complexity"] = candidate.get(
                            "complexity", pattern.get("complexity")
                        )
                        schema = experiment.get("feature_schema", {})
                        pattern["feature_profile"] = experiment.get("config", {}).get(
                            "feature_profile", pattern.get("feature_profile", "legacy")
                        )
                        pattern["feature_schema"] = schema
                        pattern["feature_mapping"] = schema.get(
                            "feature_mapping", experiment.get("feature_variables", {})
                        )
                        pattern["feature_schema_version"] = schema.get(
                            "feature_schema_version", FEATURE_SCHEMA_VERSION
                        )
                        pattern["primitive_set_version"] = schema.get(
                            "primitive_set_version", PRIMITIVE_SET_VERSION
                        )
            reason = None
            if pattern.get("current_status") in {"refuted", "insufficient_evidence"}:
                reason = f"pattern status is {pattern.get('current_status')}"
            elif pattern.get("outputs_per_event") not in {1, 6}:
                reason = "model output type is missing or unsupported"
            elif (
                int(pattern.get("feature_schema_version", 1))
                > MAX_SUPPORTED_FEATURE_SCHEMA_VERSION
            ):
                reason = "feature schema version is newer than this application"
            elif (
                pattern.get("primitive_set_version", PRIMITIVE_SET_VERSION) != PRIMITIVE_SET_VERSION
            ):
                reason = "primitive set version is incompatible"
            source_mapping = pattern.get("feature_schema", {}).get(
                "feature_mapping"
            ) or pattern.get("feature_mapping", {})
            source_profile = str(pattern.get("feature_profile", "legacy"))
            if reason is None:
                for formula in [
                    pattern.get("activation_condition", ""),
                    *pattern.get("formulas", ()),
                ]:
                    _, incompatibility = remap_formula_features(
                        str(formula), source_mapping, target_mapping, source_profile
                    )
                    if incompatibility:
                        reason = incompatibility
                        break
            if reason:
                skipped.append({"pattern_id": pattern_id, "reason": reason})
                continue
            fingerprint = str(pattern.get("canonical_fingerprint", ""))
            if fingerprint in seen:
                skipped.append({"pattern_id": pattern_id, "reason": "duplicate canonical formula"})
                continue
            seen.add(fingerprint)
            performance = pattern.get("performance", {})
            training = performance.get("training_score", performance.get("training_hit_rate"))
            validation = performance.get("validation_score", performance.get("validation_hit_rate"))
            if training is None or validation is None:
                skipped.append(
                    {"pattern_id": pattern_id, "reason": "training/validation metrics missing"}
                )
                continue
            training_score = float(training)
            validation_score = float(validation)
            if not math.isfinite(training_score) or not math.isfinite(validation_score):
                skipped.append(
                    {"pattern_id": pattern_id, "reason": "non-finite training/validation score"}
                )
                continue
            validation_activation = float(performance.get("validation_activation_rate", 0.0) or 0.0)
            if not validation_activation:
                skipped.append(
                    {"pattern_id": pattern_id, "reason": "validation activation metric missing"}
                )
                continue
            if validation_activation < 0.05:
                skipped.append(
                    {"pattern_id": pattern_id, "reason": "too few validation activations"}
                )
                continue
            complexity = int(pattern.get("complexity", 0) or 0)
            stability_penalty = min(0.2, abs(training_score - validation_score) * 0.35)
            complexity_penalty = min(0.1, complexity * 0.0005)
            reuse_penalty = min(0.05, int(pattern.get("inheritance_count", 0)) * 0.005)
            random_p = performance.get("random_control_validation_p_value")
            random_control_penalty = (
                min(0.05, max(0.0, float(random_p)) * 0.05)
                if isinstance(random_p, (int, float)) and math.isfinite(float(random_p))
                else 0.025
            )
            score = (
                validation_score
                - stability_penalty
                - complexity_penalty
                - reuse_penalty
                - random_control_penalty
            )
            pattern["elite_selection_score"] = score
            pattern["elite_selection_policy"] = (
                "validation_stability_complexity_random_control_diversity_v2"
            )
            eligible.append((score, pattern_id, pattern))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        selected: list[dict[str, Any]] = []
        origin_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        model_cap = max(1, math.ceil(max_candidates / 2))
        for _, _, pattern in eligible:
            if len(selected) >= max_candidates:
                break
            origin = str(
                pattern.get("origin_research_id") or pattern.get("experiment_id") or "legacy"
            )
            model = str(pattern.get("model_type"))
            if origin_counts.get(origin, 0) >= max_per_origin:
                continue
            if model_counts.get(model, 0) >= model_cap:
                continue
            selected.append(pattern)
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1
        selected_ids = {str(pattern["pattern_id"]) for pattern in selected}
        for _, pattern_id, _ in eligible:
            if pattern_id not in selected_ids and len(selected) >= max_candidates:
                skipped.append(
                    {"pattern_id": pattern_id, "reason": "elite candidate limit reached"}
                )
        return selected, skipped

    def list_elite_patterns(
        self, *, limit: int, target_model_type: str = "main"
    ) -> list[dict[str, Any]]:
        """Read a bounded, score-ordered candidate window for Elite selection."""
        if limit < 1:
            return []
        if target_model_type == "bonus_number":
            model_filter = "model_type = 'bonus_number'"
        elif target_model_type == "main":
            model_filter = "(model_type IS NULL OR model_type != 'bonus_number')"
        else:
            raise ValueError("Target model type must be main or bonus_number.")
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT pattern_id, payload, status FROM patterns
                    WHERE {model_filter}
                    ORDER BY (validation_score IS NULL), validation_score DESC,
                             (training_score IS NULL), training_score DESC,
                             complexity ASC, created_at DESC
                    LIMIT ?""",
                (int(limit),),
            ).fetchall()
            bonus_rows = []
            if target_model_type == "main":
                bonus_rows = connection.execute(
                    """SELECT pattern_id, payload, status FROM patterns
                       WHERE model_type = 'bonus_number'
                       ORDER BY created_at DESC LIMIT ?""",
                    (min(int(limit), 100),),
                ).fetchall()
        return [
            {**json.loads(row["payload"]), "current_status": row["status"]}
            for row in (*rows, *bonus_rows)
        ]

    @retry_sqlite_operation("record pattern reuse")
    def record_elite_use(
        self, pattern_ids: list[str], research_id: str, dataset_sha256: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            for pattern_id in sorted(set(pattern_ids)):
                row = connection.execute(
                    "SELECT payload FROM patterns WHERE pattern_id=?", (pattern_id,)
                ).fetchone()
                if row is None:
                    continue
                payload = json.loads(row["payload"])
                payload["inheritance_count"] = int(payload.get("inheritance_count", 0)) + 1
                payload["last_used_research_id"] = research_id
                payload["last_seeded_dataset_sha256"] = dataset_sha256
                connection.execute(
                    "UPDATE patterns SET payload=?, last_used_research_id=?, inheritance_count=? "
                    "WHERE pattern_id=?",
                    (
                        json.dumps(payload, ensure_ascii=False, allow_nan=False),
                        research_id,
                        payload["inheritance_count"],
                        pattern_id,
                    ),
                )
                connection.execute(
                    "INSERT INTO pattern_events VALUES (?, ?, ?, 'elite_seeded', ?)",
                    (
                        str(uuid.uuid4()),
                        pattern_id,
                        now,
                        json.dumps(
                            {
                                "research_id": research_id,
                                "dataset_sha256": dataset_sha256,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

    @retry_sqlite_operation("record formula lineage")
    def record_lineage(
        self,
        parent_pattern_id: str,
        child_pattern_id: str,
        *,
        research_id: str | None,
        generation: int,
        method: str,
        operators: list[str],
        parent_formula: str,
        child_formula: str,
    ) -> bool:
        if parent_pattern_id == child_pattern_id:
            return False
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO pattern_lineage
                   (lineage_id, parent_pattern_id, child_pattern_id, research_id,
                    generation, method, operators, parent_formula, child_formula, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    parent_pattern_id,
                    child_pattern_id,
                    research_id,
                    max(0, generation),
                    method,
                    json.dumps(sorted(set(operators)), ensure_ascii=False),
                    parent_formula,
                    child_formula,
                    now,
                ),
            )
            if cursor.rowcount:
                parent_row = connection.execute(
                    "SELECT payload, child_count FROM patterns WHERE pattern_id=?",
                    (parent_pattern_id,),
                ).fetchone()
                if parent_row is None:
                    raise KeyError(parent_pattern_id)
                parent_payload = json.loads(parent_row["payload"])
                new_count = int(parent_row["child_count"] or 0) + 1
                parent_payload["child_count"] = new_count
                connection.execute(
                    "UPDATE patterns SET child_count=?, payload=? WHERE pattern_id=?",
                    (
                        new_count,
                        json.dumps(parent_payload, ensure_ascii=False, allow_nan=False),
                        parent_pattern_id,
                    ),
                )
            return bool(cursor.rowcount)

    def list_lineage(self, pattern_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if pattern_id is None:
                rows = connection.execute(
                    "SELECT * FROM pattern_lineage ORDER BY created_at"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM pattern_lineage
                       WHERE parent_pattern_id=? OR child_pattern_id=? ORDER BY created_at""",
                    (pattern_id, pattern_id),
                ).fetchall()
        return [{**dict(row), "operators": json.loads(row["operators"])} for row in rows]

    @retry_sqlite_operation("record pattern observation")
    def record_observation(
        self,
        pattern_id: str,
        round_number: int,
        phase: str,
        payload: dict[str, Any],
    ) -> bool:
        if phase not in {"retrospective", "validation", "independent_test", "prospective"}:
            raise ValueError(f"Unsupported pattern evaluation phase: {phase}")
        if phase == "prospective" and not payload.get("prediction_created_before_draw", False):
            raise ValueError("Prospective evidence requires a prediction recorded before the draw.")
        observation_id = str(uuid.uuid4())
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO observations
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    observation_id,
                    pattern_id,
                    round_number,
                    phase,
                    datetime.now(UTC).isoformat(),
                    json.dumps(payload, ensure_ascii=False, allow_nan=False),
                ),
            )
            return bool(cursor.rowcount)

    @retry_sqlite_operation("update pattern status")
    def set_status(self, pattern_id: str, status: str, reason: str) -> None:
        if status not in PATTERN_STATUSES:
            raise ValueError(f"Unknown pattern status: {status}")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM patterns WHERE pattern_id=?", (pattern_id,)
            ).fetchone()
            if row is None:
                raise KeyError(pattern_id)
            connection.execute(
                "UPDATE patterns SET status=? WHERE pattern_id=?", (status, pattern_id)
            )
            connection.execute(
                "INSERT INTO pattern_events VALUES (?, ?, ?, 'status_change', ?)",
                (
                    str(uuid.uuid4()),
                    pattern_id,
                    datetime.now(UTC).isoformat(),
                    json.dumps({"status": status, "reason": reason}, ensure_ascii=False),
                ),
            )

    def get_pattern(self, pattern_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, status FROM patterns WHERE pattern_id=?", (pattern_id,)
            ).fetchone()
            if row is None:
                raise KeyError(pattern_id)
            pattern = json.loads(row["payload"])
            pattern["current_status"] = row["status"]
            pattern["observations"] = [
                {
                    **dict(observation),
                    "payload": json.loads(observation["payload"]),
                }
                for observation in connection.execute(
                    "SELECT * FROM observations WHERE pattern_id=? ORDER BY observed_at",
                    (pattern_id,),
                )
            ]
            pattern["events"] = [
                {
                    **dict(event),
                    "payload": json.loads(event["payload"]),
                }
                for event in connection.execute(
                    "SELECT * FROM pattern_events WHERE pattern_id=? ORDER BY created_at",
                    (pattern_id,),
                )
            ]
            return pattern

    def get_pattern_summary(self, pattern_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, status FROM patterns WHERE pattern_id=?", (pattern_id,)
            ).fetchone()
        if row is None:
            raise KeyError(pattern_id)
        payload = json.loads(row["payload"])
        return {
            key: payload.get(key)
            for key in (
                "pattern_id",
                "origin_research_id",
                "inheritance_count",
                "feature_profile",
                "model_type",
                "discovered_at",
            )
        } | {"current_status": row["status"]}

    def list_hit_pattern_candidates(
        self, *, minimum_hits: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return a compact, deduplicated view of recorded hit evidence.

        The query only reads the pattern and observation tables.  Repeated evaluations
        remain visible as counts, while round numbers are deduplicated per formula.
        A six-number exact match requires six main-number hits and matching sorted sets;
        bonus matches never count toward this filter.
        """
        if minimum_hits not in {4, 5, 6}:
            raise ValueError("minimum_hits must be 4, 5, or 6")
        if limit < 1:
            return []
        exact_only = minimum_hits == 6
        hit_filter = (
            "COALESCE(CAST(json_extract(o.payload, '$.hit_count') AS INTEGER), 0) >= ?"
        )
        exact_filter = """
            AND json_extract(o.payload, '$.full_set_match') = 1
        """ if exact_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT p.pattern_id, p.payload, p.status,
                           o.round_number, o.phase, o.payload AS observation
                    FROM patterns p JOIN observations o USING(pattern_id)
                    WHERE {hit_filter} {exact_filter}
                    ORDER BY o.round_number DESC, o.observed_at DESC
                    LIMIT ?""",
                (minimum_hits, max(limit * 20, limit)),
            ).fetchall()
            round_by_research_id: dict[str, int] = {}
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='research_rounds'"
                )
            }
            if tables:
                round_by_research_id = {
                    str(row["research_id"]): int(row["round_number"])
                    for row in connection.execute(
                        "SELECT research_id, round_number FROM research_rounds"
                    )
                }
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            pattern_id = str(row["pattern_id"])
            item = grouped.get(pattern_id)
            observation = json.loads(row["observation"])
            exact = bool(
                observation.get("full_set_match")
                and int(observation.get("hit_count", 0)) == 6
                and sorted(observation.get("predicted_numbers", []))
                == sorted(observation.get("winning_numbers", []))
            )
            if exact_only and not exact:
                continue
            if item is None:
                payload = json.loads(row["payload"])
                source_research_id = payload.get("origin_research_id") or payload.get(
                    "research_id"
                )
                outputs = int(payload.get("outputs_per_event", 0))
                model_type = payload.get("model_type") or (
                    "six_number_set" if outputs == 6 else "one_number" if outputs == 1 else None
                )
                item = {
                    "pattern_id": pattern_id,
                    "origin_research_id": source_research_id,
                    "research_round": round_by_research_id.get(str(source_research_id))
                    or payload.get("research_round"),
                    "experiment_id": payload.get("experiment_id"),
                    "feature_profile": payload.get("feature_profile")
                    or payload.get("experiment_config", {}).get("feature_profile", "legacy"),
                    "engine": payload.get("engine", "미수집"),
                    "model_type": model_type,
                    "outputs_per_event": outputs,
                    "activation_condition": payload.get("activation_condition"),
                    "formulas": payload.get("formulas", []),
                    "candidate_rank": payload.get("candidate_rank", 1),
                    "rule_index": payload.get("rule_index", 1),
                    "experiment_config": payload.get("experiment_config", {}),
                    "feature_mapping": payload.get("feature_mapping", {}),
                    "current_status": row["status"],
                    "evidence_source": "pattern_observations",
                    "hit_count": 0,
                    "unique_hit_rounds": 0,
                    "exact_six_count": 0,
                    "phases": set(),
                    "hit_rounds": [],
                    "prediction_created_before_draw": False,
                    "replayable": bool(
                        model_type == "six_number_set"
                        and outputs == 6
                        and len(payload.get("formulas", [])) == 6
                        and payload.get("activation_condition")
                    ),
                }
                grouped[pattern_id] = item
            item["hit_count"] += 1
            item["phases"].add(str(row["phase"]))
            round_number = int(row["round_number"])
            if exact:
                item["exact_six_count"] += 1
            item["prediction_created_before_draw"] |= bool(
                observation.get("prediction_created_before_draw")
            )
            if not any(hit["round_number"] == round_number for hit in item["hit_rounds"]):
                item["unique_hit_rounds"] += 1
                item["hit_rounds"].append(
                    {
                        "round_number": round_number,
                        "hit_count": int(observation.get("hit_count", 0)),
                        "phase": str(row["phase"]),
                        "exact_six": exact,
                        "prediction_created_before_draw": bool(
                            observation.get("prediction_created_before_draw")
                        ),
                        "predicted_numbers": observation.get("predicted_numbers", []),
                        "winning_numbers": observation.get("winning_numbers", []),
                        "classification": observation.get("classification"),
                    }
                )
        results = list(grouped.values())
        for item in results:
            item["phases"] = sorted(item["phases"])
            item["hit_rounds"].sort(key=lambda hit: hit["round_number"], reverse=True)
        return sorted(
            results,
            key=lambda item: (
                item["exact_six_count"],
                item["unique_hit_rounds"],
                item["hit_count"],
                item["pattern_id"],
            ),
            reverse=True,
        )[:limit]

    def list_patterns(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pattern_id, payload, status FROM patterns ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [{**json.loads(row["payload"]), "current_status": row["status"]} for row in rows]

    def research_pattern_counts(self, research_id: str) -> dict[str, Any]:
        """Return inserted patterns, duplicate attempts, and recorded lineage children."""
        with self._connect() as connection:
            pattern_row = connection.execute(
                "SELECT COUNT(*) AS count FROM patterns WHERE origin_research_id=?",
                (research_id,),
            ).fetchone()
            child_row = connection.execute(
                "SELECT COUNT(DISTINCT child_pattern_id) AS count FROM pattern_lineage "
                "WHERE research_id=?",
                (research_id,),
            ).fetchone()
            new_child_row = connection.execute(
                "SELECT COUNT(DISTINCT l.child_pattern_id) AS count "
                "FROM pattern_lineage l JOIN patterns p ON p.pattern_id=l.child_pattern_id "
                "WHERE l.research_id=? AND p.origin_research_id=?",
                (research_id, research_id),
            ).fetchone()
            modified_child_row = connection.execute(
                "SELECT COUNT(DISTINCT child_pattern_id) AS count FROM pattern_lineage "
                "WHERE research_id=? AND method!='elite_seed'",
                (research_id,),
            ).fetchone()
            new_modified_child_row = connection.execute(
                "SELECT COUNT(DISTINCT l.child_pattern_id) AS count "
                "FROM pattern_lineage l JOIN patterns p ON p.pattern_id=l.child_pattern_id "
                "WHERE l.research_id=? AND p.origin_research_id=? AND l.method!='elite_seed'",
                (research_id, research_id),
            ).fetchone()
            parent_row = connection.execute(
                "SELECT COUNT(DISTINCT parent_pattern_id) AS count FROM pattern_lineage "
                "WHERE research_id=?",
                (research_id,),
            ).fetchone()
            lineage_rows = connection.execute(
                "SELECT method, COUNT(*) AS count FROM pattern_lineage "
                "WHERE research_id=? GROUP BY method",
                (research_id,),
            ).fetchall()
            duplicate_row = connection.execute(
                "SELECT COUNT(*) AS count FROM pattern_events "
                "WHERE event_type='duplicate_discovery' "
                "AND json_extract(payload, '$.research_id')=?",
                (research_id,),
            ).fetchone()
        new_count = int(pattern_row["count"])
        duplicate_count = int(duplicate_row["count"])
        return {
            "new_pattern_count": new_count,
            "duplicate_pattern_count": duplicate_count,
            "registration_attempt_count": new_count + duplicate_count,
            "child_pattern_count": int(child_row["count"]),
            "new_child_pattern_count": int(new_child_row["count"]),
            "modified_child_pattern_count": int(modified_child_row["count"]),
            "new_modified_child_pattern_count": int(new_modified_child_row["count"]),
            "parent_pattern_count": int(parent_row["count"]),
            "lineage_event_counts": {str(row["method"]): int(row["count"]) for row in lineage_rows},
        }

    def list_exact_matches(self) -> list[dict[str, Any]]:
        matches = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT p.payload, o.payload AS observation FROM patterns p "
                "JOIN observations o USING(pattern_id) "
                "WHERE json_extract(o.payload, '$.full_set_match')=1 "
                "ORDER BY o.observed_at DESC"
            )
            for row in rows:
                matches.append(
                    {
                        "pattern": json.loads(row["payload"]),
                        "match": json.loads(row["observation"]),
                    }
                )
        return matches

    def find_pattern(self, activation_condition: str, formulas: list[str]) -> str | None:
        fingerprint = pattern_fingerprint(activation_condition, formulas)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pattern_id FROM patterns WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            return str(row["pattern_id"]) if row else None


def register_experiment_exact_matches(
    result: dict[str, Any], library: PatternLibrary
) -> list[dict[str, Any]]:
    """Preserve historical exact fits as retrospective discoveries, never as forecasts."""
    dataset = result.get("dataset", {})
    configuration = result.get("config", {})
    discovered: list[dict[str, Any]] = []
    for model_name in ("one_number_model", "six_number_set_model"):
        model = result.get(model_name, {})
        selected = model.get("selected", {})
        candidates = model.get("candidates") or [selected]
        for candidate_rank, candidate in enumerate(candidates, start=1):
            if candidate.get("outputs_per_event") != 6:
                continue
            formulas = tuple(str(formula) for formula in candidate.get("formulas", ()))
            if not formulas:
                continue
            for event in candidate.get("activation_events", ()):
                if not event.get("full_set_match"):
                    continue
                round_number = int(event["round_number"])
                rule_index = int(event["rule_index"])
                rule_formulas = formulas[(rule_index - 1) * 6 : rule_index * 6]
                pattern_payload = {
                    "activation_condition": str(candidate["activation_condition"]),
                    "formulas": list(rule_formulas),
                    "engine": "deap_intermittent_gp",
                    "experiment_id": result["experiment_id"],
                    "model": model_name,
                    "candidate_rank": candidate_rank,
                    "discovery_mode": "retrospective_search",
                    "input_range": dataset.get("round_range"),
                    "data_verification_status": dataset.get("library_enrichment_source", {}).get(
                        "verification_status", dataset.get("verification_status")
                    ),
                    "feature_mapping": result.get("feature_variables", {}),
                    "experiment_config": configuration,
                    "complexity": candidate.get("complexity"),
                    "performance": {
                        key: candidate.get(key)
                        for key in (
                            "training_hit_rate",
                            "validation_hit_rate",
                            "test_hit_rate",
                        )
                    },
                    "status": "pending_validation",
                }
                pattern_id, was_created = library.register_pattern(pattern_payload)
                observation = {
                    "full_set_match": True,
                    "classification": "retrospective_search_fit",
                    "prediction_created_before_draw": False,
                    "predicted_numbers": sorted(event["predicted_numbers"]),
                    "winning_numbers": sorted(event["winning_numbers"]),
                    "hit_count": 6,
                    "rule_index": rule_index,
                }
                library.record_observation(pattern_id, round_number, "retrospective", observation)
                discovered.append(
                    {
                        "pattern_id": pattern_id,
                        "round_number": round_number,
                        "created": was_created,
                        "research_id": result.get("research_id"),
                        "experiment_id": result.get("experiment_id"),
                        "model": model_name,
                        "candidate_rank": candidate_rank,
                        "rule_index": rule_index,
                        "winning_numbers": sorted(event.get("winning_numbers", [])),
                        "predicted_numbers": sorted(event.get("predicted_numbers", [])),
                        "formulas": list(rule_formulas),
                        "activation_condition": str(candidate["activation_condition"]),
                        "feature_profile": configuration.get("feature_profile", "legacy"),
                        "data_round_range": dataset.get("round_range"),
                        "data_verification_status": dataset.get("verification_status"),
                        "prediction_created_before_draw": False,
                        "validation_status": "retrospective_search_fit",
                    }
                )
    return discovered


def register_experiment_candidates(
    result: dict[str, Any], library: PatternLibrary
) -> list[dict[str, Any]]:
    """Persist candidate rules from every archived model, with retrospective labeling."""
    dataset = result.get("dataset", {})
    configuration = result.get("config", {})
    feature_schema = result.get("feature_schema", {})
    registered: list[dict[str, Any]] = []
    model_types = (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
        ("bonus_number_model", "bonus_number"),
    )
    for model_name, model_type in model_types:
        model = result.get(model_name, {})
        if not model:
            continue
        selected = model.get("selected", {})
        candidates = model.get("candidates") or [selected]
        for candidate_rank, candidate in enumerate(candidates, start=1):
            outputs = int(candidate.get("outputs_per_event", 0))
            if outputs not in {1, 6}:
                continue
            formulas = tuple(str(formula) for formula in candidate.get("formulas", ()))
            for rule_index in range(2):
                rule_formulas = formulas[rule_index * outputs : (rule_index + 1) * outputs]
                if len(rule_formulas) != outputs:
                    continue
                payload = {
                    "activation_condition": str(candidate["activation_condition"]),
                    "formulas": list(rule_formulas),
                    "model_type": model_type,
                    "feature_profile": feature_schema.get(
                        "feature_profile", configuration.get("feature_profile", "legacy")
                    ),
                    "feature_schema_version": feature_schema.get(
                        "feature_schema_version", FEATURE_SCHEMA_VERSION
                    ),
                    "primitive_set_version": feature_schema.get(
                        "primitive_set_version", PRIMITIVE_SET_VERSION
                    ),
                    "feature_schema": feature_schema,
                    "engine": "deap_intermittent_gp",
                    "experiment_id": result["experiment_id"],
                    "research_id": result.get("research_id"),
                    "origin_research_id": result.get("research_id"),
                    "dataset_sha256": dataset.get("dataset_sha256"),
                    "code_version": feature_schema.get("code_version", CODE_VERSION),
                    "model": model_name,
                    "candidate_rank": candidate_rank,
                    "rule_index": rule_index + 1,
                    "outputs_per_event": outputs,
                    "discovery_mode": "retrospective_search",
                    "input_range": dataset.get("round_range"),
                    "data_verification_status": dataset.get("library_enrichment_source", {}).get(
                        "verification_status", dataset.get("verification_status")
                    ),
                    "feature_mapping": result.get("feature_variables", {}),
                    "experiment_config": configuration,
                    "complexity": candidate.get("complexity"),
                    "independent_test_scored": candidate_rank == 1,
                    "performance": {
                        key: candidate.get(key)
                        for key in (
                            "training_score",
                            "validation_score",
                            "test_score",
                            "training_hit_rate",
                            "validation_hit_rate",
                            "test_hit_rate",
                            "training_activation_rate",
                            "validation_activation_rate",
                            "test_activation_rate",
                            "training_full_set_matches",
                            "validation_full_set_matches",
                            "test_full_set_matches",
                        )
                    },
                    "second_prize_by_split": (
                        candidate.get("second_prize_by_rule_split", {}).get(
                            str(rule_index + 1), {}
                        )
                        if outputs == 6
                        else None
                    ),
                    "active_rounds": [
                        int(event["round_number"])
                        for event in candidate.get("activation_events", ())
                        if event.get("active") and event.get("rule_index") == rule_index + 1
                    ],
                    "active_gaps": candidate.get("rule_active_gaps", [[], []])[rule_index],
                    "lineage": candidate.get("lineage", {}),
                    "discovered_at": datetime.now(UTC).isoformat(),
                    "status": "pending_validation",
                }
                pattern_id, created = library.register_pattern(payload)
                lineage = candidate.get("lineage", {})
                parent_ids = lineage.get("parent_pattern_ids", ())
                parent_formulas = lineage.get(
                    "parent_formulas", lineage.get("original_formulas", ())
                )
                parent_formula = ";".join(map(str, parent_formulas))
                operators = list(lineage.get("operators", ()))
                method = "+".join(operators) if operators else "elite_seed"
                for parent_pattern_id in parent_ids:
                    try:
                        library.record_lineage(
                            str(parent_pattern_id),
                            pattern_id,
                            research_id=str(result.get("research_id") or "") or None,
                            generation=int(lineage.get("generation", 0)),
                            method=method,
                            operators=operators,
                            parent_formula=parent_formula,
                            child_formula=";".join(rule_formulas),
                        )
                    except (KeyError, sqlite3.IntegrityError):
                        # A legacy or removed parent stays in its original payload and event log.
                        continue
                registered.append(
                    {
                        "pattern_id": pattern_id,
                        "model": model_name,
                        "candidate_rank": candidate_rank,
                        "rule_index": rule_index + 1,
                        "created": created,
                        "parent_pattern_ids": [str(item) for item in parent_ids],
                        **library.get_pattern_summary(pattern_id),
                    }
                )
    return registered


def register_prospective_comparison(
    recommendation: dict[str, Any],
    comparison: dict[str, Any],
    library: PatternLibrary,
) -> str | None:
    """Append a draw outcome only when the saved recommendation predates its target draw."""
    target_round = int(recommendation["target_round"])
    if recommendation.get("recommendation_status") != "pre_draw_prediction":
        return None
    if int(recommendation.get("official_latest_round_at_creation") or target_round) >= target_round:
        return None
    if int(recommendation["data_provenance"]["data_used_through_round"]) >= target_round:
        return None
    if recommendation.get("model_type") != "six_number_set":
        return None
    formulas = list(recommendation.get("used_formulas", ()))
    if len(formulas) != 6:
        return None
    pattern_id = library.find_pattern(str(recommendation["activation_condition"]), formulas)
    if pattern_id is None:
        return None
    payload = {
        **comparison,
        "recommendation_id": recommendation["recommendation_id"],
        "experiment_id": recommendation["experiment_id"],
        "prediction_created_before_draw": True,
        "full_set_match": int(comparison["ai_hit_count"]) == 6,
        "classification": "prospective_fixed_pattern_validation",
    }
    library.record_observation(pattern_id, target_round, "prospective", payload)
    return pattern_id


def import_historical_experiment(
    experiment: dict[str, Any], dataset: DrawDataset, library: PatternLibrary
) -> dict[str, Any]:
    """Enrich legacy events from their recorded rounds without editing the original JSON.

    Only rounds already present in the saved experiment's event list are evaluated. Current
    data is used to recover missing winning-number fields; every imported result remains
    explicitly retrospective and carries the selected data's verification status.
    """
    hydrated = copy.deepcopy(experiment)
    actual_by_round = {
        record.round_number: sorted(record.draw.numbers) for record in dataset.records
    }
    for model_name in ("one_number_model", "six_number_set_model"):
        model = hydrated.get(model_name, {})
        selected = model.get("selected", {})
        candidates = model.get("candidates") or [selected]
        for candidate in candidates:
            outputs = int(candidate.get("outputs_per_event", 0))
            for event in candidate.get("activation_events", ()):
                actual = actual_by_round.get(int(event["round_number"]))
                if actual is None:
                    event["full_set_match"] = False
                    continue
                event.setdefault("winning_numbers", actual)
                event["full_set_match"] = bool(
                    event.get("active")
                    and outputs == 6
                    and sorted(event.get("predicted_numbers", ())) == actual
                )
    hydrated.setdefault("dataset", {})["library_enrichment_source"] = {
        "source_kind": dataset.source_kind,
        "verification_status": dataset.verification_status,
        "source_description": dataset.source_description,
        "round_range": [dataset.records[0].round_number, dataset.records[-1].round_number],
    }
    registered = register_experiment_candidates(hydrated, library)
    exact = register_experiment_exact_matches(hydrated, library)
    return {
        "experiment_id": hydrated.get("experiment_id"),
        "source_verification_status": dataset.verification_status,
        "registered_candidates": len(registered),
        "exact_matches": len(exact),
        "patterns": registered,
        "matches": exact,
    }

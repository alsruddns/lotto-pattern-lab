"""Conservative formula fingerprints, condition-scoped evaluation cache, and quality labels."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import project_path

_COMMUTATIVE = (ast.Add, ast.Mult)
_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.Pow: "pow"}
_UNARY = {ast.USub: "neg", ast.UAdd: "pos"}
_FUNCTIONS = {"square", "abs", "sqrt", "log", "exp", "sin", "cos"}


def _ast_key(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        if not (node.id.startswith("x") and node.id[1:].isdigit()):
            raise ValueError(f"Unsupported PySR variable {node.id!r}.")
        return ("var", node.id)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return ("const", repr(float(node.value)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _ast_key(node.left), _ast_key(node.right)
        if isinstance(node.op, _COMMUTATIVE):
            left, right = sorted((left, right), key=repr)
        return (_BINARY[type(node.op)], left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return (_UNARY[type(node.op)], _ast_key(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCTIONS or node.keywords or len(node.args) != 1:
            raise ValueError("Unsupported function in formula fingerprint.")
        return (node.func.id, _ast_key(node.args[0]))
    raise ValueError("Formula is outside the safe PySR expression grammar.")


def canonical_pysr_formula(formula: str) -> str:
    """Canonicalize safe commutative pairs, preserving operation order otherwise."""
    source = formula.strip().replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid PySR formula: {error.msg}") from error
    return json.dumps(_ast_key(tree.body), separators=(",", ":"), ensure_ascii=True)


def evaluation_fingerprint(formula: str, conditions: dict[str, Any]) -> str:
    """Identify an evaluation only when formula semantics and all declared conditions match."""
    payload = {
        "canonical_formula": canonical_pysr_formula(formula),
        "conditions": conditions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EvaluationCache:
    """SQLite cache with a unique condition-scoped fingerprint and append-only quality evidence."""

    def __init__(self, path: str | Path = "data/database/patterns.sqlite") -> None:
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS formula_evaluations (
                    fingerprint TEXT PRIMARY KEY,
                    canonical_formula TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_quality (
                    fingerprint TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    PRIMARY KEY (fingerprint, policy_version, dataset_sha256)
                );
                CREATE TABLE IF NOT EXISTS symbolic_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    formula TEXT NOT NULL,
                    normalized_formula TEXT NOT NULL,
                    target TEXT NOT NULL,
                    feature_profile TEXT NOT NULL,
                    feature_schema_version INTEGER NOT NULL,
                    primitive_set_version TEXT NOT NULL,
                    numeric_semantics TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS symbolic_pattern_sources (
                    source_id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL REFERENCES symbolic_patterns(pattern_id),
                    experiment_id TEXT NOT NULL,
                    research_id TEXT,
                    dataset_sha256 TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(pattern_id, experiment_id, research_id)
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, formula: str, conditions: dict[str, Any]) -> dict[str, Any] | None:
        fingerprint = evaluation_fingerprint(formula, conditions)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM formula_evaluations WHERE fingerprint=?",
                (fingerprint,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, formula: str, conditions: dict[str, Any], result: dict[str, Any]) -> bool:
        fingerprint = evaluation_fingerprint(formula, conditions)
        canonical = canonical_pysr_formula(formula)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO formula_evaluations VALUES (?, ?, ?, ?, ?)",
                (
                    fingerprint,
                    canonical,
                    json.dumps(conditions, sort_keys=True, allow_nan=False),
                    json.dumps(result, sort_keys=True, allow_nan=False),
                    now,
                ),
            )
            return cursor.rowcount == 1

    def classify(
        self,
        fingerprint: str,
        classification: str,
        reason: str,
        *,
        policy_version: str,
        dataset_sha256: str,
    ) -> None:
        allowed = {"PROMISING", "NEEDS_REVIEW", "REDUNDANT", "LOW_VALUE", "INVALID"}
        if classification not in allowed:
            raise ValueError(f"Unsupported candidate classification: {classification}")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO candidate_quality VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fingerprint,
                    classification,
                    reason,
                    policy_version,
                    dataset_sha256,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def register_symbolic_candidate(
        self,
        candidate: dict[str, Any],
        *,
        experiment_id: str,
        research_id: str | None,
        dataset_sha256: str,
        engine_version: str,
        feature_profile: str,
        feature_schema_version: int,
    ) -> str:
        formula = str(candidate["formula"])
        normalized = canonical_pysr_formula(formula)
        identity = {
            "formula": normalized,
            "target": candidate.get("target"),
            "feature_profile": feature_profile,
            "feature_schema_version": feature_schema_version,
            "primitive_set_version": "pysr_safe_float_v1",
            "numeric_semantics": "float64",
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        pattern_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO symbolic_patterns VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern_id,
                    fingerprint,
                    formula,
                    normalized,
                    str(candidate.get("target", "")),
                    feature_profile,
                    feature_schema_version,
                    "pysr_safe_float_v1",
                    "float64",
                    str(candidate.get("classification", "NEEDS_REVIEW")),
                    json.dumps(candidate.get("evaluation", {}), sort_keys=True, allow_nan=False),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT pattern_id FROM symbolic_patterns WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
            assert row is not None
            pattern_id = str(row[0])
            connection.execute(
                "INSERT OR IGNORE INTO symbolic_pattern_sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    pattern_id,
                    experiment_id,
                    research_id,
                    dataset_sha256,
                    engine_version,
                    now,
                ),
            )
        return pattern_id


def classify_candidate(
    *,
    validation_mae: float | None,
    validation_sample_count: int,
    complexity: int,
    minimum_validation_samples: int = 20,
) -> tuple[str, str]:
    if validation_mae is None:
        return "INVALID", "validation score missing or non-finite"
    if validation_sample_count < minimum_validation_samples:
        return "NEEDS_REVIEW", f"only {validation_sample_count} validation observations"
    if complexity > 24:
        return "LOW_VALUE", f"complexity {complexity} exceeds review threshold 24"
    return (
        "PROMISING",
        "finite validation score and sufficient activation evidence; review required",
    )

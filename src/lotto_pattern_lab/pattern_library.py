"""Local SQLite registry for immutable discovered patterns and append-only evaluations."""

import ast
import copy
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import DrawDataset
from .paths import project_path

_COMMUTATIVE = {"add", "mul", "min2", "max2", "bit_xor", "bit_and", "bit_or"}
PATTERN_STATUSES = {
    "new_discovery",
    "pending_validation",
    "under_validation",
    "tentative_pass",
    "refuted",
    "insufficient_evidence",
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                    round_number INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(pattern_id, round_number, phase)
                );
                CREATE INDEX IF NOT EXISTS observations_pattern_idx
                    ON observations(pattern_id, round_number);
                CREATE TABLE IF NOT EXISTS pattern_events (
                    event_id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL REFERENCES patterns(pattern_id),
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def register_pattern(self, payload: dict[str, Any]) -> tuple[str, bool]:
        activation = str(payload["activation_condition"])
        formulas = tuple(str(formula) for formula in payload["formulas"])
        fingerprint = pattern_fingerprint(activation, formulas)
        initial_status = str(payload.get("status", "new_discovery"))
        if initial_status not in PATTERN_STATUSES:
            raise ValueError(f"Unknown pattern status: {initial_status}")
        now = datetime.now(UTC).isoformat()
        pattern_id = str(uuid.uuid4())
        stored_payload = {
            **payload,
            "pattern_id": pattern_id,
            "created_at": now,
            "canonical_fingerprint": fingerprint,
            "status": initial_status,
        }
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO patterns VALUES (?, ?, ?, ?, ?)",
                (
                    pattern_id,
                    fingerprint,
                    now,
                    json.dumps(stored_payload, ensure_ascii=False, allow_nan=False),
                    initial_status,
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
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            return str(row["pattern_id"]), False

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

    def list_patterns(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pattern_id, payload, status FROM patterns ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [{**json.loads(row["payload"]), "current_status": row["status"]} for row in rows]

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
                    {"pattern_id": pattern_id, "round_number": round_number, "created": was_created}
                )
    return discovered


def register_experiment_candidates(
    result: dict[str, Any], library: PatternLibrary
) -> list[dict[str, Any]]:
    """Persist candidate rules from every archived model, with retrospective labeling."""
    dataset = result.get("dataset", {})
    configuration = result.get("config", {})
    registered: list[dict[str, Any]] = []
    for model_name in ("one_number_model", "six_number_set_model"):
        model = result.get(model_name, {})
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
                    "engine": "deap_intermittent_gp",
                    "experiment_id": result["experiment_id"],
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
                            "training_hit_rate",
                            "validation_hit_rate",
                            "test_hit_rate",
                            "training_full_set_matches",
                            "validation_full_set_matches",
                            "test_full_set_matches",
                        )
                    },
                    "active_rounds": [
                        int(event["round_number"])
                        for event in candidate.get("activation_events", ())
                        if event.get("active") and event.get("rule_index") == rule_index + 1
                    ],
                    "active_gaps": candidate.get("rule_active_gaps", [[], []])[rule_index],
                    "status": "pending_validation",
                }
                pattern_id, created = library.register_pattern(payload)
                registered.append(
                    {
                        "pattern_id": pattern_id,
                        "model": model_name,
                        "candidate_rank": candidate_rank,
                        "rule_index": rule_index + 1,
                        "created": created,
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

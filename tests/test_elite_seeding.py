from __future__ import annotations

import json
import random
import sqlite3

import pytest
from deap import gp

from lotto_pattern_lab.backtest import chronological_split
from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.feature_schema import (
    PRIMITIVE_SET_VERSION,
    feature_mapping,
    remap_formula_features,
)
from lotto_pattern_lab.features import build_examples
from lotto_pattern_lab.gp_engine import _make_primitive_set
from lotto_pattern_lab.intermittent import (
    IntermittentConfig,
    _candidate_key,
    _initial_population,
    restore_elite_individual,
    run_intermittent_search,
)
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.research import run_research_schedule


def _seed(pattern_id: str = "source-pattern", **extra: object) -> dict[str, object]:
    return {
        "pattern_id": pattern_id,
        "activation_condition": "add(x0,1)",
        "formulas": ["abs(x1)"],
        "outputs_per_event": 1,
        "rule_index": 1,
        "feature_profile": "legacy",
        "feature_mapping": feature_mapping("legacy", 10),
        "primitive_set_version": PRIMITIVE_SET_VERSION,
        **extra,
    }


def test_elite_formula_restores_and_compiles_identically() -> None:
    pset = _make_primitive_set(70)
    restored, reason = restore_elite_individual(
        _seed(), pset, 1, IntermittentConfig(population_size=8, max_depth=4)
    )
    assert reason is None
    assert restored is not None
    original_gate = gp.compile("add(x0,1)", pset)
    restored_gate = gp.compile(restored[0], pset)
    original_rule = gp.compile("abs(x1)", pset)
    restored_rule = gp.compile(restored[1], pset)
    features = (2.0, -7.0, *([0.0] * 68))
    assert restored_gate(*features) == original_gate(*features)
    assert restored_rule(*features) == original_rule(*features)


@pytest.mark.parametrize(
    ("seed", "message"),
    [
        (_seed(activation_condition="__import__('os').system('whoami')"), "invalid"),
        (_seed(primitive_set_version="future-pset"), "primitive set"),
        (_seed(outputs_per_event=6), "output count"),
    ],
)
def test_elite_formula_restore_rejects_invalid_or_incompatible_seed(seed, message) -> None:
    restored, reason = restore_elite_individual(
        seed, _make_primitive_set(70), 1, IntermittentConfig(population_size=8, max_depth=4)
    )
    assert restored is None
    assert message in reason


def test_feature_mapping_uses_stable_ids_and_rejects_calendar_only_input() -> None:
    legacy = feature_mapping("legacy", 10)
    calendar = feature_mapping("calendar_round", 10)
    remapped, reason = remap_formula_features("add(x0,x1)", legacy, calendar, "legacy")
    assert reason is None
    assert remapped == "add(x0,x1)"

    calendar_only = "x" + str(len(legacy))
    remapped, reason = remap_formula_features(calendar_only, calendar, legacy, "calendar_round")
    assert remapped is None
    assert "unavailable" in reason


def test_real_search_uses_elites_and_keeps_random_population_slots() -> None:
    examples = build_examples(make_random_dataset(40, seed=44))
    splits = chronological_split(examples, 0.6, 0.2)
    enabled = run_intermittent_search(
        splits,
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=1,
            seed=99,
            control_runs=0,
            elite_seeding_enabled=True,
            elite_seed_ratio=0.25,
        ),
        outputs_per_event=1,
        elite_seeds=(
            _seed("elite-a"),
            _seed("elite-b", activation_condition="add(x0,2)", formulas=["abs(x2)"]),
        ),
    )
    disabled = run_intermittent_search(
        splits,
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=1,
            seed=99,
            control_runs=0,
        ),
        outputs_per_event=1,
    )
    assert enabled.seeding["seeded_count"] == 2
    assert enabled.seeding["new_random_count"] == 6
    assert enabled.seeding["source_pattern_ids"] == ["elite-a", "elite-b"]
    assert disabled.seeding["seeded_count"] == 0


def test_missing_or_duplicate_elites_fill_remaining_slots_with_random(tmp_path) -> None:
    examples = build_examples(make_random_dataset(34, seed=45))
    splits = chronological_split(examples, 0.6, 0.2)
    result = run_intermittent_search(
        splits,
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=1,
            seed=101,
            control_runs=0,
            elite_seeding_enabled=True,
            elite_seed_ratio=0.5,
        ),
        outputs_per_event=1,
        elite_seeds=(_seed("duplicate-a"), _seed("duplicate-b")),
    )
    assert result.seeding["seeded_count"] == 1
    assert result.seeding["new_random_count"] == 7
    assert result.seeding["skipped"][0]["reason"] == "duplicate initial formula"


def test_seeded_run_keeps_random_slots_paired_with_unseeded_initialization() -> None:
    pset = _make_primitive_set(70)
    disabled = IntermittentConfig(population_size=8, max_depth=3, seed=108)
    enabled = IntermittentConfig(
        population_size=8,
        max_depth=3,
        seed=108,
        elite_seeding_enabled=True,
        elite_seed_ratio=0.25,
    )
    random.seed(108)
    baseline_population, _ = _initial_population(pset, 1, disabled, ())
    random.seed(108)
    seeded_population, report = _initial_population(pset, 1, enabled, (_seed(),))
    assert report["seeded_count"] == 1
    assert [
        _candidate_key(item) for item in seeded_population[1:]
    ] == [_candidate_key(item) for item in baseline_population[1:]]


def test_windows_spawn_workers_accept_elite_seed_payloads() -> None:
    examples = build_examples(make_random_dataset(34, seed=46))
    splits = chronological_split(examples, 0.6, 0.2)
    result = run_intermittent_search(
        splits,
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=2,
            seed=102,
            control_runs=0,
            elite_seeding_enabled=True,
            elite_seed_ratio=0.25,
        ),
        outputs_per_event=1,
        elite_seeds=(_seed("spawn-elite"),),
    )
    assert result.seeding["seeded_count"] == 1
    assert result.seeding["new_random_count"] == 7


def test_pattern_lineage_and_additive_sqlite_migration_preserve_old_rows(tmp_path) -> None:
    database = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE patterns (pattern_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, "
            "created_at TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL)"
        )
        payload = {
            "activation_condition": "x0",
            "formulas": ["x1"],
            "outputs_per_event": 1,
            "experiment_id": "old-exp",
            "feature_mapping": {"x0": "round_scaled", "x1": "lag1_sum"},
        }
        connection.execute(
            "INSERT INTO patterns VALUES (?, ?, ?, ?, ?)",
            (
                "old-pattern",
                "old-fingerprint",
                "2020-01-01T00:00:00+00:00",
                json.dumps(payload),
                "pending_validation",
            ),
        )

    library = PatternLibrary(database)
    old = library.get_pattern("old-pattern")
    assert old["experiment_id"] == "old-exp"
    assert old["feature_profile"] == "legacy"
    assert old["origin_research_id"] is None

    parent_id, _ = library.register_pattern(
        {
            "activation_condition": "x0",
            "formulas": ["x2"],
            "outputs_per_event": 1,
            "feature_profile": "legacy",
            "feature_mapping": feature_mapping("legacy", 10),
        }
    )
    child_id, _ = library.register_pattern(
        {
            "activation_condition": "x0",
            "formulas": ["x3"],
            "outputs_per_event": 1,
            "feature_profile": "legacy",
            "feature_mapping": feature_mapping("legacy", 10),
        }
    )
    assert library.record_lineage(
        parent_id,
        child_id,
        research_id="research-a",
        generation=2,
        method="mutation",
        operators=["mutation"],
        parent_formula="x2",
        child_formula="x3",
    )
    assert library.list_lineage(parent_id)[0]["child_pattern_id"] == child_id
    assert library.get_pattern(parent_id)["child_count"] == 1
    assert len(library.list_patterns()) == 3


def test_elite_selection_uses_validation_not_independent_test(tmp_path) -> None:
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    ids = []
    for index, validation_score in enumerate((0.4, 0.6)):
        pattern_id, _ = library.register_pattern(
            {
                "activation_condition": f"add(x0,{index + 1})",
                "formulas": [f"x{index + 2}"],
                "outputs_per_event": 1,
                "feature_profile": "legacy",
                "feature_mapping": feature_mapping("legacy", 10),
                "performance": {
                    "training_score": validation_score,
                    "validation_score": validation_score,
                    "validation_activation_rate": 0.5,
                    "test_score": 0.0,
                },
            }
        )
        ids.append(pattern_id)
    first, _ = library.select_elite_candidates(
        feature_profile="legacy", max_lag=10, max_candidates=1
    )
    assert [item["pattern_id"] for item in first] == [ids[1]]
    with library._connect() as connection:
        for pattern_id in ids:
            row = connection.execute(
                "SELECT payload FROM patterns WHERE pattern_id=?", (pattern_id,)
            ).fetchone()
            pattern = json.loads(row["payload"])
            pattern["performance"]["test_score"] = 1_000_000.0
            connection.execute(
                "UPDATE patterns SET payload=? WHERE pattern_id=?",
                (json.dumps(pattern), pattern_id),
            )
    second, _ = library.select_elite_candidates(
        feature_profile="legacy", max_lag=10, max_candidates=1
    )
    assert [item["pattern_id"] for item in second] == [ids[1]]


def test_research_checkpoint_reuses_selected_elites_and_next_seed(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    library = PatternLibrary(database)
    source_id, _ = library.register_pattern(
        {
            "activation_condition": "add(x0,1)",
            "formulas": ["x1"],
            "outputs_per_event": 1,
            "rule_index": 1,
            "feature_profile": "legacy",
            "feature_mapping": feature_mapping("legacy", 10),
            "performance": {
                "training_score": 0.4,
                "validation_score": 0.4,
                "validation_activation_rate": 0.6,
            },
        }
    )
    dataset = make_random_dataset(40, seed=50, first_round=800)
    checkpoint = tmp_path / "checks" / "study.json"
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=3,
        workers=1,
        seed=201,
        control_runs=0,
        elite_seeding_enabled=True,
        elite_seed_ratio=0.25,
    )
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=database,
        max_runs=1,
    )
    assert first["elite_seeding"]["source_pattern_ids"] == [source_id]
    assert first["last_run_summary"]["patterns_registered"] > 0
    experiment = json.loads(
        (tmp_path / "experiments" / f"{first['last_experiment_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert experiment["elite_seeding"]["one_number_search"]["seeded_count"] == 1
    assert experiment["elite_seeding"]["one_number_search"]["new_random_count"] == 7
    assert experiment["paired_unseeded_comparison"]["same_dataset_and_splits"] is True
    assert experiment["paired_unseeded_comparison"]["same_population_and_generations"] is True

    resumed = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=database,
        resume=True,
        max_runs=2,
    )
    assert resumed["elite_seeding"]["source_pattern_ids"] == [source_id]
    assert [item["seed"] for item in resumed["completed_experiments"]] == [201, 202]


def test_calendar_feature_mapping_matches_date_names() -> None:
    legacy = feature_mapping("legacy", 10)
    calendar = feature_mapping("calendar_round", 10)
    date_variable = f"x{len(legacy)}"
    assert calendar[date_variable]["name"] == "draw_year"

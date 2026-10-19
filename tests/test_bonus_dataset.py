import hashlib
import json

import pytest

from lotto_pattern_lab.bonus_research import (
    evaluate_paired_prize_backtest,
    run_bonus_experiment,
)
from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, load_csv
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.experiment import ExperimentStore
from lotto_pattern_lab.feature_schema import (
    feature_mapping,
    feature_schema_version,
    remap_formula_features,
)
from lotto_pattern_lab.features import (
    build_bonus_examples,
    build_bonus_features_for_target,
)
from lotto_pattern_lab.gp_engine import SearchConfig
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.intermittent_experiment import run_intermittent_experiment
from lotto_pattern_lab.pattern_library import PatternLibrary, register_experiment_candidates
from lotto_pattern_lab.prize_evaluation import evaluate_prize_predictions
from lotto_pattern_lab.research import dataset_digest


def _dataset(count: int = 45) -> DrawDataset:
    records = []
    for round_number in range(753, 753 + count):
        numbers = tuple(range(1 + round_number % 30, 7 + round_number % 30))
        offset = round_number % 45
        bonus = next(
            (offset + value - 1) % 45 + 1
            for value in range(1, 46)
            if (offset + value - 1) % 45 + 1 not in numbers
        )
        records.append(
            DrawRecord(round_number, Draw.from_numbers(numbers, bonus))
        )
    return DrawDataset(tuple(records), "synthetic", "test data")


def test_old_and_bonus_csv_formats_are_both_supported(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("round,n1,n2,n3,n4,n5,n6\n1,1,2,3,4,5,6\n2,2,3,4,5,6,7\n")
    bonus = tmp_path / "bonus.csv"
    bonus.write_text(
        "draw_no,n1,n2,n3,n4,n5,n6,bonus\n"
        "1,1,2,3,4,5,6,7\n2,2,3,4,5,6,7,8\n"
    )
    assert load_csv(old).records[0].draw.bonus is None
    assert load_csv(bonus).records[0].draw.bonus == 7


def test_bonus_digest_changes_without_breaking_legacy_checkpoint_digest():
    legacy_records = tuple(
        DrawRecord(record.round_number, Draw(record.draw.numbers), record.draw_date)
        for record in _dataset().records
    )
    legacy = DrawDataset(legacy_records, "synthetic", "legacy")
    expected_legacy = hashlib.sha256(
        json.dumps(
            [
                {
                    "round": record.round_number,
                    "date": None,
                    "numbers": record.draw.numbers,
                }
                for record in legacy.records
            ],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert dataset_digest(legacy) == expected_legacy
    assert dataset_digest(_dataset()) != expected_legacy


def test_bonus_target_uses_only_lagged_bonus_values():
    dataset = _dataset()
    examples = build_bonus_examples(dataset, max_lag=3)
    target = examples[0]
    expected_lags = tuple(float(dataset.records[index].draw.bonus) for index in (2, 1, 0))
    assert target.features[-3:] == expected_lags
    assert target.winning_numbers == frozenset({dataset.records[3].draw.bonus})
    assert target.features[-3:] != tuple(
        float(dataset.records[3 + index].draw.bonus) for index in (0, 1, 2)
    )

    features_before = build_bonus_features_for_target(dataset, 763, max_lag=3)
    records = list(dataset.records)
    records[10] = DrawRecord(763, Draw.from_numbers(records[10].draw.numbers, 45), None)
    changed_target = DrawDataset(tuple(records), "synthetic", "target bonus changed")
    assert build_bonus_features_for_target(changed_target, 763, max_lag=3) == features_before


def test_bonus_target_requires_bonus_values():
    no_bonus = DrawDataset(
        tuple(
            DrawRecord(record.round_number, Draw(record.draw.numbers), record.draw_date)
            for record in _dataset().records
        ),
        "synthetic",
        "no bonus",
    )
    with pytest.raises(ValueError, match="bonus value"):
        build_bonus_examples(no_bonus)


def test_bonus_profile_accepts_shared_ids_and_rejects_missing_bonus_feature():
    old_mapping = feature_mapping("legacy", 3)
    new_mapping = feature_mapping("bonus_lag_v1", 3)
    assert remap_formula_features("add(x0,x1)", old_mapping, new_mapping, "legacy")[1] is None
    missing = {"x0": {"feature_id": "lotto.lag1_bonus", "name": "lag1_bonus"}}
    assert remap_formula_features("add(x0,x99)", missing, new_mapping, "bonus_lag_v1")[0] is None
    assert feature_schema_version("bonus_lag_v1") > feature_schema_version("legacy")


def test_first_and_second_prize_counts_are_separate():
    dataset = _dataset(2)
    first = dataset.records[0]
    second = dataset.records[1]
    results = evaluate_prize_predictions(
        dataset,
        {
            first.round_number: first.draw.numbers,
            second.round_number: (*second.draw.numbers[:5], 45),
        },
        {first.round_number: first.draw.bonus, second.round_number: second.draw.bonus},
    )
    assert results["first_prize_exact_six"] == {"hits": 1, "denominator": 2}
    assert results["second_prize_five_plus_bonus"] == {"hits": 1, "denominator": 2}


def test_bonus_experiment_records_distinct_target_and_baseline(tmp_path):
    dataset = _dataset()
    result = run_bonus_experiment(
        dataset,
        SearchConfig(
            population_size=8,
            generations=1,
            workers=1,
            max_lag=3,
            control_runs=0,
            seed=9,
        ),
        ExperimentStore(tmp_path),
    )
    assert result["model_type"] == "bonus_number"
    assert result["random_baseline_accuracy"] == pytest.approx(1 / 45)
    assert result["target_leakage_prevented"] is True


def test_paired_backtest_separates_exact_six_from_five_plus_bonus():
    records = []
    for round_number in range(753, 765):
        numbers = tuple(range(1 + round_number % 30, 7 + round_number % 30))
        records.append(DrawRecord(round_number, Draw.from_numbers(numbers, 45)))
    dataset = DrawDataset(tuple(records), "synthetic", "fixed bonus test")
    test_rounds = [762, 763, 764]
    first = records[9]
    main_predictions = {
        round_number: (
            (*first.draw.numbers[:5], 44)
            if round_number == 762
            else records[round_number - 753].draw.numbers
        )
        for round_number in test_rounds
    }
    main_experiment = {
        "experiment_id": "main",
        "config": {"feature_profile": "legacy"},
        "six_number_set_model": {
            "selected": {
                "test_rounds": test_rounds,
                "activation_events": [
                    {"round_number": round_number, "predicted_numbers": values}
                    for round_number, values in main_predictions.items()
                ],
            }
        },
    }
    bonus_experiment = {
        "experiment_id": "bonus",
        "config": {"max_lag": 3},
        "feature_variables": {
            variable: item["name"]
            for variable, item in feature_mapping("bonus_lag_v1", 3).items()
        },
        "candidate": {"test_rounds": test_rounds, "formula": "44"},
    }
    result = evaluate_paired_prize_backtest(dataset, main_experiment, bonus_experiment)
    assert result["first_prize_exact_six"] == {"hits": 2, "denominator": 3}
    assert result["second_prize_five_plus_bonus"] == {"hits": 1, "denominator": 3}
    assert result["evaluation_kind"] == "chronological_backtest_not_pre_draw_predictions"


def test_persistent_bonus_profile_runs_bonus_target_and_paired_tiers(tmp_path):
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    result = run_intermittent_experiment(
        _dataset(),
        IntermittentConfig(
            population_size=8,
            generations=1,
            workers=1,
            max_lag=3,
            control_runs=0,
            hidden_states=2,
            min_activation_rate=0.05,
            feature_profile="bonus_lag_v1",
            seed=17,
        ),
        ExperimentStore(tmp_path / "experiments"),
        pattern_library=library,
    )
    assert result["bonus_number_model"] is not None
    paired = result["prize_tier_backtest"]
    assert paired["first_prize_exact_six"]["denominator"] > 0
    assert paired["second_prize_five_plus_bonus"]["denominator"] == paired[
        "first_prize_exact_six"
    ]["denominator"]
    assert paired["bonus_elite_seeding"]["cross_target_patterns_allowed"] is False
    registered = register_experiment_candidates(result, library)
    assert any(item["model_type"] == "bonus_number" for item in registered)
    elite, skipped = library.select_elite_candidates(
        feature_profile="bonus_lag_v1", max_lag=3, max_candidates=20
    )
    assert all(item.get("model_type") != "bonus_number" for item in elite)
    assert any("bonus target is isolated" in item["reason"] for item in skipped)

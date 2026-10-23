import hashlib
import json
from datetime import date, timedelta

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
from lotto_pattern_lab.prize_evaluation import (
    classify_prize_tier,
    evaluate_combo_prize_predictions,
    evaluate_prize_predictions,
)
from lotto_pattern_lab.research import dataset_digest
from lotto_pattern_lab.research_features import (
    build_integrated_examples,
    build_integrated_features_for_target,
)


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


def test_integrated_profile_merges_three_unique_feature_groups_and_keeps_old_layouts():
    legacy = feature_mapping("legacy", 10)
    calendar = feature_mapping("calendar_round", 10)
    bonus = feature_mapping("bonus_lag_v1", 10)
    integrated = feature_mapping("integrated_v1", 10)

    assert len(legacy) == 70
    assert len(calendar) == 90
    assert len(bonus) == 80
    assert len(integrated) == 100
    ids = [item["feature_id"] for item in integrated.values()]
    assert len(ids) == len(set(ids))
    assert list(legacy.items()) == list(feature_mapping("legacy", 10).items())
    assert feature_schema_version("integrated_v1") > feature_schema_version("bonus_lag_v1")


def test_integrated_features_use_target_metadata_and_only_prior_draws():
    original = _dataset()
    dated = DrawDataset(
        tuple(
            DrawRecord(
                record.round_number,
                record.draw,
                date(2000, 1, 1) + timedelta(days=index * 7),
            )
            for index, record in enumerate(original.records)
        ),
        "synthetic",
        "dated test data",
    )
    target_round = dated.records[20].round_number
    target_date = dated.records[20].draw_date
    features = build_integrated_features_for_target(dated, target_round, target_date, 5)

    changed = list(dated.records)
    target = changed[20]
    changed[20] = DrawRecord(
        target.round_number, Draw.from_numbers((1, 2, 3, 4, 5, 6), 45), target.draw_date
    )
    changed[21] = DrawRecord(
        changed[21].round_number,
        Draw.from_numbers((7, 8, 9, 10, 11, 12), 44),
        changed[21].draw_date,
    )
    changed_dataset = DrawDataset(tuple(changed), "synthetic", "changed target and future")
    assert build_integrated_features_for_target(
        changed_dataset, target_round, target_date, 5
    ) == features

    examples = build_integrated_examples(dated, 5)
    expected_bonus_lags = tuple(
        float(dated.records[19 - lag].draw.bonus) for lag in range(5)
    )
    assert examples[15].features[-5:] == expected_bonus_lags
    assert examples[15].round_number == target_round


def test_first_and_second_prize_counts_are_separate():
    dataset = _dataset(2)
    first = dataset.records[0]
    second = dataset.records[1]
    results = evaluate_prize_predictions(
        dataset,
        {
                first.round_number: first.draw.numbers,
                second.round_number: (*second.draw.numbers[:5], second.draw.bonus),
        },
        {first.round_number: first.draw.bonus, second.round_number: second.draw.bonus},
    )
    assert results["first_prize_exact_six"] == {"hits": 1, "denominator": 2}
    assert results["second_prize_five_plus_bonus"]["hits"] == 1
    assert results["second_prize_five_plus_bonus"]["denominator"] == 2


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
            (*first.draw.numbers[:5], first.draw.bonus)
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
    assert result["second_prize_five_plus_bonus"]["hits"] == 1
    assert result["second_prize_five_plus_bonus"]["denominator"] == 3
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
    assert paired["first_prize_exact_six"]["denominator"] >= 0
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


def test_official_prize_tier_uses_main_and_actual_bonus_rules():
    winning = (1, 5, 12, 20, 33, 40)
    bonus = 17
    assert classify_prize_tier(winning, winning, bonus)["prize_tier"] == 1
    assert classify_prize_tier((1, 5, 12, 20, 33, bonus), winning, bonus)["prize_tier"] == 2
    assert classify_prize_tier((1, 5, 12, 20, 33, 41), winning, bonus)["prize_tier"] == 3
    assert classify_prize_tier((1, 5, 12, 20, bonus, 41), winning, bonus)["prize_tier"] == 4
    assert classify_prize_tier((1, 5, 12, 41, 42, 43), winning, bonus)["prize_tier"] == 5
    assert classify_prize_tier(
        (1, 5, 12, 20, 33, 41), winning, None
    )["prize_tier"] is None
    assert classify_prize_tier(
        (40, 33, 20, 12, 5, 1), winning, bonus
    )["prize_tier"] == 1
    with pytest.raises(ValueError):
        classify_prize_tier((1, 1, 2, 3, 4, 5), winning, bonus)


def test_bonus_is_not_second_prize_for_six_or_four_main_matches():
    winning, bonus = (1, 5, 12, 20, 33, 40), 17
    assert classify_prize_tier(winning, winning, bonus)["prize_tier"] != 2
    assert classify_prize_tier((1, 5, 12, 20, bonus, 41), winning, bonus)["prize_tier"] != 2


def test_combo_prize_statistics_distinguish_unknown_bonus_from_zero():
    actual = {
        1: Draw.from_numbers((1, 5, 12, 20, 33, 40), 17),
        2: Draw.from_numbers((1, 5, 12, 20, 33, 40)),
    }
    summary = evaluate_combo_prize_predictions(
        {1: (1, 5, 12, 20, 33, 41), 2: (1, 5, 12, 20, 33, 41)}, actual
    )
    assert summary["second_prize_hits"] == 0
    assert summary["second_prize_denominator"] == 1
    assert summary["second_prize_unknown_bonus_count"] == 1
    assert summary["prize_tier_hits"]["3"] == 1
    assert summary["prize_tier_hits"]["2"] == 0

    unavailable = evaluate_combo_prize_predictions({1: (1, 2, 3, 4, 5, 6)}, {})
    assert unavailable["second_prize_hits"] is None
    assert unavailable["second_prize_status"] == "판정 불가"

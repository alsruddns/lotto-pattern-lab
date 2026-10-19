import json

import pytest

from lotto_pattern_lab.backtest import chronological_split
from lotto_pattern_lab.dataset import DrawDataset, load_csv, make_random_dataset
from lotto_pattern_lab.experiment import run_experiment
from lotto_pattern_lab.features import build_examples
from lotto_pattern_lab.gp_engine import (
    SearchConfig,
    _make_primitive_set,
    _predict,
    _safe_div,
    _safe_mod,
    run_search,
)
from lotto_pattern_lab.storage import ExperimentStore


def test_feature_encoding_is_invariant_to_number_order() -> None:
    first = make_random_dataset(20, 5)
    reordered_records = tuple(
        type(record)(record.round_number, type(record.draw)(tuple(reversed(record.draw.numbers))))
        for record in first.records
    )
    second = DrawDataset(reordered_records, "synthetic", "reordered")

    assert build_examples(first) == build_examples(second)


def test_safe_math_operations_and_expression_compilation() -> None:
    assert _safe_div(9, 0) == 9
    assert _safe_div(9, 3) == 3
    assert _safe_mod(9, 0) == 0
    assert _safe_mod(9, 4) == 1

    pset = _make_primitive_set(1)
    function = __import__("deap.gp", fromlist=["compile"]).compile("safe_div(add(x0, 4), 2)", pset)
    assert function(6) == 5
    assert 1 <= _predict(function, (6,)) <= 45


def test_random_dataset_is_reproducible_and_unique_within_each_draw() -> None:
    first = make_random_dataset(25, 17)
    second = make_random_dataset(25, 17)

    assert first == second
    assert all(len(set(record.draw.numbers)) == 6 for record in first.records)
    assert first.source_kind == "synthetic"


def test_features_for_a_round_only_depend_on_prior_rounds() -> None:
    dataset = make_random_dataset(30, 9)
    examples = build_examples(dataset)
    changed_future_records = list(dataset.records)
    target_index = 20
    original_target = changed_future_records[target_index]
    changed_future_records[target_index] = type(original_target)(
        original_target.round_number,
        type(original_target.draw).from_numbers([1, 2, 3, 4, 5, 6]),
    )
    changed_dataset = DrawDataset(tuple(changed_future_records), "synthetic", "mutated future")
    changed_examples = build_examples(changed_dataset)

    assert examples[target_index - 10].features == changed_examples[target_index - 10].features
    assert (
        examples[target_index - 10].winning_numbers
        != changed_examples[target_index - 10].winning_numbers
    )


def test_backtest_preserves_order_and_keeps_test_out_of_train_validation() -> None:
    examples = build_examples(make_random_dataset(40, 2))
    splits = chronological_split(examples, 0.6, 0.2)

    assert max(example.round_number for example in splits.train) < min(
        example.round_number for example in splits.validation
    )
    assert max(example.round_number for example in splits.validation) < min(
        example.round_number for example in splits.test
    )
    assert not (
        {example.round_number for example in splits.train}
        & {example.round_number for example in splits.test}
    )


def test_small_gp_search_is_seed_reproducible() -> None:
    dataset = make_random_dataset(35, 12)
    splits = chronological_split(build_examples(dataset))
    config = SearchConfig(population_size=8, generations=2, workers=1, seed=42)

    first = run_search(splits, config)
    second = run_search(splits, config)

    assert first == second
    assert first.test_rounds[0] > first.validation_rounds[-1]
    assert 0 <= first.test_accuracy <= 1


def test_experiment_is_saved_and_loaded_with_control_results(tmp_path) -> None:
    dataset = make_random_dataset(35, 21)
    config = SearchConfig(population_size=8, generations=1, workers=1, seed=7, control_runs=1)
    store = ExperimentStore(tmp_path)

    result = run_experiment(dataset, config, store)
    loaded = store.load(str(result["experiment_id"]))

    assert loaded["status"] == "completed"
    assert loaded["test_set_used_for_selection"] is False
    assert len(loaded["random_controls"]) == 1
    assert loaded["candidate"]["formula"]
    assert loaded["candidate"]["candidate_formulas"]
    assert (
        loaded["candidate"]["candidate_formulas"][0]["test_accuracy"]
        == loaded["candidate"]["test_accuracy"]
    )
    assert all(
        formula["test_accuracy"] is None
        for formula in loaded["candidate"]["candidate_formulas"][1:]
    )
    assert "x0" in loaded["feature_variables"]
    assert loaded["random_control_empirical_p_value"] is not None
    assert store.list()[0]["experiment_id"] == result["experiment_id"]
    json.dumps(loaded)


def test_csv_loader_validates_draws_and_records_unverified_source(tmp_path) -> None:
    path = tmp_path / "draws.csv"
    path.write_text(
        "round,n1,n2,n3,n4,n5,n6,bonus\n1,1,2,3,4,5,6,7\n2,2,3,4,5,6,7,8\n",
        encoding="utf-8",
    )
    dataset = load_csv(path, "https://example.org/draws.csv")

    assert dataset.source_kind == "csv_unverified"
    assert dataset.source_url == "https://example.org/draws.csv"
    assert dataset.records[0].draw.numbers == (1, 2, 3, 4, 5, 6)

    path.write_text("round,n1,n2,n3,n4,n5,n6\n1,1,1,3,4,5,6\n2,2,3,4,5,6,7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_csv(path)

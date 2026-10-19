import json
from datetime import date, timedelta

import numpy as np
import pytest

from lotto_pattern_lab.backtest import chronological_split
from lotto_pattern_lab.dataset import (
    DrawDataset,
    DrawRecord,
    filter_dataset,
    load_csv,
    make_random_dataset,
    save_official_csv,
)
from lotto_pattern_lab.draw import Draw
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
from lotto_pattern_lab.hidden_states import (
    causal_state_probabilities,
    fit_gaussian_hmm,
    online_change_points,
)
from lotto_pattern_lab.intermittent import IntermittentConfig, run_intermittent_search
from lotto_pattern_lab.intermittent_experiment import run_intermittent_experiment
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


def test_intermittent_experiment_persists_selected_rules_and_control_status(tmp_path) -> None:
    dataset = make_random_dataset(45, 51)
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        workers=1,
        seed=14,
        control_runs=0,
    )
    store = ExperimentStore(tmp_path)

    result = run_intermittent_experiment(dataset, config, store)
    loaded = store.load(str(result["experiment_id"]))

    assert loaded["status"] == "completed"
    assert loaded["splits"]["test_set_used_for_selection"] is False
    assert loaded["inference_status"] == "tentative"
    assert loaded["actual_gp_runtime_seconds"] >= 0
    assert loaded["random_control_empirical_p_value"]["one_number"] is None
    for model_key in ("one_number_model", "six_number_set_model"):
        selected = loaded[model_key]["selected"]
        assert selected["rule_count"] == 2
        assert len(selected["rule_active_rounds"]) == 2
        assert selected["test_score"] is not None
        assert all(
            alternative["test_score"] is None for alternative in loaded[model_key]["candidates"][1:]
        )
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


def test_csv_loader_maps_draw_no_and_marks_complete_unreconciled_range(tmp_path) -> None:
    path = tmp_path / "draw_no.csv"
    records = make_random_dataset(494, 753, first_round=753)
    rows = ["draw_no,n1,n2,n3,n4,n5,n6"]
    rows.extend(
        f"{record.round_number},{','.join(map(str, record.draw.numbers))}"
        for record in records.records
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = load_csv(path)

    assert len(dataset.records) == 494
    assert dataset.records[0].round_number == 753
    assert dataset.records[-1].round_number == 1246
    assert dataset.records[0].draw_date is None
    assert dataset.records[0].draw.bonus is None
    assert dataset.source_kind == "csv_locally_validated"
    assert dataset.verification_status == "locally_validated_unreconciled"


def test_dataset_range_filter_does_not_keep_pre_range_rounds() -> None:
    first_date = date(2017, 5, 6)
    records = tuple(
        DrawRecord(
            record.round_number + 752,
            record.draw,
            first_date + timedelta(days=index * 7),
        )
        for index, record in enumerate(make_random_dataset(20, 1).records)
    )
    dataset = DrawDataset(records, "synthetic", "dated synthetic")

    selected = filter_dataset(
        dataset,
        start_date=date(2017, 6, 3),
        end_date=date(2017, 7, 1),
        start_round=757,
        end_round=760,
    )

    assert [record.round_number for record in selected.records] == [757, 758, 759, 760]
    assert selected.records[0].draw_date == date(2017, 6, 3)


def test_official_data_csv_manifest_detects_content_changes(tmp_path) -> None:
    records = (
        DrawRecord(753, make_random_dataset(2, 4).records[0].draw, date(2017, 5, 6)),
        DrawRecord(754, make_random_dataset(2, 5).records[1].draw, date(2017, 5, 13)),
    )
    dataset = DrawDataset(
        records,
        "official_verified",
        "Official result endpoint test fixture.",
        "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do",
        verified_at="2026-10-19T00:00:00+00:00",
    )
    csv_path, _ = save_official_csv(dataset, tmp_path / "official.csv")

    loaded = load_csv(csv_path)
    assert loaded.verification_status == "official_verified"
    assert loaded.records[0].draw_date == date(2017, 5, 6)
    history_path = tmp_path / "official.csv.history.jsonl"
    assert history_path.is_file()
    assert json.loads(history_path.read_text(encoding="utf-8").splitlines()[0])["new_sha256"]

    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("753", "755", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum"):
        load_csv(csv_path)


def test_intermittent_gp_is_reproducible_and_keeps_test_out_of_alternatives() -> None:
    examples = build_examples(make_random_dataset(55, 42))
    splits = chronological_split(examples, 0.6, 0.2)
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        workers=1,
        seed=91,
        top_candidates=5,
    )

    first = run_intermittent_search(splits, config)
    second = run_intermittent_search(splits, config)

    assert first == second
    assert first.selected.activation_condition
    assert first.selected.test_score is not None
    assert first.selected.activation_events[-1]["round_number"] == splits.test[-1].round_number
    for alternative in first.candidates[1:]:
        assert alternative.test_score is None
        assert all(
            int(event["round_number"]) <= splits.validation[-1].round_number
            for event in alternative.activation_events
        )


def test_six_number_gp_evaluates_unordered_unique_sets() -> None:
    examples = build_examples(make_random_dataset(45, 24))
    splits = chronological_split(examples, 0.6, 0.2)
    config = IntermittentConfig(
        population_size=8, generations=1, max_depth=2, workers=1, seed=4, top_candidates=2
    )

    result = run_intermittent_search(splits, config, outputs_per_event=6)
    active = [event for event in result.selected.activation_events if event["active"]]

    assert result.selected.outputs_per_event == 6
    assert active
    assert all(len(event["predicted_numbers"]) == 6 for event in active)
    assert all(len(set(event["predicted_numbers"])) == 6 for event in active)
    for event, example in zip(result.selected.activation_events, examples, strict=True):
        if event["active"]:
            assert event["hit_count"] == len(
                set(event["predicted_numbers"]) & example.winning_numbers
            )


def test_hmm_filtering_and_change_points_are_causal() -> None:
    dataset = make_random_dataset(70, 31)
    examples = build_examples(dataset)
    splits = chronological_split(examples, 0.6, 0.2)
    training_observations = np.asarray(
        [example.features[1:7] for example in splits.train], dtype=float
    )
    model = fit_gaussian_hmm(training_observations, state_count=3, seed=5)
    original = causal_state_probabilities(model, examples)

    changed_records = list(dataset.records)
    changed_records[-2] = DrawRecord(
        changed_records[-2].round_number,
        Draw.from_numbers((1, 2, 3, 4, 5, 6)),
        changed_records[-2].draw_date,
    )
    changed_examples = build_examples(
        DrawDataset(tuple(changed_records), "synthetic", "future edit")
    )
    changed = causal_state_probabilities(model, changed_examples)
    points = online_change_points(examples)

    assert np.allclose(original[:-1], changed[:-1])
    assert all("change_point" in point for point in points)

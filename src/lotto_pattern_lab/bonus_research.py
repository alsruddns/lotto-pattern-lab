"""Separate one-number GP research for the lottery bonus target."""

import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from .backtest import chronological_split
from .dataset import DrawDataset, dataset_integrity_report, make_random_dataset
from .experiment import dataset_metadata, result_to_dict
from .feature_schema import feature_schema_version
from .features import (
    bonus_feature_names,
    build_bonus_examples,
    build_bonus_features_for_target,
)
from .gp_engine import SearchConfig, run_search
from .prediction import evaluate_formula
from .prize_evaluation import evaluate_prize_predictions
from .storage import ExperimentStore


def run_bonus_experiment(
    dataset: DrawDataset, config: SearchConfig, store: ExperimentStore
) -> dict[str, object]:
    """Train and evaluate bonus-only predictions, separate from the six-main target."""
    integrity = dataset_integrity_report(dataset)
    if not integrity["all_draws_have_bonus"]:
        raise ValueError("모든 회차의 보너스 번호가 있어야 보너스 연구를 실행할 수 있습니다.")
    if config.max_lag > len(dataset.records) - 1:
        raise ValueError("보너스 연구에 사용할 과거 회차가 부족합니다.")
    experiment_id = str(uuid.uuid4())
    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    examples = build_bonus_examples(dataset, config.max_lag)
    splits = chronological_split(examples, config.train_ratio, config.validation_ratio)
    result = run_search(splits, config, run_label="bonus_target")
    candidate = result_to_dict(result)
    candidate["random_baseline"] = 1 / 45
    controls = []
    for index in range(config.control_runs):
        random_data = make_random_dataset(
            len(dataset.records),
            config.seed + 100_003 + index,
            dataset.records[0].round_number,
            include_bonus=True,
        )
        control_examples = build_bonus_examples(random_data, config.max_lag)
        control_splits = chronological_split(
            control_examples, config.train_ratio, config.validation_ratio
        )
        control = run_search(
            control_splits,
            SearchConfig(**{**asdict(config), "seed": config.seed + 10_007 + index}),
            run_label=f"bonus_random_{index + 1}",
        )
        control_result = result_to_dict(control)
        control_result["random_baseline"] = 1 / 45
        controls.append(control_result)
    scores = [float(item["test_accuracy"]) for item in controls]
    payload: dict[str, object] = {
        "experiment_id": experiment_id,
        "experiment_type": "bonus_target",
        "model_type": "bonus_number",
        "target_definition": "exact match against the draw's separate bonus number",
        "status": "completed",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - timer,
        "dataset": dataset_metadata(dataset),
        "config": asdict(config),
        "feature_profile": "bonus_lag_v1",
        "feature_schema_version": feature_schema_version("bonus_lag_v1"),
        "feature_variables": {
            f"x{index}": name for index, name in enumerate(bonus_feature_names(config.max_lag))
        },
        "candidate": candidate,
        "random_controls": controls,
        "random_control_test_scores": scores,
        "random_control_empirical_p_value": (
            (1 + sum(score >= result.test_accuracy for score in scores)) / (1 + len(scores))
            if scores
            else None
        ),
        "random_baseline_accuracy": 1 / 45,
        "target_leakage_prevented": True,
        "notes": "The current draw's bonus is a label only; bonus inputs use preceding rounds.",
    }
    store.save(payload)
    return payload


def evaluate_paired_prize_backtest(
    dataset: DrawDataset,
    main_experiment: dict[str, object],
    bonus_experiment: dict[str, object],
) -> dict[str, object]:
    """Pair the six-main GP and bonus GP on their shared untouched test rounds."""
    main_model = main_experiment["six_number_set_model"]
    main_selected = main_model["selected"]
    bonus_candidate = bonus_experiment["candidate"]
    test_rounds = sorted(
        set(main_selected["test_rounds"]) & set(bonus_candidate["test_rounds"])
    )
    events = {
        int(event["round_number"]): event
        for event in main_selected["activation_events"]
        if int(event["round_number"]) in test_rounds
    }
    if set(events) != set(test_rounds):
        raise ValueError("Main model is missing events for part of the shared test range.")
    main_predictions = {
        round_number: events[round_number]["predicted_numbers"] for round_number in test_rounds
    }
    bonus_predictions = {}
    for round_number in test_rounds:
        features = build_bonus_features_for_target(
            dataset, int(round_number), int(bonus_experiment["config"]["max_lag"])
        )
        bonus_predictions[int(round_number)] = evaluate_formula(
            str(bonus_candidate["formula"]),
            features,
            tuple(bonus_experiment["feature_variables"].values()),
        )["number"]
    evaluation = evaluate_prize_predictions(dataset, main_predictions, bonus_predictions)
    return {
        **evaluation,
        "evaluation_kind": "chronological_backtest_not_pre_draw_predictions",
        "main_model_experiment_id": main_experiment["experiment_id"],
        "bonus_model_experiment_id": bonus_experiment["experiment_id"],
        "main_model_profile": main_experiment["config"].get("feature_profile", "legacy"),
        "bonus_model_profile": "bonus_lag_v1",
    }

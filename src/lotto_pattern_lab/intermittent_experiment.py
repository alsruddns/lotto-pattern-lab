"""Full intermittent-rule experiments with matched random controls and causal regimes."""

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np

from .backtest import DataSplits, chronological_split
from .dataset import DrawDataset, DrawRecord, make_random_dataset
from .experiment import dataset_metadata
from .feature_schema import (
    CODE_VERSION,
    FEATURE_SCHEMA_VERSION,
    PRIMITIVE_SET_VERSION,
    feature_mapping,
)
from .features import build_examples, feature_names
from .hidden_states import causal_state_probabilities, fit_gaussian_hmm, online_change_points
from .intermittent import (
    IntermittentCandidate,
    IntermittentConfig,
    IntermittentSearchResult,
    run_intermittent_search,
)
from .pattern_library import (
    PatternLibrary,
    register_experiment_candidates,
    register_experiment_exact_matches,
)
from .research_features import (
    ResearchFeatureConfig,
    build_research_examples,
    research_feature_names,
)
from .storage import ExperimentStore


def _candidate_dict(candidate: IntermittentCandidate) -> dict[str, object]:
    match_distribution = {str(number): 0 for number in range(7)}
    for event in candidate.activation_events:
        if event["active"]:
            hit_count = int(event["hit_count"])
            match_distribution[str(hit_count)] += 1
    distribution_by_split = {
        "training": candidate.training_match_distribution,
        "validation": candidate.validation_match_distribution,
        "independent_test": candidate.test_match_distribution,
    }
    return {
        "activation_condition": candidate.activation_condition,
        "formulas": list(candidate.formulas),
        "rule_count": candidate.rule_count,
        "outputs_per_event": candidate.outputs_per_event,
        "training_activation_rate": candidate.training_activation_rate,
        "validation_activation_rate": candidate.validation_activation_rate,
        "test_activation_rate": candidate.test_activation_rate,
        "training_hit_rate": candidate.training_hit_rate,
        "validation_hit_rate": candidate.validation_hit_rate,
        "test_hit_rate": candidate.test_hit_rate,
        "training_hit_count": candidate.training_hit_count,
        "validation_hit_count": candidate.validation_hit_count,
        "test_hit_count": candidate.test_hit_count,
        "training_full_set_matches": candidate.training_full_set_matches,
        "validation_full_set_matches": candidate.validation_full_set_matches,
        "test_full_set_matches": candidate.test_full_set_matches,
        "match_distribution": match_distribution,
        "match_distribution_by_split": {
            split: {str(number): count for number, count in enumerate(values or ())}
            for split, values in distribution_by_split.items()
        },
        "at_least_3_hits_by_split": {
            split: sum(values[3:]) if values else None
            for split, values in distribution_by_split.items()
        },
        "at_least_4_hits_by_split": {
            split: sum(values[4:]) if values else None
            for split, values in distribution_by_split.items()
        },
        "at_least_5_hits_by_split": {
            split: sum(values[5:]) if values else None
            for split, values in distribution_by_split.items()
        },
        "training_score": candidate.training_score,
        "validation_score": candidate.validation_score,
        "test_score": candidate.test_score,
        "complexity": candidate.complexity,
        "height": candidate.height,
        "lineage": candidate.lineage,
        "activation_events": list(candidate.activation_events),
        "active_gaps": list(candidate.active_gaps),
        "rule_active_rounds": [list(rounds) for rounds in candidate.rule_active_rounds],
        "rule_active_gaps": [list(gaps) for gaps in candidate.rule_active_gaps],
    }


def _search_dict(result: IntermittentSearchResult) -> dict[str, object]:
    return {
        "outputs_per_event": result.outputs_per_event,
        "selected": _candidate_dict(result.selected),
        "candidates": [_candidate_dict(candidate) for candidate in result.candidates],
        "train_rounds": list(result.train_rounds),
        "validation_rounds": list(result.validation_rounds),
        "test_rounds": list(result.test_rounds),
    }


def _splits_for(dataset: DrawDataset, config: IntermittentConfig) -> tuple[tuple, DataSplits]:
    if config.feature_profile == "calendar_round":
        examples = build_research_examples(
            dataset,
            config.max_lag,
            ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
        )
    else:
        examples = build_examples(dataset, config.max_lag)
    return examples, chronological_split(examples, config.train_ratio, config.validation_ratio)


def _hypothetical_states(
    splits: DataSplits, config: IntermittentConfig
) -> tuple[list[dict[str, object]], dict[int, str]]:
    observations = np.asarray([example.features[1:7] for example in splits.train], dtype=float)
    model = fit_gaussian_hmm(observations, config.hidden_states, config.seed)
    examples = splits.train + splits.validation + splits.test
    probabilities = causal_state_probabilities(model, examples)
    changes = online_change_points(examples)
    change_by_round = {int(item["round_number"]): item for item in changes}
    state_names = {index: f"{chr(ord('A') + index)}" for index in range(config.hidden_states)}
    mapping: dict[int, str] = {}
    rows = []
    for index, example in enumerate(examples):
        state_index = int(np.argmax(probabilities[index]))
        state_name = state_names[state_index]
        mapping[example.round_number] = state_name
        rows.append(
            {
                "round_number": example.round_number,
                "state": state_name,
                "state_probabilities": {
                    state_names[state]: float(probabilities[index, state])
                    for state in range(config.hidden_states)
                },
                "online_change_point": bool(change_by_round[example.round_number]["change_point"]),
                "change_score": float(change_by_round[example.round_number]["score"]),
            }
        )
    return rows, mapping


def _state_conditioned_metrics(
    selected: IntermittentCandidate,
    splits: DataSplits,
    state_by_round: dict[int, str],
) -> dict[str, dict[str, float | int]]:
    test_rounds = {example.round_number for example in splits.test}
    events = [
        event for event in selected.activation_events if int(event["round_number"]) in test_rounds
    ]
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"rounds": 0, "active_rounds": 0, "active_hits": 0, "active_predictions": 0}
    )
    for event in events:
        state = state_by_round[int(event["round_number"])]
        row = totals[state]
        row["rounds"] += 1
        if event["active"]:
            row["active_rounds"] += 1
            row["active_hits"] += int(event["hit_count"])
            row["active_predictions"] += len(event["predicted_numbers"])
    return {state: dict(metrics) for state, metrics in sorted(totals.items())}


def _with_state_tags(
    candidate: IntermittentCandidate, state_by_round: dict[int, str]
) -> list[dict[str, object]]:
    return [
        {**event, "hypothetical_state": state_by_round[int(event["round_number"])]}
        for event in candidate.activation_events
    ]


def _empirical_p(observed: float, controls: list[float]) -> float | None:
    if not controls:
        return None
    return (1 + sum(value >= observed for value in controls)) / (1 + len(controls))


def run_intermittent_experiment(
    dataset: DrawDataset,
    config: IntermittentConfig,
    store: ExperimentStore,
    progress: Callable[[dict[str, object]], None] | None = None,
    pattern_library: PatternLibrary | None = None,
    elite_seeds: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Run one- and six-number gated GP models and identical random-data searches."""
    experiment_id = str(uuid.uuid4())
    started = datetime.now(UTC).isoformat()
    timer = time.perf_counter()
    examples, splits = _splits_for(dataset, config)
    events: list[dict[str, object]] = []

    def record_progress(event: dict[str, int | float | str]) -> None:
        entry = {
            **event,
            "timestamp": datetime.now(UTC).isoformat(),
            "experiment_id": experiment_id,
        }
        events.append(entry)
        if progress:
            progress(entry)
        store.save(
            {
                "experiment_id": experiment_id,
                "experiment_type": "intermittent",
                "status": "running",
                "started_at": started,
                "dataset": dataset_metadata(dataset),
                "config": asdict(config),
                "progress": events,
            }
        )

    actual_gp_started = time.perf_counter()
    one_number = run_intermittent_search(
        splits,
        config,
        outputs_per_event=1,
        progress=record_progress,
        run_label="actual_one_number",
        elite_seeds=elite_seeds,
    )
    set_model = run_intermittent_search(
        splits,
        IntermittentConfig(**{**asdict(config), "seed": config.seed + 1}),
        outputs_per_event=6,
        progress=record_progress,
        run_label="actual_six_number_set",
        elite_seeds=elite_seeds,
    )
    actual_gp_runtime = time.perf_counter() - actual_gp_started
    paired_unseeded = None
    paired_unseeded_runtime = None
    if one_number.seeding["seeded_count"] or set_model.seeding["seeded_count"]:
        paired_started = time.perf_counter()
        comparison_config = IntermittentConfig(
            **{
                **asdict(config),
                "elite_seeding_enabled": False,
                "elite_seed_ratio": 0.0,
            }
        )
        comparison_one = run_intermittent_search(
            splits,
            comparison_config,
            outputs_per_event=1,
            progress=record_progress,
            run_label="paired_unseeded_one_number",
        )
        comparison_set = run_intermittent_search(
            splits,
            IntermittentConfig(**{**asdict(comparison_config), "seed": config.seed + 1}),
            outputs_per_event=6,
            progress=record_progress,
            run_label="paired_unseeded_six_number_set",
        )
        paired_unseeded = {
            "same_dataset_and_splits": True,
            "same_population_and_generations": True,
            "initialization": "100% new random candidates",
            "one_number_model": _search_dict(comparison_one),
            "six_number_set_model": _search_dict(comparison_set),
        }
        paired_unseeded_runtime = time.perf_counter() - paired_started
    state_rows, state_by_round = _hypothetical_states(splits, config)
    change_points = [row for row in state_rows if row["online_change_point"]]
    one_dict = _search_dict(one_number)
    set_dict = _search_dict(set_model)
    one_dict["selected"]["activation_events"] = _with_state_tags(
        one_number.selected, state_by_round
    )
    set_dict["selected"]["activation_events"] = _with_state_tags(set_model.selected, state_by_round)
    state_metrics = {
        "one_number_test": _state_conditioned_metrics(one_number.selected, splits, state_by_round),
        "six_number_test": _state_conditioned_metrics(set_model.selected, splits, state_by_round),
    }

    random_controls: list[dict[str, object]] = []
    for control_index in range(config.control_runs):
        data_seed = config.seed + 100_003 + control_index
        control_dataset = make_random_dataset(
            len(dataset.records), data_seed, first_round=dataset.records[0].round_number
        )
        if config.feature_profile == "calendar_round":
            control_dataset = DrawDataset(
                tuple(
                    DrawRecord(random_record.round_number, random_record.draw, actual.draw_date)
                    for random_record, actual in zip(
                        control_dataset.records, dataset.records, strict=True
                    )
                ),
                "synthetic",
                f"Random 6/45 control with official date covariates (seed={data_seed}).",
            )
        control_examples, control_splits = _splits_for(control_dataset, config)
        control_config = IntermittentConfig(
            **{
                **asdict(config),
                "seed": config.seed + 10_007 + control_index,
                "elite_seeding_enabled": False,
                "elite_seed_ratio": 0.0,
            }
        )
        control_one = run_intermittent_search(
            control_splits,
            control_config,
            outputs_per_event=1,
            progress=record_progress,
            run_label=f"random_{control_index + 1}_one_number",
        )
        control_set_config = IntermittentConfig(
            **{**asdict(control_config), "seed": control_config.seed + 1}
        )
        control_set = run_intermittent_search(
            control_splits,
            control_set_config,
            outputs_per_event=6,
            progress=record_progress,
            run_label=f"random_{control_index + 1}_six_number_set",
        )
        control_states, _ = _hypothetical_states(control_splits, control_config)
        random_controls.append(
            {
                "replicate": control_index + 1,
                "data_seed": data_seed,
                "search_seed": control_config.seed,
                "one_number": _search_dict(control_one),
                "six_number_set": _search_dict(control_set),
                "hypothetical_state_count": config.hidden_states,
                "change_point_count": sum(
                    bool(row["online_change_point"]) for row in control_states
                ),
            }
        )

    one_control_scores = [
        float(control["one_number"]["selected"]["test_score"]) for control in random_controls
    ]
    set_control_scores = [
        float(control["six_number_set"]["selected"]["test_score"]) for control in random_controls
    ]
    test_start = splits.test[0].round_number
    test_end = splits.test[-1].round_number
    output: dict[str, object] = {
        "experiment_id": experiment_id,
        "experiment_type": "intermittent",
        "status": "completed",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - timer,
        "actual_gp_runtime_seconds": actual_gp_runtime,
        "dataset": dataset_metadata(dataset),
        "config": asdict(config),
        "feature_schema": {
            "feature_profile": config.feature_profile,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "primitive_set_version": PRIMITIVE_SET_VERSION,
            "feature_mapping": feature_mapping(config.feature_profile, config.max_lag),
            "code_version": CODE_VERSION,
        },
        "elite_seeding": {
            "enabled": config.elite_seeding_enabled,
            "ratio": config.elite_seed_ratio,
            "source_pattern_ids": sorted(
                set(one_number.seeding.get("source_pattern_ids", ()))
                | set(set_model.seeding.get("source_pattern_ids", ()))
            ),
            "one_number_search": one_number.seeding,
            "six_number_set_search": set_model.seeding,
            "random_controls": "unseeded controls; cumulative search history is not matched",
            "control_comparability": "limited" if elite_seeds else "no elite seeds used",
        },
        "paired_unseeded_comparison": (
            {
                **paired_unseeded,
                "runtime_seconds": paired_unseeded_runtime,
                "comparison_seed": config.seed,
                "interpretation": (
                    "paired same-data GP initialization comparison; random-data controls do not "
                    "replay cumulative history"
                ),
            }
            if paired_unseeded is not None
            else None
        ),
        "feature_variables": {
            f"x{index}": name
            for index, name in enumerate(
                feature_names(config.max_lag)
                + (
                    research_feature_names(ResearchFeatureConfig())
                    if config.feature_profile == "calendar_round"
                    else ()
                )
            )
        },
        "seed": config.seed,
        "splits": {
            "train_rounds": [splits.train[0].round_number, splits.train[-1].round_number],
            "validation_rounds": [
                splits.validation[0].round_number,
                splits.validation[-1].round_number,
            ],
            "test_rounds": [test_start, test_end],
            "test_set_used_for_selection": False,
        },
        "one_number_model": one_dict,
        "six_number_set_model": set_dict,
        "hypothetical_states": {
            "labels_are_organizational_only": True,
            "state_rows": state_rows,
            "test_metrics": state_metrics,
            "online_change_points": change_points,
            "estimation": (
                "HMM fits on training only; test states use forward filtering from prior draws."
            ),
        },
        "random_controls": random_controls,
        "random_control_replicates": len(random_controls),
        "random_control_max_test_score": {
            "one_number": max(one_control_scores, default=0.0),
            "six_number_set": max(set_control_scores, default=0.0),
        },
        "random_control_empirical_p_value": {
            "one_number": _empirical_p(float(one_number.selected.test_score), one_control_scores),
            "six_number_set": _empirical_p(
                float(set_model.selected.test_score), set_control_scores
            ),
        },
        "p_value_resolution": 1 / (1 + len(random_controls)) if random_controls else None,
        "inference_status": (
            "exploratory_elite_control_limited"
            if elite_seeds
            else "tentative"
            if len(random_controls) < 99
            else "empirical_controls_adequate"
        ),
        "progress": events,
    }
    store.save(output)
    if pattern_library is not None:
        output["pattern_library_candidates"] = register_experiment_candidates(
            output, pattern_library
        )
        output["full_set_match_discoveries"] = register_experiment_exact_matches(
            output, pattern_library
        )
        store.save(output)
    return output

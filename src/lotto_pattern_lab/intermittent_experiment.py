"""Full intermittent-rule experiments with matched random controls and causal regimes."""

import math
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
    PRIMITIVE_SET_VERSION,
    feature_mapping,
    feature_schema_version,
)
from .features import Example, build_bonus_examples, build_examples, feature_names
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
from .prize_evaluation import classify_prize_tier, evaluate_prize_predictions
from .research_features import (
    ResearchFeatureConfig,
    build_integrated_bonus_examples,
    build_integrated_examples,
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


def _annotate_candidate_prizes(
    candidate: dict[str, object], dataset: DrawDataset, split_rounds: dict[str, set[int]]
) -> None:
    draws = {record.round_number: record.draw for record in dataset.records}
    per_rule: dict[str, dict[str, dict[str, int]]] = {}
    for event in candidate.get("activation_events", []):
        if not event.get("active") or len(event.get("predicted_numbers", [])) != 6:
            continue
        round_number = int(event["round_number"])
        draw = draws[round_number]
        classification = classify_prize_tier(
            event["predicted_numbers"], tuple(draw.numbers), draw.bonus
        )
        event["actual_bonus_number"] = draw.bonus
        event["prize_tier"] = classification["prize_tier"]
        event["prize_tier_status"] = classification["prize_tier_status"]
        rule_key = str(event.get("rule_index", 0))
        phase = next(
            (name for name, values in split_rounds.items() if round_number in values), None
        )
        if phase is None:
            continue
        group = per_rule.setdefault(
            rule_key,
            {
                name: {"hits": 0, "known": 0, "unknown": 0}
                for name in split_rounds
            },
        )[phase]
        if draw.bonus is None:
            group["unknown"] += 1
        else:
            group["known"] += 1
            group["hits"] += int(classification["prize_tier"] == 2)
    candidate["second_prize_by_rule_split"] = per_rule


def _splits_for(dataset: DrawDataset, config: IntermittentConfig) -> tuple[tuple, DataSplits]:
    if config.feature_profile == "integrated_v1":
        examples = build_integrated_examples(dataset, config.max_lag)
    elif config.feature_profile == "calendar_round":
        examples = build_research_examples(
            dataset,
            config.max_lag,
            ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
        )
    elif config.feature_profile == "bonus_lag_v1":
        main_examples = build_examples(dataset, config.max_lag)
        bonus_examples = build_bonus_examples(dataset, config.max_lag)
        examples = tuple(
            Example(main.round_number, bonus.features, main.winning_numbers)
            for main, bonus in zip(main_examples, bonus_examples, strict=True)
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
    candidate: IntermittentCandidate,
    state_by_round: dict[int, str],
    dataset: DrawDataset,
) -> list[dict[str, object]]:
    draws_by_round = {record.round_number: record.draw for record in dataset.records}
    rows = []
    for event in candidate.activation_events:
        round_number = int(event["round_number"])
        draw = draws_by_round[round_number]
        row = {**event, "hypothetical_state": state_by_round[round_number]}
        if candidate.outputs_per_event == 6 and event.get("active"):
            tier = classify_prize_tier(
                event["predicted_numbers"], tuple(draw.numbers), draw.bonus
            )
            row["actual_bonus_number"] = draw.bonus
            row["prize_tier"] = tier["prize_tier"]
            row["prize_tier_status"] = tier["prize_tier_status"]
        rows.append(row)
    return rows


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
    bonus_model = None
    bonus_elite_seeds: list[dict[str, object]] = []
    bonus_elite_skipped: list[dict[str, str]] = []
    if config.feature_profile in {"bonus_lag_v1", "integrated_v1"}:
        if config.elite_seeding_enabled:
            bonus_library = pattern_library or PatternLibrary()
            bonus_elite_seeds, bonus_elite_skipped = bonus_library.select_elite_candidates(
                feature_profile=config.feature_profile,
                max_lag=config.max_lag,
                max_candidates=math.ceil(config.population_size * config.elite_seed_ratio) * 2,
                target_model_type="bonus_number",
            )
        bonus_examples = build_bonus_examples(dataset, config.max_lag)
        bonus_splits = chronological_split(
            bonus_examples, config.train_ratio, config.validation_ratio
        )
        bonus_config = IntermittentConfig(
            **{
                **asdict(config),
                "seed": config.seed + 2,
            }
        )
        bonus_model = run_intermittent_search(
            bonus_splits,
            bonus_config,
            outputs_per_event=1,
            progress=record_progress,
            run_label="actual_bonus_number",
            elite_seeds=tuple(bonus_elite_seeds),
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
    prize_split_rounds = {
        "training": {example.round_number for example in splits.train},
        "validation": {example.round_number for example in splits.validation},
        "independent_test": {example.round_number for example in splits.test},
    }
    for saved_candidate in set_dict["candidates"]:
        _annotate_candidate_prizes(saved_candidate, dataset, prize_split_rounds)
    _annotate_candidate_prizes(set_dict["selected"], dataset, prize_split_rounds)
    bonus_dict = _search_dict(bonus_model) if bonus_model else None
    if bonus_dict is not None and bonus_model is not None:
        bonus_dict["elite_seeding_compatibility"] = {
            "target_model_type": "bonus_number",
            "source_pattern_ids": bonus_model.seeding.get("source_pattern_ids", []),
            "skipped": bonus_elite_skipped,
        }
    one_dict["selected"]["activation_events"] = _with_state_tags(
        one_number.selected, state_by_round, dataset
    )
    set_dict["selected"]["activation_events"] = _with_state_tags(
        set_model.selected, state_by_round, dataset
    )
    bonus_events = {
        int(event["round_number"]): event
        for event in bonus_model.selected.activation_events
        if int(event["round_number"]) in set_model.test_rounds
    } if bonus_model is not None else {}
    main_events = {
        int(event["round_number"]): event
        for event in set_model.selected.activation_events
        if int(event["round_number"]) in set_model.test_rounds
        and event.get("active")
        and len(event.get("predicted_numbers", ())) == 6
    }
    prize_tier_backtest = evaluate_prize_predictions(
        dataset,
        {
            round_number: event["predicted_numbers"]
            for round_number, event in main_events.items()
        },
        {
            round_number: (
                bonus_events[round_number]["predicted_numbers"][0]
                if round_number in bonus_events and bonus_events[round_number]["predicted_numbers"]
                else None
            )
            for round_number in main_events
        },
    )
    prize_tier_backtest["evaluation_kind"] = (
        "chronological_backtest_not_pre_draw_predictions"
    )
    if bonus_model is not None:
        prize_tier_backtest["bonus_elite_seeding"] = {
            "enabled": config.elite_seeding_enabled,
            "source_pattern_ids": bonus_model.seeding.get("source_pattern_ids", []),
            "skipped": bonus_elite_skipped,
            "cross_target_patterns_allowed": False,
        }
    state_metrics = {
        "one_number_test": _state_conditioned_metrics(one_number.selected, splits, state_by_round),
        "six_number_test": _state_conditioned_metrics(set_model.selected, splits, state_by_round),
    }

    random_controls: list[dict[str, object]] = []
    for control_index in range(config.control_runs):
        data_seed = config.seed + 100_003 + control_index
        control_dataset = make_random_dataset(
            len(dataset.records),
            data_seed,
            first_round=dataset.records[0].round_number,
            include_bonus=config.feature_profile in {"bonus_lag_v1", "integrated_v1"},
        )
        if config.feature_profile in {"calendar_round", "integrated_v1"}:
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
        control_bonus = None
        if config.feature_profile in {"bonus_lag_v1", "integrated_v1"}:
            bonus_control_examples = (
                build_integrated_bonus_examples(control_dataset, config.max_lag)
                if config.feature_profile == "integrated_v1"
                else build_bonus_examples(control_dataset, config.max_lag)
            )
            bonus_control_splits = chronological_split(
                bonus_control_examples, config.train_ratio, config.validation_ratio
            )
            control_bonus = run_intermittent_search(
                bonus_control_splits,
                IntermittentConfig(
                    **{**asdict(control_config), "seed": control_config.seed + 2}
                ),
                outputs_per_event=1,
                progress=record_progress,
                run_label=f"random_{control_index + 1}_bonus_number",
            )
        control_states, _ = _hypothetical_states(control_splits, control_config)
        random_controls.append(
            {
                "replicate": control_index + 1,
                "data_seed": data_seed,
                "search_seed": control_config.seed,
                "one_number": _search_dict(control_one),
                "six_number_set": _search_dict(control_set),
                "bonus_number": _search_dict(control_bonus) if control_bonus else None,
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
            "feature_schema_version": feature_schema_version(config.feature_profile),
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
                | (set(bonus_model.seeding.get("source_pattern_ids", ())) if bonus_model else set())
            ),
            "one_number_search": one_number.seeding,
            "six_number_set_search": set_model.seeding,
            "bonus_number_search": bonus_model.seeding if bonus_model else None,
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
                    if config.feature_profile in {"calendar_round", "integrated_v1"}
                    else ()
                )
            + (
                tuple(f"lag{lag}_bonus" for lag in range(1, config.max_lag + 1))
                if config.feature_profile in {"bonus_lag_v1", "integrated_v1"}
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
        "bonus_number_model": bonus_dict,
        "prize_tier_backtest": prize_tier_backtest,
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

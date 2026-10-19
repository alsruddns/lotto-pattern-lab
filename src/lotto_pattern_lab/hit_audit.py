"""Read-only forensic audit of saved GP predictions against a draw dataset."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from .dataset import DrawDataset, load_csv
from .features import build_features_for_target, feature_names
from .paths import project_path
from .prediction import evaluate_formula, evaluate_gate
from .research import dataset_digest
from .research_features import (
    ResearchFeatureConfig,
    build_research_features_for_target,
    research_feature_names,
)
from .research_rounds import ResearchRoundStore
from .storage import ExperimentStore


def _event_phase(event: dict[str, Any], splits: dict[str, Any]) -> str | None:
    round_number = int(event["round_number"])
    for phase, key in (
        ("training", "train_rounds"),
        ("validation", "validation_rounds"),
        ("independent_test", "test_rounds"),
    ):
        bounds = splits.get(key)
        if bounds and int(bounds[0]) <= round_number <= int(bounds[1]):
            return phase
    return None


def _main_draw_date_digest(dataset: DrawDataset) -> str:
    payload = [
        {
            "round": row.round_number,
            "date": row.draw_date.isoformat() if row.draw_date else None,
            "numbers": row.draw.numbers,
        }
        for row in dataset.records
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_elite_seed_sources(
    record: dict[str, Any],
    *,
    target_dataset_hashes: set[str],
    target_validation_ranges: set[tuple[int, int]],
    target_test_ranges: set[tuple[int, int]],
) -> dict[str, Any]:
    pattern_ids = record.get("elite_seeding", {}).get("source_pattern_ids", [])
    if not pattern_ids:
        return {"source_pattern_count": 0, "patterns": []}
    library_path = project_path("data/database/patterns.sqlite")
    if not library_path.is_file():
        return {
            "source_pattern_count": len(pattern_ids),
            "patterns": [],
            "error": "pattern library missing",
        }
    uri = f"{library_path.as_uri()}?mode=ro"
    patterns: list[dict[str, Any]] = []
    with sqlite3.connect(uri, uri=True) as connection:
        for pattern_id in pattern_ids:
            row = connection.execute(
                "SELECT payload FROM patterns WHERE pattern_id=?", (str(pattern_id),)
            ).fetchone()
            if row:
                patterns.append(json.loads(row[0]))
    store = ExperimentStore()
    experiment_cache: dict[str, dict[str, Any] | None] = {}
    entries = []
    for pattern in patterns:
        experiment_id = str(pattern.get("experiment_id", ""))
        if experiment_id not in experiment_cache:
            try:
                experiment_cache[experiment_id] = store.load(experiment_id)
            except (OSError, ValueError, KeyError):
                experiment_cache[experiment_id] = None
        experiment = experiment_cache.get(experiment_id) or {}
        splits = experiment.get("splits", {})
        validation = tuple(splits.get("validation_rounds", ()))
        test = tuple(splits.get("test_rounds", ()))
        candidate_model = experiment.get("six_number_set_model", {})
        if pattern.get("model") == "one_number_model":
            candidate_model = experiment.get("one_number_model", {})
        selected = candidate_model.get("selected", {})
        source_validation_hits = []
        matching_validation_hits = []
        source_test_hits = []
        outputs = int(selected.get("outputs_per_event", 1))
        formulas = selected.get("formulas", [])
        for event in selected.get("activation_events", []):
            round_number = int(event.get("round_number", -1))
            active = bool(event.get("active"))
            hit_count = int(event.get("hit_count") or 0)
            if validation and validation[0] <= round_number <= validation[1] and active:
                if hit_count >= 5:
                    source_validation_hits.append(round_number)
                rule_index = int(event.get("rule_index") or 1)
                start = (rule_index - 1) * outputs
                branch = formulas[start : start + outputs]
                if branch == pattern.get("formulas") and hit_count >= 5:
                    matching_validation_hits.append(round_number)
            if test and test[0] <= round_number <= test[1] and active and hit_count >= 4:
                source_test_hits.append(round_number)
        source_hash = str(experiment.get("dataset", {}).get("dataset_sha256", ""))
        entries.append(
            {
                "pattern_id": pattern.get("pattern_id"),
                "experiment_id": experiment_id,
                "model": pattern.get("model"),
                "feature_profile": pattern.get("feature_profile", "legacy"),
                "activation_condition": pattern.get("activation_condition"),
                "formulas": pattern.get("formulas", []),
                "same_dataset_hash": source_hash in target_dataset_hashes,
                "same_validation_split": validation in target_validation_ranges,
                "same_test_split": test in target_test_ranges,
                "test_set_used_for_selection": splits.get("test_set_used_for_selection"),
                "validation_5_plus_rounds": sorted(set(source_validation_hits)),
                "seed_formula_validation_5_plus_rounds": sorted(set(matching_validation_hits)),
                "test_4_plus_rounds": sorted(set(source_test_hits)),
            }
        )
    return {"source_pattern_count": len(pattern_ids), "patterns": entries}


def _target_features(
    experiment: dict[str, Any], dataset: DrawDataset, target_round: int
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    config = experiment.get("config", {})
    max_lag = int(config.get("max_lag", 10))
    profile = str(config.get("feature_profile", "legacy"))
    if profile == "legacy":
        return build_features_for_target(dataset, target_round, max_lag), feature_names(max_lag)
    if profile == "bonus_lag_v1":
        features = build_features_for_target(dataset, target_round, max_lag)
        prior = {row.round_number: row.draw.bonus for row in dataset.records}
        bonuses = tuple(float(prior[target_round - lag]) for lag in range(1, max_lag + 1))
        return features + bonuses, (
            *feature_names(max_lag),
            *(f"lag{lag}_bonus" for lag in range(1, max_lag + 1)),
        )
    if profile == "calendar_round":
        target_date = next(
            (row.draw_date for row in dataset.records if row.round_number == target_round), None
        )
        config_features = ResearchFeatureConfig(include_calendar=True, include_round_digits=True)
        return (
            build_research_features_for_target(
                dataset, target_round, target_date, max_lag, config_features
            ),
            (*feature_names(max_lag), *research_feature_names(config_features)),
        )
    raise ValueError(f"지원하지 않는 특징 프로파일: {profile}")


def _reproduce(
    selected: dict[str, Any], features: tuple[float, ...], names: tuple[str, ...]
) -> tuple[list[int], int]:
    gate = evaluate_gate(
        float(evaluate_formula(selected["activation_condition"], features, names)["value"])
    )
    rule_index = int(gate["rule_index"])
    if not rule_index:
        return [], rule_index
    output_count = int(selected["outputs_per_event"])
    start = (rule_index - 1) * output_count
    predicted: list[int] = []
    for formula in selected["formulas"][start : start + output_count]:
        number = int(evaluate_formula(str(formula), features, names)["number"])
        while number in predicted:
            number = number % 45 + 1
        predicted.append(number)
    return predicted, rule_index


def audit_research_round(
    research_id: str,
    dataset: DrawDataset | None = None,
    *,
    experiment_store: ExperimentStore | None = None,
    rounds_store: ResearchRoundStore | None = None,
) -> dict[str, Any]:
    """Recount all linked experiments without writing to any source study files."""
    record_store = rounds_store or ResearchRoundStore()
    record = record_store.get(research_id)
    if record is None:
        raise KeyError(f"연구 ID를 찾을 수 없습니다: {research_id}")
    if dataset is None:
        dataset = load_csv(project_path("data/raw/lotto_draws_753_1246_with_bonus.csv"))
    actual_by_round = {row.round_number: tuple(sorted(row.draw.numbers)) for row in dataset.records}
    linked_ids = record.get("completed_experiment_ids") or record.get("experiment_ids") or []
    store = experiment_store or ExperimentStore()
    result: dict[str, Any] = {
        "research_id": research_id,
        "research_round": record.get("research_round"),
        "linked_experiment_count": len(linked_ids),
        "loaded_experiment_count": 0,
        "experiment_errors": [],
        "dataset_verification_status": dataset.verification_status,
        "dataset_source_kind": dataset.source_kind,
        "dataset_round_range": [dataset.records[0].round_number, dataset.records[-1].round_number],
        "recorded_dataset_sha256": record.get("dataset_sha256"),
        "current_main_draw_date_sha256": _main_draw_date_digest(dataset),
        "current_full_dataset_sha256": dataset_digest(dataset),
        "main_draw_date_hash_matches_study": (
            record.get("dataset_sha256") == _main_draw_date_digest(dataset)
        ),
        "experiment_dataset_sha256s": [],
        "models": {},
    }
    experiment_hashes: set[str] = set()
    target_dataset_hashes: set[str] = set()
    target_validation_ranges: set[tuple[int, int]] = set()
    target_test_ranges: set[tuple[int, int]] = set()
    feature_profiles: set[str] = set()
    test_selection_flags: set[bool] = set()
    for model_key, model_label in (
        ("one_number_model", "one_number"),
        ("six_number_set_model", "six_number_set"),
    ):
        repeated: dict[str, Counter[str]] = defaultdict(Counter)
        unique_rounds: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        high_hits: list[dict[str, Any]] = []
        formula_rows: dict[str, dict[str, Any]] = {}
        formula_hit_groups: dict[str, dict[str, Any]] = {}
        comparison_counts = Counter()
        formula_reproduced = 0
        formula_compared = 0
        winning_verified = 0
        winning_compared = 0
        prediction_verified = 0
        prediction_compared = 0
        for experiment_id in linked_ids:
            try:
                experiment = store.load(str(experiment_id))
            except (OSError, ValueError, KeyError) as error:
                result["experiment_errors"].append(
                    {"experiment_id": experiment_id, "error": str(error)}
                )
                continue
            if model_label == "one_number":
                result["loaded_experiment_count"] += 1
            experiment_hash = experiment.get("dataset", {}).get("dataset_sha256")
            if experiment_hash:
                experiment_hashes.add(str(experiment_hash))
                target_dataset_hashes.add(str(experiment_hash))
            experiment_splits = experiment.get("splits", {})
            for key, destination in (
                ("validation_rounds", target_validation_ranges),
                ("test_rounds", target_test_ranges),
            ):
                bounds = experiment_splits.get(key)
                if bounds and len(bounds) == 2:
                    destination.add((int(bounds[0]), int(bounds[1])))
            feature_profiles.add(str(experiment.get("config", {}).get("feature_profile", "legacy")))
            if "test_set_used_for_selection" in experiment.get("splits", {}):
                test_selection_flags.add(bool(experiment["splits"]["test_set_used_for_selection"]))
            model = experiment.get(model_key, {})
            selected = model.get("selected", {})
            if not selected:
                result["experiment_errors"].append(
                    {"experiment_id": experiment_id, "error": f"{model_key}.selected 누락"}
                )
                continue
            signature = (
                str(selected.get("activation_condition"))
                + "\n"
                + "\n".join(str(value) for value in selected.get("formulas", []))
            )
            formula_item = formula_rows.setdefault(
                signature,
                {
                    "activation_condition": selected.get("activation_condition"),
                    "formulas": selected.get("formulas", []),
                    "experiment_ids": [],
                    "independent_test_4_plus": 0,
                    "independent_test_5_plus": 0,
                    "independent_test_6_exact": 0,
                    "independent_test_hit_rounds": set(),
                    "all_4_plus_rounds": set(),
                    "validation_5_plus": 0,
                    "validation_4_plus": 0,
                    "validation_4_plus_rounds": set(),
                    "validation_hit_rounds": set(),
                    "feature_profile": experiment.get("config", {}).get(
                        "feature_profile", "legacy"
                    ),
                },
            )
            formula_item["experiment_ids"].append(str(experiment_id))
            mapping = experiment.get("feature_variables", {})
            if "feature_variables" not in result:
                result["feature_variables"] = mapping
            formula_item["variables"] = {
                name: mapping.get(name, "알 수 없음")
                for name in _referenced_variables(
                    [
                        str(selected.get("activation_condition", "")),
                        *map(str, selected.get("formulas", [])),
                    ]
                )
            }
            splits = experiment.get("splits", {})
            events = selected.get("activation_events", [])
            for event in events:
                round_number = int(event["round_number"])
                phase = _event_phase(event, splits)
                if phase is None:
                    continue
                recorded_winners = tuple(
                    sorted(int(number) for number in event.get("winning_numbers", []))
                )
                actual_winners = actual_by_round.get(round_number)
                if actual_winners is not None:
                    winning_compared += 1
                    winning_verified += int(recorded_winners == actual_winners)
                if not event.get("active"):
                    continue
                predicted = tuple(
                    sorted(int(number) for number in event.get("predicted_numbers", []))
                )
                actual_hits = len(set(predicted) & set(actual_winners or ()))
                recorded_hits = int(event.get("hit_count") or 0)
                prediction_compared += 1
                prediction_verified += int(
                    actual_winners is not None and actual_hits == recorded_hits
                )
                output_count = int(selected.get("outputs_per_event", 1))
                rule_index = int(event.get("rule_index") or 1)
                formulas = selected.get("formulas", [])
                start = (rule_index - 1) * output_count
                active_formulas = formulas[start : start + output_count]
                if (phase == "independent_test" and actual_hits >= 4) or (
                    phase == "validation" and actual_hits >= 4
                ):
                    group_key = json.dumps(
                        [selected.get("activation_condition"), rule_index, active_formulas],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    group = formula_hit_groups.setdefault(
                        group_key,
                        {
                            "activation_condition": selected.get("activation_condition"),
                            "rule_index": rule_index,
                            "active_formulas": active_formulas,
                            "independent_test_4_plus": 0,
                            "validation_4_plus": 0,
                            "validation_5_plus": 0,
                            "independent_test_rounds": set(),
                            "validation_rounds": set(),
                            "experiment_ids": set(),
                            "predictions": set(),
                            "variables": {
                                name: mapping.get(name, "알 수 없음")
                                for name in _referenced_variables(
                                    [
                                        str(selected.get("activation_condition", "")),
                                        *map(str, active_formulas),
                                    ]
                                )
                            },
                        },
                    )
                    group["experiment_ids"].add(str(experiment_id))
                    group["predictions"].add(tuple(sorted(predicted)))
                    if phase == "independent_test":
                        group["independent_test_4_plus"] += 1
                        group["independent_test_rounds"].add(round_number)
                    else:
                        group["validation_4_plus"] += 1
                        group["validation_rounds"].add(round_number)
                        if actual_hits >= 5:
                            group["validation_5_plus"] += 1
                for threshold in (3, 4, 5):
                    if actual_hits >= threshold:
                        repeated[phase][f"{threshold}_plus"] += 1
                        unique_rounds[phase][f"{threshold}_plus"].add(round_number)
                        if threshold == 5 and phase == "validation":
                            formula_item["validation_5_plus"] += 1
                            formula_item["validation_hit_rounds"].add(round_number)
                        if threshold == 4 and phase == "validation":
                            formula_item["validation_4_plus"] += 1
                            formula_item["validation_4_plus_rounds"].add(round_number)
                            formula_item["all_4_plus_rounds"].add(round_number)
                        if threshold == 4 and phase == "independent_test":
                            formula_item["independent_test_4_plus"] += 1
                            formula_item["independent_test_hit_rounds"].add(round_number)
                            formula_item["all_4_plus_rounds"].add(round_number)
                        if threshold == 5 and phase == "independent_test":
                            formula_item["independent_test_5_plus"] += 1
                        if (threshold == 5 and phase == "validation") or (
                            threshold == 4 and phase == "independent_test"
                        ):
                            high_hits.append(
                                {
                                    "experiment_id": str(experiment_id),
                                    "model": model_label,
                                    "phase": phase,
                                    "round_number": round_number,
                                    "predicted_numbers": list(predicted),
                                    "winning_numbers": list(actual_winners or recorded_winners),
                                    "hit_count": actual_hits,
                                    "recorded_hit_count": recorded_hits,
                                    "activation_condition": selected.get("activation_condition"),
                                    "formulas": selected.get("formulas", []),
                                    "rule_index": event.get("rule_index"),
                                    "active_formulas": active_formulas,
                                    "used_feature_variables": {
                                        name: mapping.get(name, "알 수 없음")
                                        for name in _referenced_variables(
                                            [
                                                str(selected.get("activation_condition", "")),
                                                *map(
                                                    str,
                                                    active_formulas,
                                                ),
                                            ]
                                        )
                                    },
                                }
                            )
                if actual_hits == 6:
                    repeated[phase]["6_exact"] += 1
                    unique_rounds[phase]["6_exact"].add(round_number)
                    if phase == "independent_test":
                        formula_item["independent_test_6_exact"] += 1
                try:
                    features, names = _target_features(experiment, dataset, round_number)
                    recreated, recreated_rule = _reproduce(selected, features, names)
                    formula_compared += 1
                    if sorted(recreated) == sorted(
                        int(number) for number in event["predicted_numbers"]
                    ) and recreated_rule == int(event.get("rule_index") or 0):
                        formula_reproduced += 1
                    else:
                        comparison_counts["formula_mismatch"] += 1
                        comparison_counts["first_formula_mismatch_round"] = round_number
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    comparison_counts["formula_errors"] += 1
                    comparison_counts["last_formula_error"] = str(error)
        ranked_formulas = []
        for item in formula_rows.values():
            test_rounds = item.pop("independent_test_hit_rounds")
            validation_rounds = item.pop("validation_hit_rounds")
            validation_4_plus_rounds = item.pop("validation_4_plus_rounds")
            all_4_plus_rounds = item.pop("all_4_plus_rounds")
            ranked_formulas.append(
                {
                    **item,
                    "independent_test_unique_hit_rounds": sorted(test_rounds),
                    "independent_test_unique_4_plus_rounds": len(test_rounds),
                    "validation_unique_5_plus_rounds": len(validation_rounds),
                    "validation_unique_hit_rounds": sorted(validation_rounds),
                    "validation_unique_4_plus_rounds": len(validation_4_plus_rounds),
                    "all_split_unique_4_plus_rounds": sorted(all_4_plus_rounds),
                    "all_split_unique_4_plus_count": len(all_4_plus_rounds),
                }
            )
        hit_formula_rankings = []
        for group in formula_hit_groups.values():
            test_rounds = group.pop("independent_test_rounds")
            validation_rounds = group.pop("validation_rounds")
            group["experiment_ids"] = sorted(group["experiment_ids"])
            group["predictions"] = [list(value) for value in sorted(group["predictions"])]
            hit_formula_rankings.append(
                {
                    **group,
                    "independent_test_unique_rounds": sorted(test_rounds),
                    "validation_unique_rounds": sorted(validation_rounds),
                }
            )
        hit_formula_rankings.sort(
            key=lambda item: (
                item["independent_test_4_plus"],
                item["validation_4_plus"],
            ),
            reverse=True,
        )
        result["models"][model_label] = {
            "repeated_hit_counts": {phase: dict(counts) for phase, counts in repeated.items()},
            "unique_round_counts": {
                phase: {threshold: len(rounds) for threshold, rounds in thresholds.items()}
                for phase, thresholds in unique_rounds.items()
            },
            "unique_hit_rounds": {
                phase: {threshold: sorted(rounds) for threshold, rounds in thresholds.items()}
                for phase, thresholds in unique_rounds.items()
            },
            "high_hits": high_hits,
            "hit_formula_rankings": hit_formula_rankings,
            "winning_numbers_match_official": {
                "verified": winning_verified,
                "compared": winning_compared,
            },
            "recorded_hit_count_matches_official": {
                "verified": prediction_verified,
                "compared": prediction_compared,
            },
            "formula_prediction_reproduction": {
                "reproduced": formula_reproduced,
                "compared": formula_compared,
                **dict(comparison_counts),
            },
            "formula_rankings": sorted(
                ranked_formulas,
                key=lambda item: (
                    item["independent_test_4_plus"],
                    item["independent_test_unique_4_plus_rounds"],
                    item["independent_test_5_plus"],
                ),
                reverse=True,
            ),
        }
    result["leakage_review"] = {
        "feature_profiles": sorted(feature_profiles),
        "test_set_used_for_selection_values": sorted(test_selection_flags),
        "interpretation": (
            "입력 특징은 target_round 이전 회차만 사용합니다. "
            "저장 당첨번호는 사후 대조에만 씁니다. "
            "test_set_used_for_selection 필드는 원본에서 읽었습니다."
        ),
    }
    result["experiment_dataset_sha256s"] = sorted(experiment_hashes)
    result["elite_seed_provenance"] = _audit_elite_seed_sources(
        record,
        target_dataset_hashes=target_dataset_hashes,
        target_validation_ranges=target_validation_ranges,
        target_test_ranges=target_test_ranges,
    )
    seed_patterns = result["elite_seed_provenance"].get("patterns", [])
    result["elite_seed_provenance"]["same_dataset_hash_count"] = sum(
        bool(item.get("same_dataset_hash")) for item in seed_patterns
    )
    result["elite_seed_provenance"]["same_validation_split_count"] = sum(
        bool(item.get("same_validation_split")) for item in seed_patterns
    )
    result["elite_seed_provenance"]["same_test_split_count"] = sum(
        bool(item.get("same_test_split")) for item in seed_patterns
    )
    for model in result["models"].values():
        for group in model.get("hit_formula_rankings", []):
            group["elite_seed_pattern_ids"] = [
                item["pattern_id"]
                for item in seed_patterns
                if item.get("activation_condition") == group.get("activation_condition")
                and item.get("formulas") == group.get("active_formulas")
            ]
    return result


def build_hit_audit_markdown(audit: dict[str, Any], *, detailed: bool = False) -> str:
    """Create a grounded ChatGPT report from a read-only audit result."""
    experiment_count = audit.get("linked_experiment_count", "미수집")
    loaded_count = audit.get("loaded_experiment_count", "미수집")
    verification = audit.get("dataset_verification_status", "미수집")
    source_kind = audit.get("dataset_source_kind", "미수집")
    round_range = audit.get("dataset_round_range", "미수집")
    lines = [
        "# 연구 고적중 정밀 재검증",
        "",
        f"- 연구 회차: {audit.get('research_round', '미수집')}회차",
        f"- 연구 ID: {audit.get('research_id', '미수집')}",
        f"- 연결/상세 로드 실험: {experiment_count} / {loaded_count}",
        f"- 기준 데이터: {source_kind} · 검증 상태 {verification}",
        f"- 기준 회차 범위: {round_range}",
        f"- 연구 저장 해시: {audit.get('recorded_dataset_sha256', '미수집')}",
        "- 현재 메인 번호·날짜 해시 일치: "
        f"{audit.get('main_draw_date_hash_matches_study', '미수집')}",
        "",
        "## 회차 단위 재집계",
        "",
    ]
    for model, label in (("six_number_set", "6개 집합 모델"), ("one_number", "한 번호 모델")):
        data = audit.get("models", {}).get(model, {})
        lines.append(f"### {label}")
        if not data:
            lines.append("- 미수집")
            continue
        repeated = data.get("repeated_hit_counts", {})
        unique = data.get("unique_round_counts", {})
        for phase, phase_label in (
            ("validation", "검증 구간"),
            ("independent_test", "독립 테스트 구간"),
        ):
            values = repeated.get(phase, {})
            distinct = unique.get(phase, {})
            repeated_counts = [
                values.get("3_plus", 0),
                values.get("4_plus", 0),
                values.get("5_plus", 0),
                values.get("6_exact", 0),
            ]
            unique_counts = [
                distinct.get("3_plus", 0),
                distinct.get("4_plus", 0),
                distinct.get("5_plus", 0),
                distinct.get("6_exact", 0),
            ]
            lines.append(f"- {phase_label}: 반복 평가 3+/4+/5+/6개 = {repeated_counts}")
            lines.append(f"  고유 회차 3+/4+/5+/6개 = {unique_counts}")
        formula_check = data.get("formula_prediction_reproduction", {})
        winner_check = data.get("winning_numbers_match_official", {})
        lines.append(
            f"- 공식 재실행 일치: {formula_check.get('reproduced', 0)} / "
            f"{formula_check.get('compared', 0)} 활성 이벤트"
        )
        lines.append(
            f"- 저장 당첨번호 대조: {winner_check.get('verified', 0)} / "
            f"{winner_check.get('compared', 0)} 이벤트"
        )
    leakage = audit.get("leakage_review", {})
    elite_sources = audit.get("elite_seed_provenance", {})
    profiles = ", ".join(leakage.get("feature_profiles", [])) or "미수집"
    test_selection = leakage.get("test_set_used_for_selection_values", "미수집")
    variable_map = audit.get("feature_variables", {})
    lines.extend(
        [
            "",
            "## 데이터 누출 및 해석",
            "",
            f"- 사용 프로파일: {profiles}",
            f"- test_set_used_for_selection 기록: {test_selection}",
            f"- Elite 원본 패턴: {elite_sources.get('source_pattern_count', '미수집')}개 · "
            f"동일 데이터 해시 {elite_sources.get('same_dataset_hash_count', '미수집')}개 · "
            f"동일 검증/테스트 분할 {elite_sources.get('same_validation_split_count', '미수집')}/"
            f"{elite_sources.get('same_test_split_count', '미수집')}개",
            "- 예측 입력은 대상 회차 전의 추첨만 사용합니다.",
            "- legacy 입력에는 대상 회차의 날짜와 보너스가 없습니다.",
            f"- 변수 의미: x0={variable_map.get('x0', '미수집')}, "
            f"x57={variable_map.get('x57', '미수집')} (10회 전 번호의 쌍별 합)",
            "- 검증 결과에는 후보 선택 편향이 포함됩니다.",
            "- 같은 검증 분할에서 성능을 얻은 Elite 원본도 있어 검증 적중은 독립 검증이 아닙니다.",
            "- 테스트 점수는 선택에 쓰지 않았지만 같은 회차가 원본 연구에 이미 기록돼 있습니다.",
            "- 따라서 59건을 독립 표본이나 연구 전체의 독립 테스트 성과로 볼 수 없습니다.",
            "- 반복 실험 적중 수와 고유 회차 수를 따로 해석해야 합니다.",
            "",
            "## 독립 테스트 4개 이상 적중 공식 순위",
            "",
        ]
    )
    ranked = audit.get("models", {}).get("six_number_set", {}).get("hit_formula_rankings", [])
    ranked = [item for item in ranked if item.get("independent_test_4_plus", 0)]
    limit = len(ranked) if detailed else min(10, len(ranked))
    if not ranked:
        lines.append("- 해당 없음")
    for index, item in enumerate(ranked[:limit], 1):
        lines.extend(
            [
                f"{index}. 독립 테스트 4+ 반복: {item.get('independent_test_4_plus')}회",
                f"   - 검증 5+ 반복: {item.get('validation_5_plus')}회",
                f"   - 테스트 회차: {item.get('independent_test_unique_rounds')}",
                f"   - 검증 4+ 회차: {item.get('validation_unique_rounds')}",
                f"   - 실험 ID: {item.get('experiment_ids')}",
                f"   - 일치 Elite 원본 패턴: {item.get('elite_seed_pattern_ids')}",
                f"   - 활성화 조건: `{item.get('activation_condition')}`",
                f"   - 실제 분기 수식: `{item.get('active_formulas')}`",
                f"   - 변수 의미: `{item.get('variables')}`",
            ]
        )
    seed_evidence = [
        item
        for item in elite_sources.get("patterns", [])
        if item.get("seed_formula_validation_5_plus_rounds") or item.get("test_4_plus_rounds")
    ]
    lines.extend(["", "## Elite 원본 수식의 과거 적중", ""])
    if seed_evidence:
        for item in seed_evidence:
            lines.append(f"- 패턴 `{item.get('pattern_id')}` / 실험 `{item.get('experiment_id')}`")
            lines.append(
                f"  - 원본 수식 검증 5+ 회차: {item.get('seed_formula_validation_5_plus_rounds')}"
            )
            lines.append(f"  - 원본 실험 테스트 4+ 회차: {item.get('test_4_plus_rounds')}")
            lines.append(
                "  - 같은 데이터·검증·테스트 분할: "
                f"{item.get('same_dataset_hash')}/"
                f"{item.get('same_validation_split')}/"
                f"{item.get('same_test_split')}"
            )
    else:
        lines.append("- 일치하는 원본 수식의 고적중 기록 미수집")
    if detailed:
        lines.extend(["", "## 각 고적중 이벤트", ""])
        for item in audit.get("models", {}).get("six_number_set", {}).get("high_hits", []):
            if item.get("phase") == "independent_test" and item.get("hit_count", 0) >= 4:
                lines.extend(
                    [
                        f"- {item['round_number']}회 · {item['hit_count']}개",
                        f"  - 예측 {item['predicted_numbers']} / 당첨 {item['winning_numbers']}",
                        f"  - 실험 {item['experiment_id']}",
                        f"  - 게이트 `{item.get('activation_condition')}`",
                        f"  - 분기 {item.get('rule_index')}",
                        f"  - 수식 `{item.get('formulas')}`",
                    ]
                )
            elif item.get("phase") == "validation":
                lines.extend(
                    [
                        f"- 검증 {item['round_number']}회 · {item['hit_count']}개",
                        f"  - 예측 {item['predicted_numbers']}",
                        f"  - 당첨 {item['winning_numbers']}",
                        f"  - 실험 {item['experiment_id']}",
                    ]
                )
    return "\n".join(lines)


def _referenced_variables(formulas: list[str]) -> set[str]:
    import re

    return set(re.findall(r"\bx\d+\b", " ".join(formulas)))

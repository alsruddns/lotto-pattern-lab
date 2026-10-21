"""Evidence-filtered replay of completed GP and PySR research candidates."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .dataset import DrawDataset
from .prediction import (
    create_pysr_recommendation,
    create_recommendation,
    pysr_replay_readiness,
)


def _split_provenance(experiment: dict[str, Any]) -> tuple[bool, int | None, int | None]:
    splits = experiment.get("splits")
    if not isinstance(splits, dict):
        return False, None, None
    train, validation, test = (
        splits.get("train_rounds"),
        splits.get("validation_rounds"),
        splits.get("test_rounds"),
    )
    if all(isinstance(value, list) and len(value) >= 2 for value in (train, validation, test)):
        try:
            ordered = max(train) < min(validation) and max(validation) < min(test)
        except (TypeError, ValueError):
            return False, None, None
        if not ordered:
            return False, None, None
        return True, len(validation), len(test)
    refs = (train, validation, test)
    if all(isinstance(value, dict) and value.get("$detail_ref") for value in refs):
        counts = (experiment.get("storage") or {}).get("detail_collection_counts") or {}
        return (
            True,
            counts.get("/splits/validation_rounds"),
            counts.get("/splits/test_rounds"),
        )
    return False, None, None


def _has_metric(candidate: dict[str, Any], *keys: str) -> bool:
    return any(candidate.get(key) is not None for key in keys)


def list_research_recommendation_candidates(
    experiment_directory: str | Path,
    *,
    feature_profile: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """List supported six-number candidates with validation and held-out evidence.

    GP and PySR optimize different metrics, so this function preserves recent-first
    order within each engine and never compares their raw scores.
    """
    root = Path(experiment_directory)
    if limit < 1 or not root.is_dir():
        return []
    paths = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            experiment = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if experiment.get("status") != "completed":
            continue
        config = experiment.get("config")
        if not isinstance(config, dict):
            continue
        profile = str(config.get("feature_profile", "legacy"))
        if feature_profile is not None and profile != feature_profile:
            continue
        has_splits, validation_count, test_count = _split_provenance(experiment)
        if (
            not has_splits
            or experiment.get("independent_test_used_for_selection") is True
            or (experiment.get("splits") or {}).get("test_set_used_for_selection") is True
        ):
            continue
        common = {
            "experiment_id": str(experiment.get("experiment_id", path.stem)),
            "research_id": experiment.get("research_id"),
            "feature_profile": profile,
            "created_at": experiment.get("created_at") or experiment.get("ended_at"),
            "source_path": str(path),
            "validation_round_count": validation_count,
            "test_round_count": test_count,
            "evidence_level": "시간순 검증 및 홀드아웃 기록",
        }
        engine = str(experiment.get("engine", ""))
        if engine in {"intermittent_gp", "deap_intermittent_gp"}:
            model = experiment.get("six_number_set_model") or {}
            if not isinstance(model, dict):
                continue
            candidates = model.get("candidates") or [model.get("selected") or {}]
            for rank, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, dict):
                    continue
                if not (
                    _has_metric(candidate, "validation_score", "validation_hit_rate")
                    and _has_metric(
                        candidate,
                        "test_score",
                        "test_hit_rate",
                        "test_full_set_matches",
                    )
                ):
                    continue
                rows.append(
                    {
                        **common,
                        "engine": "intermittent_gp",
                        "candidate_rank": rank,
                        "validation_score": candidate.get("validation_score"),
                        "validation_hit_rate": candidate.get("validation_hit_rate"),
                        "test_score": candidate.get("test_score"),
                        "test_hit_rate": candidate.get("test_hit_rate"),
                    }
                )
                break
        elif engine == "pysr_symbolic_regression":
            if (
                config.get("target") != "main_positions"
                or pysr_replay_readiness(experiment).get("status") != "verified"
            ):
                continue
            targets = {f"main_position_{index}": [] for index in range(1, 7)}
            for candidate in experiment.get("candidates", []):
                if candidate.get("target") in targets:
                    targets[candidate["target"]].append(candidate)
            common_ranks = min((len(values) for values in targets.values()), default=0)
            for rank in range(1, common_ranks + 1):
                selected = [values[rank - 1] for values in targets.values()]
                if any(
                    item.get("classification") in {"INVALID", "REDUNDANT"}
                    or not isinstance(item.get("evaluation"), dict)
                    or not isinstance(item["evaluation"].get("validation"), dict)
                    or not isinstance(item["evaluation"].get("independent_test"), dict)
                    or not isinstance(item["evaluation"]["validation"].get("mae"), (int, float))
                    or not isinstance(
                        item["evaluation"]["independent_test"].get("mae"), (int, float)
                    )
                    or not math.isfinite(item["evaluation"]["validation"]["mae"])
                    or not math.isfinite(item["evaluation"]["independent_test"]["mae"])
                    for item in selected
                ):
                    continue
                validation_maes = [
                    float(item["evaluation"]["validation"]["mae"]) for item in selected
                ]
                test_maes = [
                    float(item["evaluation"]["independent_test"]["mae"])
                    for item in selected
                ]
                rows.append(
                    {
                        **common,
                        "engine": engine,
                        "candidate_rank": rank,
                        "validation_score": None,
                        "validation_hit_rate": None,
                        "test_score": None,
                        "test_hit_rate": None,
                        "position_validation_mae": fmean(validation_maes),
                        "position_holdout_mae": fmean(test_maes),
                    }
                )
                break
    return rows


def _interleave_engines(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.get("engine")), []).append(candidate)
    result: list[dict[str, Any]] = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for engine in sorted(groups):
            if index < len(groups[engine]):
                result.append(groups[engine][index])
    return result


def generate_research_recommendations(
    candidates: list[dict[str, Any]],
    dataset: DrawDataset,
    target_round: int,
    count: int,
    *,
    feature_profile: str,
    data_file_sha256: str | None = None,
    random_seed: int = 0,
    official_latest_round: int | None = None,
    target_date=None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Replay compatible stored formulas into unique, unsaved recommendation drafts."""
    if count not in {1, 3, 5, 10}:
        raise ValueError("추천 조합 수는 1, 3, 5, 10 중에서 선택해야 합니다.")
    from .storage import ExperimentStore

    drafts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    by_numbers: dict[tuple[int, ...], dict[str, Any]] = {}
    selected_research_ids: set[str] = set()
    max_sources = max(20, count * 10)
    for candidate in _interleave_engines(candidates)[:max_sources]:
        if candidate.get("feature_profile") != feature_profile:
            continue
        experiment_id = str(candidate.get("experiment_id", ""))
        try:
            source_path = Path(candidate["source_path"])
            experiment = ExperimentStore(source_path.parent).load(experiment_id)
            if experiment.get("status") != "completed":
                raise ValueError("원본 연구가 완료 상태가 아닙니다.")
            provenance, _, _ = _split_provenance(experiment)
            if not provenance:
                raise ValueError("시간순 학습·검증·홀드아웃 구간을 확인할 수 없습니다.")
            rank = int(candidate["candidate_rank"])
            if candidate.get("engine") == "pysr_symbolic_regression":
                recommendation = create_pysr_recommendation(
                    experiment,
                    dataset,
                    target_round,
                    candidate_rank=rank,
                    data_file_sha256=data_file_sha256,
                    random_seed=random_seed + len(drafts),
                    official_latest_round=official_latest_round,
                    target_date=target_date,
                )
            else:
                experiment = copy.deepcopy(experiment)
                model = experiment.get("six_number_set_model") or {}
                model_candidates = model.get("candidates") or [model.get("selected") or {}]
                if rank < 1 or rank > len(model_candidates):
                    raise ValueError("저장된 GP 후보 순위를 찾을 수 없습니다.")
                model["selected"] = model_candidates[rank - 1]
                experiment["six_number_set_model"] = model
                experiment["selected_candidate_rank"] = rank
                recommendation = create_recommendation(
                    experiment,
                    dataset,
                    target_round,
                    "six_number_set",
                    data_file_sha256=data_file_sha256,
                    random_seed=random_seed + len(drafts),
                    official_latest_round=official_latest_round,
                    target_date=target_date,
                )
            if recommendation.get("recommendation_status") != "pre_draw_prediction":
                raise ValueError("추첨 전 사전 추천 상태로 확인되지 않았습니다.")
            numbers = tuple(sorted(int(value) for value in recommendation["recommended_numbers"]))
            invalid_numbers = (
                len(numbers) != 6
                or len(set(numbers)) != 6
                or any(not 1 <= number <= 45 for number in numbers)
            )
            if invalid_numbers:
                raise ValueError("저장된 공식이 유효한 6개 번호를 만들지 못했습니다.")
            source = {
                "experiment_id": experiment_id,
                "research_id": candidate.get("research_id"),
                "engine": candidate.get("engine"),
                "feature_profile": feature_profile,
                "candidate_rank": rank,
                "validation_score": candidate.get("validation_score"),
                "validation_hit_rate": candidate.get("validation_hit_rate"),
                "test_score": candidate.get("test_score"),
                "test_hit_rate": candidate.get("test_hit_rate"),
                "position_validation_mae": candidate.get("position_validation_mae"),
                "position_holdout_mae": candidate.get("position_holdout_mae"),
                "validation_round_count": candidate.get("validation_round_count"),
                "test_round_count": candidate.get("test_round_count"),
                "evidence_level": candidate.get("evidence_level"),
                "dataset_sha256": experiment.get("dataset_sha256"),
            }
            recommendation["research_selection"] = {
                "strategy_version": "research_candidate_replay_v1",
                "selection_reason": (
                    "프로파일 호환, 저장 수식 재실행 가능, 시간순 검증 및 홀드아웃 기록 보유. "
                    "GP와 PySR의 서로 다른 점수는 직접 비교하지 않았습니다."
                ),
                "supporting_candidates": [source],
                "actual_win_probability_claimed": False,
            }
            existing = by_numbers.get(numbers)
            if existing is not None:
                existing["research_selection"]["supporting_candidates"].append(source)
                continue
            research_id = str(candidate.get("research_id") or "")
            if research_id and research_id in selected_research_ids:
                continue
            by_numbers[numbers] = recommendation
            drafts.append(recommendation)
            if research_id:
                selected_research_ids.add(research_id)
            if len(drafts) >= count:
                break
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            failures.append({"experiment_id": experiment_id, "reason": str(error)})
    return drafts, failures

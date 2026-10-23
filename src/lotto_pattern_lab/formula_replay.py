"""Immutable full historical replay for saved intermittent GP formulas."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deap import gp

from .atomic_io import atomic_write_bytes
from .dataset import DrawDataset
from .feature_schema import PRIMITIVE_SET_VERSION, feature_mapping
from .features import build_bonus_examples, build_examples
from .gp_engine import _make_primitive_set
from .intermittent import _gate_rule, _predict_numbers, _validate_formula_ast
from .prize_evaluation import classify_prize_tier
from .research_features import (
    ResearchFeatureConfig,
    build_integrated_examples,
    build_research_examples,
)

REPLAY_VERSION = "formula-full-historical-replay-v2"
DEFAULT_START_ROUND = 753
DEFAULT_END_ROUND = 1246
_RANDOM_MEAN_HITS_SIX_NUMBERS = 36 / 45


def _examples_for_profile(dataset: DrawDataset, profile: str, max_lag: int):
    if profile == "integrated_v1":
        return build_integrated_examples(dataset, max_lag)
    if profile == "calendar_round":
        return build_research_examples(
            dataset,
            max_lag,
            ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
        )
    if profile == "bonus_lag_v1":
        main = build_examples(dataset, max_lag)
        bonus = build_bonus_examples(dataset, max_lag)
        return tuple(
            type(left)(left.round_number, right.features, left.winning_numbers)
            for left, right in zip(main, bonus, strict=True)
        )
    return build_examples(dataset, max_lag)


def _score_segment(round_number: int, model: dict[str, Any]) -> str:
    for label in ("training", "validation", "test"):
        rounds = model.get(f"{label}_rounds", ())
        if round_number in rounds:
            return {
                "training": "학습",
                "validation": "후보 선택 검증",
                "test": "원본 실험 테스트",
            }[label]
    return "연구에서 확인한 과거 회차"


def _formula_functions(
    candidate: dict[str, Any],
    profile: str,
    max_lag: int,
    primitive_version: str,
    original_mapping: dict[str, Any],
):
    if primitive_version != PRIMITIVE_SET_VERSION:
        raise ValueError(f"지원하지 않는 연산자 버전: {primitive_version}")
    formulas = candidate.get("formulas")
    outputs = int(candidate.get("outputs_per_event", 0))
    if outputs != 6 or int(candidate.get("rule_count", 0)) != 2:
        raise ValueError("전체 회차 재검증은 규칙 2개를 가진 6개 집합 GP 수식만 지원합니다.")
    if not isinstance(formulas, list) or len(formulas) != 12:
        raise ValueError("저장된 수식이 게이트 1개와 규칙별 출력 12개인지 확인할 수 없습니다.")
    mapping = feature_mapping(profile, max_lag)
    if not original_mapping:
        raise ValueError(
            "원본 실험에 저장된 특징 매핑이 없어 수식 입력을 안전하게 복원할 수 없습니다."
        )
    if original_mapping != mapping:
        raise ValueError("원본 실험의 특징 매핑이 현재 프로파일과 달라 재검증을 중단했습니다.")
    pset = _make_primitive_set(len(mapping))
    parsed: list[str] = [str(candidate.get("activation_condition", "")), *map(str, formulas)]
    trees = []
    for formula in parsed:
        _validate_formula_ast(formula, pset)
        trees.append(gp.PrimitiveTree.from_string(formula, pset))
    return tuple(gp.compile(tree, pset) for tree in trees)


def replay_formula(
    *,
    dataset: DrawDataset,
    experiment: dict[str, Any],
    model_key: str,
    candidate: dict[str, Any],
    formula_id: str,
    start_round: int = DEFAULT_START_ROUND,
    end_round: int = DEFAULT_END_ROUND,
    output_directory: str | Path = "data/replays",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay a frozen saved candidate. Source experiment JSON is never modified."""
    if model_key != "six_number_set_model":
        raise ValueError("6개 집합 모델만 전체 회차 6번호 재검증 대상입니다.")
    if start_round > end_round:
        raise ValueError("시작 회차는 종료 회차보다 클 수 없습니다.")
    config = experiment.get("config") or {}
    schema = experiment.get("feature_schema") or {}
    profile = str(schema.get("feature_profile", config.get("feature_profile", "legacy")))
    max_lag = int(config.get("max_lag", 10))
    functions = _formula_functions(
        candidate,
        profile,
        max_lag,
        str(schema.get("primitive_set_version", PRIMITIVE_SET_VERSION)),
        schema.get("feature_mapping") or {},
    )
    gate, *predictors = functions
    examples_by_round = {
        example.round_number: example
        for example in _examples_for_profile(dataset, profile, max_lag)
    }
    records_by_round = {record.round_number: record for record in dataset.records}
    target_rounds = list(range(start_round, end_round + 1))
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    matches: Counter[int] = Counter()
    second_prize_hits = 0
    second_prize_known = 0
    second_prize_unknown = 0
    training_rounds = set((experiment.get(model_key) or {}).get("train_rounds", ()))
    validation_rounds = set((experiment.get(model_key) or {}).get("validation_rounds", ()))
    test_rounds = set((experiment.get(model_key) or {}).get("test_rounds", ()))
    model_rounds = {
        "training_rounds": training_rounds,
        "validation_rounds": validation_rounds,
        "test_rounds": test_rounds,
    }

    started = time.monotonic()

    def emit_progress(row_index: int) -> None:
        if not progress_callback or (row_index % 10 and row_index != len(target_rounds)):
            return
        elapsed = time.monotonic() - started
        rate = row_index / elapsed if elapsed > 0 else 0.0
        progress_callback(
            {
                "processed_rounds": row_index,
                "total_rounds": len(target_rounds),
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    (len(target_rounds) - row_index) / rate if rate > 0 else None
                ),
            }
        )

    for row_index, round_number in enumerate(target_rounds, start=1):
        record = records_by_round.get(round_number)
        if record is None:
            counts["INPUT_UNAVAILABLE"] += 1
            rows.append(
                {
                    "round_number": round_number,
                    "formula_id": formula_id,
                    "feature_profile": profile,
                    "activation_condition_met": None,
                    "execution_success": False,
                    "predicted_numbers": [],
                    "winning_numbers": [],
                    "matched_numbers": [],
                    "match_count": None,
                    "actual_bonus_number": None,
                    "prize_tier": None,
                    "prize_tier_status": "판정 불가",
                    "evaluation_segment": _score_segment(round_number, model_rounds),
                    "validation_status": "INPUT_UNAVAILABLE",
                    "reason": "대상 회차 또는 이전 회차 데이터가 스냅샷에 없습니다.",
                    "draw_date": None,
                }
            )
            emit_progress(row_index)
            continue
        example = examples_by_round.get(round_number)
        if example is None:
            counts["INPUT_UNAVAILABLE"] += 1
            rows.append(
                {
                    "round_number": round_number,
                    "draw_date": record.draw_date.isoformat() if record.draw_date else None,
                    "formula_id": formula_id,
                    "feature_profile": profile,
                    "activation_condition_met": None,
                    "execution_success": False,
                    "predicted_numbers": [],
                    "winning_numbers": sorted(record.draw.numbers),
                    "matched_numbers": [],
                    "match_count": None,
                    "actual_bonus_number": record.draw.bonus,
                    "prize_tier": None,
                    "prize_tier_status": "판정 불가",
                    "evaluation_segment": _score_segment(round_number, model_rounds),
                    "validation_status": "INPUT_UNAVAILABLE",
                    "reason": f"{max_lag}회 lag 입력이 없거나 연속 회차가 아닙니다.",
                }
            )
            emit_progress(row_index)
            continue
        try:
            rule_index = _gate_rule(gate, example.features)
            if rule_index:
                first = (rule_index - 1) * 6
                predicted = _predict_numbers(tuple(predictors[first : first + 6]), example.features)
            else:
                predicted = ()
            if rule_index and (len(predicted) != 6 or len(set(predicted)) != 6):
                raise ValueError("원본 후처리 결과가 고유 번호 6개가 아닙니다.")
            winning = set(record.draw.numbers)
            matched = sorted(set(predicted) & winning)
            match_count = len(matched) if rule_index else None
            status = "OK" if rule_index else "INACTIVE"
            if rule_index:
                counts["EVALUATED"] += 1
                counts["ACTIVE_PREDICTION"] += 1
            else:
                counts["EVALUATED"] += 1
                counts["INACTIVE"] += 1
            if rule_index:
                matches[match_count] += 1
                tier = classify_prize_tier(predicted, tuple(record.draw.numbers), record.draw.bonus)
                if record.draw.bonus is None:
                    second_prize_unknown += 1
                else:
                    second_prize_known += 1
                    second_prize_hits += int(tier["prize_tier"] == 2)
            else:
                tier = None
            rows.append(
                {
                    "round_number": round_number,
                    "draw_date": record.draw_date.isoformat() if record.draw_date else None,
                    "formula_id": formula_id,
                    "feature_profile": profile,
                    "activation_condition_met": bool(rule_index),
                    "selected_rule": rule_index or None,
                    "execution_success": True,
                    "predicted_numbers": list(predicted),
                    "winning_numbers": sorted(winning),
                    "matched_numbers": matched,
                    "match_count": match_count,
                    "actual_bonus_number": record.draw.bonus,
                    "prize_tier": tier["prize_tier"] if tier is not None else None,
                    "prize_tier_status": (
                        tier["prize_tier_status"] if tier is not None else "활성 예측 없음"
                    ),
                    "evaluation_segment": _score_segment(round_number, model_rounds),
                    "validation_status": status,
                    "reason": None,
                }
            )
        except Exception as error:  # preserve a row and continue the full range
            counts["EXECUTION_ERROR"] += 1
            rows.append(
                {
                    "round_number": round_number,
                    "draw_date": record.draw_date.isoformat() if record.draw_date else None,
                    "formula_id": formula_id,
                    "feature_profile": profile,
                    "activation_condition_met": None,
                    "execution_success": False,
                    "predicted_numbers": [],
                    "winning_numbers": sorted(record.draw.numbers),
                    "matched_numbers": [],
                    "match_count": None,
                    "actual_bonus_number": record.draw.bonus,
                    "prize_tier": None,
                    "prize_tier_status": "판정 불가",
                    "evaluation_segment": _score_segment(round_number, model_rounds),
                    "validation_status": "EXECUTION_ERROR",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        emit_progress(row_index)

    evaluable = counts["EVALUATED"]
    active_predictions = counts["ACTIVE_PREDICTION"]
    distribution = {str(index): matches[index] for index in range(7)}
    record_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "round": item.round_number,
                    "numbers": item.draw.numbers,
                    "bonus": item.draw.bonus,
                    "date": item.draw_date.isoformat() if item.draw_date else None,
                }
                for item in dataset.records
                if start_round <= item.round_number <= end_round
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": REPLAY_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "Full Historical Replay",
        "formula_id": formula_id,
        "source_experiment_id": experiment.get("experiment_id"),
        "research_id": experiment.get("research_id"),
        "model_key": model_key,
        "feature_profile": profile,
        "primitive_set_version": schema.get("primitive_set_version", PRIMITIVE_SET_VERSION),
        "range": {"start": start_round, "end": end_round},
        "target_round_count": len(target_rounds),
        "evaluable_round_count": evaluable,
        "active_prediction_count": active_predictions,
        "input_unavailable_count": counts["INPUT_UNAVAILABLE"],
        "execution_error_count": counts["EXECUTION_ERROR"],
        "inactive_round_count": counts["INACTIVE"],
        "match_distribution_exact": distribution,
        "at_least_3": sum(matches[index] for index in range(3, 7)),
        "at_least_4": sum(matches[index] for index in range(4, 7)),
        "at_least_5": sum(matches[index] for index in range(5, 7)),
        "at_least_6": matches[6],
        "second_prize_exact": second_prize_hits if second_prize_known else None,
        "second_prize_evaluable_count": second_prize_known,
        "second_prize_unavailable_count": second_prize_unknown,
        "mean_hits_when_active": (
            sum(index * matches[index] for index in range(7)) / active_predictions
            if active_predictions
            else None
        ),
        "random_baseline_mean_hits": _RANDOM_MEAN_HITS_SIX_NUMBERS,
        "mean_hits_minus_random_baseline": (
            sum(index * matches[index] for index in range(7)) / active_predictions
            - _RANDOM_MEAN_HITS_SIX_NUMBERS
        )
        if active_predictions
        else None,
        "runtime_seconds": time.monotonic() - started,
        "dataset_snapshot_hash": record_hash,
        "rows": rows,
    }
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['run_id']}.json"
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        raise FileExistsError(path)
    atomic_write_bytes(path, payload)
    result["result_path"] = str(path)
    return result

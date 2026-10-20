"""Optional supervised baselines for six-number set prediction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .atomic_io import atomic_write_bytes
from .backtest import chronological_split
from .dataset import DrawDataset
from .experiment import dataset_metadata
from .features import build_examples
from .intermittent import IntermittentConfig
from .paths import project_path
from .storage import ExperimentStore

MODEL_IDS = ("random_forest", "xgboost", "torch_mlp")
MODEL_EXTRAS = {
    "random_forest": ".[classical]",
    "xgboost": ".[boosting]",
    "torch_mlp": ".[neural]",
}


def model_availability() -> dict[str, dict[str, Any]]:
    modules = {"random_forest": "sklearn", "xgboost": "xgboost", "torch_mlp": "torch"}
    return {
        model_id: {
            "available": importlib.util.find_spec(module_name) is not None,
            "module": module_name,
            "install_extra": MODEL_EXTRAS[model_id],
        }
        for model_id, module_name in modules.items()
    }


def _targets(examples) -> np.ndarray:
    targets = np.zeros((len(examples), 45), dtype=np.float32)
    for index, example in enumerate(examples):
        for number in example.winning_numbers:
            targets[index, int(number) - 1] = 1.0
    return targets


def _build_estimator(model_id: str, *, seed: int, workers: int, input_width: int):
    if model_id == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=128,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=min(max(1, workers), 4),
            random_state=seed,
        )
    if model_id == "xgboost":
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=96,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=seed,
                verbosity=0,
            ),
            n_jobs=1,
        )
    if model_id == "torch_mlp":
        import torch
        from torch import nn

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(input_width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 45),
            nn.Sigmoid(),
        )
    raise ValueError(f"지원하지 않는 모델입니다: {model_id}")


def _fit_torch(model, train_x: np.ndarray, train_y: np.ndarray, seed: int) -> None:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device="cpu")
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(30):
        order = torch.randperm(len(x_tensor), generator=generator)
        for batch in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()


def _predict_scores(model_id: str, model: Any, features: np.ndarray) -> np.ndarray:
    if model_id == "torch_mlp":
        import torch

        with torch.no_grad():
            return model(torch.as_tensor(features, dtype=torch.float32, device="cpu")).numpy()
    return np.asarray(model.predict(features), dtype=float)


def _evaluate_rows(examples, scores: np.ndarray) -> dict[str, Any]:
    distributions = Counter()
    predictions = []
    for example, row_scores in zip(examples, scores, strict=True):
        chosen = tuple(sorted((np.argsort(row_scores, kind="stable")[-6:] + 1).tolist()))
        matched = sorted(set(chosen) & set(example.winning_numbers))
        distributions[len(matched)] += 1
        predictions.append(
            {
                "round_number": example.round_number,
                "predicted_numbers": list(chosen),
                "winning_numbers": sorted(example.winning_numbers),
                "matched_numbers": matched,
                "match_count": len(matched),
                "number_scores": [float(value) for value in row_scores],
            }
        )
    count = len(predictions)
    return {
        "evaluation_round_count": count,
        "average_hits": (
            sum(item["match_count"] for item in predictions) / count if count else None
        ),
        "at_least_3": sum(value for hits, value in distributions.items() if hits >= 3),
        "at_least_4": sum(value for hits, value in distributions.items() if hits >= 4),
        "at_least_5": sum(value for hits, value in distributions.items() if hits >= 5),
        "exactly_6": distributions[6],
        "match_distribution": {str(hits): distributions[hits] for hits in range(7)},
        "predictions": predictions,
    }


def run_tabular_model(
    dataset: DrawDataset,
    model_id: str,
    *,
    config: IntermittentConfig | None = None,
    directory: str | Path = "experiments/model_comparisons",
    model_directory: str | Path = "models",
) -> dict[str, Any]:
    """Fit one optional model on the chronological training split and save its evaluation."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"지원하는 모델은 {', '.join(MODEL_IDS)}입니다.")
    availability = model_availability()[model_id]
    if not availability["available"]:
        raise RuntimeError(
            f"{model_id} 모델에 필요한 의존성이 없습니다. 설치 선택 항목: "
            f"{availability['install_extra']}"
        )
    config = config or IntermittentConfig(workers=1)
    examples = build_examples(dataset, config.max_lag)
    splits = chronological_split(examples, config.train_ratio, config.validation_ratio)
    features_by_round = {item.round_number: item.features for item in examples}
    train_x = np.asarray([features_by_round[item.round_number] for item in splits.train])
    train_y = _targets(splits.train)
    model = _build_estimator(
        model_id,
        seed=config.seed,
        workers=config.workers,
        input_width=int(train_x.shape[1]),
    )
    started = time.perf_counter()
    if model_id == "torch_mlp":
        _fit_torch(model, train_x, train_y, config.seed)
    else:
        model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - started
    model_root = project_path(model_directory)
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / f"{model_id}_{uuid.uuid4()}.pkl"
    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(model_path, model_bytes)

    def predict_split(split) -> dict[str, Any]:
        matrix = np.asarray([features_by_round[item.round_number] for item in split])
        return _evaluate_rows(split, _predict_scores(model_id, model, matrix))

    result = {
        "experiment_id": str(uuid.uuid4()),
        "experiment_type": "supervised_baseline",
        "engine": model_id,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "round": record.round_number,
                        "date": record.draw_date.isoformat() if record.draw_date else None,
                        "numbers": record.draw.numbers,
                        "bonus": record.draw.bonus,
                    }
                    for record in dataset.records
                ],
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "dataset": dataset_metadata(dataset),
        "code_version": "tabular-baseline-v1",
        "target_type": "six_number_set",
        "target_encoding": "45-output multi-label presence scores; top six unique main numbers",
        "data_round_range": [
            dataset.records[0].round_number,
            dataset.records[-1].round_number,
        ],
        "dataset_source_kind": dataset.source_kind,
        "config": {
            "max_lag": config.max_lag,
            "train_ratio": config.train_ratio,
            "validation_ratio": config.validation_ratio,
            "seed": config.seed,
            "workers": min(max(1, config.workers), 4),
        },
        "split_rounds": {
            "training": [splits.train[0].round_number, splits.train[-1].round_number],
            "validation": [
                splits.validation[0].round_number,
                splits.validation[-1].round_number,
            ],
            "independent_test": [splits.test[0].round_number, splits.test[-1].round_number],
            "test_used_for_selection": False,
        },
        "training_seconds": training_seconds,
        "model_path": str(model_path),
        "feature_count": int(train_x.shape[1]),
        "model_size_bytes": len(model_bytes),
        "validation": predict_split(splits.validation),
        "independent_test": predict_split(splits.test),
        "limitations": [
            "이 모델의 과거 백테스트는 미래 당첨확률을 보장하지 않습니다.",
            "훈련 대상은 45개 번호의 회차별 출현 여부이며, 최종 출력은 점수 상위 6개입니다.",
        ],
    }
    ExperimentStore(directory).save(result)
    return result


def validation_ensemble_weights(
    results: dict[str, dict[str, Any]], model_ids: tuple[str, ...] | None = None
) -> dict[str, float]:
    """Derive nonnegative combination weights using validation average hits only."""
    selected = model_ids or tuple(results)
    raw = {
        model_id: max(
            0.001,
            float((results[model_id].get("validation") or {}).get("average_hits") or 0.0),
        )
        for model_id in selected
        if model_id in results
    }
    total = sum(raw.values())
    return {model_id: value / total for model_id, value in raw.items()} if total else {}


def evaluate_weighted_ensemble(
    results: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    phase: str = "independent_test",
) -> dict[str, Any]:
    """Combine saved 45-number scores; caller must derive weights from validation only."""
    if phase not in {"validation", "independent_test"}:
        raise ValueError("앙상블 평가 구간은 validation 또는 independent_test여야 합니다.")
    active_weights = {
        key: float(value)
        for key, value in weights.items()
        if key in results and float(value) >= 0
    }
    if not active_weights or sum(active_weights.values()) <= 0:
        raise ValueError("양수 가중치가 있는 비교 모델이 없습니다.")
    total_weight = sum(active_weights.values())
    active_weights = {key: value / total_weight for key, value in active_weights.items()}
    by_model = {
        model_id: {
            int(row["round_number"]): row
            for row in (results[model_id].get(phase) or {}).get("predictions", [])
        }
        for model_id in active_weights
    }
    common_rounds = set.intersection(*(set(rows) for rows in by_model.values()))
    combined = []
    for round_number in sorted(common_rounds):
        base = next(iter(by_model.values()))[round_number]
        score_vector = np.zeros(45, dtype=float)
        for model_id, weight in active_weights.items():
            row = by_model[model_id][round_number]
            values = row.get("number_scores")
            if not values or len(values) != 45:
                values = [
                    1.0 if number in row.get("predicted_numbers", []) else 0.0
                    for number in range(1, 46)
                ]
            score_vector += weight * np.asarray(values, dtype=float)
        prediction = sorted((np.argsort(score_vector, kind="stable")[-6:] + 1).tolist())
        winning = sorted(map(int, base["winning_numbers"]))
        matched = sorted(set(prediction) & set(winning))
        combined.append(
            {
                "round_number": round_number,
                "predicted_numbers": prediction,
                "winning_numbers": winning,
                "matched_numbers": matched,
                "match_count": len(matched),
            }
        )
    distribution = Counter(row["match_count"] for row in combined)
    count = len(combined)
    return {
        "phase": phase,
        "evaluation_round_count": count,
        "average_hits": sum(row["match_count"] for row in combined) / count if count else None,
        "at_least_3": sum(value for hits, value in distribution.items() if hits >= 3),
        "at_least_4": sum(value for hits, value in distribution.items() if hits >= 4),
        "at_least_5": sum(value for hits, value in distribution.items() if hits >= 5),
        "exactly_6": distribution[6],
        "weights": active_weights,
        "weight_selection_phase": "validation",
        "independent_test_used_for_weight_selection": False,
        "predictions": combined,
    }


def load_model_artifact(path: str | Path) -> Any:
    """Load a locally created trusted model artifact."""
    source = project_path(path)
    with source.open("rb") as model_file:
        return pickle.load(model_file)

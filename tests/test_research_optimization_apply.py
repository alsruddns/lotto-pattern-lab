from __future__ import annotations

from pathlib import Path

from lotto_pattern_lab.research_optimization_apply import (
    OPTUNA_TO_RESEARCH_KEYS,
    changed_optimization_settings,
    optimization_origin_record,
    validate_optimization_compatibility,
)


def _study() -> dict[str, object]:
    return {
        "study_name": "gp-tuning-test",
        "optuna_study_id": 42,
        "status": "completed",
        "engine": "intermittent_gp",
        "dataset_range": [753, 1246],
        "feature_profile": "bonus_lag_v1",
        "best_trial_number": 2,
        "best_parameters": {
            "population_size": 56,
            "generations": 14,
            "crossover_probability": 0.77,
            "mutation_probability": 0.39,
            "max_depth": 10,
            "max_lag": 8,
            "min_activation_rate": 0.21,
        },
    }


def _compatibility(study: dict[str, object], engines: tuple[str, ...] = ("intermittent_gp",)):
    return validate_optimization_compatibility(
        study,
        current_engines=engines,
        current_dataset_range=[753, 1246],
        dataset_integrity={"all_draws_have_bonus": True, "all_draws_dated": True},
    )


def test_optimization_parameters_apply_and_normalize_for_gp() -> None:
    params, error = _compatibility(_study(), ("intermittent_gp", "pysr_symbolic_regression"))

    assert error is None
    assert params is not None
    assert params["population_size"] == 56
    assert isinstance(params["population_size"], int)
    assert set(params) == set(OPTUNA_TO_RESEARCH_KEYS)


def test_optimization_apply_rejects_engine_and_dataset_mismatch() -> None:
    params, error = _compatibility(_study(), ("pysr_symbolic_regression",))
    assert params is None
    assert error and "GP" in error

    changed = _study()
    changed["dataset_range"] = [1, 1246]
    params, error = _compatibility(changed)
    assert params is None
    assert error and "회차 범위" in error


def test_integrated_profile_optimization_requires_both_dates_and_bonus_values() -> None:
    study = _study()
    study["feature_profile"] = "integrated_v1"
    assert _compatibility(study)[1] is None

    params, error = validate_optimization_compatibility(
        study,
        current_engines=("intermittent_gp",),
        current_dataset_range=[753, 1246],
        dataset_integrity={"all_draws_have_bonus": True, "all_draws_dated": False},
    )
    assert params is None
    assert error is not None


def test_manual_edits_are_detected_without_changing_optuna_snapshot() -> None:
    applied = {key: 1 for key in OPTUNA_TO_RESEARCH_KEYS.values()}
    current = dict(applied)
    assert not changed_optimization_settings(applied, current)
    current["new_max_depth"] = 2
    assert changed_optimization_settings(applied, current)
    assert applied["new_max_depth"] == 1


def test_research_metadata_keeps_optuna_origin_snapshot() -> None:
    study = _study()
    parameters, error = _compatibility(study)
    assert error is None and parameters is not None
    settings = {target: parameters[source] for source, target in OPTUNA_TO_RESEARCH_KEYS.items()}
    origin = optimization_origin_record(
        study=study,  # type: ignore[arg-type]
        parameters=parameters,
        research_settings=settings,
        applied_at="2026-10-20T00:00:00+00:00",
        modified=True,
    )
    assert origin["optuna_study_id"] == 42
    assert origin["trial_number"] == 2
    assert origin["modified_by_user"] is True
    assert origin["parameters"] == parameters


def test_continuous_research_page_has_compact_setup_and_apply_state() -> None:
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    page = next((root / "src/lotto_pattern_lab/pages").glob("3_*.py"))
    controls = {
        "new_population_size": 56,
        "new_generations": 14,
        "new_crossover_probability": 0.77,
        "new_mutation_probability": 0.39,
        "new_max_depth": 10,
        "new_max_lag": 8,
        "new_min_activation_rate": 0.21,
        "new_profile": "bonus_lag_v1",
    }
    params = {source: controls[target] for source, target in OPTUNA_TO_RESEARCH_KEYS.items()}
    app = AppTest.from_file(page)
    app.session_state["optimization_apply_pending"] = {
        "values": controls,
        "context": {
            "study_name": "gp-tuning-test",
            "optuna_study_id": 42,
            "trial_number": 2,
            "parameters": params,
            "research_settings": controls,
            "expected_controls": controls,
            "feature_profile": "bonus_lag_v1",
            "dataset_range": [753, 1246],
            "applied_at": "2026-10-20T00:00:00+00:00",
            "started_research_ids": [],
            "application_verified": False,
        },
    }
    app.run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.session_state["optimization_applied_context"]["application_verified"] is True
    assert any(
        "최적화 설정이 적용되었습니다" in item.value for item in app.success
    )
    labels = {item.label for item in app.selectbox}
    assert "탐색 엔진" in labels
    assert "특징 프로파일" in labels
    assert "연구 시간 프리셋" in labels
    assert any(button.label == "새 연구 시작" for button in app.button)

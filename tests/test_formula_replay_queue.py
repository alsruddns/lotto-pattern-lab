from __future__ import annotations

import hashlib

from lotto_pattern_lab.formula_replay_queue import (
    _execute_job,
    discover_high_hit_candidates,
    enqueue_formula_replay,
    list_replay_jobs,
)
from lotto_pattern_lab.storage import ExperimentStore


def _candidate(rounds: tuple[int, ...]) -> dict:
    return {
        "activation_condition": "x0",
        "formulas": ["x0"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
        "at_least_5_hits_by_split": {"training": len(rounds)},
        "activation_events": [
            {"round_number": round_number, "hit_count": 5} for round_number in rounds
        ],
    }


def _save_experiment(root, experiment_id: str, rounds: tuple[int, ...]) -> None:
    record = {
        "experiment_id": experiment_id,
        "research_id": f"research-{experiment_id}",
        "experiment_type": "intermittent",
        "status": "completed",
        "config": {"feature_profile": "legacy", "max_lag": 10},
        "feature_schema": {"feature_profile": "legacy"},
        "six_number_set_model": {
            "candidates": [_candidate(rounds)],
            "selected": _candidate(rounds),
            "train_rounds": [1, 2],
            "validation_rounds": [3],
            "test_rounds": [4],
        },
    }
    ExperimentStore(root, minimum_compress_bytes=1).save(record)


def test_high_hit_candidates_deduplicate_formulas_and_distinguish_rounds(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    _save_experiment(experiments, "experiment-one", (10, 10))
    _save_experiment(experiments, "experiment-two", (10, 11))

    candidates = discover_high_hit_candidates(experiments, minimum_hits=5)

    assert len(candidates) == 1
    assert candidates[0]["origin_experiment_ids"] == ["experiment-one", "experiment-two"]
    assert candidates[0]["repeated_hit_events"] == 4
    assert candidates[0]["original_hit_rounds"] == [10, 11]


def test_replay_queue_persists_jobs_and_deduplicates(tmp_path) -> None:
    source = tmp_path / "draws.csv"
    source.write_text(
        "round,n1,n2,n3,n4,n5,n6\n1,1,2,3,4,5,6\n2,2,3,4,5,6,7\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = _candidate((10,))
    experiment = {
        "experiment_id": "one",
        "experiment_type": "intermittent",
        "status": "completed",
        "config": {"feature_profile": "legacy", "max_lag": 10},
        "feature_schema": {"feature_profile": "legacy"},
        "six_number_set_model": {"train_rounds": [], "validation_rounds": [], "test_rounds": []},
    }
    first = enqueue_formula_replay(
        dataset_path=source,
        dataset_sha256=digest,
        experiment=experiment,
        model_key="six_number_set_model",
        candidate=candidate,
        formula_id="formula-a",
        start_round=1,
        end_round=2,
        directory=tmp_path / "jobs",
        dispatch=False,
    )
    duplicate = enqueue_formula_replay(
        dataset_path=source,
        dataset_sha256=digest,
        experiment=experiment,
        model_key="six_number_set_model",
        candidate=candidate,
        formula_id="formula-b",
        start_round=1,
        end_round=2,
        directory=tmp_path / "jobs",
        dispatch=False,
    )

    assert first["status"] == "queued"
    assert duplicate["deduplicated"]
    assert duplicate["duplicate_request_count"] == 1
    assert len(list_replay_jobs(tmp_path / "jobs")) == 1


def test_replay_worker_persists_progress_and_completion(tmp_path, monkeypatch) -> None:
    import lotto_pattern_lab.formula_replay_queue as queue

    source = tmp_path / "draws.csv"
    source.write_text(
        "round,n1,n2,n3,n4,n5,n6\n1,1,2,3,4,5,6\n2,2,3,4,5,6,7\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    experiment = {
        "experiment_id": "worker-test",
        "config": {"feature_profile": "legacy", "max_lag": 10},
        "feature_schema": {"feature_profile": "legacy"},
        "six_number_set_model": {},
    }
    job = enqueue_formula_replay(
        dataset_path=source,
        dataset_sha256=digest,
        experiment=experiment,
        model_key="six_number_set_model",
        candidate=_candidate((1,)),
        formula_id="formula-worker",
        start_round=1,
        end_round=2,
        directory=tmp_path / "jobs",
        dispatch=False,
    )

    def fake_replay(**kwargs):
        kwargs["progress_callback"](
            {"processed_rounds": 2, "total_rounds": 2, "elapsed_seconds": 0.5}
        )
        return {
            "result_path": str(tmp_path / "result.json"),
            "run_id": "run-worker",
            "runtime_seconds": 0.5,
            "target_round_count": 2,
            "evaluable_round_count": 1,
            "active_prediction_count": 1,
            "input_unavailable_count": 1,
            "execution_error_count": 0,
            "at_least_3": 0,
            "at_least_4": 0,
            "at_least_5": 0,
            "at_least_6": 0,
            "mean_hits_when_active": 0.0,
        }

    monkeypatch.setattr(queue, "replay_formula", fake_replay)
    _execute_job(job["job_id"], tmp_path / "jobs")

    saved = list_replay_jobs(tmp_path / "jobs")[0]
    assert saved["status"] == "completed"
    assert saved["processed_rounds"] == 2
    assert saved["run_id"] == "run-worker"
    assert saved["result_summary"]["evaluable_round_count"] == 1

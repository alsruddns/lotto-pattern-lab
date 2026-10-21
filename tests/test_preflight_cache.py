from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from lotto_pattern_lab import preflight, preflight_cache
from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.preflight_cache import (
    load_preflight_cache,
    preflight_cache_key,
    save_preflight_cache,
)


def test_preflight_cache_persists_across_calls_and_keys_configuration(tmp_path) -> None:
    settings = {"data_sha256": "abc", "workers": 4}
    key = preflight_cache_key(settings)
    report = {"status": "passed", "engines": {"gp": {"status": "passed"}}}
    cache_path = tmp_path / "preflight.json"

    save_preflight_cache(key, report, cache_path)

    cached = load_preflight_cache(key, cache_path)
    assert cached["report"] == report
    assert cached["checked_at"]
    assert load_preflight_cache(preflight_cache_key({**settings, "workers": 2}), cache_path) is None


def test_corrupt_or_wrong_version_cache_is_ignored(tmp_path) -> None:
    cache_path = tmp_path / "preflight.json"
    cache_path.write_text('{"cache_version":99,"entries":{}}', encoding="utf-8")
    assert load_preflight_cache("key", cache_path) is None
    cache_path.write_text("not json", encoding="utf-8")
    assert load_preflight_cache("key", cache_path) is None


def test_success_and_failure_entries_use_different_ttls(tmp_path) -> None:
    cache_path = tmp_path / "preflight.json"
    key = "ttl-key"
    now = datetime.now(UTC)
    report = {"status": "passed"}
    save_preflight_cache(key, report, cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["entries"][key]["checked_at"] = (now - timedelta(hours=25)).isoformat()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_preflight_cache(key, cache_path) is None

    report = {"status": "failed"}
    save_preflight_cache(key, report, cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["entries"][key]["checked_at"] = (now - timedelta(minutes=3)).isoformat()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_preflight_cache(key, cache_path) is None


def test_julia_changes_invalidate_only_pysr_cache(monkeypatch) -> None:
    gp_settings = {"engines": ["intermittent_gp"]}
    pysr_settings = {"engines": ["pysr_symbolic_regression"]}
    first_gp = preflight_cache.preflight_cache_key(gp_settings)
    first_pysr = preflight_cache.preflight_cache_key(pysr_settings)
    monkeypatch.setenv("JULIA_PROJECT", "alternate-project")
    assert preflight_cache.preflight_cache_key(gp_settings) == first_gp
    assert preflight_cache.preflight_cache_key(pysr_settings) != first_pysr


def test_cache_key_rejects_runtime_objects_instead_of_stringifying() -> None:
    with pytest.raises(TypeError):
        preflight_cache.preflight_cache_key({"process": object()})


def test_corrupt_json_shape_is_ignored_and_recoverable(tmp_path) -> None:
    cache_path = tmp_path / "preflight.json"
    cache_path.write_text("[]", encoding="utf-8")
    assert load_preflight_cache("key", cache_path) is None
    save_preflight_cache("key", {"status": "passed"}, cache_path)
    assert load_preflight_cache("key", cache_path)["report"]["status"] == "passed"


def test_concurrent_cache_writes_keep_valid_entries(tmp_path) -> None:
    cache_path = tmp_path / "preflight.json"

    def save(index: int) -> None:
        save_preflight_cache(f"key-{index}", {"status": "passed", "index": index}, cache_path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(8)))

    for index in range(8):
        assert load_preflight_cache(f"key-{index}", cache_path)["report"]["index"] == index


def test_get_or_run_preflight_reuses_cache_and_force_refreshes(monkeypatch, tmp_path) -> None:
    dataset = make_random_dataset(24, 21)
    config = IntermittentConfig(population_size=8, generations=1, workers=1, control_runs=0)
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        return {"status": "passed", "engines": {"intermittent_gp": {"status": "passed"}}}

    monkeypatch.setattr(preflight, "run_research_preflight", fake_run)
    cache_path = tmp_path / "preflight.json"
    first = preflight.get_or_run_research_preflight(
        dataset, config, ("intermittent_gp",), cache_path=cache_path
    )
    second = preflight.get_or_run_research_preflight(
        dataset, config, ("intermittent_gp",), cache_path=cache_path
    )
    forced = preflight.get_or_run_research_preflight(
        dataset, config, ("intermittent_gp",), force=True, cache_path=cache_path
    )

    assert first["cache_used"] is False
    assert second["cache_used"] is True
    assert forced["cache_used"] is False
    assert len(calls) == 2


def test_research_preflight_key_tracks_only_smoke_relevant_gp_settings() -> None:
    dataset = make_random_dataset(24, 22)
    config = IntermittentConfig(population_size=8, generations=1, workers=1, control_runs=0)
    original = preflight.research_preflight_cache_key(
        dataset, config, ("intermittent_gp",)
    )
    cheaper = preflight.research_preflight_cache_key(
        dataset,
        replace(config, population_size=64, generations=20, workers=4),
        ("intermittent_gp",),
    )
    changed_smoke_input = preflight.research_preflight_cache_key(
        dataset, replace(config, max_depth=config.max_depth + 1), ("intermittent_gp",)
    )
    assert original == cheaper
    assert original != changed_smoke_input

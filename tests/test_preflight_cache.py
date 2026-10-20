from __future__ import annotations

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

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from lotto_pattern_lab.research_rounds import ResearchRoundStore
from lotto_pattern_lab.revalidation_catalog import (
    build_experiment_catalog,
    experiment_choice_label,
    filter_experiment_catalog,
    sort_experiment_catalog,
)


def test_catalog_uses_read_only_compact_summary_and_preserves_uuid_selection(
    tmp_path: Path,
) -> None:
    database = tmp_path / "patterns.sqlite"
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    (experiments / "exp-1.json").write_text(
        json.dumps({"experiment_id": "exp-1", "large_detail": [1, 2, 3]}), encoding="utf-8"
    )
    (experiments / "orphan.json").write_text("{}", encoding="utf-8")
    registry = ResearchRoundStore(database)
    registry.ensure_round(
        "research-1",
        {
            "status": "completed",
            "started_at": "2026-10-20T12:00:00+09:00",
            "config": {"feature_profile": "bonus_lag_v1"},
            "dashboard_summary": {
                "summary_schema_version": 3,
                "completed_experiment_count": 1,
                "experiment_summaries": [
                    {
                        "experiment_id": "exp-1",
                        "status": "completed",
                        "engine": "intermittent_gp",
                        "models": {
                            "six_number_set": {
                                "independent_test_activation_count": 20,
                                "independent_test_target_round_count": 25,
                                "independent_test_score": 0.125,
                            }
                        },
                        "hits": {
                            "six_number_set:independent_test": {
                                "3_plus": 4,
                                "4_plus": 2,
                                "5_plus": 1,
                                "6_exact": 0,
                                "2_exact": 1,
                                "2_known": 18,
                            }
                        },
                        "p_values": {"six_number_set": 0.2},
                    }
                ],
            },
        },
        checkpoints,
        experiments,
    )
    before = registry.get("research-1")

    catalog = build_experiment_catalog(database, experiments)
    linked = next(item for item in catalog if item["experiment_id"] == "exp-1")
    orphan = next(item for item in catalog if item["experiment_id"] == "orphan")

    assert linked["research_round"] == 1
    assert linked["engine"] == "GP"
    assert linked["feature_profile"] == "bonus_lag_v1"
    assert linked["hits_4_plus"] == 2
    assert linked["second_prize"] == 1
    assert linked["second_prize_known"] == 18
    assert linked["rate_4_plus"] == 0.1
    assert linked["second_prize_rate"] == 1 / 18
    assert linked["independent_test_available"] is True
    assert orphan["validation_status"] == "요약 정보 미수집"
    label = experiment_choice_label(linked)
    assert "4개+ 2건" in label
    assert "bonus_lag_v1" in label
    assert "2026-10-20" in label
    assert registry.get("research-1") == before


def test_catalog_filter_and_sort_keep_unknown_values_separate() -> None:
    rows = [
        {
            "experiment_id": "recent",
            "engine": "gp",
            "feature_profile": "p1",
            "validation_status": "verified",
            "independent_test_available": True,
            "research_date": "2026-10-20",
            "rate_4_plus": 0.05,
            "hits_4_plus": 2,
        },
        {
            "experiment_id": "older",
            "engine": "pysr",
            "feature_profile": "p2",
            "validation_status": "verified",
            "independent_test_available": True,
            "research_date": "2026-10-01",
            "rate_4_plus": 0.1,
            "hits_4_plus": 1,
        },
        {
            "experiment_id": "unknown",
            "engine": "gp",
            "feature_profile": "p1",
            "validation_status": "missing",
            "independent_test_available": False,
            "research_date": None,
            "rate_4_plus": None,
            "hits_4_plus": None,
        },
    ]

    filtered = filter_experiment_catalog(
        rows,
        engine="gp",
        independent_test=True,
        hit_key="hits_4_plus",
        start_date=date(2026, 10, 10),
    )
    assert [item["experiment_id"] for item in filtered] == ["recent"]
    sorted_rows = sort_experiment_catalog(rows, key="rate_4_plus")
    assert [item["experiment_id"] for item in sorted_rows] == ["older", "recent", "unknown"]
    sorted_random_p = sort_experiment_catalog(rows, key="random_comparison_p", descending=False)
    assert sorted_random_p[-1]["experiment_id"] == "unknown"


def test_catalog_choice_and_nullable_numeric_table_render_with_apptest(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    page = tmp_path / "catalog_component.py"
    page.write_text(
        """import streamlit as st
from lotto_pattern_lab.revalidation_catalog import experiment_choice_label
items = [
    {"experiment_id": "uuid-a", "research_round": 4, "hits_4_plus": 2,
     "hits_5_plus": 0, "second_prize": None, "engine": "gp",
     "feature_profile": "p1", "validation_status": "missing", "research_date": "2026-10-20"},
    {"experiment_id": "uuid-b", "research_round": 4, "hits_4_plus": 2,
     "hits_5_plus": 0, "second_prize": None, "engine": "gp",
     "feature_profile": "p1", "validation_status": "missing", "research_date": "2026-10-20"},
]
by_id = {item["experiment_id"]: item for item in items}
selected = st.selectbox(
    "Source",
    list(by_id),
    format_func=lambda key: experiment_choice_label(by_id[key]),
)
st.dataframe([{"4+": 2, "Second prize": None}], hide_index=True)
st.caption(selected)
""",
        encoding="utf-8",
    )

    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.selectbox[0].options) == 2
    assert app.selectbox[0].value == "uuid-a"
    assert all("4개+ 2건" in option for option in app.selectbox[0].options)

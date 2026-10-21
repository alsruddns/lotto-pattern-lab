from __future__ import annotations

from pathlib import Path

from lotto_pattern_lab.research_rounds import ResearchRoundStore


def test_dashboard_overview_renders_from_temporary_database(monkeypatch, tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    import lotto_pattern_lab.paths as paths
    import lotto_pattern_lab.research_rounds as research_rounds

    project_root = tmp_path / "project"
    for name in ("data/database", "checkpoints", "experiments", "predictions"):
        (project_root / name).mkdir(parents=True, exist_ok=True)

    def temporary_project_path(value):
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    monkeypatch.setattr(paths, "project_path", temporary_project_path)
    monkeypatch.setattr(research_rounds, "project_path", temporary_project_path)

    database = project_root / "data/database/patterns.sqlite"
    store = ResearchRoundStore(database)
    store.migrate_legacy(project_root / "checkpoints", project_root / "experiments")
    store.ensure_round(
        "temporary-round",
        {
            "status": "completed",
            "started_at": "2026-10-20T12:00:00+09:00",
            "engines": ["intermittent_gp"],
            "config": {"feature_profile": "legacy"},
            "total_elapsed_seconds": 60,
            "completed_runs": 1,
            "failed_runs": 0,
            "dashboard_summary": {
                "summary_schema_version": 3,
                "completed_experiment_count": 1,
                "failed_experiment_count": 0,
                "new_pattern_count": 1,
                "hit_totals": {"six_number_set:independent_test": {"3_plus": 0}},
                "model_metrics": {"six_number_set": [{"independent_test_score": 0.0}]},
                "experiment_summaries": [{"experiment_id": "temporary-exp"}],
            },
        },
        project_root / "checkpoints",
        project_root / "experiments",
    )

    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("5_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert any(item.key == "research_dashboard_selected_round" for item in app.selectbox)
    assert any(item.key == "load_research_round_detail" for item in app.button)
    assert len(app.dataframe) == 1
    assert store.get("temporary-round")["status"] == "completed"

"""Project-root path resolution and checkpoint recovery lookup tests."""

from pathlib import Path

from lotto_pattern_lab import paths
from lotto_pattern_lab.paths import locate_checkpoint, project_path
from lotto_pattern_lab.research import _atomic_json, load_checkpoint


def test_relative_paths_are_anchored_at_project_root_from_other_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(elsewhere)

    assert project_path("checkpoints/research.json") == project_root / "checkpoints/research.json"


def test_checkpoint_lookup_distinguishes_relocated_file_from_missing_file(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    checkpoints = project_root / "checkpoints"
    checkpoints.mkdir(parents=True)
    actual = checkpoints / "research.json"
    actual.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", project_root)

    requested, matches = locate_checkpoint("elsewhere/research.json")
    missing, no_matches = locate_checkpoint("checkpoints/missing.json")

    assert requested == project_root / "elsewhere/research.json"
    assert matches == [actual]
    assert missing == checkpoints / "missing.json"
    assert no_matches == []


def test_checkpoint_loader_explains_relocated_and_missing_paths(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    actual = project_root / "checkpoints" / "research.json"
    actual.parent.mkdir(parents=True)
    actual.write_text('{"research_id":"r-1"}', encoding="utf-8")
    monkeypatch.setattr(paths, "PROJECT_ROOT", project_root)

    try:
        load_checkpoint("elsewhere/research.json")
    except FileNotFoundError as error:
        assert "path is incorrect" in str(error)
        assert str(actual) in str(error)
    else:
        raise AssertionError("A relocated checkpoint should require explicit path correction.")

    try:
        load_checkpoint("checkpoints/missing.json")
    except FileNotFoundError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError("A nonexistent checkpoint should be reported as missing.")


def test_atomic_checkpoint_write_preserves_previous_valid_file_on_serialization_error(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "research.json"
    _atomic_json(checkpoint, {"status": "paused"})

    try:
        _atomic_json(checkpoint, {"status": "running", "invalid": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite checkpoint data should fail JSON serialization.")

    assert load_checkpoint(checkpoint) == {"status": "paused"}
    assert list(tmp_path.glob("*.tmp")) == []

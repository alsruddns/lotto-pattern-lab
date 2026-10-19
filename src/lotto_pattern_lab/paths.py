"""Resolve application data paths independently of the process working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    """Return an absolute path, anchoring relative paths at the repository root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def locate_checkpoint(path: str | Path) -> tuple[Path, list[Path]]:
    """Resolve a checkpoint and find same-named project checkpoints if its path moved."""
    requested = project_path(path)
    if requested.is_file():
        return requested, []
    directory = PROJECT_ROOT / "checkpoints"
    matches = sorted(directory.rglob(requested.name)) if directory.is_dir() else []
    matches = [candidate.resolve() for candidate in matches if candidate.is_file()]
    return requested, matches

"""Immutable Korean reports for completed research runs."""

from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .paths import project_path


def build_research_report(state: dict[str, Any]) -> str:
    """Render the canonical research-round report without recomputing results."""
    from .research_rounds import build_chatgpt_report

    report = build_chatgpt_report(state, detailed=True)
    queued = (state.get("candidate_revalidation") or {}).get("queued_jobs", 0)
    return (
        report
        + f"\n\n## \uc790\ub3d9 \uc7ac\uac80\uc99d\n- \ub300\uae30 \uc791\uc5c5: {queued}"
    )


def save_completed_research_report(
    state: dict[str, Any], *, directory: str | Path = "reports/research"
) -> Path:
    """Save once per research ID; existing reports are never overwritten."""
    if state.get("status") != "completed":
        raise ValueError(
            "\uc644\ub8cc\ub41c \uc5f0\uad6c\ub9cc \ubcf4\uace0\uc11c\ub85c \uc800\uc7a5\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4."  # noqa: E501
        )
    research_id = str(state.get("research_id") or "").strip()
    if not research_id or any(char in research_id for char in ("/", "\\", "..")):
        raise ValueError(
            "\uc5f0\uad6c ID\uac00 \uc5c6\uac70\ub098 \uc548\uc804\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."  # noqa: E501
        )
    target_dir = project_path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{research_id}.md"
    with interprocess_lock(target, scope="research-report"):
        if not target.exists():
            atomic_write_bytes(target, build_research_report(state).encode("utf-8"))
    return target

"""Small browser-side clipboard button with an explicit success/failure callback."""

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "lotto_research_report_clipboard",
    path=str(Path(__file__).with_name("clipboard_component")),
)


def report_copy_button(markdown: str, *, key: str, detailed: bool = False) -> dict[str, Any] | None:
    """Render a browser clipboard control and return the confirmed browser result."""
    return _COMPONENT(
        markdown=markdown,
        label="전체 상세 보고서 복사" if detailed else "ChatGPT 분석 보고서 복사",
        key=key,
        default=None,
        height=52,
    )

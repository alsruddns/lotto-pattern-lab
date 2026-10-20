from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "lotto_pattern_lab"


def _ui_files() -> list[Path]:
    return [ROOT / "app.py", ROOT / "home.py", *sorted((ROOT / "pages").glob("*.py"))]


def test_streamlit_strings_are_utf8_and_free_of_known_corruption() -> None:
    problems = []
    for path in _ui_files():
        raw = path.read_bytes()
        try:
            source = raw.decode("utf-8")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError) as error:
            problems.append(f"{path}: {error}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            if "\ufffd" in value or re.search(r"\?{2,}", value):
                problems.append(f"{path.name}:{node.lineno}: {value!r}")
            if any("\u4e00" <= char <= "\u9fff" for char in value):
                problems.append(f"{path.name}:{node.lineno}: CJK text {value!r}")
            lowered = value.lower()
            if "lotto" in lowered and not any(
                token in lowered
                for token in (
                    "lotto_pattern_lab",
                    "lotto_draws_",
                    "lotto-pattern-lab-locks",
                )
            ):
                problems.append(f"{path.name}:{node.lineno}: user-facing Lotto text {value!r}")
    assert not problems, "\n".join(problems)


def test_navigation_has_the_required_nine_unique_visible_pages() -> None:
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    visible_titles = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "Page":
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        hidden = keywords.get("visibility")
        if isinstance(hidden, ast.Constant) and hidden.value == "hidden":
            continue
        title = keywords.get("title")
        if isinstance(title, ast.Constant) and isinstance(title.value, str):
            visible_titles.append(title.value)
    assert visible_titles == [
        "연구실 홈",
        "단일 탐색",
        "지속적 패턴 연구",
        "수학 공식 라이브러리",
        "전체 회차 재검증",
        "미래 독립 검증",
        "연구 회차 대시보드",
        "데이터 및 저장 공간 관리",
        "설정",
    ]
    assert len(set(visible_titles)) == 9

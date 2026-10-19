"""Fetch and validate draw results directly from the Donghaeng Lottery website."""

import json
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .dataset import DrawDataset, DrawRecord
from .draw import Draw

OFFICIAL_RESULT_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do"
OFFICIAL_HISTORY_URL = "https://www.dhlottery.co.kr/lt645/result"
FIRST_SUPPORTED_ROUND = 753
FIRST_SUPPORTED_DATE = date(2017, 5, 6)


def fetch_official_draw(round_number: int, timeout: float = 20.0) -> DrawRecord | None:
    """Fetch one official result; return None when the requested round is unpublished."""
    if round_number < FIRST_SUPPORTED_ROUND:
        raise ValueError(f"Official research range starts at draw {FIRST_SUPPORTED_ROUND}.")
    request = Request(
        f"{OFFICIAL_RESULT_URL}?srchLtEpsd={round_number}",
        headers={
            "Accept": "application/json",
            "Referer": OFFICIAL_HISTORY_URL,
            "User-Agent": "LottoPatternLab/0.2 (official result verification)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Official result request failed for round {round_number}: {error}"
        ) from error
    rows = payload.get("data", {}).get("list", [])
    if not rows:
        return None
    row = rows[0]
    returned_round = int(row["ltEpsd"])
    if returned_round != round_number:
        raise ValueError(
            f"Requested round {round_number}, official source returned {returned_round}."
        )
    draw_date = datetime.strptime(str(row["ltRflYmd"]), "%Y%m%d").date()
    if draw_date < FIRST_SUPPORTED_DATE:
        raise ValueError(f"Official source date {draw_date} is earlier than supported start date.")
    numbers = tuple(int(row[f"tm{index}WnNo"]) for index in range(1, 7))
    return DrawRecord(returned_round, Draw.from_numbers(numbers), draw_date)


def fetch_official_dataset(
    start_round: int = FIRST_SUPPORTED_ROUND,
    end_round: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    delay_seconds: float = 0.08,
    progress: Callable[[int, int], None] | None = None,
) -> DrawDataset:
    """Fetch an inclusive round interval; omitted end round resolves to official latest."""
    if start_round < FIRST_SUPPORTED_ROUND:
        raise ValueError(f"Start round must be at least {FIRST_SUPPORTED_ROUND}.")
    if delay_seconds < 0:
        raise ValueError("Request delay cannot be negative.")
    if start_date and end_date and start_date > end_date:
        raise ValueError("Start date must not be after end date.")

    if end_round is None:
        # The site's no-parameter result returns its current latest published draw.
        end_round = _fetch_latest_round()
    if end_round < start_round:
        raise ValueError("End round must not be earlier than start round.")

    records: list[DrawRecord] = []
    total = end_round - start_round + 1
    for completed, round_number in enumerate(range(start_round, end_round + 1), start=1):
        record = fetch_official_draw(round_number)
        if record is None:
            raise RuntimeError(
                f"Official source has no published result for expected round {round_number}."
            )
        if (start_date is None or record.draw_date >= start_date) and (
            end_date is None or record.draw_date <= end_date
        ):
            records.append(record)
        if delay_seconds:
            time.sleep(delay_seconds)
        if progress:
            progress(completed, total)
    if len(records) < 2:
        raise ValueError("The requested official range contains fewer than two draws.")
    return DrawDataset(
        tuple(records),
        "official_verified",
        "Fetched draw numbers and dates from the Donghaeng Lottery official results endpoint.",
        OFFICIAL_RESULT_URL,
        verified_at=datetime.now(UTC).isoformat(),
    )


def _fetch_latest_round(timeout: float = 20.0) -> int:
    request = Request(
        OFFICIAL_RESULT_URL,
        headers={
            "Accept": "application/json",
            "Referer": OFFICIAL_HISTORY_URL,
            "User-Agent": "LottoPatternLab/0.2 (official result verification)",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not determine latest official draw: {error}") from error
    rows = payload.get("data", {}).get("list", [])
    if not rows:
        raise RuntimeError("Official result endpoint returned no latest draw.")
    return int(rows[0]["ltEpsd"])

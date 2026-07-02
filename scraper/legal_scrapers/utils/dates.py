"""Date conversions for the various Georgian legal sources.

Spiders take ``start_date``/``end_date`` as ISO ``YYYY-MM-DD`` strings (see
``BaseLegalSpider``). Each site wants a different format on the wire, so the
conversions are centralized here to keep the spiders declarative.
"""

import re
from datetime import UTC, date, datetime


def iso_to_dotted(d: date) -> str:
    """``date(2024, 3, 9)`` -> ``"09-03-2024"`` (constcourt dateFrom/dateTo)."""
    return d.strftime("%d-%m-%Y")


def iso_to_slashed(d: date) -> str:
    """``date(2024, 3, 9)`` -> ``"09/03/2024"`` (napr fdate/tdate)."""
    return d.strftime("%d/%m/%Y")


def iso_to_year_slashed(d: date) -> str:
    """``date(2024, 3, 9)`` -> ``"2024/03/09"`` (supremecourt tarigiDan/tarigiMde)."""
    return d.strftime("%Y/%m/%d")


_DOTNET_RE = re.compile(r"/Date\((-?\d+)(?:[+-]\d+)?\)/")


def dotnet_date_to_iso(value: str | None) -> str | None:
    """``"/Date(1588260918000)/"`` -> ``"2020-04-30"`` (ecd DecisionDate).

    Returns ``None`` for empty or unparseable input.
    """
    if not value:
        return None
    match = _DOTNET_RE.search(value)
    if not match:
        return None
    millis = int(match.group(1))
    return datetime.fromtimestamp(millis / 1000, tz=UTC).date().isoformat()


def date_part(value):
    """Take the ``YYYY-MM-DD`` prefix of a datetime-ish string.

    ``"2024-06-03 08:00:07+00:00"`` -> ``"2024-06-03"``. Non-string or non-matching
    values pass through (trimmed). Used by the JSON sources (ecd/napr/tas) whose dates
    arrive as full timestamps.
    """
    if not isinstance(value, str):
        return value
    match = re.match(r"\s*(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else value.strip()


def parse_dotted(value: str | None) -> date | None:
    """Parse a displayed Georgian date to a ``date`` for in-spider filtering.

    Accepts ``DD-MM-YYYY`` / ``DD/MM/YYYY`` / ``DD.MM.YYYY`` (tbappeal ``span.date``).
    Returns ``None`` when it cannot be parsed.
    """
    value = (value or "").strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None

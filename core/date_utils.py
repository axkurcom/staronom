from __future__ import annotations

import datetime as dt
from typing import Iterator


def iso_to_date(value: str) -> dt.date:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).date()


def utc_midnight_ts(day: dt.date) -> int:
    midnight = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    return int(midnight.timestamp())


def daterange(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def utc_today() -> dt.date:
    return dt.datetime.now(tz=dt.timezone.utc).date()

from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from core.date_utils import daterange, utc_today

SECONDS_PER_DAY = 24 * 60 * 60
MIN_CURRENT_DAY_MEAN_EXPOSURE = 3.0 / 24.0


def normalize_utc(now_utc: Optional[dt.datetime]) -> dt.datetime:
    if now_utc is None:
        return dt.datetime.now(tz=dt.timezone.utc)
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=dt.timezone.utc)
    return now_utc.astimezone(dt.timezone.utc)


def current_day_fraction(now_utc: Optional[dt.datetime] = None) -> float:
    now = normalize_utc(now_utc)
    midnight = dt.datetime(now.year, now.month, now.day, tzinfo=dt.timezone.utc)
    elapsed = (now - midnight).total_seconds()
    return max(0.0, min(elapsed / SECONDS_PER_DAY, 1.0))


def build_daily_counts(
    star_dates: Sequence[dt.date],
    *,
    empty_day: Optional[dt.date] = None,
    end_day: Optional[dt.date] = None,
) -> Tuple[List[dt.date], List[int]]:
    daily_counter = collections.Counter(star_dates)
    final_day = end_day or utc_today()
    if not daily_counter:
        baseline_day = empty_day or final_day
        return [baseline_day], [0]

    first = min(daily_counter)
    last_seen = max(daily_counter)
    if last_seen > final_day:
        raise ValueError("star_dates must not contain dates after end_day")

    last = max(last_seen, final_day)
    days = list(daterange(first, last))
    counts = [daily_counter.get(day, 0) for day in days]
    return days, counts


@dataclass(frozen=True)
class DailySeries:
    days: List[dt.date]
    counts: List[int]
    exposures: List[float]
    mean_exposures: List[float]
    model_days: List[dt.date]
    model_counts: List[int]
    current_day_hours_elapsed: Optional[float]

    @property
    def average_effective_days(self) -> float:
        return sum(self.mean_exposures)

    @property
    def forecast_start_after(self) -> dt.date:
        return self.days[-1]


def _exposures_for_days(days: Sequence[dt.date], now_utc: dt.datetime) -> List[float]:
    today = now_utc.date()
    fraction = current_day_fraction(now_utc)
    exposures: List[float] = []
    for day in days:
        if day < today:
            exposures.append(1.0)
        elif day == today:
            exposures.append(fraction)
        else:
            exposures.append(0.0)
    return exposures


def _mean_exposures(counts: Sequence[int], exposures: Sequence[float]) -> List[float]:
    out: List[float] = []
    for count, exposure in zip(counts, exposures):
        if count > 0 and 0.0 < exposure < MIN_CURRENT_DAY_MEAN_EXPOSURE:
            out.append(MIN_CURRENT_DAY_MEAN_EXPOSURE)
        else:
            out.append(exposure)
    return out


def _model_history(
    days: Sequence[dt.date], counts: Sequence[int], exposures: Sequence[float]
) -> Tuple[List[dt.date], List[int]]:
    completed = [
        (day, count)
        for day, count, exposure in zip(days, counts, exposures)
        if exposure >= 1.0
    ]
    if completed:
        model_days, model_counts = zip(*completed)
        return list(model_days), list(model_counts)

    return list(days), list(counts)


def series_from_counts(
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    now_utc: Optional[dt.datetime] = None,
) -> DailySeries:
    if not days or not counts:
        raise ValueError("days and counts must not be empty")
    if len(days) != len(counts):
        raise ValueError("days and counts length mismatch")

    now = normalize_utc(now_utc)
    day_list = list(days)
    count_list = list(counts)
    exposures = _exposures_for_days(day_list, now)
    if any(day > now.date() for day in day_list):
        raise ValueError("days must not contain dates after now_utc date")
    model_days, model_counts = _model_history(day_list, count_list, exposures)
    current_day_hours_elapsed = None
    if now.date() in day_list:
        current_day_hours_elapsed = current_day_fraction(now) * 24.0

    return DailySeries(
        days=day_list,
        counts=count_list,
        exposures=exposures,
        mean_exposures=_mean_exposures(count_list, exposures),
        model_days=model_days,
        model_counts=model_counts,
        current_day_hours_elapsed=current_day_hours_elapsed,
    )


def extend_daily_counts(
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    end_day: dt.date,
) -> Tuple[List[dt.date], List[int]]:
    if not days or not counts:
        raise ValueError("days and counts must not be empty")
    if len(days) != len(counts):
        raise ValueError("days and counts length mismatch")

    day_list = list(days)
    count_list = list(counts)
    if day_list[-1] > end_day:
        raise ValueError("days must not contain dates after end_day")
    for idx, day in enumerate(day_list):
        if idx > 0 and day != day_list[idx - 1] + dt.timedelta(days=1):
            raise ValueError("days must be sorted and contiguous")
    for day in daterange(day_list[-1] + dt.timedelta(days=1), end_day):
        day_list.append(day)
        count_list.append(0)
    return day_list, count_list


def series_from_daily_counts(
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    now_utc: Optional[dt.datetime] = None,
) -> DailySeries:
    now = normalize_utc(now_utc)
    days, counts = extend_daily_counts(days, counts, end_day=now.date())
    return series_from_counts(days, counts, now_utc=now)


def build_daily_series(
    star_dates: Sequence[dt.date],
    *,
    now_utc: Optional[dt.datetime] = None,
    empty_day: Optional[dt.date] = None,
) -> DailySeries:
    now = normalize_utc(now_utc)
    days, counts = build_daily_counts(
        star_dates,
        empty_day=empty_day,
        end_day=now.date(),
    )
    return series_from_counts(days, counts, now_utc=now)

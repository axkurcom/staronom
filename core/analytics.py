from __future__ import annotations

import collections
import datetime as dt
import math
import statistics
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from core.models import AnalysisResult, Streak, WeekStat
from core.series import DailySeries, build_daily_counts, series_from_counts


def _exposure_adjusted_mean(values: Sequence[float], exposures: Sequence[float]) -> float:
    total_exposure = sum(exposures)
    if total_exposure <= 0.0:
        return 0.0
    return sum(float(value) for value in values) / total_exposure


def moving_average(
    series: Sequence[float], window: int, exposures: Optional[Sequence[float]] = None
) -> List[Optional[float]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if exposures is not None and len(exposures) != len(series):
        raise ValueError("series and exposures length mismatch")

    out: List[Optional[float]] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
            continue

        sample = series[i - window + 1 : i + 1]
        if exposures is None:
            out.append(sum(sample) / window)
            continue

        sample_exposures = exposures[i - window + 1 : i + 1]
        out.append(_exposure_adjusted_mean(sample, sample_exposures))
    return out


def moving_median(series: Sequence[float], window: int) -> List[Optional[float]]:
    if window <= 0:
        raise ValueError("window must be > 0")

    out: List[Optional[float]] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(None)
            continue

        sample = series[i - window + 1 : i + 1]
        out.append(statistics.median(sample))
    return out


def linear_regression(series: Sequence[float]) -> Tuple[float, float, float]:
    n = len(series)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, float(series[0]), 1.0

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(series) / n

    num = sum((xs[i] - mean_x) * (series[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    slope = num / den if den else 0.0
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in series)
    ss_res = sum((series[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return slope, intercept, r2


def percentiles(values: Sequence[int], requested: Sequence[int]) -> Dict[int, int]:
    if not values:
        return {p: 0 for p in requested}

    ordered = sorted(values)
    n = len(ordered)
    result: Dict[int, int] = {}
    for p in requested:
        if p < 1 or p > 100:
            raise ValueError("Percentile must be in [1, 100]")
        k = max(1, math.ceil((p / 100) * n))
        result[p] = ordered[k - 1]
    return result


def weekly_stats(
    days: Sequence[dt.date],
    counts: Sequence[int],
    exposures: Optional[Sequence[float]] = None,
) -> List[WeekStat]:
    if len(days) != len(counts):
        raise ValueError("days and counts length mismatch")
    if exposures is not None and len(exposures) != len(counts):
        raise ValueError("days and exposures length mismatch")

    buckets: Dict[Tuple[int, int], List[Tuple[int, float]]] = collections.defaultdict(list)
    effective_exposures = exposures if exposures is not None else [1.0] * len(counts)
    for day, count, exposure in zip(days, counts, effective_exposures):
        iso = day.isocalendar()
        buckets[(iso.year, iso.week)].append((count, exposure))

    result: List[WeekStat] = []
    for key in sorted(buckets.keys()):
        vals = buckets[key]
        total = sum(count for count, _ in vals)
        total_exposure = sum(exposure for _, exposure in vals)
        avg = (total / total_exposure) if total_exposure > 0.0 else 0.0
        result.append((key, avg, total))
    return result


def longest_streak(values: Sequence[int], predicate: Callable[[int], bool]) -> Streak:
    best_len = 0
    best = (0, 0, -1)
    cur_len = 0
    cur_start = 0

    for i, value in enumerate(values):
        if predicate(value):
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best = (best_len, cur_start, i)
            continue
        cur_len = 0

    return best


def build_daily_series(
    star_dates: Sequence[dt.date],
    *,
    empty_day: Optional[dt.date] = None,
    end_day: Optional[dt.date] = None,
) -> Tuple[List[dt.date], List[int]]:
    return build_daily_counts(star_dates, empty_day=empty_day, end_day=end_day)


def analyze_series(
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    now_utc: Optional[dt.datetime] = None,
) -> AnalysisResult:
    return analyze_daily_series(series_from_counts(days, counts, now_utc=now_utc))


def analyze_daily_series(series: DailySeries) -> AnalysisResult:
    days = series.days
    counts = series.counts
    if not days or not counts:
        raise ValueError("days and counts must not be empty")
    if len(days) != len(counts):
        raise ValueError("days and counts length mismatch")

    stats_days = series.model_days
    stats_counts = series.model_counts
    exposures = series.mean_exposures
    effective_days = series.average_effective_days
    total_sum = sum(counts)
    avg = _exposure_adjusted_mean(counts, exposures)
    median = statistics.median(stats_counts)
    stdev = statistics.pstdev(stats_counts)
    nonzero = sum(1 for x in stats_counts if x > 0)

    max_day = max(stats_counts)
    max_dates = [
        stats_days[i]
        for i, value in enumerate(stats_counts)
        if value == max_day
    ]

    percentile_by_p = percentiles(stats_counts, [50, 75, 90, 95, 99])

    ma7 = moving_average(counts, 7, exposures)
    ma28 = moving_average(counts, 28, exposures)
    med7 = moving_median(stats_counts, 7)
    med28 = moving_median(stats_counts, 28)

    slope, intercept, r2 = linear_regression(stats_counts)
    streak_nonzero = longest_streak(stats_counts, lambda x: x > 0)
    streak_zero = longest_streak(stats_counts, lambda x: x == 0)
    weekly = weekly_stats(days, counts, exposures)

    return AnalysisResult(
        first_day=days[0],
        last_day=days[-1],
        days=list(days),
        counts=list(counts),
        total_stars=total_sum,
        average_per_day=avg,
        median_per_day=median,
        stdev_per_day=stdev,
        nonzero_days=nonzero,
        max_day_count=max_day,
        max_day_dates=max_dates,
        percentile_by_p=percentile_by_p,
        ma7=ma7,
        ma28=ma28,
        med7=med7,
        med28=med28,
        slope=slope,
        intercept=intercept,
        r2=r2,
        streak_nonzero=streak_nonzero,
        streak_zero=streak_zero,
        weekly=weekly,
        average_effective_days=effective_days,
        current_day_hours_elapsed=series.current_day_hours_elapsed,
        completed_day_count=len(stats_counts),
    )

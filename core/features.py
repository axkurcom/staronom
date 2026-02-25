from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass
from typing import Counter, Dict, List, Sequence, Tuple

BASE_EVENT_KEYS = ("release", "commits", "issues", "prs")


@dataclass(frozen=True)
class EventSignals:
    releases: Counter[dt.date]
    commits: Counter[dt.date]
    issues: Counter[dt.date]
    prs: Counter[dt.date]


def empty_event_signals() -> EventSignals:
    return EventSignals(
        releases=collections.Counter(),
        commits=collections.Counter(),
        issues=collections.Counter(),
        prs=collections.Counter(),
    )


def _align_counter(days: Sequence[dt.date], values: Counter[dt.date]) -> List[float]:
    return [float(values.get(day, 0)) for day in days]


def _rolling_sum(series: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float] = []
    running = 0.0
    for i, value in enumerate(series):
        running += value
        if i >= window:
            running -= series[i - window]
        out.append(running)
    return out


def build_event_rows(days: Sequence[dt.date], signals: EventSignals) -> List[Dict[str, float]]:
    releases = _align_counter(days, signals.releases)
    commits = _align_counter(days, signals.commits)
    issues = _align_counter(days, signals.issues)
    prs = _align_counter(days, signals.prs)

    release_roll14 = _rolling_sum(releases, 14)
    commits_roll3 = _rolling_sum(commits, 3)
    issues_roll7 = _rolling_sum(issues, 7)
    prs_roll7 = _rolling_sum(prs, 7)

    rows: List[Dict[str, float]] = []
    for i, day in enumerate(days):
        row = {
            "weekday": float(day.weekday()),
            "release": releases[i],
            "commits": commits[i],
            "issues": issues[i],
            "prs": prs[i],
            "activity": releases[i] + commits[i] + issues[i] + prs[i],
            "release_roll14": release_roll14[i],
            "commits_roll3": commits_roll3[i],
            "issues_roll7": issues_roll7[i],
            "prs_roll7": prs_roll7[i],
        }
        rows.append(row)
    return rows


def event_coverage_ratio(rows: Sequence[Dict[str, float]]) -> float:
    if not rows:
        return 0.0
    active_days = sum(1 for row in rows if row.get("activity", 0.0) > 0.0)
    return active_days / len(rows)


def _weekday_means(history_days: Sequence[dt.date], values: Sequence[float]) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = collections.defaultdict(list)
    for day, value in zip(history_days, values):
        buckets[day.weekday()].append(value)
    out: Dict[int, float] = {}
    for weekday in range(7):
        vals = buckets.get(weekday)
        out[weekday] = sum(vals) / len(vals) if vals else 0.0
    return out


def estimate_future_event_rows(
    history_days: Sequence[dt.date],
    history_rows: Sequence[Dict[str, float]],
    horizon_days: int,
) -> Tuple[List[dt.date], List[Dict[str, float]]]:
    if horizon_days <= 0:
        return [], []
    if not history_days:
        raise ValueError("history_days must not be empty")
    if len(history_days) != len(history_rows):
        raise ValueError("history days and rows length mismatch")

    history_by_key: Dict[str, List[float]] = {key: [] for key in BASE_EVENT_KEYS}
    for row in history_rows:
        for key in BASE_EVENT_KEYS:
            history_by_key[key].append(float(row.get(key, 0.0)))

    recent = min(84, len(history_days))
    recent_days = list(history_days[-recent:])
    recent_by_key = {key: values[-recent:] for key, values in history_by_key.items()}

    weekday_mean_by_key = {
        key: _weekday_means(recent_days, values)
        for key, values in recent_by_key.items()
    }
    global_mean_by_key = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in recent_by_key.items()
    }

    full_releases = list(history_by_key["release"])
    full_commits = list(history_by_key["commits"])
    full_issues = list(history_by_key["issues"])
    full_prs = list(history_by_key["prs"])

    future_days: List[dt.date] = []
    future_rows: List[Dict[str, float]] = []
    last_day = history_days[-1]
    for step in range(1, horizon_days + 1):
        day = last_day + dt.timedelta(days=step)
        weekday = day.weekday()

        release = weekday_mean_by_key["release"].get(weekday, global_mean_by_key["release"])
        commits = weekday_mean_by_key["commits"].get(weekday, global_mean_by_key["commits"])
        issues = weekday_mean_by_key["issues"].get(weekday, global_mean_by_key["issues"])
        prs = weekday_mean_by_key["prs"].get(weekday, global_mean_by_key["prs"])

        full_releases.append(release)
        full_commits.append(commits)
        full_issues.append(issues)
        full_prs.append(prs)

        row = {
            "weekday": float(weekday),
            "release": release,
            "commits": commits,
            "issues": issues,
            "prs": prs,
            "activity": release + commits + issues + prs,
            "release_roll14": sum(full_releases[-14:]),
            "commits_roll3": sum(full_commits[-3:]),
            "issues_roll7": sum(full_issues[-7:]),
            "prs_roll7": sum(full_prs[-7:]),
        }
        future_days.append(day)
        future_rows.append(row)

    return future_days, future_rows

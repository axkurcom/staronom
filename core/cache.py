from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from core.models import RepoRef

STARS_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StarsCache:
    repo: str
    fetched_at_utc: dt.datetime
    days: List[dt.date]
    counts: List[int]


def _normalize_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _format_utc(value: dt.datetime) -> str:
    return _normalize_utc(value).isoformat().replace("+00:00", "Z")


def _parse_utc_datetime(value: object) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("fetched_at_utc must be an ISO datetime string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError("fetched_at_utc must be a valid ISO datetime") from None
    return _normalize_utc(parsed)


def _parse_date(value: object) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("cache days must be ISO date strings")
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"invalid cache day: {value!r}") from None


def _parse_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("cache counts must be non-negative integers")
    if value < 0:
        raise ValueError("cache counts must be non-negative integers")
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "repo"


def default_stars_cache_path(out_dir: str | os.PathLike[str], repo: RepoRef) -> Path:
    filename = f"stars_{_safe_name(repo.owner)}__{_safe_name(repo.repo)}.json"
    return Path(out_dir) / "cache" / filename


def validate_daily_counts(
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    now_utc: dt.datetime | None = None,
) -> Tuple[List[dt.date], List[int]]:
    if not days or not counts:
        raise ValueError("cache days and counts must not be empty")
    if len(days) != len(counts):
        raise ValueError("cache days and counts length mismatch")

    day_list = list(days)
    count_list = [_parse_count(count) for count in counts]
    for idx, day in enumerate(day_list):
        if not isinstance(day, dt.date) or isinstance(day, dt.datetime):
            raise ValueError("cache days must be date values")
        if idx > 0 and day != day_list[idx - 1] + dt.timedelta(days=1):
            raise ValueError("cache days must be sorted and contiguous")

    if now_utc is not None:
        today = _normalize_utc(now_utc).date()
        if day_list[-1] > today:
            raise ValueError("cache days must not contain dates after now_utc date")

    return day_list, count_list


def load_stars_cache(
    path: str | os.PathLike[str],
    repo: RepoRef,
    *,
    now_utc: dt.datetime | None = None,
) -> StarsCache:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("stars cache must contain a JSON object")
    if payload.get("schema_version") != STARS_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported stars cache schema_version")
    if payload.get("repo") != repo.full_name:
        raise ValueError("stars cache repo does not match requested repo")

    raw_days = payload.get("days")
    raw_counts = payload.get("counts")
    if not isinstance(raw_days, list) or not isinstance(raw_counts, list):
        raise ValueError("stars cache days and counts must be lists")

    days = [_parse_date(value) for value in raw_days]
    counts = [_parse_count(value) for value in raw_counts]
    days, counts = validate_daily_counts(days, counts, now_utc=now_utc)

    return StarsCache(
        repo=repo.full_name,
        fetched_at_utc=_parse_utc_datetime(payload.get("fetched_at_utc")),
        days=days,
        counts=counts,
    )


def save_stars_cache(
    path: str | os.PathLike[str],
    repo: RepoRef,
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    fetched_at_utc: dt.datetime,
) -> None:
    days, counts = validate_daily_counts(days, counts, now_utc=fetched_at_utc)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": STARS_CACHE_SCHEMA_VERSION,
        "repo": repo.full_name,
        "fetched_at_utc": _format_utc(fetched_at_utc),
        "days": [day.isoformat() for day in days],
        "counts": counts,
    }

    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def stars_cache_is_fresh(
    cache: StarsCache,
    *,
    now_utc: dt.datetime,
    ttl_hours: float,
) -> bool:
    now = _normalize_utc(now_utc)
    fetched_at = _normalize_utc(cache.fetched_at_utc)
    age = now - fetched_at
    if age.total_seconds() < 0:
        return False
    if fetched_at.date() != now.date():
        return False
    return age <= dt.timedelta(hours=ttl_hours)

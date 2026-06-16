from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from core.models import RepoRef

STARS_CACHE_SCHEMA_VERSION = 2
LEGACY_STARS_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StarsCache:
    repo: str
    fetched_at_utc: dt.datetime
    checked_at_utc: dt.datetime
    full_scan_at_utc: dt.datetime
    start_day: dt.date
    counts: List[int]
    current_stargazers_count: Optional[int] = None
    repo_etag: Optional[str] = None
    migrated_from_schema: Optional[int] = None

    @property
    def days(self) -> List[dt.date]:
        return [
            self.start_day + dt.timedelta(days=idx)
            for idx in range(len(self.counts))
        ]

    @property
    def last_day(self) -> dt.date:
        return self.start_day + dt.timedelta(days=len(self.counts) - 1)

    @property
    def total_count(self) -> int:
        return sum(self.counts)

    def with_metadata(
        self,
        *,
        checked_at_utc: dt.datetime,
        current_stargazers_count: Optional[int] = None,
        repo_etag: Optional[str] = None,
    ) -> "StarsCache":
        return replace(
            self,
            checked_at_utc=_normalize_utc(checked_at_utc),
            current_stargazers_count=current_stargazers_count,
            repo_etag=repo_etag,
            migrated_from_schema=None,
        )


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


def _parse_optional_count(value: object) -> Optional[int]:
    if value is None:
        return None
    return _parse_count(value)


def _parse_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cache optional string fields must be strings")
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


def counts_from_start_day(
    start_day: dt.date,
    counts: Sequence[int],
    *,
    now_utc: dt.datetime | None = None,
) -> Tuple[List[dt.date], List[int]]:
    if not isinstance(start_day, dt.date) or isinstance(start_day, dt.datetime):
        raise ValueError("cache start_day must be a date value")
    if not counts:
        raise ValueError("cache counts must not be empty")
    count_list = [_parse_count(count) for count in counts]
    days = [
        start_day + dt.timedelta(days=idx)
        for idx in range(len(count_list))
    ]
    return validate_daily_counts(days, count_list, now_utc=now_utc)


def _cache_from_v1(
    payload: dict,
    repo: RepoRef,
    *,
    now_utc: dt.datetime | None,
) -> StarsCache:
    raw_days = payload.get("days")
    raw_counts = payload.get("counts")
    if not isinstance(raw_days, list) or not isinstance(raw_counts, list):
        raise ValueError("stars cache days and counts must be lists")

    days = [_parse_date(value) for value in raw_days]
    counts = [_parse_count(value) for value in raw_counts]
    days, counts = validate_daily_counts(days, counts, now_utc=now_utc)
    fetched_at = _parse_utc_datetime(payload.get("fetched_at_utc"))
    return StarsCache(
        repo=repo.full_name,
        fetched_at_utc=fetched_at,
        checked_at_utc=fetched_at,
        full_scan_at_utc=fetched_at,
        start_day=days[0],
        counts=counts,
        current_stargazers_count=sum(counts),
        migrated_from_schema=LEGACY_STARS_CACHE_SCHEMA_VERSION,
    )


def _cache_from_v2(
    payload: dict,
    repo: RepoRef,
    *,
    now_utc: dt.datetime | None,
) -> StarsCache:
    start_day = _parse_date(payload.get("start_day"))
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, list):
        raise ValueError("stars cache counts must be a list")
    _, counts = counts_from_start_day(start_day, raw_counts, now_utc=now_utc)
    return StarsCache(
        repo=repo.full_name,
        fetched_at_utc=_parse_utc_datetime(payload.get("fetched_at_utc")),
        checked_at_utc=_parse_utc_datetime(payload.get("checked_at_utc")),
        full_scan_at_utc=_parse_utc_datetime(payload.get("full_scan_at_utc")),
        start_day=start_day,
        counts=counts,
        current_stargazers_count=_parse_optional_count(
            payload.get("current_stargazers_count")
        ),
        repo_etag=_parse_optional_str(payload.get("repo_etag")),
    )


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
    schema_version = payload.get("schema_version")
    if schema_version not in (
        LEGACY_STARS_CACHE_SCHEMA_VERSION,
        STARS_CACHE_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported stars cache schema_version")
    if payload.get("repo") != repo.full_name:
        raise ValueError("stars cache repo does not match requested repo")

    if schema_version == LEGACY_STARS_CACHE_SCHEMA_VERSION:
        return _cache_from_v1(payload, repo, now_utc=now_utc)
    return _cache_from_v2(payload, repo, now_utc=now_utc)


def save_stars_cache(
    path: str | os.PathLike[str],
    repo: RepoRef,
    days: Sequence[dt.date],
    counts: Sequence[int],
    *,
    fetched_at_utc: dt.datetime,
    checked_at_utc: dt.datetime | None = None,
    full_scan_at_utc: dt.datetime | None = None,
    current_stargazers_count: Optional[int] = None,
    repo_etag: Optional[str] = None,
) -> None:
    days, counts = validate_daily_counts(days, counts, now_utc=fetched_at_utc)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checked_at = checked_at_utc or fetched_at_utc
    full_scan_at = full_scan_at_utc or fetched_at_utc
    payload = {
        "schema_version": STARS_CACHE_SCHEMA_VERSION,
        "repo": repo.full_name,
        "fetched_at_utc": _format_utc(fetched_at_utc),
        "checked_at_utc": _format_utc(checked_at),
        "full_scan_at_utc": _format_utc(full_scan_at),
        "start_day": days[0].isoformat(),
        "counts": counts,
        "current_stargazers_count": (
            sum(counts)
            if current_stargazers_count is None
            else _parse_count(current_stargazers_count)
        ),
        "repo_etag": repo_etag,
    }

    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, target)


def save_stars_cache_object(
    path: str | os.PathLike[str],
    repo: RepoRef,
    cache: StarsCache,
) -> None:
    save_stars_cache(
        path,
        repo,
        cache.days,
        cache.counts,
        fetched_at_utc=cache.fetched_at_utc,
        checked_at_utc=cache.checked_at_utc,
        full_scan_at_utc=cache.full_scan_at_utc,
        current_stargazers_count=cache.current_stargazers_count,
        repo_etag=cache.repo_etag,
    )


def stars_cache_is_fresh(
    cache: StarsCache,
    *,
    now_utc: dt.datetime,
    ttl_hours: float,
) -> bool:
    now = _normalize_utc(now_utc)
    checked_at = _normalize_utc(cache.checked_at_utc)
    age = now - checked_at
    if age.total_seconds() < 0:
        return False
    if checked_at.date() != now.date():
        return False
    return age <= dt.timedelta(hours=ttl_hours)

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import requests

from core.analytics import analyze_daily_series
from core.backtest import run_backtest
from core.cache import (
    default_stars_cache_path,
    load_stars_cache,
    save_stars_cache,
    stars_cache_is_fresh,
)
from core.date_utils import utc_midnight_ts, utc_now
from core.forecast import (
    MAX_HORIZON_DAYS,
    generate_forecast,
    parse_interval_levels,
    save_forecast,
)
from core.github_client import fetch_event_signals, fetch_stars
from core.graphs import graph_advanced, graph_total_only
from core.models import RepoRef
from core.reporting import (
    print_backtest_summary,
    print_forecast_summary,
    print_summary,
)
from core.rrd import create_rrd, rrd_update
from core.series import (
    DailySeries,
    build_daily_counts,
    build_daily_series,
    series_from_daily_counts,
)

DEFAULT_STARS_CACHE_TTL_HOURS = 6.0


class StarCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class StarSeriesLoadResult:
    series: DailySeries
    source: str
    cache_path: Optional[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", default="./out")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--with-events", action="store_true")
    parser.add_argument("--intervals", default="0.8,0.95")
    parser.add_argument("--quantiles", help=argparse.SUPPRESS)
    parser.add_argument("--forecast-out")
    parser.add_argument("--drop-alert", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=DEFAULT_STARS_CACHE_TTL_HOURS,
    )
    parser.add_argument("--stars-cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--offline-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def load_star_series(
    *,
    repo: RepoRef,
    token: Optional[str],
    out_dir: str,
    now_utc: dt.datetime,
    cache_ttl_hours: float,
    stars_cache_path: Optional[str],
    refresh_cache: bool,
    offline_cache: bool,
    no_cache: bool,
    fetcher: Callable[[RepoRef, Optional[str]], Sequence[dt.date]] = fetch_stars,
) -> StarSeriesLoadResult:
    cache_path = (
        Path(stars_cache_path)
        if stars_cache_path
        else default_stars_cache_path(out_dir, repo)
    )

    if no_cache:
        star_dates = fetcher(repo, token)
        return StarSeriesLoadResult(
            series=build_daily_series(star_dates, now_utc=now_utc),
            source="api",
            cache_path=None,
        )

    cache_error: Optional[Exception] = None
    if not refresh_cache:
        try:
            cached = load_stars_cache(cache_path, repo, now_utc=now_utc)
            if offline_cache or stars_cache_is_fresh(
                cached,
                now_utc=now_utc,
                ttl_hours=cache_ttl_hours,
            ):
                return StarSeriesLoadResult(
                    series=series_from_daily_counts(
                        cached.days,
                        cached.counts,
                        now_utc=now_utc,
                    ),
                    source="cache",
                    cache_path=cache_path,
                )
        except (FileNotFoundError, ValueError, OSError) as exc:
            cache_error = exc

    if offline_cache:
        if cache_error is not None:
            raise StarCacheError(f"Offline cache is unavailable: {cache_error}") from cache_error
        raise StarCacheError("Offline cache is stale and refresh is disabled")

    star_dates = fetcher(repo, token)
    days, counts = build_daily_counts(star_dates, end_day=now_utc.date())
    try:
        save_stars_cache(cache_path, repo, days, counts, fetched_at_utc=now_utc)
    except OSError as exc:
        raise StarCacheError(f"Could not write stars cache: {exc}") from exc
    return StarSeriesLoadResult(
        series=series_from_daily_counts(days, counts, now_utc=now_utc),
        source="api",
        cache_path=cache_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cache_ttl_hours < 0 or not math.isfinite(args.cache_ttl_hours):
        print("--cache-ttl-hours must be a finite value >= 0", file=sys.stderr)
        return 2
    if args.no_cache and (args.offline_cache or args.refresh_cache):
        print(
            "--no-cache cannot be combined with --offline-cache or --refresh-cache",
            file=sys.stderr,
        )
        return 2
    if args.offline_cache and args.refresh_cache:
        print("--offline-cache cannot be combined with --refresh-cache", file=sys.stderr)
        return 2
    if args.horizon <= 0:
        print("--horizon must be > 0", file=sys.stderr)
        return 2
    if args.horizon > MAX_HORIZON_DAYS:
        print(f"--horizon must be <= {MAX_HORIZON_DAYS}", file=sys.stderr)
        return 2
    try:
        raw_intervals = args.intervals
        if args.quantiles:
            raw_intervals = args.quantiles
        interval_levels = parse_interval_levels(raw_intervals)
    except ValueError as exc:
        print(f"Invalid --intervals: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    try:
        repo = RepoRef.parse(args.repo)
    except ValueError as exc:
        print(f"Invalid --repo value: {exc}", file=sys.stderr)
        return 2

    try:
        now_utc = utc_now()
        print("Loading stars...")
        stars = load_star_series(
            repo=repo,
            token=args.token,
            out_dir=args.out,
            now_utc=now_utc,
            cache_ttl_hours=args.cache_ttl_hours,
            stars_cache_path=args.stars_cache,
            refresh_cache=args.refresh_cache,
            offline_cache=args.offline_cache,
            no_cache=args.no_cache,
        )
        daily_series = stars.series
        if stars.source == "cache" and stars.cache_path is not None:
            print(f"Using cached stars: {stars.cache_path}")
        elif stars.cache_path is not None:
            print(f"Fetched stars from GitHub; cache saved: {stars.cache_path}")
        else:
            print("Fetched stars from GitHub; cache disabled.")
        if stars.source == "api" and sum(daily_series.counts) == 0:
            print("No stars found in GitHub response; using a one-day zero baseline.")

        analysis = analyze_daily_series(daily_series)
        print_summary(repo.full_name, analysis)

        event_signals = None
        if args.with_events and (args.forecast or args.backtest):
            print()
            print("Fetching repo events (releases/commits/issues/PRs)...")
            event_signals = fetch_event_signals(repo, args.token)

        forecast = None
        if args.forecast:
            forecast = generate_forecast(
                repo=repo.full_name,
                history_days=daily_series.model_days,
                history_counts=daily_series.model_counts,
                horizon_days=args.horizon,
                interval_levels=interval_levels,
                with_events=args.with_events,
                event_signals=event_signals,
                with_drop_alert=args.drop_alert,
                forecast_start_after=daily_series.forecast_start_after,
            )
            print_forecast_summary(forecast)
            if args.forecast_out:
                save_forecast(forecast, args.forecast_out)
                print(f"Forecast saved: {args.forecast_out}")

        if args.backtest:
            backtest = run_backtest(
                repo=repo.full_name,
                history_days=daily_series.model_days,
                history_counts=daily_series.model_counts,
                horizon_days=args.horizon,
                interval_levels=interval_levels,
                with_events=args.with_events,
                event_signals=event_signals,
            )
            print_backtest_summary(backtest)

        rrd_path = os.path.join(args.out, "stars_daily.rrd")
        create_rrd(rrd_path, utc_midnight_ts(analysis.first_day))

        cumulative = 0
        skipped_updates = 0
        for day, daily in zip(daily_series.days, daily_series.counts):
            cumulative += daily
            updated = rrd_update(rrd_path, day, daily, cumulative)
            if not updated:
                skipped_updates += 1

        start_ts = str(utc_midnight_ts(analysis.first_day))
        advanced_path = os.path.join(args.out, "stars_advanced.png")
        total_path = os.path.join(args.out, "stars_total.png")
        advanced_size = graph_advanced(
            rrd_path, advanced_path, start_ts, analysis.slope, analysis.r2
        )
        total_size = graph_total_only(rrd_path, total_path, start_ts)

        print()
        print("Artifacts:")
        print(f"  rrd:         {rrd_path}")
        print(f"  graph:       {advanced_path} ({advanced_size})")
        print(f"  graph:       {total_path} ({total_size})")
        if skipped_updates:
            print(f"  note:        skipped {skipped_updates} RRD updates with old timestamps")
        return 0
    except requests.RequestException as exc:
        print(f"GitHub API request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Invalid data: {exc}", file=sys.stderr)
        return 1
    except StarCacheError as exc:
        print(f"Stars cache failed: {exc}", file=sys.stderr)
        return 1
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        print(f"RRD processing failed: {exc}", file=sys.stderr)
        return 1

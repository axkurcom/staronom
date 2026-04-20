from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional, Sequence

import requests

from core.analytics import analyze_series, build_daily_series
from core.backtest import run_backtest
from core.date_utils import utc_midnight_ts
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", default="./out")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--with-events", action="store_true")
    parser.add_argument("--quantiles", default="0.5,0.8,0.95")
    parser.add_argument("--forecast-out")
    parser.add_argument("--drop-alert", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.horizon <= 0:
        print("--horizon must be > 0", file=sys.stderr)
        return 2
    if args.horizon > MAX_HORIZON_DAYS:
        print(f"--horizon must be <= {MAX_HORIZON_DAYS}", file=sys.stderr)
        return 2
    try:
        interval_levels = parse_interval_levels(args.quantiles)
    except ValueError as exc:
        print(f"Invalid --quantiles: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    try:
        repo = RepoRef.parse(args.repo)
    except ValueError as exc:
        print(f"Invalid --repo value: {exc}", file=sys.stderr)
        return 2

    try:
        print("Fetching stars...")
        star_dates = fetch_stars(repo, args.token)
        if not star_dates:
            print("No stars found in GitHub response; using a one-day zero baseline.")

        days, counts = build_daily_series(star_dates)
        analysis = analyze_series(days, counts)
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
                history_days=analysis.days,
                history_counts=analysis.counts,
                horizon_days=args.horizon,
                interval_levels=interval_levels,
                with_events=args.with_events,
                event_signals=event_signals,
                with_drop_alert=args.drop_alert,
            )
            print_forecast_summary(forecast)
            if args.forecast_out:
                save_forecast(forecast, args.forecast_out)
                print(f"Forecast saved: {args.forecast_out}")

        if args.backtest:
            backtest = run_backtest(
                repo=repo.full_name,
                history_days=analysis.days,
                history_counts=analysis.counts,
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
        for day, daily in zip(analysis.days, analysis.counts):
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
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        print(f"RRD processing failed: {exc}", file=sys.stderr)
        return 1

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional, Sequence

import requests

from core.analytics import analyze_series, build_daily_series
from core.date_utils import utc_midnight_ts
from core.github_client import fetch_stars
from core.graphs import graph_advanced, graph_total_only
from core.models import RepoRef
from core.reporting import print_summary
from core.rrd import create_rrd, rrd_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", default="./out")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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

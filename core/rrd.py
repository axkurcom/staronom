from __future__ import annotations

import datetime as dt
import os
import subprocess

from core.date_utils import utc_midnight_ts


def rrd_has_correct_ds(path: str) -> bool:
    if not os.path.exists(path):
        return False

    try:
        out = subprocess.check_output(
            ["rrdtool", "info", path], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    return "ds[daily]" in out and "ds[total]" in out


def create_rrd(path: str, start_ts: int) -> None:
    if rrd_has_correct_ds(path):
        return

    if os.path.exists(path):
        os.remove(path)

    subprocess.run(
        [
            "rrdtool",
            "create",
            path,
            "--start",
            str(start_ts - 86400),
            "--step",
            "86400",
            "DS:daily:GAUGE:172800:0:U",
            "DS:total:GAUGE:172800:0:U",
            "RRA:AVERAGE:0.5:1:5000",
        ],
        check=True,
    )


def rrd_update(path: str, day: dt.date, daily: int, total: int) -> bool:
    ts = utc_midnight_ts(day)
    try:
        subprocess.run(
            ["rrdtool", "update", path, f"{ts}:{daily}:{total}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        if "illegal attempt to update using time" in stderr:
            return False
        raise RuntimeError(f"rrdtool update failed for {day}: {exc.stderr.strip()}") from exc

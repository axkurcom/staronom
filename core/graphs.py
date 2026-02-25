from __future__ import annotations

import subprocess

COLORS = {
    "daily": "#1F77B4",
    "ma7": "#D62728",
    "ma28": "#2CA02C",
    "total": "#111111",
}

GRAPH_WIDTH = 2400
GRAPH_HEIGHT = 1000


def _run_graph(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def graph_advanced(rrd_path: str, out_path: str, start_ts: str, slope: float, r2: float) -> str:
    command = [
        "rrdtool",
        "graph",
        out_path,
        "--start",
        start_ts,
        "--end",
        "now",
        "--width",
        str(GRAPH_WIDTH),
        "--height",
        str(GRAPH_HEIGHT),
        "--lower-limit",
        "0",
        "--alt-autoscale",
        "--units-exponent",
        "0",
        "--font",
        "DEFAULT:14",
        f"DEF:d={rrd_path}:daily:AVERAGE",
        f"DEF:t={rrd_path}:total:AVERAGE",
        "CDEF:ma7=d,604800,TREND",
        "CDEF:ma28=d,2419200,TREND",
        f"LINE2:d{COLORS['daily']}:daily",
        f"LINE2:ma7{COLORS['ma7']}:MA7",
        f"LINE2:ma28{COLORS['ma28']}:MA28",
        f"LINE2:t{COLORS['total']}:Total",
        f"COMMENT:Linear slope={slope:.4f} stars/day^2\\n",
        f"COMMENT:R^2={r2:.4f}\\n",
    ]
    return _run_graph(command)


def graph_total_only(rrd_path: str, out_path: str, start_ts: str) -> str:
    command = [
        "rrdtool",
        "graph",
        out_path,
        "--start",
        start_ts,
        "--end",
        "now",
        "--width",
        str(GRAPH_WIDTH),
        "--height",
        str(GRAPH_HEIGHT),
        "--alt-autoscale",
        "--units-exponent",
        "0",
        f"DEF:t={rrd_path}:total:AVERAGE",
        f"LINE3:t{COLORS['total']}:Total stars",
    ]
    return _run_graph(command)

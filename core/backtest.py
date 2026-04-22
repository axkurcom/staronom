from __future__ import annotations

import statistics
from typing import List, Optional, Sequence

from core.alerts import drop_reference_baseline, is_drop_event
from core.features import EventSignals
from core.forecast import generate_forecast
from core.models import BacktestResult


def _pinball_loss(actual: float, predicted: float, q: float) -> float:
    error = actual - predicted
    return q * error if error >= 0 else (q - 1.0) * error


def _select_cutoffs(
    n: int, horizon_days: int, min_train_days: int = 120, max_windows: int = 12
) -> List[int]:
    max_cutoff = n - horizon_days - 1
    if max_cutoff < min_train_days:
        return []

    available = max_cutoff - min_train_days + 1
    if available <= max_windows:
        return list(range(min_train_days, max_cutoff + 1))

    step = available / max_windows
    cutoffs = []
    for i in range(max_windows):
        cutoff = int(min_train_days + i * step)
        cutoff = min(cutoff, max_cutoff)
        cutoffs.append(cutoff)
    if cutoffs[-1] != max_cutoff:
        cutoffs[-1] = max_cutoff
    return sorted(set(cutoffs))


def run_backtest(
    repo: str,
    history_days: Sequence,
    history_counts: Sequence[int],
    horizon_days: int,
    interval_levels: Sequence[float],
    with_events: bool,
    event_signals: Optional[EventSignals],
) -> BacktestResult:
    cutoffs = _select_cutoffs(len(history_counts), horizon_days)
    if not cutoffs:
        return BacktestResult(
            windows=0,
            horizon_days=horizon_days,
            metrics={
                "pinball_mean": 0.0,
                "coverage_80": 0.0,
                "coverage_95": 0.0,
                "mase": 0.0,
                "drop_precision": 0.0,
                "drop_recall": 0.0,
            },
        )

    pinball_sum = 0.0
    pinball_n = 0
    cov80_hits = 0
    cov95_hits = 0
    cov_n = 0
    mase_values: List[float] = []

    tp = fp = fn = 0

    for cutoff in cutoffs:
        train_days = history_days[: cutoff + 1]
        train_counts = history_counts[: cutoff + 1]
        actual = history_counts[cutoff + 1 : cutoff + 1 + horizon_days]
        if len(actual) < horizon_days:
            continue

        forecast = generate_forecast(
            repo=repo,
            history_days=train_days,
            history_counts=train_counts,
            horizon_days=horizon_days,
            interval_levels=interval_levels,
            with_events=with_events,
            event_signals=event_signals,
            with_drop_alert=True,
            n_sims=500,
            weight_eval_points=10,
            weight_simulations=90,
        )

        denom = statistics.mean(
            [abs(train_counts[i] - train_counts[i - 1]) for i in range(1, len(train_counts))]
        )
        denom = max(denom, 1.0)
        drop_baseline = drop_reference_baseline(train_counts)

        abs_err_sum = 0.0
        for row, y in zip(forecast.rows, actual):
            y_float = float(y)

            pinball_sum += _pinball_loss(y_float, row.yhat_p50, 0.5)
            pinball_sum += _pinball_loss(y_float, row.yhat_p80_lo, 0.1)
            pinball_sum += _pinball_loss(y_float, row.yhat_p80_hi, 0.9)
            pinball_sum += _pinball_loss(y_float, row.yhat_p95_lo, 0.025)
            pinball_sum += _pinball_loss(y_float, row.yhat_p95_hi, 0.975)
            pinball_n += 5

            cov80_hits += int(row.yhat_p80_lo <= y_float <= row.yhat_p80_hi)
            cov95_hits += int(row.yhat_p95_lo <= y_float <= row.yhat_p95_hi)
            cov_n += 1

            abs_err_sum += abs(y_float - row.yhat_p50)

            is_drop = is_drop_event(y_float, drop_baseline)
            predicted_drop = bool(row.drop_alert)
            if predicted_drop and is_drop:
                tp += 1
            elif predicted_drop and not is_drop:
                fp += 1
            elif (not predicted_drop) and is_drop:
                fn += 1

        mase_values.append((abs_err_sum / horizon_days) / denom)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pinball_mean = pinball_sum / pinball_n if pinball_n else 0.0

    return BacktestResult(
        windows=len(cutoffs),
        horizon_days=horizon_days,
        metrics={
            "pinball_mean": pinball_mean,
            "coverage_80": (cov80_hits / cov_n) if cov_n else 0.0,
            "coverage_95": (cov95_hits / cov_n) if cov_n else 0.0,
            "mase": statistics.mean(mase_values) if mase_values else 0.0,
            "drop_precision": precision,
            "drop_recall": recall,
        },
    )

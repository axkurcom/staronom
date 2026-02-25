from __future__ import annotations

from typing import Optional

from core.models import AnalysisResult, BacktestResult, ForecastResult


def _streak_text(label: str, analysis: AnalysisResult, streak: tuple[int, int, int]) -> Optional[str]:
    length, start_idx, end_idx = streak
    if length <= 0:
        return None
    start_day = analysis.days[start_idx]
    end_day = analysis.days[end_idx]
    return f"{label}: {length} days ({start_day} .. {end_day})"


def _peak_text(analysis: AnalysisResult) -> str:
    if len(analysis.max_day_dates) == 1:
        return f"{analysis.max_day_count} on {analysis.max_day_dates[0]}"
    return (
        f"{analysis.max_day_count} on {analysis.max_day_dates[0]} "
        f"(+{len(analysis.max_day_dates) - 1} more days)"
    )


def print_summary(repo: str, analysis: AnalysisResult) -> None:
    span_days = len(analysis.days)
    nonzero_pct = (analysis.nonzero_days / span_days * 100.0) if span_days else 0.0

    print()
    print(f"Repo: {repo}")
    print(
        f"Range (UTC): {analysis.first_day} .. {analysis.last_day} "
        f"({span_days} days)"
    )
    print(f"Total stars (in range): {analysis.total_stars}")
    print(f"Days with >0 stars: {analysis.nonzero_days} ({nonzero_pct:.1f}%)")
    print()
    print("Daily delta stats:")
    print(f"  avg/day: {analysis.average_per_day:.3f}")
    print(f"  median:  {analysis.median_per_day:.3f}")
    print(f"  stdev:   {analysis.stdev_per_day:.3f}")
    print(f"  max/day: {_peak_text(analysis)}")
    p = analysis.percentile_by_p
    print(
        "  percentiles (stars/day): "
        f"p50={p[50]} p75={p[75]} p90={p[90]} p95={p[95]} p99={p[99]}"
    )

    nonzero_streak = _streak_text(
        "Longest streak with stars (>0)", analysis, analysis.streak_nonzero
    )
    zero_streak = _streak_text(
        "Longest streak with ZERO stars", analysis, analysis.streak_zero
    )
    if nonzero_streak or zero_streak:
        print()
    if nonzero_streak:
        print(nonzero_streak)
    if zero_streak:
        print(zero_streak)

    ma7_now = analysis.ma7[-1]
    ma7_prev = analysis.ma7[-8] if len(analysis.ma7) >= 8 else None
    if ma7_now is not None and ma7_prev is not None:
        delta = ma7_now - ma7_prev
        print()
        print(
            f"Momentum (MA7 7d ago -> now): {ma7_prev:.2f} -> {ma7_now:.2f} "
            f"(delta {delta:+.2f})"
        )

    ma28_now = analysis.ma28[-1]
    med7_now = analysis.med7[-1]
    if ma28_now is not None:
        print(f"MA28 latest: {ma28_now:.2f}")
    if med7_now is not None:
        print(f"Median7 latest: {med7_now:.2f}")

    print()
    print("Weekly averages (avg stars/day per ISO week) - last 10 weeks:")
    for (year, week), avg_w, total_w in analysis.weekly[-10:]:
        print(f"  {year}-W{week:02d}: avg/day={avg_w:.2f}  total={total_w}")


def print_forecast_summary(forecast: ForecastResult) -> None:
    if not forecast.rows:
        print()
        print("Forecast: no rows generated")
        return

    first = forecast.rows[0]
    last = forecast.rows[-1]
    print()
    print(
        f"Forecast ({forecast.horizon_days}d): "
        f"{first.forecast_date} .. {last.forecast_date}"
    )
    print(
        f"  day+1 median={first.yhat_p50:.2f} "
        f"p80=[{first.yhat_p80_lo:.2f}, {first.yhat_p80_hi:.2f}] "
        f"p95=[{first.yhat_p95_lo:.2f}, {first.yhat_p95_hi:.2f}]"
    )
    print(
        f"  day+{forecast.horizon_days} median={last.yhat_p50:.2f} "
        f"p80=[{last.yhat_p80_lo:.2f}, {last.yhat_p80_hi:.2f}] "
        f"p95=[{last.yhat_p95_lo:.2f}, {last.yhat_p95_hi:.2f}]"
    )
    if first.drop_prob is not None:
        print(
            f"  day+1 drop_prob={first.drop_prob:.3f} "
            f"drop_alert={'yes' if first.drop_alert else 'no'}"
        )
    print(
        "  weights: "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(forecast.model_weights.items()))
    )
    print(
        "  diagnostics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(forecast.diagnostics.items()))
    )


def print_backtest_summary(backtest: BacktestResult) -> None:
    print()
    print(f"Backtest windows: {backtest.windows}")
    for key in sorted(backtest.metrics.keys()):
        print(f"  {key}: {backtest.metrics[key]:.6f}")

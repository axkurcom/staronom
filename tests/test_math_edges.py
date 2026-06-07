from __future__ import annotations

import datetime as dt
import io
import unittest
from contextlib import redirect_stdout

from core.analytics import analyze_daily_series, sustained_momentum_score
from core.backtest import _select_cutoffs
from core.forecast import (
    ZINBParams,
    _simulate_zinb,
    generate_forecast,
    parse_interval_levels,
)
from core.reporting import print_summary
from core.series import build_daily_counts, build_daily_series, series_from_counts


class MathEdgeCaseTests(unittest.TestCase):
    def test_zinb_nonzero_branch_is_strictly_positive(self) -> None:
        days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(200)]
        params = ZINBParams(
            zero_prob=0.0,
            mean_nonzero=1.0,
            weekday_factors=[1.0] * 7,
            dispersion=120.0,
            event_activity_coef=0.0,
        )

        paths = _simulate_zinb(params, days, [{} for _ in days], n_sims=1, seed=1)

        self.assertTrue(paths)
        self.assertTrue(all(value > 0 for value in paths[0]))

    def test_short_momentum_history_is_not_sustained(self) -> None:
        one_day = sustained_momentum_score([10])
        seven_days = sustained_momentum_score([1] * 7)

        self.assertEqual(one_day.label, "insufficient")
        self.assertLess(one_day.score, 75.0)
        self.assertNotEqual(seven_days.label, "sustained")
        self.assertLess(seven_days.score, 75.0)

    def test_backtest_cutoffs_allow_exact_minimum_training_window(self) -> None:
        self.assertEqual(_select_cutoffs(150, 30), [119])
        self.assertEqual(_select_cutoffs(121, 1), [119])
        self.assertEqual(_select_cutoffs(120, 1), [])

    def test_interval_levels_must_be_prediction_intervals(self) -> None:
        with self.assertRaises(ValueError):
            parse_interval_levels("0.2")
        with self.assertRaises(ValueError):
            parse_interval_levels("0.5")

        days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(30)]
        counts = [0] * 30
        forecast = generate_forecast(
            "owner/repo",
            days,
            counts,
            horizon_days=1,
            interval_levels=[0.9],
            n_sims=60,
            weight_simulations=10,
        )

        self.assertEqual(forecast.quantiles, [0.8, 0.9, 0.95])
        self.assertEqual(set(forecast.rows[0].intervals), set(forecast.quantiles))

    def test_future_star_dates_are_rejected(self) -> None:
        today = dt.date(2026, 1, 1)
        tomorrow = today + dt.timedelta(days=1)
        now = dt.datetime(2026, 1, 1, 12, tzinfo=dt.timezone.utc)

        with self.assertRaises(ValueError):
            build_daily_counts([tomorrow], end_day=today)
        with self.assertRaises(ValueError):
            build_daily_series([tomorrow], now_utc=now)
        with self.assertRaises(ValueError):
            series_from_counts([tomorrow], [1], now_utc=now)

    def test_summary_nonzero_percentage_uses_model_basis_denominator(self) -> None:
        now = dt.datetime(2026, 1, 2, 12, tzinfo=dt.timezone.utc)
        series = build_daily_series(
            [dt.date(2026, 1, 1), dt.date(2026, 1, 2)],
            now_utc=now,
        )
        analysis = analyze_daily_series(series)

        output = io.StringIO()
        with redirect_stdout(output):
            print_summary("owner/repo", analysis)

        self.assertIn(
            "Days with >0 stars (distribution/model basis): 1/1 (100.0%)",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()

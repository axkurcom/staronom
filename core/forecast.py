from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.alerts import compute_drop_alerts
from core.analytics import linear_regression
from core.features import (
    EventSignals,
    build_event_rows,
    empty_event_signals,
    estimate_future_event_rows,
    event_coverage_ratio,
)
from core.models import ForecastResult, ForecastRow

DEFAULT_N_SIMS = 1200
DEFAULT_WEIGHT_EVAL_POINTS = 30
DEFAULT_WEIGHT_SIMULATIONS = 140
MAX_HORIZON_DAYS = 3650
_TREND_DECAY_TAU = 90.0
_MU_MIN = 0.01
_MU_MAX = 1e7


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _validate_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite and in range (0, 1): {value!r}")
    return float(value)


def _normalize_interval_levels(levels: Sequence[float]) -> List[float]:
    normalized = [_validate_probability(level, name="quantile") for level in levels]
    if not normalized:
        raise ValueError("quantiles list is empty")
    return sorted(set(normalized))


def _validate_history_counts(history_counts: Sequence[int]) -> None:
    def _value_for_error(value: object) -> str:
        if isinstance(value, int):
            sign = "-" if value < 0 else ""
            return f"{sign}<int bits={abs(value).bit_length()}>"
        try:
            rendered = repr(value)
        except Exception:
            return f"<unreprable {type(value).__name__}>"
        if len(rendered) > 120:
            return rendered[:117] + "..."
        return rendered

    for idx, count in enumerate(history_counts):
        count_text = _value_for_error(count)
        try:
            value = float(count)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "history_counts must contain finite non-negative whole-number values; "
                f"got {count_text} at index {idx}"
            ) from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "history_counts must contain finite non-negative whole-number values; "
                f"got {count_text} at index {idx}"
            )
        if not value.is_integer():
            raise ValueError(
                "history_counts must contain finite non-negative whole-number values; "
                f"got {count_text} at index {idx}"
            )


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")

    out: List[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        width = min(window, i + 1)
        out.append(running / width)
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))

    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _pinball_loss(actual: float, predicted: float, q: float) -> float:
    error = actual - predicted
    return q * error if error >= 0 else (q - 1.0) * error


def _poisson_sample(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0

    if lam < 30.0:
        limit = math.exp(-lam)
        prod = 1.0
        k = 0
        while prod > limit:
            prod *= rng.random()
            k += 1
        return max(0, k - 1)

    # Gaussian approximation is stable enough for large lambda.
    draw = rng.gauss(lam, math.sqrt(lam))
    return max(0, int(round(draw)))


def _nb_sample(mean: float, dispersion: float, rng: random.Random) -> int:
    if mean <= 0:
        return 0
    if dispersion <= 0:
        return _poisson_sample(mean, rng)
    if dispersion >= 1e5:
        return _poisson_sample(mean, rng)

    lam = rng.gammavariate(dispersion, mean / dispersion)
    return _poisson_sample(lam, rng)


def _regularized_weekday_factors(days: Sequence[dt.date], counts: Sequence[int]) -> List[float]:
    overall = statistics.mean(counts) if counts else 0.0
    overall = max(overall, 0.2)
    by_day: Dict[int, List[int]] = {i: [] for i in range(7)}
    for day, count in zip(days, counts):
        by_day[day.weekday()].append(count)

    factors: List[float] = []
    for weekday in range(7):
        sample = by_day[weekday]
        if not sample:
            factors.append(1.0)
            continue
        weekday_mean = sum(sample) / len(sample)
        # Shrink toward 1.0 to avoid unstable high-variance weekday effects.
        factor = (weekday_mean + 2.0 * overall) / (3.0 * overall)
        factors.append(_clip(factor, 0.3, 3.5))
    return factors


@dataclass(frozen=True)
class DynamicNBParams:
    level: float
    trend: float
    weekday_factors: List[float]
    dispersion: float
    ar: float
    event_coef: Dict[str, float]
    event_scale: Dict[str, float]


def _fit_dynamic_nb(
    days: Sequence[dt.date],
    counts: Sequence[int],
    event_rows: Sequence[Dict[str, float]],
    use_events: bool,
) -> DynamicNBParams:
    if not counts:
        return DynamicNBParams(
            level=0.1,
            trend=0.0,
            weekday_factors=[1.0] * 7,
            dispersion=2.0,
            ar=0.0,
            event_coef={},
            event_scale={},
        )

    alpha = 0.22
    level = float(counts[0])
    for value in counts[1:]:
        level = alpha * value + (1.0 - alpha) * level

    smoothed = _moving_average([float(c) for c in counts], 7)
    lookback = min(120, len(smoothed))
    log_smoothed = [math.log1p(v) for v in smoothed[-lookback:]]
    trend, _, _ = linear_regression(log_smoothed)
    trend = _clip(trend, -0.06, 0.06)

    weekday_factors = _regularized_weekday_factors(days, counts)

    mu = statistics.mean(counts)
    var = statistics.pvariance(counts) if len(counts) > 1 else mu
    if var > mu and mu > 0:
        dispersion = (mu * mu) / max(var - mu, 1e-6)
    else:
        dispersion = 250.0
    dispersion = _clip(dispersion, 0.6, 500.0)

    ar = 0.0
    if len(counts) > 3:
        x = counts[:-1]
        y = counts[1:]
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        if var_x > 1e-9:
            cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            ar = _clip(cov_xy / var_x, 0.0, 0.75)

    event_coef: Dict[str, float] = {}
    event_scale: Dict[str, float] = {}
    if use_events and event_rows:
        base_features = ("release_roll14", "commits_roll3", "issues_roll7", "prs_roll7")
        residual = [counts[i] - smoothed[i] for i in range(len(counts))]
        residual_center = statistics.mean(residual)
        for key in base_features:
            xs = [float(row.get(key, 0.0)) for row in event_rows]
            scale = statistics.mean(xs) if xs else 0.0
            scale = max(scale, 1.0)
            mean_x = statistics.mean(xs) if xs else 0.0
            centered_x = [x - mean_x for x in xs]
            var_x = sum(v * v for v in centered_x)
            if var_x < 1e-9:
                event_coef[key] = 0.0
                event_scale[key] = scale
                continue
            cov = sum((cx) * (r - residual_center) for cx, r in zip(centered_x, residual))
            coef = cov / var_x
            coef /= max(mu, 1.0)
            event_coef[key] = _clip(coef, -0.40, 0.40)
            event_scale[key] = scale

    return DynamicNBParams(
        level=max(level, 0.05),
        trend=trend,
        weekday_factors=weekday_factors,
        dispersion=dispersion,
        ar=ar,
        event_coef=event_coef,
        event_scale=event_scale,
    )


def _dynamic_nb_mean(
    params: DynamicNBParams,
    day: dt.date,
    step: int,
    prev: float,
    event_row: Dict[str, float],
) -> float:
    event_effect = 0.0
    for key, coef in params.event_coef.items():
        scale = params.event_scale.get(key, 1.0)
        event_effect += coef * (event_row.get(key, 0.0) / scale)

    step_float = max(float(step), 0.0)
    trend_effect = params.trend * _TREND_DECAY_TAU * (1.0 - math.exp(-step_float / _TREND_DECAY_TAU))
    weekday_factor = max(params.weekday_factors[day.weekday()], 1e-9)
    log_base = math.log(max(params.level, 1e-9)) + math.log(weekday_factor) + trend_effect + event_effect
    log_base = _clip(log_base, math.log(_MU_MIN), math.log(_MU_MAX))
    base = math.exp(log_base)

    safe_prev = prev if math.isfinite(prev) else base
    mu = (1.0 - params.ar) * base + params.ar * safe_prev
    return _clip(mu, _MU_MIN, _MU_MAX)


def _simulate_dynamic_nb(
    params: DynamicNBParams,
    future_days: Sequence[dt.date],
    future_event_rows: Sequence[Dict[str, float]],
    n_sims: int,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(seed)
    out: List[List[int]] = []
    for _ in range(n_sims):
        prev = params.level
        path: List[int] = []
        for i, day in enumerate(future_days, start=1):
            event_row = future_event_rows[i - 1] if i - 1 < len(future_event_rows) else {}
            mu = _dynamic_nb_mean(params, day, i, prev, event_row)
            value = _nb_sample(mu, params.dispersion, rng)
            path.append(value)
            prev = float(value)
        out.append(path)
    return out


@dataclass(frozen=True)
class ZINBParams:
    zero_prob: float
    mean_nonzero: float
    weekday_factors: List[float]
    dispersion: float
    event_activity_coef: float


def _fit_zinb(
    days: Sequence[dt.date],
    counts: Sequence[int],
    event_rows: Sequence[Dict[str, float]],
    use_events: bool,
) -> ZINBParams:
    if not counts:
        return ZINBParams(0.8, 0.2, [1.0] * 7, 2.0, 0.0)

    recent = counts[-90:] if len(counts) > 90 else counts
    zero_prob = sum(1 for x in recent if x == 0) / len(recent)
    zero_prob = _clip(zero_prob, 0.05, 0.98)

    nonzero = [x for x in counts if x > 0]
    mean_nonzero = statistics.mean(nonzero) if nonzero else 0.5
    mean_nonzero = max(mean_nonzero, 0.1)

    nz_var = statistics.pvariance(nonzero) if len(nonzero) > 1 else mean_nonzero
    if nz_var > mean_nonzero:
        dispersion = (mean_nonzero * mean_nonzero) / max(nz_var - mean_nonzero, 1e-6)
    else:
        dispersion = 120.0
    dispersion = _clip(dispersion, 0.5, 500.0)

    weekday_factors = _regularized_weekday_factors(days, [max(1, c) for c in counts])
    event_activity_coef = 0.0
    if use_events and event_rows:
        activity = [row.get("activity", 0.0) for row in event_rows]
        if activity:
            activity_mean = statistics.mean(activity)
            if activity_mean > 0:
                ratio = activity_mean / (activity_mean + 5.0)
                event_activity_coef = _clip(0.25 * ratio, 0.0, 0.25)

    return ZINBParams(
        zero_prob=zero_prob,
        mean_nonzero=mean_nonzero,
        weekday_factors=weekday_factors,
        dispersion=dispersion,
        event_activity_coef=event_activity_coef,
    )


def _simulate_zinb(
    params: ZINBParams,
    future_days: Sequence[dt.date],
    future_event_rows: Sequence[Dict[str, float]],
    n_sims: int,
    seed: int,
) -> List[List[int]]:
    rng = random.Random(seed)
    out: List[List[int]] = []
    for _ in range(n_sims):
        path: List[int] = []
        for i, day in enumerate(future_days):
            event_row = future_event_rows[i] if i < len(future_event_rows) else {}
            activity = event_row.get("activity", 0.0)
            zero_prob = params.zero_prob * math.exp(-params.event_activity_coef * activity)
            zero_prob = _clip(zero_prob, 0.02, 0.995)
            if rng.random() < zero_prob:
                path.append(0)
                continue
            mu = params.mean_nonzero * params.weekday_factors[day.weekday()]
            path.append(_nb_sample(mu, params.dispersion, rng))
        out.append(path)
    return out


@dataclass(frozen=True)
class BaselineParams:
    weekday_means: List[float]
    dispersion: float


def _fit_baseline(days: Sequence[dt.date], counts: Sequence[int]) -> BaselineParams:
    if not counts:
        return BaselineParams([0.2] * 7, 2.0)

    lookback = min(84, len(counts))
    ref_days = days[-lookback:]
    ref_counts = counts[-lookback:]
    by_day: Dict[int, List[int]] = {i: [] for i in range(7)}
    for day, count in zip(ref_days, ref_counts):
        by_day[day.weekday()].append(count)

    global_mean = statistics.mean(ref_counts)
    weekday_means: List[float] = []
    for weekday in range(7):
        sample = by_day[weekday]
        weekday_means.append((sum(sample) / len(sample)) if sample else global_mean)

    var = statistics.pvariance(ref_counts) if len(ref_counts) > 1 else global_mean
    if var > global_mean:
        dispersion = (global_mean * global_mean) / max(var - global_mean, 1e-6)
    else:
        dispersion = 200.0
    dispersion = _clip(dispersion, 0.5, 500.0)

    return BaselineParams(weekday_means=weekday_means, dispersion=dispersion)


def _simulate_baseline(
    params: BaselineParams, future_days: Sequence[dt.date], n_sims: int, seed: int
) -> List[List[int]]:
    rng = random.Random(seed)
    out: List[List[int]] = []
    for _ in range(n_sims):
        path = [
            _nb_sample(params.weekday_means[day.weekday()], params.dispersion, rng)
            for day in future_days
        ]
        out.append(path)
    return out


def _pinball_targets_from_levels(levels: Sequence[float]) -> List[float]:
    qs = {0.5}
    for level in levels:
        if level <= 0.5 or level >= 1.0:
            continue
        tail = (1.0 - level) / 2.0
        qs.add(tail)
        qs.add(1.0 - tail)
    return sorted(qs)


def _loss_for_model(
    model_name: str,
    days: Sequence[dt.date],
    counts: Sequence[int],
    event_rows: Sequence[Dict[str, float]],
    use_events: bool,
    eval_points: int,
    quantiles: Sequence[float],
    simulations: int,
    seed: int,
) -> float:
    if len(counts) < 20:
        return 1.0

    start = max(14, len(counts) - eval_points - 1)
    end = len(counts) - 1
    losses: List[float] = []
    for split in range(start, end):
        hist_days = days[: split + 1]
        hist_counts = counts[: split + 1]
        hist_events = event_rows[: split + 1]
        next_day = days[split + 1]
        next_event = event_rows[split + 1]

        if model_name == "dynamic_nb":
            params = _fit_dynamic_nb(hist_days, hist_counts, hist_events, use_events)
            sims = _simulate_dynamic_nb(
                params, [next_day], [next_event], simulations, seed + split
            )
        elif model_name == "zinb":
            params = _fit_zinb(hist_days, hist_counts, hist_events, use_events)
            sims = _simulate_zinb(params, [next_day], [next_event], simulations, seed + split)
        elif model_name == "baseline":
            params = _fit_baseline(hist_days, hist_counts)
            sims = _simulate_baseline(params, [next_day], simulations, seed + split)
        else:
            raise ValueError(f"unknown model: {model_name}")

        predictions = [path[0] for path in sims]
        actual = float(counts[split + 1])
        loss = 0.0
        for q in quantiles:
            pred_q = _quantile(predictions, q)
            loss += _pinball_loss(actual, pred_q, q)
        losses.append(loss / len(quantiles))

    return statistics.mean(losses) if losses else 1.0


def _compute_model_weights(
    days: Sequence[dt.date],
    counts: Sequence[int],
    event_rows: Sequence[Dict[str, float]],
    use_events: bool,
    interval_levels: Sequence[float],
    eval_points: int,
    simulations: int,
    seed: int = 12345,
) -> Dict[str, float]:
    quantiles = _pinball_targets_from_levels(interval_levels)
    model_names = ("dynamic_nb", "zinb", "baseline")
    losses = {
        model: _loss_for_model(
            model,
            days,
            counts,
            event_rows,
            use_events,
            eval_points,
            quantiles,
            simulations,
            seed + i * 1000,
        )
        for i, model in enumerate(model_names)
    }
    scores = {model: 1.0 / max(loss, 1e-6) for model, loss in losses.items()}
    total = sum(scores.values())
    if total <= 0:
        return {model: 1.0 / len(model_names) for model in model_names}
    return {model: scores[model] / total for model in model_names}


def _collect_day_samples(paths: Sequence[Sequence[int]], day_index: int) -> List[float]:
    return [float(path[day_index]) for path in paths if day_index < len(path)]


def _interval(samples: Sequence[float], level: float) -> Tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return _quantile(samples, tail), _quantile(samples, 1.0 - tail)


def _allocate_simulations(weights: Dict[str, float], total_sims: int) -> Dict[str, int]:
    names = list(weights.keys())
    allocations = {
        name: max(1, int(math.floor(total_sims * max(weights[name], 0.0)))) for name in names
    }
    current = sum(allocations.values())
    while current < total_sims:
        name = max(names, key=lambda key: weights[key])
        allocations[name] += 1
        current += 1
    while current > total_sims:
        name = max(names, key=lambda key: allocations[key])
        if allocations[name] > 1:
            allocations[name] -= 1
            current -= 1
        else:
            break
    return allocations


def generate_forecast(
    repo: str,
    history_days: Sequence[dt.date],
    history_counts: Sequence[int],
    horizon_days: int = 30,
    interval_levels: Optional[Sequence[float]] = None,
    with_events: bool = False,
    event_signals: Optional[EventSignals] = None,
    with_drop_alert: bool = False,
    n_sims: int = DEFAULT_N_SIMS,
    weight_eval_points: int = DEFAULT_WEIGHT_EVAL_POINTS,
    weight_simulations: int = DEFAULT_WEIGHT_SIMULATIONS,
) -> ForecastResult:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if horizon_days > MAX_HORIZON_DAYS:
        raise ValueError(f"horizon_days must be <= {MAX_HORIZON_DAYS}")
    if not history_days or not history_counts:
        raise ValueError("history_days and history_counts must not be empty")
    if len(history_days) != len(history_counts):
        raise ValueError("history_days and history_counts length mismatch")
    _validate_history_counts(history_counts)

    levels = list(interval_levels or [0.5, 0.8, 0.95])
    if 0.5 not in levels:
        levels.append(0.5)
    if 0.8 not in levels:
        levels.append(0.8)
    if 0.95 not in levels:
        levels.append(0.95)
    levels = _normalize_interval_levels(levels)

    signals = event_signals or empty_event_signals()
    history_event_rows = build_event_rows(history_days, signals)
    coverage = event_coverage_ratio(history_event_rows)
    use_events = with_events and coverage >= 0.03
    future_days, future_event_rows = estimate_future_event_rows(
        history_days, history_event_rows, horizon_days
    )

    weights = _compute_model_weights(
        history_days,
        history_counts,
        history_event_rows,
        use_events=use_events,
        interval_levels=levels,
        eval_points=min(weight_eval_points, max(len(history_counts) - 2, 1)),
        simulations=weight_simulations,
    )

    dynamic_params = _fit_dynamic_nb(history_days, history_counts, history_event_rows, use_events)
    zinb_params = _fit_zinb(history_days, history_counts, history_event_rows, use_events)
    baseline_params = _fit_baseline(history_days, history_counts)

    allocation = _allocate_simulations(weights, total_sims=max(n_sims, 60))
    sim_paths: List[List[int]] = []
    seed = 90210
    if allocation["dynamic_nb"] > 0:
        sim_paths.extend(
            _simulate_dynamic_nb(
                dynamic_params,
                future_days,
                future_event_rows,
                allocation["dynamic_nb"],
                seed,
            )
        )
    if allocation["zinb"] > 0:
        sim_paths.extend(
            _simulate_zinb(
                zinb_params,
                future_days,
                future_event_rows,
                allocation["zinb"],
                seed + 17,
            )
        )
    if allocation["baseline"] > 0:
        sim_paths.extend(
            _simulate_baseline(
                baseline_params,
                future_days,
                allocation["baseline"],
                seed + 41,
            )
        )

    if not sim_paths:
        rng = random.Random(seed)
        fallback = [
            [
                _poisson_sample(max(statistics.mean(history_counts), 0.1), rng)
                for _ in range(horizon_days)
            ]
            for _ in range(60)
        ]
        sim_paths = fallback

    p50_series: List[float] = []
    rows: List[ForecastRow] = []
    for day_index, day in enumerate(future_days):
        day_samples = _collect_day_samples(sim_paths, day_index)
        p50 = _quantile(day_samples, 0.5)
        p80_lo, p80_hi = _interval(day_samples, 0.8)
        p95_lo, p95_hi = _interval(day_samples, 0.95)
        p50_series.append(p50)
        rows.append(
            ForecastRow(
                forecast_date=day,
                yhat_p50=p50,
                yhat_p80_lo=p80_lo,
                yhat_p80_hi=p80_hi,
                yhat_p95_lo=p95_lo,
                yhat_p95_hi=p95_hi,
            )
        )

    diagnostics: Dict[str, float] = {
        "event_coverage_ratio": coverage,
        "events_used_in_model": 1.0 if use_events else 0.0,
        "simulations": float(len(sim_paths)),
    }

    if with_drop_alert:
        drop_prob, drop_alert, alert_diag = compute_drop_alerts(history_counts, p50_series)
        diagnostics.update(alert_diag)
        updated_rows: List[ForecastRow] = []
        for row, prob, alert in zip(rows, drop_prob, drop_alert):
            updated_rows.append(
                ForecastRow(
                    forecast_date=row.forecast_date,
                    yhat_p50=row.yhat_p50,
                    yhat_p80_lo=row.yhat_p80_lo,
                    yhat_p80_hi=row.yhat_p80_hi,
                    yhat_p95_lo=row.yhat_p95_lo,
                    yhat_p95_hi=row.yhat_p95_hi,
                    drop_prob=prob,
                    drop_alert=alert,
                )
            )
        rows = updated_rows

    return ForecastResult(
        repo=repo,
        generated_at_utc=dt.datetime.now(tz=dt.timezone.utc),
        horizon_days=horizon_days,
        quantiles=levels,
        rows=rows,
        model_weights=weights,
        diagnostics=diagnostics,
    )


def save_forecast(result: ForecastResult, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix.lower() == ".csv":
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "forecast_date",
                    "yhat_p50",
                    "yhat_p80_lo",
                    "yhat_p80_hi",
                    "yhat_p95_lo",
                    "yhat_p95_hi",
                    "drop_prob",
                    "drop_alert",
                ]
            )
            for row in result.rows:
                writer.writerow(
                    [
                        row.forecast_date.isoformat(),
                        f"{row.yhat_p50:.6f}",
                        f"{row.yhat_p80_lo:.6f}",
                        f"{row.yhat_p80_hi:.6f}",
                        f"{row.yhat_p95_lo:.6f}",
                        f"{row.yhat_p95_hi:.6f}",
                        "" if row.drop_prob is None else f"{row.drop_prob:.6f}",
                        "" if row.drop_alert is None else int(row.drop_alert),
                    ]
                )
        return

    payload = {
        "repo": result.repo,
        "generated_at_utc": result.generated_at_utc.isoformat(),
        "horizon_days": result.horizon_days,
        "quantiles": result.quantiles,
        "model_weights": result.model_weights,
        "diagnostics": result.diagnostics,
        "rows": [
            {
                "forecast_date": row.forecast_date.isoformat(),
                "yhat_p50": row.yhat_p50,
                "yhat_p80_lo": row.yhat_p80_lo,
                "yhat_p80_hi": row.yhat_p80_hi,
                "yhat_p95_lo": row.yhat_p95_lo,
                "yhat_p95_hi": row.yhat_p95_hi,
                "drop_prob": row.drop_prob,
                "drop_alert": row.drop_alert,
            }
            for row in result.rows
        ],
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def parse_interval_levels(raw: str) -> List[float]:
    if not raw:
        return [0.5, 0.8, 0.95]
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return _normalize_interval_levels(values)

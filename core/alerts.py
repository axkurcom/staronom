from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence, Tuple


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    if window <= 0:
        raise ValueError("window must be > 0")

    out: List[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        width = min(i + 1, window)
        out.append(running / width)
    return out


def _markov_drop_probabilities(history_counts: Sequence[int], horizon: int) -> List[float]:
    if horizon <= 0:
        return []
    if len(history_counts) < 10:
        return [0.1] * horizon

    ma7 = _moving_average([float(x) for x in history_counts], 7)
    slopes = [ma7[i] - ma7[i - 1] for i in range(1, len(ma7))]
    if not slopes:
        return [0.1] * horizon

    slope_std = statistics.pstdev(slopes) if len(slopes) > 1 else abs(slopes[0])
    threshold = -max(0.15, 0.35 * slope_std)
    states = [1 if slope < threshold else 0 for slope in slopes]

    c00 = c01 = c10 = c11 = 1.0
    for prev, nxt in zip(states, states[1:]):
        if prev == 0 and nxt == 0:
            c00 += 1
        elif prev == 0 and nxt == 1:
            c01 += 1
        elif prev == 1 and nxt == 0:
            c10 += 1
        else:
            c11 += 1

    p01 = c01 / (c00 + c01)
    p11 = c11 / (c10 + c11)
    prob_drop = 0.8 if states[-1] == 1 else 0.2

    projected: List[float] = []
    for _ in range(horizon):
        prob_drop = prob_drop * p11 + (1.0 - prob_drop) * p01
        projected.append(prob_drop)
    return projected


def _cusum_drop_probability(history_counts: Sequence[int]) -> float:
    if len(history_counts) < 14:
        return 0.1

    ma7 = _moving_average([float(x) for x in history_counts], 7)
    residuals = [history_counts[i] - ma7[i] for i in range(len(history_counts))]
    recent = residuals[-120:]
    residual_std = statistics.pstdev(recent) if len(recent) > 1 else 1.0
    residual_std = max(residual_std, 1e-6)

    drift = 0.5 * residual_std
    stat = 0.0
    for residual in recent:
        stat = min(0.0, stat + residual + drift)

    # Strongly negative one-sided CUSUM means a likely slowdown regime.
    return _sigmoid((-stat) / (3.0 * residual_std))


def compute_drop_alerts(
    history_counts: Sequence[int], forecast_p50: Sequence[float]
) -> Tuple[List[float], List[bool], Dict[str, float]]:
    horizon = len(forecast_p50)
    if horizon == 0:
        return [], [], {"markov_p1": 0.0, "cusum_prob": 0.0}

    markov_probs = _markov_drop_probabilities(history_counts, horizon)
    cusum_prob = _cusum_drop_probability(history_counts)

    baseline_window = history_counts[-28:] if len(history_counts) >= 28 else history_counts
    baseline = statistics.mean(baseline_window) if baseline_window else 0.0
    scale_window = history_counts[-56:] if len(history_counts) >= 56 else history_counts
    scale = statistics.pstdev(scale_window) if len(scale_window) > 1 else max(baseline, 1.0)
    scale = max(scale, 1.0)

    probs: List[float] = []
    alerts: List[bool] = []
    for i, pred in enumerate(forecast_p50):
        pressure = _sigmoid((baseline - pred) / scale)
        prob = 0.45 * markov_probs[i] + 0.35 * cusum_prob + 0.20 * pressure
        alert = prob >= 0.82 and markov_probs[i] >= 0.70 and cusum_prob >= 0.65
        probs.append(prob)
        alerts.append(alert)

    return probs, alerts, {"markov_p1": markov_probs[0], "cusum_prob": cusum_prob}

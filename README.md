# Staronom

Staronom is a GitHub stars analytics tool. It fetches starred-at timestamps,
builds a UTC daily time series, produces operational summary statistics and
graphs, and can run simulation-based forecasts with backtest diagnostics.

It is designed for repository growth analysis: momentum, quiet periods, weekday
shape, forecast intervals, and early slowdown signals. It is not a causal model
of popularity and it does not promise exact future star counts.

## Quick Start

On Debian:

```bash
apt install -y python3 python3-requests rrdtool
git clone https://github.com/axkurcom/staronom
cd staronom
python3 staronom.py --repo telemt/telemt
```

For higher GitHub API limits, pass a token or set `GITHUB_TOKEN`:

```bash
GITHUB_TOKEN=ghp_... python3 staronom.py --repo owner/repo
```

Forecast with intervals, drop alerts, and backtest:

```bash
python3 staronom.py \
  --repo owner/repo \
  --forecast \
  --horizon 30 \
  --intervals 0.8,0.95,0.99 \
  --drop-alert \
  --backtest \
  --forecast-out ./out/forecast.json
```

## What It Produces

The default run prints:

- UTC range and total stars in the range.
- Average stars/day, median, standard deviation, max day, and percentiles.
- Nonzero and zero-day streaks.
- MA7, MA28, Median7, and weekly averages.
- RRD-backed graphs in `./out`:
  - `stars_advanced.png`: daily stars plus MA7 and MA28.
  - `stars_total.png`: cumulative total stars.
  - `stars_daily.rrd`: local RRD data store.

With `--forecast`, it also prints forecast medians, interval bands, model
weights, and diagnostics. With `--forecast-out`, it writes JSON by default or
CSV when the output path ends in `.csv`.

## Current Analytical Approach

### Daily Series

Staronom works in UTC days. The daily series is closed through the current UTC
day so that quiet trailing days are not silently ignored.

The current partial day is treated explicitly:

- Raw counts are kept for totals, RRD updates, and graph continuity.
- Average-style metrics use exposure-adjusted denominators. A day that is 6
  hours old contributes 0.25 effective days.
- To avoid first-minutes flapping, nonzero current-day rates use a 3-hour
  minimum exposure floor.
- Forecasting, backtesting, and drop alerts use completed UTC days only. This
  prevents a normal partial day from looking like a full-day slowdown.
- Forecast dates are still anchored after the visible current day, so a run on
  today forecasts from tomorrow onward.

This split is implemented around `DailySeries`: raw series, reporting exposure,
and model-safe history are separate fields instead of implicit conventions.

### Descriptive Analytics

The descriptive layer is intentionally conservative:

- Averages and moving averages account for current-day exposure.
- Distribution metrics such as median, standard deviation, percentiles, max day,
  and streaks are computed on completed-day history.
- Weekly averages use effective exposure for the current week.
- Linear trend and graph annotations are based on completed-day statistics, not
  intraday noise.

The goal is to make the summary stable enough for repeated daily use while still
showing what has happened so far today.

### Forecasting

Forecasts are simulation-based and use an ensemble of three models:

- Dynamic negative-binomial model: smoothed level, bounded log trend,
  regularized weekday factors, autoregressive carryover, and optional event
  covariates.
- Zero-inflated negative-binomial model: separates quiet days from active days
  and models weekday effects for nonzero activity.
- Recent weekday baseline: uses recent weekday means with negative-binomial
  dispersion.

Model weights are learned from recent rolling pinball-loss evaluation, not fixed
by hand. Forecast intervals are empirical quantiles from the combined simulation
paths. The default intervals are 80% and 95%; custom levels such as `0.99` are
propagated into JSON, CSV, and backtest coverage metrics.

The forecast horizon is capped at 3650 days to avoid numerically meaningless
long-range runs.

### Event-Aware Mode

With `--with-events`, Staronom fetches recent GitHub repository events:

- releases;
- push activity;
- opened issues;
- opened pull requests.

These signals are converted into rolling features and used only when recent
event coverage is high enough. Coverage is measured over the recent observable
window rather than the full lifetime of the repository, so older repositories do
not lose event-aware behavior just because their star history is long.

Event features are useful as weak context, not as causal proof. GitHub's events
API is recent-window limited, and the model treats events as explanatory signals
inside a bounded forecast heuristic.

### Backtesting

`--backtest` runs rolling historical forecast windows and reports:

- mean pinball loss;
- 80% and 95% coverage;
- coverage for extra requested intervals;
- MASE;
- drop-alert precision and recall.

Backtests truncate event signals at each historical cutoff to avoid future-event
leakage. Drop labels use the same baseline logic as production drop alerts.

### Drop Alerts

`--drop-alert` combines:

- a Markov-style slowdown state from MA7 slope behavior;
- one-sided CUSUM pressure;
- forecast pressure versus a recent baseline.

Alerts are intentionally conservative: probability alone is not enough; the
Markov and CUSUM components must also agree.

## Strength And Depth Today

Staronom is strongest as an operational analytics tool for GitHub star history:

- It handles trailing zero days and the current partial day explicitly.
- It separates reporting math from model training history.
- It has bounded numerical behavior for horizons, probabilities, trend growth,
  and invalid history counts.
- It supports interval forecasts and backtests instead of only point estimates.
- It includes basic event-aware context without leaking future events into
  backtests.

The current depth is solid for repository-level growth monitoring, comparing
recent momentum to historical behavior, and detecting broad slowdown risk. It is
not yet a full statistical forecasting platform:

- Forecast intervals are simulation heuristics, not calibrated Bayesian
  posteriors.
- GitHub event data is incomplete for long history because the events API is
  recent-window based.
- Event features should be interpreted as correlations, not causal drivers.
- Very young repositories and sparse histories still have high uncertainty.
- Graph generation depends on local `rrdtool`.

## CLI Options

```text
--repo OWNER/REPO       Required GitHub repository.
--out PATH             Output directory, default ./out.
--token TOKEN          GitHub token; defaults to GITHUB_TOKEN.
--forecast             Run forecast.
--horizon DAYS         Forecast horizon, default 30, max 3650.
--with-events          Include recent repository event features.
--intervals LIST       Forecast interval levels, default 0.8,0.95.
--forecast-out PATH    Save forecast JSON or CSV.
--drop-alert           Add slowdown/drop alert probabilities.
--backtest             Run rolling historical backtest.
```

`--quantiles` is retained as a hidden compatibility alias for `--intervals`.

## Example Graph

Example of advanced analytics on the
[telemt](https://github.com/telemt/telemt) repository:

<img width="2517" height="1116" alt="stars_advanced" src="https://github.com/user-attachments/assets/e67e3b31-b245-4bda-93dc-16d20568cc92" />

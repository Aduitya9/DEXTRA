"""
Fixed-length windows for the sequence models: 14 days of every variate in, three
days of AQI out.

Notebook 03's tabular samples flatten history into lags and rolling means, which
is what a tree wants. A transformer wants the series itself, so this builds
(lookback, variates) tensors instead. The guarantees are the same ones and for
the same reasons:

  A window never spans two cities. Windows are cut inside each city's own sorted
  series, so one city's last fortnight cannot become another's first.

  Nothing after day t is in the input. The window ends at t and the targets start
  at t+1, with no overlap of even one day.

  A sample belongs to the split of its furthest target. Taking the origin instead
  would let a window made in December 2023 put its January 2024 answers into the
  training set, and taking the nearest target would do the same for the two days
  behind it.

Gaps are filled forward inside each city and then from the pooled training
median, never from the city's own future and never from a city's own held-out
years. A training median is the only summary available at the time a forecast
would really be made.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import ALL_FEATURES, FORECAST_HORIZON, LOOKBACK_WINDOW

TARGET = 'AQI'
FORWARD_FILL_LIMIT = 7


def pooled_training_medians(daily, columns, train_split='train'):
    """One median per column from the training split, for cities and days with gaps.

    Pooled across the training cities rather than taken per city: the five unseen
    cities have no training rows at all, and filling their gaps from their own
    record would hand them years they are not allowed to see.
    """
    train = daily[daily['split'] == train_split]
    source = train if len(train) else daily
    return source[columns].median(numeric_only=True)


def fill_gaps(daily, columns, medians, limit=FORWARD_FILL_LIMIT):
    """Carry the last reading forward within a city, then fall back to the median."""
    out = daily.sort_values(['city', 'date']).copy()
    out[columns] = out.groupby('city', sort=False)[columns].ffill(limit=limit)
    out[columns] = out[columns].fillna(medians)
    remaining = [c for c in columns if out[c].isna().any()]
    if remaining:
        out[remaining] = out[remaining].fillna(0.0)
    return out


def build_sequences(daily, lookback=LOOKBACK_WINDOW, horizon=FORECAST_HORIZON,
                    target=TARGET, features=None, medians=None):
    """Windows for every city-day with a full history behind it and real answers ahead.

    Returns (X, y, index) where X is (n, lookback, variates) float32, y is
    (n, horizon) float32, and index is a frame of city, date (the day the forecast
    is made), last_target_date and split.

    The date column is named last_target_date rather than target_date because one
    row carries every horizon at once: y[:, 0] lands a day after `date` and
    y[:, -1] lands on last_target_date. Joining per-horizon output against
    anything else means rebuilding the date for that horizon, not reusing this one.
    """
    features = [f for f in (features or ALL_FEATURES) if f in daily.columns]
    if target not in features:
        features = features + [target]          # the model reads the target's own history

    daily = daily.sort_values(['city', 'date']).reset_index(drop=True)
    medians = pooled_training_medians(daily, features) if medians is None else medians
    filled = fill_gaps(daily, features, medians)

    windows, answers, rows = [], [], []
    for city, group in filled.groupby('city', sort=False):
        group = group.sort_values('date').reset_index(drop=True)
        total = len(group)
        count = total - lookback - horizon + 1
        if count <= 0:
            continue

        values = group[features].to_numpy(dtype=np.float32)
        # The answers come from the ORIGINAL column, so a gap that was filled for
        # the input never becomes something the model is scored on having predicted.
        truth = daily.loc[daily['city'] == city].sort_values('date')[target] \
                     .to_numpy(dtype=np.float32)

        strided = sliding_window_view(values, lookback, axis=0)[:count]   # (n, N, L)
        windows.append(strided.transpose(0, 2, 1))                       # (n, L, N)
        answers.append(sliding_window_view(truth, horizon)[lookback:lookback + count])

        last_input = np.arange(count) + lookback - 1
        last_target = last_input + horizon
        rows.append(pd.DataFrame({
            'city': city,
            'date': group['date'].to_numpy()[last_input],
            'last_target_date': group['date'].to_numpy()[last_target],
            'split': group['split'].to_numpy()[last_target] if 'split' in group else 'unassigned',
        }))

    if not windows:
        empty = np.empty((0, lookback, len(features)), dtype=np.float32)
        return empty, np.empty((0, horizon), dtype=np.float32), pd.DataFrame()

    X = np.concatenate(windows)
    y = np.concatenate(answers)
    index = pd.concat(rows, ignore_index=True)

    usable = np.isfinite(y).all(axis=1) & np.isfinite(X).all(axis=(1, 2))
    return X[usable], y[usable], index[usable].reset_index(drop=True)


def feature_names(daily, features=None, target=TARGET):
    """The variate order the tensors use, so a saved model can be read back."""
    names = [f for f in (features or ALL_FEATURES) if f in daily.columns]
    return names if target in names else names + [target]

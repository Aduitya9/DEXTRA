"""
Turn the city-day table into supervised samples: history up to day t, AQI at t+1..t+3.

Three rules keep the samples honest, and all three are easy to get wrong:

  Nothing after t may describe the input. Every rolling window ends at t, every
  lag reaches backwards, and the shifts are taken inside each city so one city's
  last days never become another city's history.

  A sample belongs to the split of the day it predicts, not the day it is made
  from. A forecast issued on 30 December 2023 for 2 January 2024 is a validation
  sample that happens to read training history -- which is exactly the position a
  real forecaster is in. Assigning it by its origin would put a 2024 answer in
  the training set.

  The calendar is known in advance. Day of week, month, Diwali and the stubble
  season are taken at the target date, because a forecaster on 1 November knows
  perfectly well that 4 November is Diwali. Everything measured -- pollutants,
  meteorology, fire -- is taken at t, because that is all that has happened.
"""

import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, FORECAST_HORIZON, LOOKBACK_WINDOW
from src.data.calendar_events import is_diwali, is_stubble_season

TARGET = 'AQI'
HISTORY_LAGS = (0, 1, 2, 3, 6, 13)
ANOMALY_COLUMNS = ('inv_idx', 'blh_min')
CALENDAR_AT_TARGET = ['day_of_week', 'month', 'day_of_year', 'is_diwali', 'is_stubble_season']


def history_features(daily, target=TARGET, lookback=LOOKBACK_WINDOW):
    """What the target itself has been doing, up to and including day t."""
    grouped = daily.groupby('city', sort=False)[target]
    out = pd.DataFrame(index=daily.index)
    for lag in HISTORY_LAGS:
        out[f'{target}_lag{lag}'] = grouped.shift(lag)
    for window in (3, 7, lookback):
        out[f'{target}_mean{window}'] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).mean())
    out[f'{target}_std7'] = grouped.transform(lambda s: s.rolling(7, min_periods=3).std())
    out[f'{target}_max7'] = grouped.transform(lambda s: s.rolling(7, min_periods=3).max())
    # Where the latest day sits against its own recent week: a rising week and a
    # falling week can pass through the same value on the way.
    out[f'{target}_trend7'] = out[f'{target}_lag0'] - out[f'{target}_mean7']
    return out


def meteorology_anomalies(daily, columns=ANOMALY_COLUMNS):
    """Each city's departure from its own last week, so stagnation is judged locally."""
    out = pd.DataFrame(index=daily.index)
    for column in columns:
        if column in daily.columns:
            weekly = daily.groupby('city', sort=False)[column].transform(
                lambda s: s.rolling(7, min_periods=3).mean())
            out[f'{column}_anom7'] = daily[column] - weekly
    return out


def _calendar_at(dates):
    """Calendar features for a future date, which a forecaster knows in advance."""
    return pd.DataFrame({
        'day_of_week': dates.dt.day_of_week,
        'month': dates.dt.month,
        'day_of_year': dates.dt.day_of_year,
        'is_diwali': is_diwali(dates).astype('int64'),
        'is_stubble_season': is_stubble_season(dates).astype('int64'),
    }, index=dates.index)


def make_supervised(daily, target=TARGET, lookback=LOOKBACK_WINDOW, horizon=FORECAST_HORIZON,
                    features=None):
    """Long frame: one row per city, origin day and horizon, ready to fit or score.

    Columns are city, date (the day the forecast is made), target_date, horizon,
    split (of the target day), y, and the feature columns.
    """
    features = [f for f in (features or ALL_FEATURES) if f in daily.columns]
    daily = daily.sort_values(['city', 'date']).reset_index(drop=True)

    measured = daily[features]                       # observed at t
    history = history_features(daily, target, lookback)
    anomalies = meteorology_anomalies(daily)
    base = pd.concat([daily[['city', 'date']], measured, history, anomalies], axis=1)
    base = base.drop(columns=[c for c in CALENDAR_AT_TARGET if c in base.columns])

    grouped_target = daily.groupby('city', sort=False)[target]
    grouped_split = daily.groupby('city', sort=False)['split'] if 'split' in daily.columns else None

    last_day = daily['date'].max()
    frames = []
    for step in range(1, horizon + 1):
        frame = base.copy()
        frame['horizon'] = step
        frame['target_date'] = daily['date'] + pd.Timedelta(days=step)
        # The final days of the record forecast into a year the data does not
        # cover. Those rows have no answer and are not samples; dropping them
        # here also keeps the calendar from being asked about a year it has
        # no Diwali date for.
        frame = frame[frame['target_date'] <= last_day]
        frame['y'] = grouped_target.shift(-step).to_numpy()[frame.index]
        if grouped_split is not None:
            frame['split'] = grouped_split.shift(-step).to_numpy()[frame.index]
        frames.append(pd.concat([frame, _calendar_at(frame['target_date'])], axis=1))

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['city', 'date', 'horizon']).reset_index(drop=True)


def feature_matrix(samples, target=TARGET):
    """The columns a model may read, in a fixed order."""
    excluded = {'city', 'date', 'target_date', 'split', 'y', target}
    return [c for c in samples.columns if c not in excluded]


def usable(samples):
    """Samples whose target is known and whose own history has started."""
    return samples[samples['y'].notna() & samples[f'{TARGET}_lag0'].notna()].copy()

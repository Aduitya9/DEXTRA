"""
The floor every DEXTRA model has to clear.

Three baselines, in order of how embarrassing it would be to lose to them:

  Persistence. Tomorrow is today. At one day ahead on a quantity as autocorrelated
  as AQI this is genuinely hard to beat, and a model that only matches it has
  learned nothing but the obvious.

  Seasonal climatology. Each city's own average for that day of the year, smoothed
  across a fortnight. It answers the question a reviewer will ask first: does the
  model know anything the calendar does not?

  Gradient boosting on the same features the deep model gets. Deliberately
  untuned -- the point is a floor, not a rival -- but it reads every variate the
  backbone reads, so if it comes close then the architecture is not what is doing
  the work, and that is worth finding out before building the rest of it.

Unseen cities get no special treatment here. Persistence needs no history at all,
climatology falls back to the pooled training seasonality because a thirty-day
support window cannot describe a year, and the booster is fitted on the ten
training cities and applied unchanged. That is the transfer number the adapter
has to improve on.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

AQI_RANGE = (0.0, 500.0)
CLIMATOLOGY_SMOOTHING_DAYS = 15
GBM_SETTINGS = dict(
    loss='squared_error',
    learning_rate=0.05,
    max_iter=300,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=False,     # the internal split is random, which temporal data breaks
    random_state=42,
)


def clip_to_aqi_range(values, limits=AQI_RANGE):
    """AQI is defined on 0-500; predictions outside it are not forecasts of anything."""
    return pd.Series(values).clip(*limits)


# --------------------------------------------------------------------------- persistence


def persistence(samples, target='AQI'):
    """Tomorrow, and the day after, look like today."""
    return clip_to_aqi_range(samples[f'{target}_lag0'].to_numpy())


# --------------------------------------------------------------------------- climatology


def _smooth_circular(values, window=CLIMATOLOGY_SMOOTHING_DAYS):
    """Moving average around the year's end, ignoring days the city never reported."""
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    present = np.isfinite(padded)
    kernel = np.ones(window)
    total = np.convolve(np.where(present, padded, 0.0), kernel, mode='valid')
    count = np.convolve(present.astype(float), kernel, mode='valid')
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


class SeasonalClimatology:
    """Each city's own day-of-year average, fitted on training rows only."""

    def __init__(self, window=CLIMATOLOGY_SMOOTHING_DAYS):
        self.window = window
        self.by_city = {}
        self.pooled = None

    def fit(self, train_samples, truth='y'):
        days = np.arange(1, 367)
        frame = train_samples[['city', 'day_of_year', truth]].dropna(subset=[truth])

        pooled = frame.groupby('day_of_year')[truth].mean().reindex(days).to_numpy(dtype=float)
        self.pooled = _smooth_circular(pooled, self.window)
        for city, group in frame.groupby('city', sort=False):
            raw = group.groupby('day_of_year')[truth].mean().reindex(days).to_numpy(dtype=float)
            self.by_city[city] = _smooth_circular(raw, self.window)
        return self

    def predict(self, samples):
        """A city the fit never saw falls back to the pooled seasonality."""
        index = samples['day_of_year'].to_numpy(dtype=int) - 1
        out = np.full(len(samples), np.nan)
        for city, positions in samples.groupby('city', sort=False).indices.items():
            curve = self.by_city.get(city, self.pooled)
            values = curve[index[positions]]
            out[positions] = np.where(np.isfinite(values), values, self.pooled[index[positions]])
        return clip_to_aqi_range(out)


# --------------------------------------------------------------------------- gradient boosting


def fit_gradient_boosting(train_samples, features, settings=None, truth='y'):
    """One regressor per horizon, fitted on the training split alone.

    Separate models rather than one with the horizon as a feature: three days out
    is a different problem from one day out, and keeping them apart means a poor
    long-horizon fit cannot quietly borrow from the easy short-horizon rows.
    """
    settings = {**GBM_SETTINGS, **(settings or {})}
    models = {}
    for horizon, group in train_samples.groupby('horizon', sort=True):
        rows = group.dropna(subset=[truth])
        model = HistGradientBoostingRegressor(**settings)
        model.fit(rows[features], rows[truth])
        models[int(horizon)] = model
    return models


def predict_gradient_boosting(models, samples, features):
    out = np.full(len(samples), np.nan)
    for horizon, positions in samples.groupby('horizon', sort=False).indices.items():
        model = models.get(int(horizon))
        if model is not None:
            out[positions] = model.predict(samples.iloc[positions][features])
    return clip_to_aqi_range(out)


# --------------------------------------------------------------------------- all together


def run_baselines(samples, features, train_split='train', truth='y'):
    """Fit on the training split, predict everywhere, return the samples with columns added."""
    out = samples.copy()
    train_rows = out[out['split'] == train_split]

    out['persistence'] = persistence(out).to_numpy()
    out['climatology'] = SeasonalClimatology().fit(train_rows, truth).predict(out).to_numpy()
    models = fit_gradient_boosting(train_rows, features, truth=truth)
    out['gbm'] = predict_gradient_boosting(models, out, features).to_numpy()
    return out, models

"""
Time-series foundation models as zero-shot baselines.

Chronos-2 and Chronos-Bolt were pretrained on large and varied collections of
series and forecast one they have never seen with no fitting at all. That is
exactly the situation of an unseen city, so they are the first thing a reviewer
will ask a cross-city model to beat -- and if one already closes the severe-day
gap, the right paper is about that instead.

Each forecast gets the city's own history up to the day it is issued and
nothing after it: up to `length` days, far more than the fortnight our models
read, because long context is what these models were built for. Histories come
from the same gap-filled table our windows are cut from. The covariate variant
adds the measured variables our models see -- pollutants, weather, fire,
calendar -- as past covariates only, so no future value ever reaches it.

The chronos package is imported only when a model is loaded, so everything else
in the project runs without it.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

QUANTILE_LEVELS = (0.1, 0.5, 0.9)


def history_windows(filled, origins, column, length):
    """For each origin (city, date), the last `length` values of `column` up to that day.

    `filled` is a city-day table with no gaps in its dates; `origins` has city and
    date columns. Shorter histories are padded on the left with NaN, which the
    Chronos models mask out. Returns (len(origins), length) float32.
    """
    out = np.full((len(origins), length), np.nan, dtype=np.float32)
    rows = origins.reset_index(drop=True)
    for city, wanted in rows.groupby('city', sort=False).groups.items():
        series = filled[filled['city'] == city].sort_values('date')
        values = np.concatenate([np.full(length - 1, np.nan, dtype=np.float32),
                                 series[column].to_numpy(dtype=np.float32)])
        windows = sliding_window_view(values, length)            # row i ends on day i
        position = pd.Index(pd.to_datetime(series['date'])).get_indexer(
            pd.to_datetime(rows.loc[wanted, 'date']))
        if (position < 0).any():
            raise ValueError(f"{city}: forecast dates missing from the history table")
        out[np.asarray(wanted)] = windows[position]
    return out


def load(model_id, device):
    """A Chronos pipeline on `device`, in float32 (the T4 has no fast bfloat16)."""
    import torch
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map=str(device), torch_dtype=torch.float32)


def predict_quantiles(pipeline, target, covariates=None, horizon=3, levels=QUANTILE_LEVELS,
                      batch_size=256):
    """(n, horizon, len(levels)) quantile forecasts from (n, length) histories.

    Chronos-2 takes a list of series, each with optional past covariates, and
    counts every series -- target and covariate alike -- towards `batch_size`.
    Chronos-Bolt takes a batch tensor of the target alone.
    """
    import torch
    levels = list(levels)
    if type(pipeline).__name__ == 'Chronos2Pipeline':
        names = list(covariates or {})
        inputs = [{'target': target[i], 'past_covariates': {k: covariates[k][i] for k in names}}
                  if names else {'target': target[i]} for i in range(len(target))]
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=horizon,
                                                  quantile_levels=levels, batch_size=batch_size)
        return np.stack([q[0].float().cpu().numpy() for q in quantiles])
    if covariates:
        raise ValueError(f"{type(pipeline).__name__} reads the target alone; covariates need Chronos-2")
    out = []
    for start in range(0, len(target), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(target[start:start + batch_size]))
        quantiles, _ = pipeline.predict_quantiles(batch, prediction_length=horizon, quantile_levels=levels)
        out.append(quantiles.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, horizon, len(levels)), dtype=np.float32)


def forecast(pipeline, filled, origins, target_column, covariate_columns=(), length=512,
             horizon=3, levels=QUANTILE_LEVELS, chunk=2048, batch_size=256, log=print):
    """Quantile forecasts for every origin, built a chunk at a time to bound memory.

    With twenty-odd covariates a full set of 512-day histories for ten thousand
    origins would need half a gigabyte; a chunk of two thousand needs a tenth.
    """
    parts = []
    for start in range(0, len(origins), chunk):
        part = origins.iloc[start:start + chunk]
        target = history_windows(filled, part, target_column, length)
        covariates = {c: history_windows(filled, part, c, length) for c in covariate_columns}
        parts.append(predict_quantiles(pipeline, target, covariates or None, horizon, levels, batch_size))
        log(f"  {min(start + chunk, len(origins)):,} of {len(origins):,} forecasts")
    return np.concatenate(parts) if parts else np.empty((0, horizon, len(levels)), dtype=np.float32)

"""
Scoring for every DEXTRA forecast, baseline or model.

The headline number for this project is not accuracy. A forecaster that predicts
the median every day scores well on RMSE across 38,355 city-days and is useless
on the days anyone cares about, because days above AQI 200 are 11% of the record
and days above 300 are 5%. So error and rare-event detection are always reported
together, and both are always reported per city.

Rare-event recall is undefined where there is nothing to recall. Hyderabad never
exceeds AQI 200 in seven years and Bengaluru does so once, so a recall of 0.0
there would be an artefact of the city, not a property of the model. Those come
back as NaN with their support shown, which a mean over cities then skips
instead of being dragged down by them.
"""

import numpy as np
import pandas as pd

from src.config import HAZARDOUS_AQI_THRESHOLD


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and bias over the rows where both are present."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    both = np.isfinite(true) & np.isfinite(pred)
    if not both.any():
        return {'n': 0, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan}
    error = pred[both] - true[both]
    return {
        'n': int(both.sum()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
        'bias': float(np.mean(error)),
    }


def rare_event_metrics(y_true, y_pred, threshold=HAZARDOUS_AQI_THRESHOLD):
    """Recall, precision and F1 for crossing `threshold`, plus how many days there were.

    `support` is the number of true exceedances. With none, recall and F1 are NaN
    rather than 0, because a city with no hazardous days cannot fail to find one.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    both = np.isfinite(true) & np.isfinite(pred)
    actual, forecast = true[both] > threshold, pred[both] > threshold

    hits = int(np.sum(actual & forecast))
    support = int(np.sum(actual))
    flagged = int(np.sum(forecast))
    recall = hits / support if support else np.nan
    precision = hits / flagged if flagged else (np.nan if not support else 0.0)
    if support and flagged and (recall + precision):
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = np.nan if not support else 0.0
    return {'support': support, 'flagged': flagged, 'hits': hits,
            'recall': recall, 'precision': precision, 'f1': f1}


def score(frame, truth='y', prediction='y_hat', threshold=HAZARDOUS_AQI_THRESHOLD):
    """Both families of metric for one set of forecasts."""
    out = regression_metrics(frame[truth], frame[prediction])
    out.update(rare_event_metrics(frame[truth], frame[prediction], threshold))
    return out


def score_by(frame, by, truth='y', prediction='y_hat', threshold=HAZARDOUS_AQI_THRESHOLD):
    """One row of metrics per group -- per city is the one the paper needs.

    A mean transfer gain hides which cities carry it, and with five unseen cities
    of very different difficulty that average is the easiest number to mislead with.
    """
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, group in frame.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(score(group, truth, prediction, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def compare(scored, index='model', columns=None):
    """Side-by-side table of several models' scores, for the paper's results section."""
    columns = columns or ['n', 'rmse', 'mae', 'bias', 'support', 'recall', 'precision', 'f1']
    return pd.DataFrame(scored).set_index(index)[columns]

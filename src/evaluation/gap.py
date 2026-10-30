"""
Scoring every forecaster the same way on the severe-day gap.

One rule for all of them -- persistence, gradient boosting, the all-token head
with or without a rare-day fix, and the foundation models: the alarm level is
chosen on that protocol's validation rows, and everything is then read on its
test rows. In the leave-one-city-out protocol that happens fold by fold, since
each fold has its own validation cities and its own trained models, and the
held-out cities' counts are pooled afterwards.

Forecasters are compared on the rows every one of them has a forecast for, so a
model can never look better by being scored on easier days.
"""

import re

import numpy as np
import pandas as pd

from src.config import RARE_AQI_THRESHOLD
from src.evaluation import alarms

SEEDED = re.compile(r'^(?P<name>.+)_seed(?P<seed>\d+)$')


def forecaster(column):
    """'plain_seed42' -> ('plain', 42); 'persistence' -> ('persistence', None)."""
    match = SEEDED.match(column)
    return (match['name'], int(match['seed'])) if match else (column, None)


def long_frame(rows, answers, today, forecasts):
    """One row per forecast origin and horizon, with every forecaster as a column.

    `rows` holds city, date and split per origin; `answers` and every array in
    `forecasts` are (origins, horizon); `today` is the AQI on the day of issue,
    which is persistence's forecast for every horizon.
    """
    rows = rows[['city', 'date', 'split']].reset_index(drop=True)
    parts = []
    for step in range(answers.shape[1]):
        part = rows.assign(horizon=step + 1, y=answers[:, step], persistence=np.asarray(today))
        for name, values in forecasts.items():
            part[name] = np.asarray(values)[:, step]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def city_counts(rows, column, level, threshold=RARE_AQI_THRESHOLD):
    """Per city: severe days, alarms raised at `level`, and the hits among them."""
    actual, raised = rows['y'] > threshold, rows[column] > level
    return (pd.DataFrame({'city': rows['city'].to_numpy(), 'hits': (actual & raised).to_numpy(),
                          'severe': actual.to_numpy(), 'raised': raised.to_numpy()})
            .groupby('city', sort=True).sum().astype(float).reset_index())


def score_column(frame, column, tune, test, threshold=RARE_AQI_THRESHOLD, level=None):
    """Alarm level from the `tune` rows; everything else read on the `test` rows.

    A fixed `level` skips the tuning -- for persistence at the official line,
    which is how an agency would actually run it. Returns a summary and the
    per-city counts at the level used.
    """
    if level is None:
        level = alarms.tune_alarm(frame.loc[tune, 'y'], frame.loc[tune, column], threshold=threshold)
    rows = frame[test]
    at_line = alarms.alarm_scores(rows['y'], rows[column], threshold, threshold)
    tuned = alarms.alarm_scores(rows['y'], rows[column], level, threshold)
    summary = {'alarm level': level,
               'rmse': float(np.sqrt(np.mean((rows[column] - rows['y']) ** 2))),
               'hits': tuned['hits'], 'severe': tuned['severe'], 'false alarms': tuned['false_alarms'],
               'f1 at 300': at_line['f1'], 'f1': tuned['f1'],
               'bias on severe days': alarms.severe_bias(rows['y'], rows[column], threshold)}
    return summary, city_counts(rows, column, level, threshold)


def score_folds(folds, column, threshold=RARE_AQI_THRESHOLD, level=None):
    """Leave-one-city-out: each fold tunes on its own validation rows, then counts pool.

    `folds` maps a held-out city to its frame, whose split is 'val' or
    'test_loco'. The summary is over every held-out city's test rows together.
    """
    counts, tested, levels = [], [], []
    for frame in folds.values():
        summary, part = score_column(frame, column, frame['split'] == 'val',
                                     frame['split'] == 'test_loco', threshold, level)
        counts.append(part)
        tested.append(frame[frame['split'] == 'test_loco'])
        levels.append(summary['alarm level'])
    counts, rows = pd.concat(counts, ignore_index=True), pd.concat(tested, ignore_index=True)
    hits, severe, raised = counts['hits'].sum(), counts['severe'].sum(), counts['raised'].sum()
    summary = {'alarm level': float(np.median(levels)),
               'rmse': float(np.sqrt(np.mean((rows[column] - rows['y']) ** 2))),
               'hits': int(hits), 'severe': int(severe), 'false alarms': int(raised - hits),
               'f1 at 300': alarms.alarm_scores(rows['y'], rows[column], threshold, threshold)['f1'],
               'f1': alarms.pooled_f1(counts),
               'bias on severe days': alarms.severe_bias(rows['y'], rows[column], threshold)}
    return summary, counts


def combine_seeds(results):
    """Average a seeded forecaster's runs: summaries by mean, F1 with its spread, counts per city.

    `results` maps a column to (summary, counts). Columns sharing a name across
    seeds become one row; an unseeded column passes through with a spread of 0.
    """
    grouped = {}
    for column, (summary, counts) in results.items():
        grouped.setdefault(forecaster(column)[0], []).append((summary, counts))
    table, pooled = [], {}
    for name, runs in grouped.items():
        frame = pd.DataFrame([summary for summary, _ in runs])
        row = {'forecaster': name, 'runs': len(runs), **frame.mean(numeric_only=True).to_dict(),
               'f1 sd': float(frame['f1'].std(ddof=1)) if len(runs) > 1 else 0.0}
        table.append(row)
        pooled[name] = (pd.concat([counts for _, counts in runs])
                        .groupby('city', sort=True).mean().reset_index())
    return pd.DataFrame(table).set_index('forecaster'), pooled


def paired_difference(counts_a, counts_b, draws=2000, seed=0):
    """Pooled F1 of a minus that of b, with a 95% interval from resampling whole cities."""
    merged = counts_a.merge(counts_b, on='city', suffixes=('_a', '_b'))

    def difference(c):
        def f1(side):
            denominator = c[f'severe{side}'].sum() + c[f'raised{side}'].sum()
            return 2 * c[f'hits{side}'].sum() / denominator if denominator else np.nan
        return f1('_a') - f1('_b')

    low, high = alarms.bootstrap_groups(merged, difference, draws=draws, seed=seed)
    return float(difference(merged)), low, high

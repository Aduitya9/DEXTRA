"""
Severe-day alarms: did the forecast see the day coming?

RMSE weighs every day alike, and on a quantity this autocorrelated it rewards a
forecast that stays near the recent level. The days that matter most -- AQI
above 300, CPCB's 'Very Poor', where health advisories and GRAP actions begin
-- are a few percent of the record and the first casualty of that habit. These
score a forecast as an alarm: of the days that crossed the line, how many it
flagged, and how many of its flags were wrong.

The alarm level is a decision, not a law of nature. A forecast trained on
squared error sits below the truth on peaks, so it can rank severe days well and
still fall short of 300. tune_alarm picks the level on validation rows only, so
a test split is always scored at a level chosen before anyone looked at it.
"""

import numpy as np
import pandas as pd

from src.config import RARE_AQI_THRESHOLD

# Candidate alarm levels, AQI points. 300 itself is one of them, so tuning can
# always fall back to the official line when nothing else does better. The range
# reaches well above 300 as well as below: a forecast trained or read at its 90th
# percentile is meant to sit high, and its best alarm sits high with it.
ALARM_LEVELS = np.arange(150.0, 452.0, 2.0)


def alarm_scores(truth, forecast, level=RARE_AQI_THRESHOLD, threshold=RARE_AQI_THRESHOLD):
    """Hits, misses and false alarms for 'the truth crosses `threshold`', raised at `level`.

    F1 is 2 x hits / (severe days + alarms raised), which is the usual F1 and
    stays defined when the forecast raises nothing (0) or there was nothing to
    find (NaN, since a period with no severe days cannot fail to find one).
    """
    truth, forecast = np.asarray(truth, dtype=float), np.asarray(forecast, dtype=float)
    keep = np.isfinite(truth) & np.isfinite(forecast)
    actual, raised = truth[keep] > threshold, forecast[keep] > level
    hits, severe, flagged = int((actual & raised).sum()), int(actual.sum()), int(raised.sum())
    return {'hits': hits, 'severe': severe, 'false_alarms': flagged - hits,
            'recall': hits / severe if severe else np.nan,
            'precision': hits / flagged if flagged else np.nan,
            'f1': 2 * hits / (severe + flagged) if severe + flagged else np.nan}


def tune_alarm(truth, forecast, levels=ALARM_LEVELS, threshold=RARE_AQI_THRESHOLD):
    """The alarm level with the best F1 on the rows given -- validation rows, never test.

    Ties go to the level nearest the threshold itself, so a tuned alarm moves
    off the official line only when moving pays.
    """
    levels = np.asarray(levels, dtype=float)
    f1 = np.array([alarm_scores(truth, forecast, level, threshold)['f1'] for level in levels])
    f1 = np.nan_to_num(f1, nan=-1.0)
    tied = levels[f1 == f1.max()]
    return float(tied[np.argmin(np.abs(tied - threshold))])


def severe_bias(truth, forecast, threshold=RARE_AQI_THRESHOLD):
    """Mean forecast minus truth on the severe days alone, in AQI points."""
    truth, forecast = np.asarray(truth, dtype=float), np.asarray(forecast, dtype=float)
    severe = (truth > threshold) & np.isfinite(forecast)
    return float(np.mean(forecast[severe] - truth[severe])) if severe.any() else np.nan


def return_to_normal(frame, forecasts, today='persistence', threshold=RARE_AQI_THRESHOLD):
    """For forecasts issued on a severe day: the mean truth and mean forecast, by horizon.

    A model that has learned that episodes end forecasts a fall that the truth
    does not take. `frame` holds horizon, y and today's AQI under `today`.
    """
    issued = frame[frame[today] > threshold]
    table = issued.groupby('horizon')[['y'] + list(forecasts)].mean()
    table.insert(0, 'forecasts', issued.groupby('horizon').size())
    return table


def pooled_f1(counts):
    """F1 over a table of per-group hits, severe days and alarms raised."""
    hits, severe, raised = counts['hits'].sum(), counts['severe'].sum(), counts['raised'].sum()
    return 2 * hits / (severe + raised) if severe + raised else np.nan


def bootstrap_groups(counts, statistic=pooled_f1, group='city', draws=2000, seed=0):
    """Percentile interval of `statistic` with whole groups resampled.

    Days inside a city are not independent -- an episode lasts several -- so the
    unit resampled is the city, which keeps every episode whole. `counts` has
    one row per group; for a paired comparison, give `statistic` a frame that
    carries both forecasters' columns, so both see the same resample.
    """
    rng = np.random.default_rng(seed)
    groups = counts[group].to_numpy()
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        values.append(statistic(counts.iloc[pick]))
    low, high = np.nanpercentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(low), float(high)

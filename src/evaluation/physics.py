"""
Does each feature relate to AQI the way the physics says it should?

A single correlation over the whole year is the wrong test for this, and asking
it produces a false alarm on every run. India has four meteorological regimes
and some features mean opposite things in two of them. Measured on the full
seven years:

    blh_anomaly   winter -0.172   pre-monsoon +0.209   monsoon +0.010

Both numbers are correct. In winter a shallower-than-usual boundary layer is a
lid and pollution builds beneath it. In the pre-monsoon the layer is deep
because the surface is baking at 30 C, and the same heating lifts mineral dust:
on the deepest-anomaly days the PM10/PM2.5 ratio is 2.49 against 2.16 overall.
The layer is not dispersing the dust, it is a symptom of what raised it. Over a
year the two regimes cancel to +0.068, which looks like a broken feature and is
not one.

So a feature is only reported as wrong when its sign is consistent across
seasons and still contradicts the physics. A sign that changes with the season
is reported as seasonal, because that is a fact about the atmosphere rather than
a fault in the data -- and it is the measurement this project's gating is for.
"""

import numpy as np
import pandas as pd

IMD_SEASONS = {12: 'winter', 1: 'winter', 2: 'winter',
               3: 'pre-monsoon', 4: 'pre-monsoon', 5: 'pre-monsoon',
               6: 'monsoon', 7: 'monsoon', 8: 'monsoon', 9: 'monsoon',
               10: 'post-monsoon', 11: 'post-monsoon'}
SEASON_ORDER = ['winter', 'pre-monsoon', 'monsoon', 'post-monsoon']

# +1 means the feature should rise with AQI, -1 that it should fall with it.
EXPECTED_SIGN = {
    'inv_idx': (+1, 'a warm lid traps what is under it'),
    'blh': (-1, 'a deep mixing layer dilutes'),
    'blh_min': (-1, 'a shallow night layer traps'),
    'blh_anomaly': (-1, 'a shallower day than this city is used to traps'),
    'wind_speed': (-1, 'wind ventilates'),
    'precip': (-1, 'rain washes particles out'),
    'fire_count_500km': (+1, 'smoke upwind adds particles'),
    'frp_500km': (+1, 'fiercer fires add more'),
    'temp2m': (0, 'no single expectation: cold traps in winter, heat lifts dust in summer'),
}

WEAK_CORRELATION = 0.05
MIN_DAYS = 200


def season_of(dates):
    return pd.Series(pd.to_datetime(dates).dt.month.map(IMD_SEASONS), index=dates.index)


def physics_report(daily, target='AQI', expectations=None, weak=WEAK_CORRELATION,
                   min_days=MIN_DAYS):
    """One row per feature: the all-year correlation, each season's, and a verdict.

    Verdicts
        consistent  same sign in every season that has enough data, as expected
        seasonal    the sign changes between seasons -- real, and what gating is for
        unexpected  one sign throughout, and it contradicts the physics: a fault
        weak        nowhere strong enough for its sign to mean anything
    """
    expectations = expectations or EXPECTED_SIGN
    frame = daily.copy()
    frame['season'] = season_of(frame['date'])

    rows = []
    for feature, (expected, reason) in expectations.items():
        if feature not in frame.columns:
            continue
        row = {'feature': feature, 'all year': frame[feature].corr(frame[target])}
        by_season = {}
        for season in SEASON_ORDER:
            part = frame[frame['season'] == season]
            value = part[feature].corr(part[target]) if len(part) >= min_days else np.nan
            row[season] = value
            by_season[season] = value

        # A correlation only counts if it clears both an absolute floor and about
        # two standard errors for the number of days behind it. Without the second
        # term a season of noise can pass the floor by luck, and three such seasons
        # agreeing by chance would be reported as a contradiction of the physics.
        strong = {}
        for season, value in by_season.items():
            days = int(frame[frame['season'] == season][[feature, target]].dropna().shape[0])
            floor = max(weak, 2.0 / np.sqrt(days)) if days else np.inf
            if np.isfinite(value) and abs(value) >= floor:
                strong[season] = value
        signs = {np.sign(v) for v in strong.values()}
        if not strong:
            verdict, note = 'weak', 'no season reaches |r| = %.2f' % weak
        elif len(signs) > 1:
            against = [s for s, v in strong.items() if np.sign(v) != expected and expected != 0]
            verdict = 'seasonal'
            note = 'reverses in ' + ', '.join(against) if against else 'sign varies by season'
        elif expected == 0:
            verdict, note = 'consistent', reason
        elif signs == {expected}:
            verdict, note = 'consistent', reason
        else:
            verdict, note = 'unexpected', 'contradicts: ' + reason
        row['expected'] = {1: 'rises with AQI', -1: 'falls with AQI', 0: 'either'}[expected]
        row['verdict'] = verdict
        row['note'] = note
        rows.append(row)

    columns = ['feature', 'all year'] + SEASON_ORDER + ['expected', 'verdict', 'note']
    return pd.DataFrame(rows)[columns]


def failures(report):
    """Only the rows that mean something is wrong."""
    return report[report['verdict'] == 'unexpected']

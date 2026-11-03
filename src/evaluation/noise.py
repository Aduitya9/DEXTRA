"""
The noise ceiling: how good could a forecaster be, given how noisy its target is?

A city's AQI is computed from the median of its stations, so it carries whatever those
stations happen to disagree about on the day. No forecaster reading city-level inputs
can predict that part, and every error score includes it. The idea comes from
neuroscience, where split-half reliability bounds how well a model of noisy recordings
can do: here the repeated measurements are the stations themselves.

Each city's stations are divided once into two halves, kept fixed over time, and an AQI
is computed from each. A lasting difference between the halves says one part of town
is dirtier, which a forecaster can learn, so a rolling median of the difference is
removed; what remains, day to day, is the noise. For an average of k stations with
independent errors, two halves of k/2 differ with four times the variance of the full
estimate's noise, so the noise SD is SD(difference) / 2. With a median and few stations
this runs a little low (about 9% in SD at worst, three to five stations), which makes
the floor conservative.

It bounds what city-level inputs can reach. It is not the limit of predictability: the
weather's own uncertainty sits on top of it, and some of what it calls noise is real
air that a model reading each station might partly foresee.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import POLLUTANTS, RARE_AQI_THRESHOLD
from src.utils.aqi_calculator import calculate_aqi_from_pollutants

LEVELS = [0, 100, 150, 200, 250, 280, 300, 320, 350, 400, 501]
PERSISTENT_WINDOW_DAYS = 31


def split_network(stations, seed=0, pollutants=POLLUTANTS):
    """The AQI of two fixed halves of each city's stations, per day both halves reported.

    `stations` has city, date, site_id and pollutant columns, one row per station-day.
    Cities with a single station cannot be split and are left out.
    """
    rng = np.random.default_rng(seed)
    side = {}
    for city, sites in stations.dropna(subset=['PM2.5']).groupby('city')['site_id'].unique().items():
        if len(sites) < 2:
            continue
        for rank, site in enumerate(rng.permutation(sites)):
            side[(city, site)] = 'A' if rank < len(sites) / 2 else 'B'
    keep = stations[[key in side for key in zip(stations['city'], stations['site_id'])]].copy()
    keep['half'] = [side[key] for key in zip(keep['city'], keep['site_id'])]
    present = [p for p in pollutants if p in keep.columns]
    halves = keep.groupby(['city', 'date', 'half'])[present].median().reset_index()
    halves['aqi'] = [calculate_aqi_from_pollutants(row) for row in halves[present].to_dict('records')]
    wide = halves.pivot_table(index=['city', 'date'], columns='half', values='aqi').dropna()
    return wide.reset_index().rename_axis(columns=None)


def day_to_day_noise(halves, window=PERSISTENT_WINDOW_DAYS):
    """Each day's noise in a city's AQI: half the difference, its lasting part removed."""
    out = halves.sort_values(['city', 'date']).copy()
    difference = out['A'] - out['B']
    lasting = difference.groupby(out['city']).transform(
        lambda d: d.rolling(window, center=True, min_periods=max(2, window // 4)).median())
    out['noise'] = (difference - lasting) / 2
    return out.dropna(subset=['noise'])


def noise_by_level(noise, aqi_column='AQI', levels=LEVELS):
    """How large the noise is at each level of AQI: SD, and a version robust to odd days."""
    bands = pd.cut(noise[aqi_column], levels, right=False)
    grouped = noise.groupby(bands, observed=True)['noise']
    return pd.DataFrame({'days': grouped.size(), 'noise_sd': grouped.std(),
                         'noise_robust': (grouped.quantile(0.84) - grouped.quantile(0.16)) / 2})


def draw_noise(values, pool, rng, levels=LEVELS):
    """A noise value for each AQI, from the days measured at its level. `pool` has level and noise."""
    edges = np.asarray(levels, dtype=float)
    position = np.clip(np.searchsorted(edges, np.asarray(values, dtype=float), side='right') - 1, 0, len(edges) - 2)
    samples = {level: part['noise'].to_numpy() for level, part in pool.groupby('level')}
    out = np.zeros(len(position))
    for i, level in enumerate(edges[:-1]):
        chosen = position == i
        if chosen.any() and level in samples:
            out[chosen] = rng.choice(samples[level], size=int(chosen.sum()))
    return out


def ceilings(y, today, pool, draws=200, seed=0, threshold=RARE_AQI_THRESHOLD):
    """Mean over noise draws of the best scores a perfect forecast could reach.

    Every day is scored by its measured AQI, as a forecaster who knew the true air
    would nearly score it, and the label is that value plus fresh noise of the size
    measured at its level. Returns the RMSE floor and the AP ceilings on severe days and
    on onsets, the severe days that were not severe on the day of issue.
    """
    y, today = np.asarray(y, dtype=float), np.asarray(today, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    first = today <= threshold
    for _ in range(draws):
        noise = draw_noise(y, pool, rng)
        label = y + noise > threshold
        rows.append((float(np.sqrt(np.mean(noise ** 2))), average_precision_score(label, y),
                     average_precision_score(label[first], y[first])))
    return dict(zip(('rmse_floor', 'ap_ceiling', 'onset_ap_ceiling'), np.mean(rows, axis=0)))


def reducible_share(mse_model, mse_reference, mse_floor):
    """How much of the error that can be removed at all, beyond a reference, a model has removed."""
    return (mse_reference - mse_model) / (mse_reference - mse_floor)

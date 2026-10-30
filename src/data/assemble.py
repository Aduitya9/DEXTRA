"""
Merge the daily sources into the DEXTRA feature table, then split it.

Order matters here, because two things decide whether the headline result is
believable:

  Leakage. Normalisation statistics and the per-city static features are fitted
  on the training split alone. For the five unseen cities, static features come
  from their few-shot support window only -- using their whole history would
  quietly undo the "adapts with 30 days" claim.

  Honesty about thin data. n_stations rides along with every row, so a Guwahati
  day backed by one sensor is distinguishable from a Delhi day backed by forty.
"""

import numpy as np
import pandas as pd

from src.config import (
    ARCHETYPE_CLUSTERS,
    CITY_META_FEATURES,
    FIRE_FEATURES,
    HAZARDOUS_AQI_THRESHOLD,
    METEO_FEATURES,
    MIN_EPISODE_DURATION_DAYS,
    POLLUTANTS,
    SUPPORT_END_DATE,
    SUPPORT_START_DATE,
    TARGET_CITIES,
    TEMPORAL_FEATURES,
    TEST_UNSEEN_CITIES,
    TRAIN_CITIES,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
from src.data.calendar_events import is_diwali, is_stubble_season

ARCHETYPE_ID = {name: index for index, name in enumerate(ARCHETYPE_CLUSTERS)}


# --------------------------------------------------------------------------- merge


def _require_one_row_per_city_day(name, frame):
    """Refuse a source that repeats a city-day.

    A left join against a duplicated key multiplies rows instead of failing, so a
    single repeated day in any one source would quietly inflate the dataset and
    reweight whatever is trained on it. Sources built from per-year files are the
    ones at risk: UTC-stamped data shifted into IST spills across the New Year.
    """
    duplicated = frame.duplicated(['city', 'date'])
    if duplicated.any():
        examples = ', '.join(f"{row.city} {row.date:%Y-%m-%d}"
                             for row in frame.loc[duplicated].head(3).itertuples())
        raise ValueError(
            f"{name} repeats {int(duplicated.sum())} city-day(s) ({examples}). "
            "Every source must hold one row per city per day before merging."
        )


def merge_sources(pollutant_daily, meteo_daily, fire_daily):
    """One row per city-day: pollutants, meteorology and fire counts."""
    for name, frame in (('pollutant_daily', pollutant_daily),
                        ('meteo_daily', meteo_daily),
                        ('fire_daily', fire_daily)):
        _require_one_row_per_city_day(name, frame)
    merged = pollutant_daily.merge(meteo_daily, on=['city', 'date'], how='left')
    return merged.merge(fire_daily, on=['city', 'date'], how='left')


def add_calendar_features(daily):
    """Day-of-week, month, day-of-year, and the two India-specific event flags."""
    out = daily.copy()
    out['day_of_week'] = out['date'].dt.day_of_week
    out['month'] = out['date'].dt.month
    out['day_of_year'] = out['date'].dt.day_of_year
    out['is_diwali'] = is_diwali(out['date']).astype('int64')
    out['is_stubble_season'] = is_stubble_season(out['date']).astype('int64')
    return out


# --------------------------------------------------------------------------- static city features


def city_static_features(daily, train_cities=None, train_end=TRAIN_END_DATE,
                         support_start=SUPPORT_START_DATE, support_end=SUPPORT_END_DATE):
    """Per-city static features, each computed only from data that city is allowed to see.

    Training cities may use the whole training period. Unseen cities may use only
    the support window, which is all the few-shot protocol gives them; letting them
    use seven years would quietly undo the 30-day claim.

    The window is the same calendar month for every unseen city. Taking each
    city's own first 30 days instead made this feature a record of when its
    pollution board installed the equipment: three of the five came online on
    1 January 2019 and saw nothing but winter smog.
    """
    train_cities = set(train_cities or TRAIN_CITIES)
    train_end = pd.Timestamp(train_end)
    support_start, support_end = pd.Timestamp(support_start), pd.Timestamp(support_end)

    rows = []
    for city, group in daily.groupby('city', sort=False):
        group = group.sort_values('date')
        if city in train_cities:
            allowed = group[group['date'] <= train_end]
        else:
            allowed = group[group['date'].between(support_start, support_end)]

        meta = TARGET_CITIES.get(city, {})
        ratio = np.nan
        if {'SO2', 'NO2'} <= set(allowed.columns):
            usable = allowed[allowed['NO2'] > 0]
            if usable['SO2'].notna().any():
                ratio = float((usable['SO2'] / usable['NO2']).median(skipna=True))
        rows.append({
            'city': city,
            'archetype_id': ARCHETYPE_ID.get(meta.get('archetype_cluster'), -1),
            'coastal_flag': int(bool(meta.get('coastal', False))),
            'industrial_index': ratio,
            'static_window_days': int(allowed['date'].nunique()),
        })
    return pd.DataFrame(rows)


def add_city_meta(daily, static):
    """Attach static city features, and station_density measured per day from the data itself."""
    columns = ['city', 'archetype_id', 'coastal_flag', 'industrial_index']
    out = daily.merge(static[columns], on='city', how='left')
    out['station_density'] = out['n_stations'].fillna(0).astype(float) if 'n_stations' in out.columns else 0.0
    return out


# --------------------------------------------------------------------------- episodes


def extreme_episodes(daily, threshold=HAZARDOUS_AQI_THRESHOLD, min_days=MIN_EPISODE_DURATION_DAYS):
    """Contiguous runs of hazardous days, one row per episode."""
    episodes = []
    for city, group in daily.groupby('city', sort=False):
        group = group.sort_values('date')
        over = (group['AQI'] > threshold).fillna(False).to_numpy()
        dates = group['date'].to_numpy()
        aqi = group['AQI'].to_numpy(dtype=float)
        start = None
        for i, flag in enumerate(list(over) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_days:
                    episodes.append({
                        'city': city,
                        'start_date': dates[start],
                        'end_date': dates[i - 1],
                        'duration_days': i - start,
                        'peak_aqi': float(np.nanmax(aqi[start:i])),
                    })
                start = None
    return pd.DataFrame(episodes)


# --------------------------------------------------------------------------- splits


def assign_split(daily, train_cities=None, unseen_cities=None,
                 train_end=TRAIN_END_DATE, val_end=VAL_END_DATE,
                 support_start=SUPPORT_START_DATE, support_end=SUPPORT_END_DATE):
    """Six labels: three years for the training cities, three roles for the unseen ones.

        train / val / test_seen    ten training cities, 2019-2023 / 2024 / 2025
        support_unseen             five unseen cities, the 31 days of December 2024
        test_unseen                five unseen cities, 2025
        zeroshot_unseen            five unseen cities, everything before the support window

    test_unseen covers the same year as test_seen on purpose. Scoring the unseen
    cities across all seven years instead would have mixed the question being
    asked -- how much is lost on a city the model has never seen -- with the
    question of which years were dirtier, and the two are not separable after
    the fact.

    zeroshot_unseen is held back rather than thrown away: it is six years of
    cities the model never trains on, which is a second, weaker transfer number
    for a city it cannot have adapted to, since that data precedes the support.
    """
    train_cities = set(train_cities or TRAIN_CITIES)
    unseen_cities = set(unseen_cities or TEST_UNSEEN_CITIES)
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    support_start, support_end = pd.Timestamp(support_start), pd.Timestamp(support_end)

    split = pd.Series('unassigned', index=daily.index, dtype=object)
    date = daily['date']
    is_unseen = daily['city'].isin(unseen_cities)
    is_train_city = daily['city'].isin(train_cities)

    split[is_train_city & (date <= train_end)] = 'train'
    split[is_train_city & (date > train_end) & (date <= val_end)] = 'val'
    split[is_train_city & (date > val_end)] = 'test_seen'

    split[is_unseen] = 'zeroshot_unseen'
    split[is_unseen & date.between(support_start, support_end)] = 'support_unseen'
    split[is_unseen & (date > val_end)] = 'test_unseen'
    return split


def leave_one_out_split(daily, held_out, train_end=TRAIN_END_DATE, val_end=VAL_END_DATE):
    """Split labels for one leave-one-city-out fold, over all fifteen cities.

        train      every other city, through train_end (2019-2023)
        val        every other city, the year after (2024): early stopping, alarm levels
        test_loco  the held-out city, everything after train_end (2024-2025)
        unused     the held-out city's earlier years, and the other cities' 2025

    The main split tests transfer on five fixed cities, and their 2025 holds about
    forty severe days -- too few to rest a claim on. Holding out each city in turn
    makes every city an unseen one once, and scoring two years instead of one
    gives each held-out city two winters. The held-out city contributes nothing
    to training, validation, gap-filling medians or alarm levels; its own history
    reaches a forecast only as the input a forecaster would really have.
    """
    if held_out not in set(daily['city']):
        raise ValueError(f"{held_out!r} is not a city in the table")
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    split = pd.Series('unused', index=daily.index, dtype=object)
    others = daily['city'] != held_out
    split[others & (daily['date'] <= train_end)] = 'train'
    split[others & (daily['date'] > train_end) & (daily['date'] <= val_end)] = 'val'
    split[~others & (daily['date'] > train_end)] = 'test_loco'
    return split


# --------------------------------------------------------------------------- normalisation


def fit_scalers(train_rows, pollutants=None, meteo=None, fire=None):
    """Scaling parameters from the training split only.

    Pollutants are scaled per city, because Delhi's ordinary winter day is Chennai's
    emergency; meteorology globally, since it is one physical scale everywhere; fire
    counts through log1p first, being counts that run from zero to many hundreds.
    """
    pollutants = pollutants or POLLUTANTS
    meteo = meteo or METEO_FEATURES
    fire = fire or [f for f in FIRE_FEATURES if not f.startswith('is_')]

    params = {'pollutants_minmax_per_city': {}, 'meteo_standard': {}, 'fire_log1p_minmax': {}}
    for city, group in train_rows.groupby('city', sort=False):
        params['pollutants_minmax_per_city'][city] = {
            p: {'min': float(group[p].min()), 'max': float(group[p].max())}
            for p in pollutants if p in group.columns and group[p].notna().any()
        }
    for column in meteo:
        if column in train_rows.columns and train_rows[column].notna().any():
            std = float(train_rows[column].std(ddof=0))
            params['meteo_standard'][column] = {'mean': float(train_rows[column].mean()),
                                                'std': std if std else 1.0}
    for column in fire:
        if column in train_rows.columns and train_rows[column].notna().any():
            logged = np.log1p(train_rows[column].clip(lower=0))
            params['fire_log1p_minmax'][column] = {'min': float(logged.min()), 'max': float(logged.max())}
    return params


def _pooled_bounds(per_city):
    """Training range across all cities, used for a city the scalers never saw."""
    pooled = {}
    for city_params in per_city.values():
        for pollutant, bounds in city_params.items():
            entry = pooled.setdefault(pollutant, {'min': bounds['min'], 'max': bounds['max']})
            entry['min'] = min(entry['min'], bounds['min'])
            entry['max'] = max(entry['max'], bounds['max'])
    return pooled


def apply_scalers(daily, params):
    """Apply fitted parameters. Unseen cities fall back to the pooled training range."""
    out = daily.copy()
    per_city = params['pollutants_minmax_per_city']
    pooled = _pooled_bounds(per_city)

    for pollutant, pooled_bounds in pooled.items():
        if pollutant not in out.columns:
            continue
        scaled = pd.Series(np.nan, index=out.index, dtype=float)
        for city, index in out.groupby('city', sort=False).groups.items():
            bounds = per_city.get(city, {}).get(pollutant) or pooled_bounds
            span = bounds['max'] - bounds['min'] or 1.0
            scaled.loc[index] = (out.loc[index, pollutant] - bounds['min']) / span
        out[pollutant] = scaled

    for column, stats in params['meteo_standard'].items():
        if column in out.columns:
            out[column] = (out[column] - stats['mean']) / (stats['std'] or 1.0)
    for column, bounds in params['fire_log1p_minmax'].items():
        if column in out.columns:
            span = bounds['max'] - bounds['min'] or 1.0
            out[column] = (np.log1p(out[column].clip(lower=0)) - bounds['min']) / span
    return out


def feature_columns():
    """The model variates, in a fixed order (N_VARIATES of them)."""
    return (list(POLLUTANTS) + list(METEO_FEATURES) + list(FIRE_FEATURES)
            + list(TEMPORAL_FEATURES) + list(CITY_META_FEATURES))

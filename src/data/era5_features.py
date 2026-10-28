"""
ERA5 hourly fields -> daily meteorological features, in Indian Standard Time.

Everything is shifted to IST before grouping, so a day here means the same span
CPCB uses for its daily AQI.

Two features carry the physics the Mixture-of-Experts gate routes on:
  inv_idx      temperature at the city's inversion level minus 2 m temperature,
               taken in the early morning when inversions are strongest. Positive
               means warm air sits over cool air and pollution cannot rise.
  blh_anomaly  boundary-layer height minus its own 30-day rolling mean, so a
               stagnant day is measured against that city's own normal rather
               than against a coastal city's.
"""

import numpy as np
import pandas as pd

from src.config import IST_OFFSET
from src.data.download_era5 import select_inversion_level

BLH_ANOMALY_WINDOW_DAYS = 30


def to_ist(frame, column='time_utc'):
    """Add an IST timestamp and its calendar date."""
    out = frame.copy()
    out['time_ist'] = out[column] + pd.Timedelta(IST_OFFSET)
    out['date'] = out['time_ist'].dt.normalize()
    return out


def circular_mean_degrees(degrees):
    """Mean wind direction. Averaging 350 and 10 arithmetically gives 180, which is backwards."""
    values = pd.Series(degrees).dropna()
    if values.empty:
        return np.nan
    radians = np.deg2rad(values.to_numpy(dtype=float))
    angle = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    # Round before wrapping: averaging 350 and 10 lands a hair below zero, and a bare
    # modulo would report 360.0 for what is due north.
    return float(round(angle % 360.0, 9) % 360.0)


def surface_daily(surface):
    """Daily surface features for one city's hourly ERA5 series.

    wind_speed is the daily MEAN, not the daily maximum the workflow document
    suggested: dispersion depends on how ventilated the whole day was, and a
    maximum would make a day with one windy hour look ventilated after 23 calm ones.
    """
    df = to_ist(surface)
    grouped = df.groupby('date')
    daily = pd.DataFrame({
        'blh': grouped['blh'].mean(),
        'blh_min': grouped['blh'].min(),
        'temp2m': grouped['temp2m'].mean(),
        'precip': grouped['precip'].sum(min_count=1),
        'wind_speed': grouped['wind_speed'].mean(),
        'hours': grouped['temp2m'].count(),
    })
    daily['wind_dir'] = grouped['wind_dir'].apply(circular_mean_degrees)
    daily = daily.reset_index()
    daily.insert(0, 'city', surface['city'].iloc[0])
    return daily


def add_blh_anomaly(daily, window_days=BLH_ANOMALY_WINDOW_DAYS):
    """Boundary-layer height minus its own trailing 30-day mean, per city."""
    out = daily.sort_values(['city', 'date']).copy()
    rolling = (out.groupby('city')['blh']
                  .transform(lambda s: s.rolling(window_days, min_periods=window_days // 2).mean()))
    out['blh_anomaly'] = out['blh'] - rolling
    return out


def inversion_index(pressure, surface_daily_frame, levels_by_city):
    """Strongest early-morning inversion per city-day.

    `pressure` holds temperature on several pressure levels at 00 and 03 UTC;
    each city uses the level ~75 hPa above its own surface. The 2 m temperature
    comes from the hourly surface series at those same timestamps.
    """
    rows = []
    for city, level in levels_by_city.items():
        column = f'temp_{level}hPa'
        city_pressure = pressure[pressure['city'] == city]
        if city_pressure.empty or column not in city_pressure.columns:
            continue
        merged = city_pressure[['city', 'time_utc', column]].merge(
            surface_daily_frame[['city', 'time_utc', 'temp2m']], on=['city', 'time_utc'], how='inner'
        )
        merged['inv'] = merged[column] - merged['temp2m']
        merged = to_ist(merged)
        strongest = merged.groupby(['city', 'date'])['inv'].max().reset_index(name='inv_idx')
        strongest['inversion_level_hpa'] = level
        rows.append(strongest)
    if not rows:
        return pd.DataFrame(columns=['city', 'date', 'inv_idx', 'inversion_level_hpa'])
    return pd.concat(rows, ignore_index=True)


def inversion_levels_from_surface(surface):
    """{city: pressure level} from each city's mean grid-cell surface pressure."""
    means = surface.groupby('city')['surface_pressure'].mean()
    return {city: select_inversion_level(value) for city, value in means.items()}


def build_meteo_daily(surface_by_city, pressure):
    """Daily meteorological table for every city: surface features, inversion index, BLH anomaly."""
    surface_all = pd.concat(surface_by_city, ignore_index=True)
    daily = pd.concat([surface_daily(frame) for frame in surface_by_city], ignore_index=True)
    daily = add_blh_anomaly(daily)

    levels = inversion_levels_from_surface(surface_all)
    inversions = inversion_index(pressure, surface_all, levels)
    return daily.merge(inversions, on=['city', 'date'], how='left')

"""
CPCB hourly station readings -> daily city series, by the National AQI methodology.

Three steps, each kept separate so the rules stay visible:
  1. station_daily  - one station's hourly file to daily values. PM2.5, PM10, NO2,
     SO2 and NH3 take the 24-hour mean; CO and O3 take the highest 8-hour running
     mean of the day, which is what CPCB's breakpoints for those two are defined on.
     A day needs 16 valid hours, so a mostly-dead sensor does not produce a number.
  2. city_daily     - the median across a city's stations, plus how many stations
     actually reported. The count matters: Guwahati and Bhopal run on one station
     for years, so their "city median" is a single sensor and the paper must say so.
  3. add_aqi        - CPCB AQI and category from the daily values.
"""

import numpy as np
import pandas as pd

from src.config import (
    MIN_HOURS_FOR_8H,
    MIN_HOURS_FOR_DAILY,
    MIN_PM25_PM10_RATIO,
    PHYSICAL_LIMITS,
    POLLUTANTS,
    POLLUTANTS_24H,
    POLLUTANTS_8H_MAX,
)
from src.utils.aqi_calculator import calculate_aqi_from_pollutants, get_aqi_category

ALL_AGGREGATED = POLLUTANTS_24H + POLLUTANTS_8H_MAX


def clip_to_physical_limits(hourly, limits=PHYSICAL_LIMITS):
    """Blank out readings outside each pollutant's instrument range (faults, not air quality)."""
    out = hourly.copy()
    for pollutant, (low, high) in limits.items():
        if pollutant in out.columns:
            out.loc[~out[pollutant].between(low, high), pollutant] = pd.NA
            out[pollutant] = pd.to_numeric(out[pollutant], errors='coerce')
    return out


def drop_inconsistent_pm(daily, min_ratio=MIN_PM25_PM10_RATIO):
    """Blank both PM channels on a station-day where they contradict each other.

    PM2.5 is measured as part of PM10, so it cannot exceed it, and it cannot be a
    negligible share of it either: whatever throws coarse particles into the air
    throws fine ones too. Guwahati on 2019-05-25 reported PM10 673 against PM2.5
    19.7, a ratio of 0.03, which carried that day to AQI 500 in a city whose
    median is 83 -- on one station, with no second sensor to outvote it.

    Which of the two is wrong cannot be known, so both go. The day then falls back
    on this station's other pollutants, and on whatever other stations reported.
    """
    if not {'PM2.5', 'PM10'} <= set(daily.columns):
        return daily
    out = daily.copy()
    fine, coarse = out['PM2.5'], out['PM10']
    both = fine.notna() & coarse.notna()
    impossible = both & (fine > coarse)
    implausible = both & (coarse > 0) & ((fine / coarse) < min_ratio)
    out.loc[impossible | implausible, ['PM2.5', 'PM10']] = np.nan
    return out


def _daily_mean(series, dates, min_hours):
    counts = series.groupby(dates).count()
    means = series.groupby(dates).mean()
    return means.where(counts >= min_hours)


def _daily_max_8h(series, dates, min_hours_8h, min_hours_24h):
    """Highest 8-hour running mean of each day, as CPCB defines CO and O3."""
    rolling = series.rolling('8h', min_periods=min_hours_8h).mean()
    daily_max = rolling.groupby(dates).max()
    # Still require a reasonably complete day, so one stray window cannot carry a day
    return daily_max.where(series.groupby(dates).count() >= min_hours_24h)


def station_daily(hourly, min_hours_24h=MIN_HOURS_FOR_DAILY, min_hours_8h=MIN_HOURS_FOR_8H):
    """Daily values for one station's hourly frame (from cpcb_loader.load_station_file)."""
    df = clip_to_physical_limits(hourly).sort_values('timestamp').set_index('timestamp')
    dates = df.index.normalize()

    out = {}
    for pollutant in POLLUTANTS_24H:
        if pollutant in df.columns:
            out[pollutant] = _daily_mean(df[pollutant], dates, min_hours_24h)
    for pollutant in POLLUTANTS_8H_MAX:
        if pollutant in df.columns:
            out[pollutant] = _daily_max_8h(df[pollutant], dates, min_hours_8h, min_hours_24h)

    daily = pd.DataFrame(out)
    daily.index.name = 'date'
    daily = daily.reset_index()
    daily = drop_inconsistent_pm(daily)
    present = [c for c in ALL_AGGREGATED if c in daily.columns]
    daily['station_reported'] = daily[present].notna().any(axis=1) if present else False
    for column in ('site_id', 'station'):
        if column in hourly.columns:
            daily.insert(1, column, hourly[column].iloc[0])
    return daily


def city_daily(station_dailies, city):
    """Median across a city's stations per day, with the number of stations reporting."""
    if not len(station_dailies):
        return pd.DataFrame()
    stacked = pd.concat(station_dailies, ignore_index=True)
    present = [c for c in ALL_AGGREGATED if c in stacked.columns]

    grouped = stacked.groupby('date')
    daily = grouped[present].median()
    daily['n_stations'] = grouped['station_reported'].sum().astype('int64')
    daily['n_stations_pm25'] = grouped['PM2.5'].count().astype('int64') if 'PM2.5' in present else 0

    daily = daily.reset_index()
    daily.insert(0, 'city', city)
    return daily


def add_aqi(daily, pollutants=POLLUTANTS):
    """CPCB AQI and 0-5 category from daily pollutant values."""
    present = [p for p in pollutants if p in daily.columns]
    values = daily[present].to_dict('records')
    out = daily.copy()
    out['AQI'] = [calculate_aqi_from_pollutants(row) for row in values]
    out['AQI_category'] = [get_aqi_category(aqi) for aqi in out['AQI']]
    return out


def reindex_to_calendar(daily, start, end):
    """One row per calendar day per city, so gaps are explicit rather than missing rows."""
    frames = []
    full = pd.date_range(start, end, freq='D')
    for city, group in daily.groupby('city', sort=False):
        filled = group.set_index('date').reindex(full)
        filled['city'] = city
        for column in ('n_stations', 'n_stations_pm25'):
            if column in filled.columns:
                filled[column] = filled[column].fillna(0).astype('int64')
        filled.index.name = 'date'
        frames.append(filled.reset_index())
    out = pd.concat(frames, ignore_index=True)
    return out[['city', 'date'] + [c for c in out.columns if c not in ('city', 'date')]]

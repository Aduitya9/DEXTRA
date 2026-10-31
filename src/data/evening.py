"""
The end of the day, kept: evening pollution and weather for every city-day.

The daily table averages each day into one number, and a forecast for tomorrow
is issued when today ends. Notebook 06 found that most severe days outside Delhi
are onsets -- severe on the target day, not on the day the forecast is made --
and that among days in the 200-300 band, where onsets start, the evening's PM2.5
separated them from other days better than the day's AQI did (AUC 0.763 against
0.724, hourly data for eleven cities, 2023-2025). A daily mean dilutes an evening
surge with a clean morning. These keep the shape of the day:

  pm25_evening, pm10_evening   mean of the hours stamped 18:00 to 23:00, local time
  pm25_late, pm10_late         mean of 21:00 to 23:00, the last thing seen
  pm25_rise, pm10_rise         the evening minus the same day's morning (00:00 to 11:00)
  pm_index_evening             the larger CPCB sub-index of the evening's PM2.5 and
                               PM10: the evening, on the AQI scale
  blh_evening, wind_evening    ERA5 boundary-layer height and wind speed over the
                               same evening hours, in Indian Standard Time

Every value comes from day t's own hours, so a forecast issued at the end of day t
could have read it. Station hours are cleaned as the daily pipeline cleans them --
instrument limits, and a station-day whose PM channels contradict each other loses
both -- and stations are combined by their median, as the daily table combines them.
"""

import numpy as np
import pandas as pd

from src.data import daily_cpcb, era5_features
from src.utils.aqi_calculator import calculate_sub_index

# window -> (first hour, last hour, hours that must have reported for a value)
WINDOWS = {'evening': (18, 23, 3), 'late': (21, 23, 2), 'morning': (0, 11, 6)}
PM = {'PM2.5': 'pm25', 'PM10': 'pm10'}
FEATURES = ['pm25_evening', 'pm10_evening', 'pm25_late', 'pm10_late', 'pm25_rise', 'pm10_rise',
            'pm_index_evening', 'blh_evening', 'wind_evening']


def _window_mean(values, dates, hours, first, last, required):
    """Each day's mean over hours first..last, where at least `required` of them reported."""
    inside = (hours >= first) & (hours <= last) & values.notna()
    grouped = values[inside].groupby(dates[inside])
    return grouped.mean().where(grouped.count() >= required)


def station_evening(hourly):
    """One station's hourly frame (cpcb_loader.load_station_file) -> its PM windows, per day."""
    df = daily_cpcb.clip_to_physical_limits(hourly).sort_values('timestamp').reset_index(drop=True)
    present = [p for p in PM if p in df.columns]
    if not present:
        return pd.DataFrame(columns=['date'])
    dates, hours = df['timestamp'].dt.normalize(), df['timestamp'].dt.hour
    if len(present) == 2:
        # A station-day whose PM channels contradict each other loses both, as in the daily table.
        daily = df.groupby(dates)[present].mean()
        cleaned = daily_cpcb.drop_inconsistent_pm(daily)
        contradicted = daily.notna().all(axis=1) & cleaned.isna().all(axis=1)
        df.loc[dates.isin(contradicted.index[contradicted]).to_numpy(), present] = np.nan
    windows = {f'{PM[p]}_{name}': _window_mean(df[p], dates, hours, *bounds)
               for p in present for name, bounds in WINDOWS.items()}
    # Every day the file covers keeps a row, with NaN where a window could not be read.
    frame = pd.DataFrame(windows).reindex(pd.DatetimeIndex(dates.unique()))
    frame.index.name = 'date'
    return frame.reset_index()


def evening_index(pm25, pm10):
    """The larger CPCB sub-index of the evening's PM2.5 and PM10; NaN where neither reported."""
    subs = np.array([[calculate_sub_index(a, 'PM2.5'), calculate_sub_index(b, 'PM10')]
                     for a, b in zip(pm25, pm10)], dtype=float).reshape(-1, 2)
    reported = np.isfinite(subs)
    return np.where(reported.any(axis=1), np.where(reported, subs, -np.inf).max(axis=1), np.nan)


def city_evening(station_frames, city):
    """The median across a city's stations per day, then the evening's rise and its sub-index."""
    frames = [f for f in station_frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=['city', 'date'])
    stacked = pd.concat(frames, ignore_index=True)
    daily = stacked.groupby('date')[[c for c in stacked.columns if c != 'date']].median()
    for short in PM.values():
        if f'{short}_evening' in daily and f'{short}_morning' in daily:
            daily[f'{short}_rise'] = daily[f'{short}_evening'] - daily[f'{short}_morning']
    nothing = pd.Series(np.nan, index=daily.index)
    daily['pm_index_evening'] = evening_index(daily.get('pm25_evening', nothing),
                                              daily.get('pm10_evening', nothing))
    daily = daily.reset_index()
    daily.insert(0, 'city', city)
    return daily


def era5_evening(surface, patch=None):
    """ERA5 boundary-layer height and wind speed over the evening hours, in IST, per city-day.

    `surface` is the hourly ERA5 series (city, time_utc, blh, wind_speed, ...);
    `patch` fills the boundary-layer hours Open-Meteo's archive lacks, as for the daily table.
    """
    df = era5_features.to_ist(era5_features.apply_blh_patch(surface, patch))
    first, last, required = WINDOWS['evening']
    frames = []
    for city, part in df.groupby('city', sort=False):
        hours = part['time_ist'].dt.hour
        frame = pd.DataFrame({
            'blh_evening': _window_mean(part['blh'], part['date'], hours, first, last, required),
            'wind_evening': _window_mean(part['wind_speed'], part['date'], hours, first, last, required),
        })
        frame.index.name = 'date'
        frames.append(frame.reset_index().assign(city=city))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['city', 'date'])


def build_table(city_frames, weather, start, end):
    """One row per city and calendar day from start to end, every evening feature a column.

    Days a city's stations did not cover stay NaN rather than being dropped, so
    the table lines up row for row with the daily table it is merged into.
    """
    pollution = pd.concat([f for f in city_frames if len(f)], ignore_index=True)
    table = pollution.merge(weather, on=['city', 'date'], how='outer')
    table = daily_cpcb.reindex_to_calendar(table, start, end)
    for column in FEATURES:
        if column not in table:
            table[column] = np.nan
    return table[['city', 'date'] + FEATURES]

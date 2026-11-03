"""
Hourly series per city, and the shape of each day, from the CPCB station files.

The daily table reduces a day to one mean. These keep what that loses, for the final
model and for the gradient-boosting baseline it has to beat:

  city_hourly  one city's stations combined by their median, hour by hour, cleaned as
               the daily table is cleaned, on an unbroken hourly grid.
  dynamics     one row per day, from that day's hours and those before it: the level,
               peak and spread of PM2.5, each part of the day, how fast the evening is
               rising, and the last three days' level and spread. Everything is known
               by the end of the day it describes.
  add_night_anchors  what tonight's last hours imply for each coming day, in AQI points,
               from the ratio that has lately held in that city. The final model weighs
               this as one of the places a forecast can start from.
"""

import numpy as np
import pandas as pd

from src.data import daily_cpcb
from src.utils.aqi_calculator import calculate_sub_index

CHANNELS = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'site_rel_humidity', 'site_air_temp', 'site_wind_speed']
DAY_PARTS = {'night': (0, 5), 'morning': (6, 11), 'afternoon': (12, 17), 'evening': (18, 23), 'late': (21, 23), 'last': (23, 23)}
NIGHT_HOURS = (21, 23)
ANCHOR_MEMORY_DAYS = 14
ANCHOR_MIN_DAYS = 5
# How far the anchor moves from today towards tonight's projection, chosen on the training
# years alone: on 2019-2023, all cities, next day, it reached AP 0.850 (0.848 at 0.5, 0.819
# at 0.75, persistence 0.820), and at two and three days it was the best of those tried.
ANCHOR_WEIGHT = 0.25


def city_hourly(station_frames, city, channels=CHANNELS):
    """The median across a city's stations per hour, every hour present, for every channel."""
    frames = []
    for frame in station_frames:
        clean = daily_cpcb.blank_contradicted_pm(daily_cpcb.clip_to_physical_limits(frame))
        frames.append(clean[['timestamp'] + [c for c in channels if c in clean.columns]])
    if not frames:
        return pd.DataFrame(columns=['city', 'timestamp'] + list(channels))
    merged = (pd.concat(frames, ignore_index=True).groupby('timestamp').median(numeric_only=True)
              .reindex(columns=channels).sort_index().asfreq('h'))
    merged.index.name = 'timestamp'
    merged = merged.reset_index()
    merged.insert(0, 'city', city)
    return merged


def _night_and_day(series, night=NIGHT_HOURS):
    """A pollutant's mean over the last hours of each day, and its mean over each whole day."""
    stamps = series.index
    day, hour = stamps.normalize(), stamps.hour
    inside = (hour >= night[0]) & (hour <= night[1])
    days = pd.DatetimeIndex(day.unique())
    late = series[inside].groupby(day[inside]).mean().reindex(days)
    grouped = series.groupby(day)
    return late, grouped.mean().where(grouped.count() >= daily_cpcb.MIN_HOURS_FOR_DAILY).reindex(days)


def night_anchor(series, horizon, memory=ANCHOR_MEMORY_DAYS, min_periods=ANCHOR_MIN_DAYS, weight=ANCHOR_WEIGHT):
    """What tonight implies for the daily mean `horizon` days on, per day, in the series' own units.

    Tonight's level times the median, over the last `memory` nights whose outcome is
    known, of (daily mean `horizon` days later) / (that night's level). At the end of day
    t the latest such night is t - horizon, whose outcome is day t, so nothing unknown at
    the moment of issue enters. One odd night would swing that projection too far, so it
    is blended with today's mean, geometrically and in concentration, because the AQI
    scale bends at each breakpoint: today ** (1 - weight) * projection ** weight. It reads
    only this city's own recent past, so it needs no training and exists for a city no
    model has seen once it has a few weeks of record.
    """
    late, mean = _night_and_day(series)
    ratio = mean.shift(-horizon) / late.clip(lower=1.0)
    usual = ratio.shift(horizon).rolling(memory, min_periods=min_periods).median()
    return mean.clip(lower=1.0) ** (1 - weight) * (late * usual).clip(lower=1.0) ** weight


def add_night_anchors(daily, table, horizons=(1, 2, 3)):
    """`daily` with anchor1.. per horizon, in AQI points, one row per city-day unchanged.

    PM2.5 and PM10 are projected separately and the larger sub-index taken. Where another
    pollutant set today's AQI the anchor does not go below it, and where a city has too
    little history for a ratio the anchor is today's AQI: persistence.
    """
    table = table.copy()
    table['timestamp'] = pd.to_datetime(table['timestamp'])
    frames = []
    for city, part in table.groupby('city', sort=False):
        part = part.sort_values('timestamp').set_index('timestamp')
        anchors = {}
        for horizon in horizons:
            fine = night_anchor(part['PM2.5'], horizon)
            coarse = night_anchor(part['PM10'], horizon)
            anchors[f'pm_anchor{horizon}'] = np.fmax(_sub_indices(fine, 'PM2.5'), _sub_indices(coarse, 'PM10'))
        frame = pd.DataFrame(anchors, index=fine.index).rename_axis('date').reset_index()
        frame.insert(0, 'city', city)
        frames.append(frame)
    anchors = pd.concat(frames, ignore_index=True)
    anchors['date'] = anchors['date'].astype('datetime64[ns]')
    out = daily.merge(anchors, on=['city', 'date'], how='left')
    pm_today = np.fmax(_sub_indices(out['PM2.5'], 'PM2.5'), _sub_indices(out['PM10'], 'PM10'))
    aqi = out['AQI'].to_numpy(dtype=float)
    other = np.where(aqi > pm_today + 1e-6, aqi, 0.0)
    for horizon in horizons:
        projected = out.pop(f'pm_anchor{horizon}').to_numpy(dtype=float)
        out[f'anchor{horizon}'] = np.where(np.isfinite(projected), np.maximum(projected, other), aqi)
    return out


def _sub_indices(values, pollutant):
    return np.array([calculate_sub_index(float(v), pollutant) if np.isfinite(v) and v >= 0 else np.nan
                     for v in np.asarray(values, dtype=float)], dtype=float)


def _slope(values):
    """Change per hour over the part of the day given, where at least three hours reported."""
    seen = values.notna().to_numpy()
    if seen.sum() < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(values))[seen], values.to_numpy()[seen], 1)[0])


def dynamics(hourly):
    """One city's hourly frame (city_hourly) -> the shape of each day, in columns named h_*."""
    frame = hourly.set_index('timestamp').sort_index()
    pm25, day, hour = frame['PM2.5'], frame.index.normalize(), frame.index.hour
    grouped = pm25.groupby(day)
    out = pd.DataFrame({'h_pm25_max': grouped.max(), 'h_pm25_min': grouped.min(),
                        'h_pm25_mean': grouped.mean(), 'h_pm25_std': grouped.std()})
    out['h_pm25_amp'] = out['h_pm25_max'] - out['h_pm25_min']
    out['h_pm25_peakhour'] = grouped.apply(lambda s: s.idxmax().hour if s.notna().any() else np.nan)
    for line in (90, 120):
        out[f'h_pm25_hours_over{line}'] = grouped.apply(lambda s, v=line: (s > v).sum() if s.notna().any() else np.nan)
    for name, (first, last) in DAY_PARTS.items():
        inside = (hour >= first) & (hour <= last)
        out[f'h_pm25_{name}'] = pm25[inside].groupby(day[inside]).mean()
    for name, first in (('slope6', 18), ('slope12', 12)):
        inside = hour >= first
        out[f'h_pm25_{name}'] = pm25[inside].groupby(day[inside]).apply(_slope)
    out['h_evening_minus_day'] = out['h_pm25_evening'] - out['h_pm25_mean']
    out['h_evening_change'] = out['h_pm25_evening'] - out['h_pm25_evening'].shift(1)
    out['h_night_change'] = out['h_pm25_night'] - out['h_pm25_night'].shift(1)
    out['h_afternoon_clean'] = out['h_pm25_afternoon'] / out['h_pm25_mean'].clip(lower=1)
    last_of_day = frame.index.hour == 23
    stamp = frame.index[last_of_day].normalize()
    rolling = pm25.rolling(72, min_periods=36)
    out['h_72h_mean'] = rolling.mean()[last_of_day].set_axis(stamp).reindex(out.index)
    out['h_72h_std'] = rolling.std()[last_of_day].set_axis(stamp).reindex(out.index)
    out['h_48h_change'] = pm25.diff(48)[last_of_day].set_axis(stamp).reindex(out.index)
    if 'PM10' in frame:
        inside = hour >= 18
        out['h_ratio_evening'] = pm25[inside].groupby(day[inside]).mean() / frame['PM10'][inside].groupby(day[inside]).mean().clip(lower=1)
    out.index.name = 'date'
    return out.reset_index()

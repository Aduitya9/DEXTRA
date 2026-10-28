"""
NASA FIRMS detections -> daily per-city fire features.

Delhi's worst air arrives from crop fires 200-400 km upwind in Punjab and Haryana,
so a city's own municipal area says nothing useful. Each city therefore counts
detections inside two rings: 100 km for local burning, 500 km for the regional
transport that drives the November smog.

Only "presumed vegetation fire" detections are kept. The other FIRMS types are
volcanoes, offshore flares and other static land sources: a gas flare near an
industrial city burns every day and would read as a permanently elevated fire
count rather than a burning season.
"""

import numpy as np
import pandas as pd

from src.config import (
    FIRE_RADII_KM,
    FIRMS_MIN_CONFIDENCE,
    FIRMS_VEGETATION_TYPE,
    FRP_RADIUS_KM,
    IST_OFFSET,
    TARGET_CITIES,
)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalars for the city, arrays for the detections."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def filter_detections(fires, min_confidence=FIRMS_MIN_CONFIDENCE, fire_type=FIRMS_VEGETATION_TYPE):
    """Keep confident vegetation-fire detections only."""
    keep = pd.Series(True, index=fires.index)
    if 'confidence' in fires.columns:
        keep &= pd.to_numeric(fires['confidence'], errors='coerce') >= min_confidence
    if 'type' in fires.columns and fire_type is not None:
        keep &= pd.to_numeric(fires['type'], errors='coerce') == fire_type
    return fires[keep]


def _acquisition_ist(fires):
    """FIRMS acq_date/acq_time are UTC; return the IST calendar date of each detection."""
    minutes = pd.to_numeric(fires['acq_time'], errors='coerce').fillna(0)
    utc = pd.to_datetime(fires['acq_date']) + pd.to_timedelta(minutes // 100, unit='h') \
        + pd.to_timedelta(minutes % 100, unit='m')
    return (utc + pd.Timedelta(IST_OFFSET)).dt.normalize()


def city_fire_daily(fires, cities=None, radii=FIRE_RADII_KM, frp_radius=FRP_RADIUS_KM):
    """Daily fire counts and radiative power per city, from one or more years of detections."""
    cities = cities or TARGET_CITIES
    kept = filter_detections(fires)
    if kept.empty:
        return pd.DataFrame(columns=['city', 'date'] + list(radii) + ['frp_500km'])

    dates = _acquisition_ist(kept)
    lat = kept['latitude'].to_numpy(dtype=float)
    lon = kept['longitude'].to_numpy(dtype=float)
    frp = pd.to_numeric(kept.get('frp'), errors='coerce').fillna(0.0).to_numpy(dtype=float)

    frames = []
    for city, meta in cities.items():
        distance = haversine_km(meta['lat'], meta['lon'], lat, lon)
        per_city = pd.DataFrame({'date': dates.to_numpy()})
        for column, radius in radii.items():
            per_city[column] = (distance <= radius).astype('int64')
        within_frp = distance <= frp_radius
        per_city['frp_500km'] = np.where(within_frp, frp, 0.0)
        per_city = per_city[per_city[list(radii)].to_numpy().any(axis=1) | within_frp]

        daily = per_city.groupby('date').sum().reset_index()
        daily.insert(0, 'city', city)
        frames.append(daily)

    return pd.concat(frames, ignore_index=True)


def fill_calendar(fire_daily, start, end, cities=None):
    """Days with no detections are real zeros, not missing data."""
    cities = cities or TARGET_CITIES
    index = pd.MultiIndex.from_product(
        [list(cities), pd.date_range(start, end, freq='D')], names=['city', 'date']
    )
    value_columns = [c for c in fire_daily.columns if c not in ('city', 'date')]
    out = (fire_daily.set_index(['city', 'date'])
                     .reindex(index)
                     .fillna({c: 0 for c in value_columns})
                     .reset_index())
    for column in value_columns:
        if column.startswith('fire_count'):
            out[column] = out[column].astype('int64')
    return out

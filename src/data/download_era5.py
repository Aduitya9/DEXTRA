"""
ERA5 Meteorological Data Downloader for the 15 DEXTRA cities.

Surface variables come as hourly point series from the Open-Meteo ERA5 archive:
the model needs 15 grid cells, not the India grid, so a point API replaces a
~20-35 GB CDS download. Temperature on pressure levels (for the inversion index)
is not served there, so it comes from the Copernicus CDS. 2m temperature and
surface pressure at the same hours come from the same CDS chain, so both sides
of the inversion difference share one processing chain.

Open-Meteo's ERA5 boundary-layer height has a hole: every hour from 2024-01-01
to 2024-06-30 is missing, for all cities (checked 2026-10-26). Those months are
patched from CDS, together with one overlapping month to verify both sources agree.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    CDS_AREA,
    ERA5_PRESSURE_LEVELS,
    INVERSION_HOURS_UTC,
    INVERSION_OFFSET_HPA,
    OPEN_METEO_ARCHIVE_URL,
    TARGET_CITIES,
)

# Open-Meteo variable -> DEXTRA column name
SURFACE_VARIABLES = {
    'boundary_layer_height': 'blh',
    'temperature_2m': 'temp2m',
    'dew_point_2m': 'dewpoint2m',
    'surface_pressure': 'surface_pressure',
    'precipitation': 'precip',
    'wind_speed_10m': 'wind_speed',
    'wind_direction_10m': 'wind_dir',
}

KELVIN = 273.15


def _get_with_retry(session, url, params, retries=5, backoff=5.0):
    """GET with linear backoff on rate limiting and transient server errors."""
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=120)
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
            return resp
        time.sleep(backoff * (attempt + 1))
    resp.raise_for_status()
    return resp


def fetch_surface_hourly(lat, lon, start_date, end_date, session=None):
    """Hourly ERA5 surface variables at the grid cell nearest (lat, lon), UTC timestamps."""
    session = session or requests.Session()
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(SURFACE_VARIABLES),
        'models': 'era5',          # pure ERA5, not Open-Meteo's best-match blend
        'timezone': 'GMT',
        'wind_speed_unit': 'ms',   # Open-Meteo defaults to km/h
    }
    payload = _get_with_retry(session, OPEN_METEO_ARCHIVE_URL, params).json()
    hourly = payload['hourly']

    df = pd.DataFrame({'time_utc': pd.to_datetime(hourly['time'])})
    for api_name, col in SURFACE_VARIABLES.items():
        df[col] = pd.to_numeric(pd.Series(hourly[api_name]), errors='coerce')
    df['grid_lat'] = payload.get('latitude')
    df['grid_lon'] = payload.get('longitude')
    return df


def fetch_surface_city(city, years, session=None, pause=1.0):
    """All requested years of hourly surface data for one target city."""
    meta = TARGET_CITIES[city]
    session = session or requests.Session()
    frames = []
    for year in years:
        frames.append(fetch_surface_hourly(meta['lat'], meta['lon'], f'{year}-01-01', f'{year}-12-31', session))
        time.sleep(pause)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, 'city', city)
    return df


def select_inversion_level(mean_surface_pressure_hpa, levels=ERA5_PRESSURE_LEVELS, offset=INVERSION_OFFSET_HPA):
    """Pressure level closest to `offset` hPa above the surface (a fixed 925 hPa is underground in Bengaluru)."""
    target = mean_surface_pressure_hpa - offset
    return min(levels, key=lambda level: abs(level - target))


def cds_request(year, kind, months=None):
    """(dataset, request) for one year of the inversion inputs.

    kind='pressure' -> temperature on ERA5_PRESSURE_LEVELS
    kind='surface'  -> 2m temperature and surface pressure at the same hours
    """
    months = months or list(range(1, 13))
    base = {
        'product_type': ['reanalysis'],
        'year': [str(year)],
        'month': [f'{m:02d}' for m in months],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': list(INVERSION_HOURS_UTC),
        'data_format': 'netcdf',
        'download_format': 'unarchived',
        'area': list(CDS_AREA),
    }
    if kind == 'pressure':
        return 'reanalysis-era5-pressure-levels', {
            **base,
            'variable': ['temperature'],
            'pressure_level': [str(level) for level in ERA5_PRESSURE_LEVELS],
        }
    if kind == 'surface':
        return 'reanalysis-era5-single-levels', {
            **base,
            'variable': ['2m_temperature', 'surface_pressure'],
        }
    if kind == 'blh':
        return 'reanalysis-era5-single-levels', {
            **base,
            'variable': ['boundary_layer_height'],
            'time': [f'{h:02d}:00' for h in range(24)],
        }
    raise ValueError(f"Unknown CDS request kind: {kind}")


def extract_city_points(ds, cities=None):
    """Nearest-grid-cell series for each city from a CDS ERA5 dataset, one wide row per (city, time)."""
    cities = cities or TARGET_CITIES
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    ds = ds.drop_vars([v for v in ('number', 'expver') if v in ds.variables])
    level_dim = next((d for d in ('pressure_level', 'level') if d in ds.dims), None)

    frames = []
    for city, meta in cities.items():
        point = ds.sel(latitude=meta['lat'], longitude=meta['lon'], method='nearest')
        grid_lat, grid_lon = float(point.latitude), float(point.longitude)
        df = point.drop_vars(['latitude', 'longitude']).to_dataframe()
        if level_dim:
            df = df.unstack(level_dim)
            df.columns = [f'{var}_{int(level)}hPa' for var, level in df.columns]
        df = df.reset_index().rename(columns={'time': 'time_utc'})
        df.insert(0, 'city', city)
        df['grid_lat'], df['grid_lon'] = grid_lat, grid_lon
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _to_dextra_units(df):
    """Kelvin -> Celsius and Pa -> hPa, with DEXTRA column names."""
    renames = {}
    for col in df.columns:
        if col.startswith('t_') and col.endswith('hPa'):
            df[col] = df[col] - KELVIN
            renames[col] = 'temp_' + col[2:]
    if 't2m' in df.columns:
        df['t2m'] = df['t2m'] - KELVIN
        renames['t2m'] = 'temp2m'
    if 'sp' in df.columns:
        df['sp'] = df['sp'] / 100.0
        renames['sp'] = 'surface_pressure'
    return df.rename(columns=renames)


def _month_blocks(months, months_per_request):
    months = sorted(months)
    return [months[i:i + months_per_request] for i in range(0, len(months), months_per_request)]


def _retrieve_points(client, year, kind, months, workdir, months_per_request):
    """Retrieve one request kind for the given months of a year and keep only the city grid cells."""
    import xarray as xr

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for block in _month_blocks(months, months_per_request):
        dataset, request = cds_request(year, kind, block)
        target = workdir / f'era5_{kind}_{year}_{block[0]:02d}.nc'
        client.retrieve(dataset, request, str(target))
        with xr.open_dataset(target) as ds:
            frames.append(extract_city_points(ds.load()))
        target.unlink()
    return pd.concat(frames, ignore_index=True)


def download_inversion_year(client, year, workdir, months_per_request=12):
    """One year of pressure-level temperature plus 2m temperature / surface pressure at the 15 cities.

    `client` is a cdsapi.Client. Lower `months_per_request` (6, 3, 1) if CDS rejects a request as too large.
    """
    months = list(range(1, 13))
    pressure = _retrieve_points(client, year, 'pressure', months, workdir, months_per_request)
    surface = _retrieve_points(client, year, 'surface', months, workdir, months_per_request)
    merged = pressure.merge(surface.drop(columns=['grid_lat', 'grid_lon']), on=['city', 'time_utc'], how='outer')
    return _to_dextra_units(merged).sort_values(['city', 'time_utc']).reset_index(drop=True)


def blh_gap_months(surface):
    """{year: [months]} in which any city's Open-Meteo boundary-layer height has missing hours."""
    missing = surface.loc[surface['blh'].isna(), 'time_utc']
    gaps = {}
    for year, month in sorted(set(zip(missing.dt.year, missing.dt.month))):
        gaps.setdefault(int(year), []).append(int(month))
    return gaps


def overlap_month(gap_months):
    """A month next to the gap, fetched too so the two sources can be compared."""
    return max(gap_months) + 1 if max(gap_months) < 12 else min(gap_months) - 1


def download_blh_months(client, year, months, workdir, months_per_request=12):
    """Hourly ERA5 boundary-layer height from CDS at the 15 cities for the given months of one year."""
    df = _retrieve_points(client, year, 'blh', months, workdir, months_per_request)
    return df.sort_values(['city', 'time_utc']).reset_index(drop=True)


def blh_agreement(open_meteo, cds):
    """Per-city agreement between the two BLH sources on hours where both have a value."""
    both = open_meteo[['city', 'time_utc', 'blh']].merge(
        cds[['city', 'time_utc', 'blh']], on=['city', 'time_utc'], suffixes=('_open_meteo', '_cds')
    ).dropna()
    if both.empty:
        return pd.DataFrame(columns=['city', 'hours', 'mean_abs_diff_m', 'correlation'])
    rows = []
    for city, g in both.groupby('city'):
        rows.append({
            'city': city,
            'hours': len(g),
            'mean_abs_diff_m': (g['blh_open_meteo'] - g['blh_cds']).abs().mean(),
            'correlation': g['blh_open_meteo'].corr(g['blh_cds']),
        })
    return pd.DataFrame(rows)

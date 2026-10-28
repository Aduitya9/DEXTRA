"""
ERA5 Meteorological Data Downloader for the 15 DEXTRA cities.

Surface variables come as hourly point series from the Open-Meteo ERA5 archive:
the model needs 15 grid cells, not the India grid, so a point API replaces a
~20-35 GB CDS download. Open-Meteo is asked for raw grid-cell values
(elevation=nan): by default it rescales temperature and pressure to a 90 m terrain
model, which shifted Pune by 0.9 C and 16 hPa against the ERA5 grid cell
(checked 2026-10-27) and would no longer match pressure-level temperatures.

Temperature on pressure levels (for the inversion index) is not served by
Open-Meteo, so it comes from the Copernicus CDS. Open-Meteo's ERA5 boundary-layer
height also has a hole: every hour from 2024-01-01 to 2024-06-30 is missing, for
all cities (checked 2026-10-26). Those months are patched from CDS, with one
overlapping month fetched to verify the two sources agree.

CDS asks users to submit requests sequentially, not in parallel or in batches
(Data Stores protection measures, October 2026), and can suspend accounts that
don't. CdsJobRunner therefore runs one request at a time, records each request ID
the moment it is submitted, and re-attaches to it after a Colab disconnect
instead of submitting again.
"""

import time
from datetime import datetime, timezone
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


# --------------------------------------------------------------------------- Open-Meteo


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
    """Hourly ERA5 surface variables of the grid cell nearest (lat, lon), raw grid values, UTC timestamps."""
    session = session or requests.Session()
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(SURFACE_VARIABLES),
        'models': 'era5',          # pure ERA5, not Open-Meteo's best-match blend
        'elevation': 'nan',        # raw grid-cell values, no terrain rescaling
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
    df['grid_elevation'] = payload.get('elevation')
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


def inversion_levels(surface):
    """{city: pressure level} from each city's mean grid-cell surface pressure."""
    mean_sp = surface.groupby('city')['surface_pressure'].mean()
    return {city: select_inversion_level(sp) for city, sp in mean_sp.items()}


# --------------------------------------------------------------------------- CDS requests


def cds_request(year, kind, months=None):
    """(dataset, request) for one year of CDS data.

    kind='pressure' -> temperature on ERA5_PRESSURE_LEVELS at INVERSION_HOURS_UTC
    kind='blh'      -> hourly boundary-layer height (gap fill for Open-Meteo)
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
    if kind == 'blh':
        return 'reanalysis-era5-single-levels', {
            **base,
            'variable': ['boundary_layer_height'],
            'time': [f'{h:02d}:00' for h in range(24)],
        }
    raise ValueError(f"Unknown CDS request kind: {kind}")


def _normalise(value):
    if isinstance(value, (list, tuple)):
        return sorted(_normalise(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def same_request(dataset, request, other_dataset, other_request,
                 keys=('variable', 'year', 'month', 'time', 'pressure_level', 'area')):
    """Whether a job already on CDS asked for the same data, ignoring formatting differences."""
    if dataset != other_dataset or not isinstance(other_request, dict):
        return False
    return all(_normalise(request.get(k, [])) == _normalise(other_request.get(k, [])) for k in keys)


class CdsJobRunner:
    """Runs CDS requests one at a time and survives notebook disconnects.

    `client` is an ecmwf.datastores.Client; `store` is an HFStore, where the ID of
    every submitted request is recorded before waiting starts.
    """

    ACTIVE = ('accepted', 'running', 'successful')

    def __init__(self, client, store, record_path='meta/cds_jobs.json', poll_seconds=30,
                 poll_max_seconds=300, log=print):
        self.client, self.store, self.record_path = client, store, record_path
        self.poll_seconds, self.poll_max_seconds, self.log = poll_seconds, poll_max_seconds, log
        self.records = store.read_json(record_path, default={}) if store.exists(record_path) else {}

    def _save(self, message):
        self.store.upload_json(self.records, self.record_path, message)

    def _recorded(self, key):
        request_id = self.records.get(key, {}).get('request_id')
        if not request_id:
            return None
        remote = self.client.get_remote(request_id)
        return remote if remote.status in self.ACTIVE else None

    def _already_on_cds(self, dataset, request, limit=50):
        """A matching job on this account, e.g. one submitted before a disconnect or by an older notebook."""
        jobs = self.client.get_jobs(limit=limit, sortby='-created', status=list(self.ACTIVE))
        for request_id in jobs.request_ids:
            remote = self.client.get_remote(request_id)
            if same_request(dataset, request, remote.collection_id, remote.request):
                return remote
        return None

    def _wait_until_ready(self, remote):
        """Poll until the job succeeds, returning the seconds waited.

        One status call per poll, with a widening interval: short at first so a
        job that runs straight away is picked up quickly, then up to poll_max
        seconds apart, because a queued job can sit for hours and ECMWF watches
        for heavy API use.
        """
        status, waited, interval = None, 0, self.poll_seconds
        while True:
            current = remote.status
            if current != status:
                status = current
                self.log(f"  {datetime.now(timezone.utc):%H:%M} UTC  status: {status}")
            if status not in ('accepted', 'running'):
                remote.results_ready   # raises with CDS's reason unless it succeeded
                return waited
            time.sleep(interval)
            waited += interval
            interval = min(interval * 1.5, self.poll_max_seconds)

    def run(self, key, dataset, request, target, reuse_existing=True):
        """Download the result of `request` to `target`, submitting only if no matching job exists."""
        remote = (self._recorded(key) or self._already_on_cds(dataset, request)) if reuse_existing else None
        reattached = remote is not None
        if reattached:
            self.log(f"  re-attached to CDS request {remote.request_id} ({remote.status})")
        else:
            remote = self.client.submit(dataset, request)
            self.log(f"  submitted CDS request {remote.request_id}")
        self.records[key] = {'request_id': remote.request_id, 'dataset': dataset,
                             'recorded_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        self._save(f"CDS job record: {key}")

        waited = self._wait_until_ready(remote)
        self.log(f"  ready after {waited / 3600:.1f} h of waiting here; downloading")
        try:
            path = remote.download(str(target))
        except Exception as err:
            if not reattached:
                raise
            # Results of an old successful job may have expired on CDS: submit once, fresh.
            self.log(f"  could not download the earlier result ({type(err).__name__}); submitting a new request")
            return self.run(key, dataset, request, target, reuse_existing=False)

        self.records.pop(key, None)
        self._save(f"CDS job done: {key}")
        return path


def _points_from_file(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return extract_city_points(ds.load())


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


def download_pressure_year(runner, year, workdir):
    """Temperature on the ERA5 pressure levels at INVERSION_HOURS_UTC, 15 cities, one year (Celsius)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dataset, request = cds_request(year, 'pressure')
    target = workdir / f'era5_pressure_{year}.nc'
    runner.run(f'pressure-{year}', dataset, request, target)
    df = _to_dextra_units(_points_from_file(target))
    target.unlink()
    return df.sort_values(['city', 'time_utc']).reset_index(drop=True)


def pressure_from_inversion_file(df):
    """Pressure-level temperature columns from an earlier combined inversion file (no new CDS request)."""
    keep = ['city', 'time_utc', *[c for c in df.columns if c.startswith('temp_') and c.endswith('hPa')], 'grid_lat', 'grid_lon']
    return df[keep].sort_values(['city', 'time_utc']).reset_index(drop=True)


# --------------------------------------------------------------------------- BLH gap fill


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


def download_blh_months(runner, year, months, workdir):
    """Hourly ERA5 boundary-layer height from CDS at the 15 cities for the given months of one year."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dataset, request = cds_request(year, 'blh', sorted(months))
    target = workdir / f'era5_blh_{year}.nc'
    runner.run(f'blh-{year}', dataset, request, target)
    df = _points_from_file(target)
    target.unlink()
    return df.sort_values(['city', 'time_utc']).reset_index(drop=True)


def series_agreement(a, b, column, a_name='open_meteo', b_name='cds'):
    """Per-city agreement of one variable between two sources, on hours where both have a value."""
    both = a[['city', 'time_utc', column]].merge(
        b[['city', 'time_utc', column]], on=['city', 'time_utc'], suffixes=(f'_{a_name}', f'_{b_name}')
    ).dropna()
    if both.empty:
        return pd.DataFrame(columns=['city', 'hours', 'mean_abs_diff', 'correlation'])
    rows = []
    for city, g in both.groupby('city'):
        x, y = g[f'{column}_{a_name}'], g[f'{column}_{b_name}']
        rows.append({'city': city, 'hours': len(g), 'mean_abs_diff': (x - y).abs().mean(), 'correlation': x.corr(y)})
    return pd.DataFrame(rows)


def blh_agreement(open_meteo, cds):
    return series_agreement(open_meteo, cds, 'blh').rename(columns={'mean_abs_diff': 'mean_abs_diff_m'})

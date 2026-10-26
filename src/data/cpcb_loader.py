"""
Loader for CPCB CAAQMS Data Repository files.

The repository (airquality.cpcb.gov.in/ccr/#/repository/data) serves one CSV per
station x frequency x year, named Raw_Data_{year}_site_{id}_{station}_{freq}.csv:
a `Timestamp` column (local time, assumed IST) followed by measured variables whose
names carry their units, e.g. "PM2.5 (µg/m³)", with the literal string NA for
missing readings.

Two repository quirks handled here:
  - Empty placeholder files. The repository lists years in which a station has no
    readings at all; those files hold a full hourly grid with every reading NA.
  - TOT-RF (cumulative rainfall) is computed, and reads 0.00 even in empty files,
    so it is dropped rather than counted as a measurement.
"""

import re
from pathlib import Path

import pandas as pd

from src.config import POLLUTANTS

FILENAME_RE = re.compile(
    r'Raw_Data_(?P<year>\d{4})_site_(?P<site_id>\d+)_(?P<station>.+)_(?P<freq>15M|1Hr|8Hr|1D)\.csv$'
)

# Repository variable name (without unit) -> DEXTRA column name
VARIABLE_NAMES = {
    'PM2.5': 'PM2.5', 'PM10': 'PM10', 'NO': 'NO', 'NO2': 'NO2', 'NOx': 'NOx', 'NH3': 'NH3',
    'SO2': 'SO2', 'CO': 'CO', 'Ozone': 'O3',
    'Benzene': 'benzene', 'Toluene': 'toluene', 'Xylene': 'xylene', 'O Xylene': 'o_xylene',
    'Eth-Benzene': 'ethylbenzene', 'MP-Xylene': 'mp_xylene',
    'AT': 'site_air_temp', 'RH': 'site_rel_humidity', 'WS': 'site_wind_speed', 'WD': 'site_wind_dir',
    'RF': 'site_rainfall', 'SR': 'site_solar_rad', 'BP': 'site_pressure_mmhg', 'VWS': 'site_vertical_wind',
}
COMPUTED_COLUMNS = {'TOT-RF'}

# Byte sizes of the repository's all-NA hourly templates (8,760 and 8,784 rows),
# identical across stations. A fast pre-filter only: partly filled years still
# need value counts.
EMPTY_TEMPLATE_BYTES = {823773, 826029}
HEADER_ONLY_MAX_BYTES = 1000


def parse_filename(path):
    m = FILENAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Not a CPCB repository filename: {Path(path).name}")
    return {'year': int(m['year']), 'site_id': int(m['site_id']), 'station': m['station'], 'freq': m['freq']}


def looks_like_empty_template(path):
    size = Path(path).stat().st_size
    return size in EMPTY_TEMPLATE_BYTES or size < HEADER_ONLY_MAX_BYTES


def _column_name(raw):
    base = raw.split(' (')[0].strip()
    if base in VARIABLE_NAMES:
        return VARIABLE_NAMES[base]
    return re.sub(r'[^0-9a-z]+', '_', base.lower()).strip('_')


def load_station_file(path):
    """One repository file as a tidy frame: timestamp, identifiers, then numeric variables."""
    meta = parse_filename(path)
    df = pd.read_csv(path, encoding='utf-8-sig', na_values=['NA'], keep_default_na=True)
    df = df.drop(columns=[c for c in df.columns if c.split(' (')[0].strip() in COMPUTED_COLUMNS])
    df = df.rename(columns={c: _column_name(c) for c in df.columns if c != 'Timestamp'})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.rename(columns={'Timestamp': 'timestamp'})

    value_cols = [c for c in df.columns if c != 'timestamp']
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce')
    df.insert(1, 'site_id', meta['site_id'])
    df.insert(2, 'station', meta['station'])
    return df


def pollutant_coverage(df, pollutants=POLLUTANTS):
    """Share of rows with a reading, per pollutant present in the file."""
    return df[[p for p in pollutants if p in df.columns]].notna().mean()


def is_empty(df, pollutants=POLLUTANTS):
    present = [p for p in pollutants if p in df.columns]
    return not present or not df[present].notna().any().any()

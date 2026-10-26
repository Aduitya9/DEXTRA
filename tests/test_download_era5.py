import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import CDS_AREA, ERA5_PRESSURE_LEVELS, INVERSION_HOURS_UTC
from src.data import download_era5 as era5

CITIES = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090},
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
}
TIMES = pd.to_datetime(['2024-11-10 00:00', '2024-11-10 03:00'])
LATS = np.arange(30.0, 11.75, -0.25)   # CDS grids run north -> south
LONS = np.arange(71.5, 93.25, 0.25)


@pytest.mark.parametrize("surface_pressure, expected_level", [
    (989.0, 925),    # Delhi
    (914.0, 850),    # Bengaluru: 925 hPa is below ground here
    (1008.0, 925),   # Mumbai
    (950.0, 875),    # Hyderabad
])
def test_inversion_level_sits_above_ground(surface_pressure, expected_level):
    assert era5.select_inversion_level(surface_pressure) == expected_level


def test_cds_pressure_request_shape():
    dataset, request = era5.cds_request(2024, 'pressure')
    assert dataset == 'reanalysis-era5-pressure-levels'
    assert request['variable'] == ['temperature']
    assert request['pressure_level'] == [str(level) for level in ERA5_PRESSURE_LEVELS]
    assert request['time'] == list(INVERSION_HOURS_UTC)
    assert request['area'] == list(CDS_AREA)
    assert len(request['month']) == 12 and request['data_format'] == 'netcdf'


def test_cds_surface_request_can_be_split_by_month():
    dataset, request = era5.cds_request(2024, 'surface', months=[1, 2, 3])
    assert dataset == 'reanalysis-era5-single-levels'
    assert request['variable'] == ['2m_temperature', 'surface_pressure']
    assert request['month'] == ['01', '02', '03']


def test_unknown_request_kind_raises():
    with pytest.raises(ValueError):
        era5.cds_request(2024, 'wind')


def test_cds_blh_request_is_hourly():
    dataset, request = era5.cds_request(2024, 'blh', months=[1, 2])
    assert dataset == 'reanalysis-era5-single-levels'
    assert request['variable'] == ['boundary_layer_height']
    assert len(request['time']) == 24 and request['month'] == ['01', '02']


def test_month_blocks_split_requests():
    assert era5._month_blocks([1, 2, 3, 4, 5, 6, 7], 3) == [[1, 2, 3], [4, 5, 6], [7]]


def test_blh_gap_detection_and_overlap_month():
    times = pd.date_range('2024-05-31', '2024-07-01 23:00', freq='h')
    blh = np.where(times < pd.Timestamp('2024-07-01'), np.nan, 300.0)
    surface = pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': blh})
    gaps = era5.blh_gap_months(surface)
    assert gaps == {2024: [5, 6]}
    assert era5.overlap_month(gaps[2024]) == 7
    assert era5.overlap_month([11, 12]) == 10


def test_blh_agreement_compares_only_hours_both_sources_have():
    times = pd.date_range('2024-07-01', periods=4, freq='h')
    open_meteo = pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': [100.0, 200.0, 300.0, np.nan]})
    cds = pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': [110.0, 210.0, 310.0, 400.0]})
    stats = era5.blh_agreement(open_meteo, cds).iloc[0]
    assert stats['hours'] == 3
    assert stats['mean_abs_diff_m'] == pytest.approx(10.0)
    assert stats['correlation'] == pytest.approx(1.0)


class _FakeCdsClient:
    """Writes synthetic CDS-shaped NetCDF files instead of contacting CDS."""

    def __init__(self):
        self.requests = []

    def retrieve(self, dataset, request, target):
        self.requests.append((dataset, request['variable'], request['month']))
        if dataset == 'reanalysis-era5-pressure-levels':
            _pressure_dataset().to_netcdf(target)
        else:
            _surface_dataset().to_netcdf(target)


def test_download_inversion_year_end_to_end(tmp_path):
    client = _FakeCdsClient()
    df = era5.download_inversion_year(client, 2024, tmp_path, months_per_request=6)

    assert len(client.requests) == 4  # 2 kinds x 2 half-year blocks
    assert {'temp_925hPa', 'temp2m', 'surface_pressure'} <= set(df.columns)
    assert df['temp_925hPa'].iloc[0] == pytest.approx(21.85)
    assert df['surface_pressure'].iloc[0] == pytest.approx(989.0)
    assert not list(tmp_path.glob('*.nc'))  # temporary NetCDF files are cleaned up


def _pressure_dataset():
    shape = (len(TIMES), len(ERA5_PRESSURE_LEVELS), len(LATS), len(LONS))
    t = np.full(shape, 290.0)
    t[:, ERA5_PRESSURE_LEVELS.index(925)] = 295.0  # warm layer aloft = inversion
    return xr.Dataset(
        {'t': (('valid_time', 'pressure_level', 'latitude', 'longitude'), t)},
        coords={'valid_time': TIMES, 'pressure_level': ERA5_PRESSURE_LEVELS,
                'latitude': LATS, 'longitude': LONS, 'number': 0, 'expver': ('valid_time', ['0001', '0001'])},
    )


def _surface_dataset():
    shape = (len(TIMES), len(LATS), len(LONS))
    return xr.Dataset(
        {'t2m': (('valid_time', 'latitude', 'longitude'), np.full(shape, 288.15)),
         'sp': (('valid_time', 'latitude', 'longitude'), np.full(shape, 98900.0))},
        coords={'valid_time': TIMES, 'latitude': LATS, 'longitude': LONS, 'number': 0},
    )


def test_extract_pressure_points_is_wide_per_city_and_time():
    df = era5.extract_city_points(_pressure_dataset(), CITIES)
    assert len(df) == len(CITIES) * len(TIMES)
    assert {f't_{level}hPa' for level in ERA5_PRESSURE_LEVELS} <= set(df.columns)
    assert set(df['city']) == set(CITIES)
    delhi = df[df.city == 'Delhi'].iloc[0]
    assert delhi['grid_lat'] == pytest.approx(28.5) and delhi['grid_lon'] == pytest.approx(77.25)


def test_units_are_converted_after_extraction():
    pres = era5.extract_city_points(_pressure_dataset(), CITIES)
    surf = era5.extract_city_points(_surface_dataset(), CITIES)
    merged = pres.merge(surf.drop(columns=['grid_lat', 'grid_lon']), on=['city', 'time_utc'])
    out = era5._to_dextra_units(merged)
    row = out.iloc[0]
    assert row['temp_925hPa'] == pytest.approx(21.85)
    assert row['temp2m'] == pytest.approx(15.0)
    assert row['surface_pressure'] == pytest.approx(989.0)
    assert 't2m' not in out.columns and 'sp' not in out.columns


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeSession:
    def __init__(self, payload):
        self.payload, self.params = payload, None

    def get(self, url, params, timeout):
        self.params = params
        return _FakeResponse(self.payload)


def test_fetch_surface_hourly_parses_open_meteo_and_keeps_gaps_as_nan():
    hourly = {'time': ['2024-11-10T00:00', '2024-11-10T01:00']}
    for name in era5.SURFACE_VARIABLES:
        hourly[name] = [1.0, None]
    session = _FakeSession({'latitude': 28.5, 'longitude': 77.25, 'hourly': hourly})

    df = era5.fetch_surface_hourly(28.61, 77.21, '2024-11-10', '2024-11-10', session)

    assert session.params['models'] == 'era5'
    assert session.params['wind_speed_unit'] == 'ms'
    assert df.columns[0] == 'time_utc'
    assert df['blh'].tolist()[0] == 1.0 and np.isnan(df['blh'].iloc[1])
    assert df['grid_lat'].iloc[0] == 28.5

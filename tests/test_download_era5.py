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


# --------------------------------------------------------------------------- inversion level


@pytest.mark.parametrize("surface_pressure, expected_level", [
    (989.0, 925),    # Delhi
    (914.0, 850),    # Bengaluru: 925 hPa is below ground here
    (1008.0, 925),   # Mumbai
    (950.0, 875),    # Hyderabad
])
def test_inversion_level_sits_above_ground(surface_pressure, expected_level):
    assert era5.select_inversion_level(surface_pressure) == expected_level


def test_inversion_levels_use_each_citys_mean_pressure():
    surface = pd.DataFrame({'city': ['Delhi', 'Delhi', 'Bengaluru'], 'surface_pressure': [988.0, 990.0, 914.0]})
    assert era5.inversion_levels(surface) == {'Delhi': 925, 'Bengaluru': 850}


# --------------------------------------------------------------------------- CDS requests


def test_cds_pressure_request_shape():
    dataset, request = era5.cds_request(2024, 'pressure')
    assert dataset == 'reanalysis-era5-pressure-levels'
    assert request['variable'] == ['temperature']
    assert request['pressure_level'] == [str(level) for level in ERA5_PRESSURE_LEVELS]
    assert request['time'] == list(INVERSION_HOURS_UTC)
    assert request['area'] == list(CDS_AREA)
    assert len(request['month']) == 12 and request['data_format'] == 'netcdf'


def test_cds_blh_request_is_hourly():
    dataset, request = era5.cds_request(2024, 'blh', months=[1, 2])
    assert dataset == 'reanalysis-era5-single-levels'
    assert request['variable'] == ['boundary_layer_height']
    assert len(request['time']) == 24 and request['month'] == ['01', '02']


def test_unknown_request_kind_raises():
    with pytest.raises(ValueError):
        era5.cds_request(2024, 'wind')


def test_same_request_ignores_formatting_but_not_content():
    dataset, request = era5.cds_request(2021, 'pressure')
    reformatted = {**request, 'area': [float(v) for v in reversed(request['area'])],
                   'pressure_level': list(reversed(request['pressure_level']))}
    assert era5.same_request(dataset, request, dataset, reformatted)
    assert not era5.same_request(dataset, request, dataset, {**request, 'year': ['2022']})
    assert not era5.same_request(dataset, request, 'reanalysis-era5-single-levels', request)
    assert not era5.same_request(dataset, request, dataset, None)


# --------------------------------------------------------------------------- point extraction


def _pressure_dataset():
    shape = (len(TIMES), len(ERA5_PRESSURE_LEVELS), len(LATS), len(LONS))
    t = np.full(shape, 290.0)
    t[:, ERA5_PRESSURE_LEVELS.index(925)] = 295.0  # warm layer aloft = inversion
    return xr.Dataset(
        {'t': (('valid_time', 'pressure_level', 'latitude', 'longitude'), t)},
        coords={'valid_time': TIMES, 'pressure_level': ERA5_PRESSURE_LEVELS,
                'latitude': LATS, 'longitude': LONS, 'number': 0, 'expver': ('valid_time', ['0001', '0001'])},
    )


def _blh_dataset():
    shape = (len(TIMES), len(LATS), len(LONS))
    return xr.Dataset(
        {'blh': (('valid_time', 'latitude', 'longitude'), np.full(shape, 350.0))},
        coords={'valid_time': TIMES, 'latitude': LATS, 'longitude': LONS, 'number': 0},
    )


def test_extract_pressure_points_is_wide_per_city_and_time():
    df = era5.extract_city_points(_pressure_dataset(), CITIES)
    assert len(df) == len(CITIES) * len(TIMES)
    assert {f't_{level}hPa' for level in ERA5_PRESSURE_LEVELS} <= set(df.columns)
    delhi = df[df.city == 'Delhi'].iloc[0]
    assert delhi['grid_lat'] == pytest.approx(28.5) and delhi['grid_lon'] == pytest.approx(77.25)


def test_units_are_converted():
    out = era5._to_dextra_units(era5.extract_city_points(_pressure_dataset(), CITIES))
    assert out['temp_925hPa'].iloc[0] == pytest.approx(21.85)
    assert not any(c.startswith('t_') for c in out.columns)


def test_pressure_columns_can_be_taken_from_an_old_inversion_file():
    old = era5._to_dextra_units(era5.extract_city_points(_pressure_dataset(), CITIES))
    old['temp2m'], old['surface_pressure'] = 15.0, 989.0
    df = era5.pressure_from_inversion_file(old)
    assert 'temp2m' not in df.columns and 'temp_925hPa' in df.columns


# --------------------------------------------------------------------------- CDS job runner


class _FakeStore:
    def __init__(self, records=None):
        self.json = {} if records is None else {'meta/cds_jobs.json': records}
        self.uploads = []

    def exists(self, path):
        return path in self.json

    def read_json(self, path, default=None):
        return self.json.get(path, default)

    def upload_json(self, obj, path, message):
        self.json[path] = dict(obj)
        self.uploads.append(message)


class _FakeRemote:
    def __init__(self, request_id, dataset, request, statuses, writer, fail_download=False):
        self.request_id, self.collection_id, self.request = request_id, dataset, request
        self._statuses, self._writer, self._fail = list(statuses), writer, fail_download
        self.polls = 0

    @property
    def status(self):
        return self._statuses[min(self.polls, len(self._statuses) - 1)]

    @property
    def results_ready(self):
        status = self.status
        self.polls += 1
        if status in ('failed', 'rejected'):
            raise RuntimeError(f'CDS job {status}')
        return status == 'successful'

    def download(self, target):
        if self._fail:
            raise OSError('results expired')
        self._writer().to_netcdf(target)
        return target


class _FakeJobs:
    def __init__(self, ids):
        self.request_ids = ids


class _FakeCdsClient:
    def __init__(self, existing=(), writer=_pressure_dataset, final_status='successful'):
        self.remotes = {r.request_id: r for r in existing}
        self.writer, self.final_status, self.submitted = writer, final_status, []

    def submit(self, dataset, request):
        remote = _FakeRemote(f'new-{len(self.submitted)}', dataset, request,
                             ['accepted', 'running', self.final_status], self.writer)
        self.remotes[remote.request_id] = remote
        self.submitted.append(remote.request_id)
        return remote

    def get_remote(self, request_id):
        return self.remotes[request_id]

    def get_jobs(self, limit, sortby, status):
        return _FakeJobs([rid for rid, r in self.remotes.items() if r.status in status])


def _runner(client, store):
    return era5.CdsJobRunner(client, store, poll_seconds=0, log=lambda *_: None)


def test_runner_submits_one_request_and_clears_its_record(tmp_path):
    client, store = _FakeCdsClient(), _FakeStore()
    df = era5.download_pressure_year(_runner(client, store), 2022, tmp_path)

    assert client.submitted == ['new-0']
    assert df['temp_925hPa'].iloc[0] == pytest.approx(21.85)
    assert store.json['meta/cds_jobs.json'] == {}   # recorded while waiting, cleared when done
    assert store.uploads[0] == 'CDS job record: pressure-2022'
    assert not list(tmp_path.glob('*.nc'))


def test_runner_reattaches_to_recorded_request_instead_of_resubmitting(tmp_path):
    dataset, request = era5.cds_request(2023, 'pressure')
    earlier = _FakeRemote('earlier-id', dataset, request, ['running', 'successful'], _pressure_dataset)
    client = _FakeCdsClient(existing=[earlier])
    store = _FakeStore({'pressure-2023': {'request_id': 'earlier-id'}})

    era5.download_pressure_year(_runner(client, store), 2023, tmp_path)
    assert client.submitted == []


def test_runner_adopts_matching_job_already_on_cds(tmp_path):
    dataset, request = era5.cds_request(2021, 'pressure')
    queued_by_old_notebook = _FakeRemote('old-nb-id', dataset, request, ['successful'], _pressure_dataset)
    other_year = _FakeRemote('other-id', dataset, era5.cds_request(2020, 'pressure')[1], ['successful'], _pressure_dataset)
    client = _FakeCdsClient(existing=[other_year, queued_by_old_notebook])

    era5.download_pressure_year(_runner(client, _FakeStore()), 2021, tmp_path)
    assert client.submitted == []
    assert queued_by_old_notebook.polls > 0 and other_year.polls == 0


def test_runner_resubmits_once_when_an_adopted_result_has_expired(tmp_path):
    dataset, request = era5.cds_request(2021, 'pressure')
    expired = _FakeRemote('expired-id', dataset, request, ['successful'], _pressure_dataset, fail_download=True)
    client = _FakeCdsClient(existing=[expired])

    era5.download_pressure_year(_runner(client, _FakeStore()), 2021, tmp_path)
    assert client.submitted == ['new-0']


def test_runner_surfaces_a_rejected_request(tmp_path):
    client = _FakeCdsClient(final_status='rejected')
    with pytest.raises(RuntimeError, match='rejected'):
        era5.download_pressure_year(_runner(client, _FakeStore()), 2022, tmp_path)


def test_blh_months_go_through_the_runner(tmp_path):
    client = _FakeCdsClient(writer=_blh_dataset)
    df = era5.download_blh_months(_runner(client, _FakeStore()), 2024, [7, 1, 2], tmp_path)
    assert client.remotes['new-0'].request['month'] == ['01', '02', '07']
    assert df['blh'].iloc[0] == 350.0


# --------------------------------------------------------------------------- gap fill and agreement


def test_blh_gap_detection_and_overlap_month():
    times = pd.date_range('2024-05-31', '2024-07-01 23:00', freq='h')
    blh = np.where(times < pd.Timestamp('2024-07-01'), np.nan, 300.0)
    gaps = era5.blh_gap_months(pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': blh}))
    assert gaps == {2024: [5, 6]}
    assert era5.overlap_month(gaps[2024]) == 7
    assert era5.overlap_month([11, 12]) == 10


def test_agreement_compares_only_hours_both_sources_have():
    times = pd.date_range('2024-07-01', periods=4, freq='h')
    open_meteo = pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': [100.0, 200.0, 300.0, np.nan]})
    cds = pd.DataFrame({'city': 'Delhi', 'time_utc': times, 'blh': [110.0, 210.0, 310.0, 400.0]})
    stats = era5.blh_agreement(open_meteo, cds).iloc[0]
    assert stats['hours'] == 3
    assert stats['mean_abs_diff_m'] == pytest.approx(10.0)
    assert stats['correlation'] == pytest.approx(1.0)


# --------------------------------------------------------------------------- Open-Meteo


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


def test_fetch_surface_hourly_asks_for_raw_grid_values_and_keeps_gaps_as_nan():
    hourly = {'time': ['2024-11-10T00:00', '2024-11-10T01:00']}
    for name in era5.SURFACE_VARIABLES:
        hourly[name] = [1.0, None]
    session = _FakeSession({'latitude': 28.5, 'longitude': 77.25, 'elevation': 220.0, 'hourly': hourly})

    df = era5.fetch_surface_hourly(28.61, 77.21, '2024-11-10', '2024-11-10', session)

    assert session.params['models'] == 'era5'
    assert session.params['elevation'] == 'nan'
    assert session.params['wind_speed_unit'] == 'ms'
    assert df['blh'].tolist()[0] == 1.0 and np.isnan(df['blh'].iloc[1])
    assert df['grid_elevation'].iloc[0] == 220.0

import pandas as pd
import pytest

from src.data import calendar_events, cpcb_loader
from src.utils.hf_store import HFStore

# --------------------------------------------------------------------------- calendar


def test_diwali_window_moves_with_the_lunar_calendar():
    flags = calendar_events.is_diwali(['2023-11-09', '2023-11-12', '2023-11-15', '2023-11-16', '2024-11-12'])
    assert flags.tolist() == [True, True, True, False, False]


def test_unlisted_diwali_year_raises_instead_of_reading_false():
    with pytest.raises(ValueError, match='2018'):
        calendar_events.is_diwali(['2018-11-07'])


def test_stubble_season_bounds_are_inclusive():
    flags = calendar_events.is_stubble_season(['2024-10-14', '2024-10-15', '2024-11-30', '2024-12-01'])
    assert flags.tolist() == [False, True, True, False]


# --------------------------------------------------------------------------- CPCB loader

HEADER = ('Timestamp,PM2.5 (µg/m³),PM10 (µg/m³),NO2 (µg/m³),SO2 (µg/m³),CO (mg/m³),'
          'Ozone (µg/m³),Eth-Benzene (µg/m³),AT (°C),TOT-RF (mm)')


def _write(tmp_path, rows, name='Raw_Data_2021_site_5417_pan_bazaar_guwahati_pcba_1Hr.csv'):
    path = tmp_path / name
    path.write_text('\n'.join([HEADER, *rows]) + '\n', encoding='utf-8')
    return path


def test_parse_filename():
    meta = cpcb_loader.parse_filename('Raw_Data_2019_site_5247_t_t_nagar_bhopal_mppcb_1Hr.csv')
    assert meta == {'year': 2019, 'site_id': 5247, 'station': 't_t_nagar_bhopal_mppcb', 'freq': '1Hr'}


def test_load_renames_units_maps_ozone_and_drops_computed_rainfall(tmp_path):
    path = _write(tmp_path, [
        '2021-01-01 00:00:00,85.2,140.0,40.1,9.0,1.2,30.0,0.4,12.5,0.00',
        '2021-01-01 01:00:00,NA,NA,NA,NA,NA,NA,NA,NA,0.00',
    ])
    df = cpcb_loader.load_station_file(path)

    assert list(df.columns[:3]) == ['timestamp', 'site_id', 'station']
    assert {'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'ethylbenzene', 'site_air_temp'} <= set(df.columns)
    assert 'TOT-RF' not in df.columns and 'tot_rf' not in df.columns
    assert df['PM2.5'].iloc[0] == 85.2 and pd.isna(df['PM2.5'].iloc[1])
    assert df['site_id'].iloc[0] == 5417
    assert cpcb_loader.pollutant_coverage(df)['PM2.5'] == 0.5
    assert not cpcb_loader.is_empty(df)


def test_all_na_file_is_empty_even_though_rainfall_reads_zero(tmp_path):
    path = _write(tmp_path, ['2019-01-01 00:00:00,NA,NA,NA,NA,NA,NA,NA,NA,0.00'] * 3)
    assert cpcb_loader.is_empty(cpcb_loader.load_station_file(path))


# --------------------------------------------------------------------------- HF store


class _FakeApi:
    def __init__(self, exists=True, files=()):
        self._exists, self.files, self.uploads = exists, list(files), []

    def repo_exists(self, repo_id, repo_type):
        return self._exists

    def list_repo_files(self, repo_id, repo_type):
        return self.files

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.uploads.append(path_in_repo)


def test_store_refuses_missing_repo_instead_of_creating_one():
    with pytest.raises(RuntimeError, match='not found'):
        HFStore('someone/typo-repo', token='t', api=_FakeApi(exists=False))


def test_store_tracks_uploads_for_resume():
    api = _FakeApi(files=['raw/firms/modis_2019.parquet'])
    store = HFStore('someone/dextra', token='t', api=api)
    assert store.exists('raw/firms/modis_2019.parquet')
    assert not store.exists('raw/firms/modis_2020.parquet')

    store.upload_df(pd.DataFrame({'a': [1]}), 'raw/firms/modis_2020.parquet', 'msg')
    assert store.exists('raw/firms/modis_2020.parquet')
    assert api.uploads == ['raw/firms/modis_2020.parquet']
    assert store.files('raw/firms/') == ['raw/firms/modis_2019.parquet', 'raw/firms/modis_2020.parquet']

from datetime import date

import pandas as pd
import pytest
import requests

from src.data import download_firms as firms

SECRET_KEY = 'secret-key-0123456789'


def test_plan_covers_every_day_of_the_year_once():
    plan = firms.plan_requests(2024)
    days = sum(n for _, n, _ in plan)
    assert days == 366  # leap year
    assert all(n <= 5 for _, n, _ in plan)
    assert {source for _, _, source in plan} == {firms.SP_SOURCE}


def test_plan_switches_to_nrt_after_sp_archive_end_and_splits_the_boundary_window():
    plan = firms.plan_requests(2025, sp_end_date=date(2025, 9, 2))
    assert sum(n for _, n, _ in plan) == 365
    sp_days = sum(n for _, n, s in plan if s == firms.SP_SOURCE)
    assert sp_days == (date(2025, 9, 2) - date(2025, 1, 1)).days + 1
    first_nrt = next(start for start, _, s in plan if s == firms.NRT_SOURCE)
    assert first_nrt == date(2025, 9, 3)


def test_archive_end_date_reads_availability_table():
    table = pd.DataFrame({
        'data_id': ['MODIS_NRT', 'MODIS_SP'],
        'min_date': ['2025-09-01', '2000-11-01'],
        'max_date': ['2026-10-25', '2025-08-31'],
    })
    assert firms.archive_end_date(table) == date(2025, 8, 31)
    assert firms.archive_end_date(table, 'VIIRS_SNPP_SP') is None


class _Resp:
    def __init__(self, status_code, text):
        self.status_code, self.text = status_code, text


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, url, timeout):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_error_messages_never_contain_the_map_key():
    session = _Session([_Resp(200, 'Invalid MAP_KEY.')])
    with pytest.raises(RuntimeError) as err:
        firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2024, 1, 1), 5, session=session)
    assert SECRET_KEY not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_connection_errors_are_sanitised():
    url_with_key = f'https://firms.example/api/area/csv/{SECRET_KEY}/MODIS_SP'
    session = _Session([requests.ConnectionError(url_with_key)] * 2)
    with pytest.raises(RuntimeError) as err:
        firms._get_csv(session, url_with_key, retries=2, backoff=0)
    assert SECRET_KEY not in str(err.value)


def test_fetch_area_parses_csv_and_handles_empty_windows():
    csv = 'latitude,longitude,brightness,acq_date,acq_time,confidence,frp\n30.9,75.8,320.1,2024-11-01,530,85,12.4\n'
    df = firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2024, 11, 1), 5, session=_Session([_Resp(200, csv)]))
    assert len(df) == 1 and df['frp'].iloc[0] == 12.4

    empty = firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2024, 7, 1), 5, session=_Session([_Resp(200, '')]))
    assert empty.empty

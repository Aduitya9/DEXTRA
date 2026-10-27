from datetime import date

import pandas as pd
import pytest
import requests

from src.config import FIRMS_STATUS_URL
from src.data import download_firms as firms

SECRET_KEY = 'secret-key-0123456789'
CSV = 'latitude,longitude,brightness,acq_date,acq_time,confidence,frp\n30.9,75.8,320.1,2024-11-01,530,85,12.4\n'


@pytest.fixture(autouse=True)
def fresh_key_state(monkeypatch):
    monkeypatch.setattr(firms, '_successful_requests', 0)


# --------------------------------------------------------------------------- planning


def test_plan_covers_every_day_of_the_year_once():
    plan = firms.plan_requests(2024)
    assert sum(n for _, n, _ in plan) == 366  # leap year
    assert all(n <= 5 for _, n, _ in plan)
    assert {source for _, _, source in plan} == {firms.SP_SOURCE}


def test_plan_switches_to_nrt_after_sp_archive_end_and_splits_the_boundary_window():
    plan = firms.plan_requests(2025, sp_end_date=date(2025, 9, 2))
    assert sum(n for _, n, _ in plan) == 365
    sp_days = sum(n for _, n, s in plan if s == firms.SP_SOURCE)
    assert sp_days == (date(2025, 9, 2) - date(2025, 1, 1)).days + 1
    assert next(start for start, _, s in plan if s == firms.NRT_SOURCE) == date(2025, 9, 3)


def test_archive_end_date_reads_availability_table():
    table = pd.DataFrame({
        'data_id': ['MODIS_NRT', 'MODIS_SP'],
        'min_date': ['2025-09-01', '2000-11-01'],
        'max_date': ['2026-10-25', '2025-08-31'],
    })
    assert firms.archive_end_date(table) == date(2025, 8, 31)
    assert firms.archive_end_date(table, 'VIIRS_SNPP_SP') is None


# --------------------------------------------------------------------------- fakes


class _Resp:
    def __init__(self, status_code, text='', payload=None):
        self.status_code, self.text, self._payload = status_code, text, payload

    def json(self):
        if self._payload is None:
            raise ValueError('not json')
        return self._payload


class _Session:
    """Area requests pop from `area`; status requests return `status` (or 404)."""

    def __init__(self, area=(), status=None):
        self.area, self.status, self.area_calls = list(area), status, 0

    def get(self, url, timeout, params=None):
        if url == FIRMS_STATUS_URL:
            return _Resp(200, payload=self.status) if self.status else _Resp(404, 'not found')
        self.area_calls += 1
        item = self.area.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# --------------------------------------------------------------------------- errors


def test_firms_400_explanation_is_kept_and_the_key_scrubbed():
    session = _Session([_Resp(400, f'Invalid date for {SECRET_KEY}.')])
    with pytest.raises(firms.FirmsError) as err:
        firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2025, 1, 1), 5, session=session)
    assert 'HTTP 400' in str(err.value) and 'Invalid date' in str(err.value)
    assert SECRET_KEY not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_connection_errors_are_sanitised():
    url_with_key = f'https://firms.example/api/area/csv/{SECRET_KEY}/MODIS_SP'
    session = _Session([requests.ConnectionError(url_with_key)] * 2)
    with pytest.raises(firms.FirmsError) as err:
        firms._get_csv(session, url_with_key, secret=SECRET_KEY, retries=2, backoff=0)
    assert SECRET_KEY not in str(err.value)


def test_transaction_limit_reply_is_waited_out_and_retried():
    session = _Session([_Resp(400, 'Exceeding allowed transaction limit.'), _Resp(200, CSV)])
    text = firms._get_csv(session, 'url', secret=SECRET_KEY, limit_wait=0)
    assert text == CSV and session.area_calls == 2


def test_invalid_key_on_the_very_first_request_fails_fast():
    session = _Session([_Resp(400, 'Invalid MAP_KEY.')])
    with pytest.raises(firms.FirmsError, match='Invalid MAP_KEY'):
        firms.fetch_year(SECRET_KEY, 2025, session=session, pause=0, limit_wait=0, log=lambda *_: None)
    assert session.area_calls == 1


def test_invalid_key_after_the_key_has_worked_is_treated_as_quota_and_retried():
    session = _Session([_Resp(200, CSV), _Resp(400, 'Invalid MAP_KEY.'), _Resp(200, CSV)] + [_Resp(200, '')] * 80)
    df = firms.fetch_year(SECRET_KEY, 2025, session=session, pause=0, limit_wait=0, log=lambda *_: None)
    assert len(df) == 2 and set(df['source']) == {firms.SP_SOURCE}


def test_persistent_failure_names_the_window_that_failed():
    session = _Session([_Resp(200, CSV)] + [_Resp(400, 'Invalid MAP_KEY.')] * 5)
    with pytest.raises(firms.FirmsError, match=r'window 2025-01-06 \+ 5 days, MODIS_SP'):
        firms.fetch_year(SECRET_KEY, 2025, session=session, pause=0, retries=2, limit_wait=0, log=lambda *_: None)


# --------------------------------------------------------------------------- pacing


def test_transaction_status_parses_firms_fields_and_tolerates_absence():
    assert firms.transaction_status(SECRET_KEY, _Session(status={'current_transactions': '120', 'transaction_limit': 5000,
                                                                 'transaction_interval': '10 minutes'})) == (120.0, 5000.0)
    assert firms.transaction_status(SECRET_KEY, _Session(status=None)) is None
    assert firms.transaction_status(SECRET_KEY, _Session(status={'unexpected': 1})) is None


def test_wait_for_transactions_pauses_only_near_the_limit():
    class Draining(_Session):
        def __init__(self):
            super().__init__()
            self.readings = [4900, 4900, 50]

        def get(self, url, timeout, params=None):
            used = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
            return _Resp(200, payload={'current_transactions': used, 'transaction_limit': 5000})

    session, logs = Draining(), []
    firms.wait_for_transactions(SECRET_KEY, session, poll_seconds=0, log=logs.append)
    assert session.readings == [50] and len(logs) == 1

    calm, logs = _Session(status={'current_transactions': 10, 'transaction_limit': 5000}), []
    firms.wait_for_transactions(SECRET_KEY, calm, poll_seconds=0, log=logs.append)
    assert logs == []


def test_fetch_area_parses_csv_and_handles_empty_windows():
    df = firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2024, 11, 1), 5, session=_Session([_Resp(200, CSV)]))
    assert len(df) == 1 and df['frp'].iloc[0] == 12.4
    empty = firms.fetch_area(SECRET_KEY, firms.SP_SOURCE, date(2024, 7, 1), 5, session=_Session([_Resp(200, '')]))
    assert empty.empty

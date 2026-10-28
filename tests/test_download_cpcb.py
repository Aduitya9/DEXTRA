import pandas as pd
import pytest
import requests

from src.data import download_cpcb
from src.data.cpcb_loader import EMPTY_TEMPLATE_BYTES, parse_filename

# Long enough to clear the header-only cutoff: a few bytes of CSV would itself
# look like a year the repository has nothing for.
REAL = b'Timestamp,PM2.5 (ug/m3)\n' + b'2021-01-01 00:00:00,42\n' * 500


def _manifest():
    return pd.DataFrame([
        {'city': 'Bhopal', 'site_id': 5247, 'station_name': 'T T Nagar, Bhopal - MPPCB',
         'years': '2021 2022'},
        {'city': 'Guwahati', 'site_id': 5417, 'station_name': 'Pan Bazaar, Guwahati - PCBA',
         'years': '2022'},
    ])


class FakeResponse:
    def __init__(self, content=REAL, status_code=200):
        self.content, self.status_code = content, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses=None):
        self.responses, self.urls = responses or {}, []

    def get(self, url, timeout=None):
        self.urls.append(url)
        for key, response in self.responses.items():
            if key in url:
                if isinstance(response, Exception):
                    raise response
                return response
        return FakeResponse()


class FakeStore:
    def __init__(self, existing=()):
        self.existing, self.uploaded = set(existing), {}

    def files(self, prefix=''):
        return [p for p in self.existing if p.startswith(prefix)]

    def upload_bytes(self, content, path, message):
        self.uploaded[path] = content
        self.existing.add(path)


def test_filename_matches_what_the_portal_would_have_saved():
    name = download_cpcb.repository_filename(2021, 301, 'Anand Vihar, Delhi - DPCC')
    assert name == 'Raw_Data_2021_site_301_anand_vihar_delhi_dpcc_1Hr.csv'
    # and the loader must be able to read it back
    assert parse_filename(name) == {'year': 2021, 'site_id': 301,
                                    'station': 'anand_vihar_delhi_dpcc', 'freq': '1Hr'}


def test_the_all_na_template_is_recognised_by_its_exact_size():
    for size in EMPTY_TEMPLATE_BYTES:
        assert download_cpcb.is_placeholder(b'x' * size)
    assert download_cpcb.is_placeholder(b'Timestamp\n')          # header only
    # a station reporting fewer species is small but real -- size thresholds would drop it
    assert not download_cpcb.is_placeholder(b'x' * 747_000)


def test_a_year_with_no_readings_comes_back_as_nothing():
    empty = FakeResponse(b'x' * min(EMPTY_TEMPLATE_BYTES))
    session = FakeSession({'2020': empty, '2009': FakeResponse(b'', 404)})
    assert download_cpcb.fetch_station_year(session, 2020, 301) is None
    assert download_cpcb.fetch_station_year(session, 2009, 301) is None
    assert download_cpcb.fetch_station_year(session, 2021, 301) == REAL


def test_a_network_error_is_retried_then_reported_with_its_reason():
    session = FakeSession({'2021': requests.ConnectionError('reset')})
    with pytest.raises(download_cpcb.CpcbError, match='reset'):
        download_cpcb.fetch_station_year(session, 2021, 301, retries=2, backoff=0)
    assert len(session.urls) == 2


def test_jobs_expand_the_manifest_year_list():
    jobs = download_cpcb.manifest_jobs(_manifest())
    assert [j['year'] for j in jobs] == [2021, 2022, 2022]
    assert download_cpcb.store_path(jobs[0]) == \
        'raw/cpcb/bhopal/Raw_Data_2021_site_5247_t_t_nagar_bhopal_mppcb_1Hr.csv'


def test_a_rerun_skips_what_already_landed():
    done = 'raw/cpcb/bhopal/Raw_Data_2021_site_5247_t_t_nagar_bhopal_mppcb_1Hr.csv'
    store = FakeStore([done])
    summary = download_cpcb.download_manifest(
        _manifest(), store, session=FakeSession(), pause=0, log=lambda *a: None)

    assert summary == {'uploaded': 2, 'skipped': 1, 'empty': 0, 'failed': 0}
    assert done not in store.uploaded          # the finished one was not fetched again
    assert len(store.uploaded) == 2


def test_an_empty_year_is_counted_rather_than_uploaded():
    store = FakeStore()
    session = FakeSession({'2022': FakeResponse(b'x' * min(EMPTY_TEMPLATE_BYTES))})
    summary = download_cpcb.download_manifest(
        _manifest(), store, session=session, pause=0, log=lambda *a: None)

    assert summary['empty'] == 2 and summary['uploaded'] == 1
    assert all('2021' in path for path in store.uploaded)

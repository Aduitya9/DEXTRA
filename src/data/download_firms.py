"""
NASA FIRMS Active Fire Downloader for India.

Pulls MODIS Collection 6.1 fire detections through the FIRMS Area API (free MAP_KEY,
at most 5 days per request). Standard-processing (SP) data is the quality-controlled
archive but lags real time by a few months, so dates past the SP archive's end are
filled from near-real-time (NRT) and tagged in a `source` column.

The MAP_KEY travels inside the request URL, so no error raised here ever includes
the URL: a failed request must not print the key into notebook output.
"""

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import FIRMS_API_URL, FIRMS_BBOX, FIRMS_DAY_RANGE

SP_SOURCE = 'MODIS_SP'
NRT_SOURCE = 'MODIS_NRT'
_RETRY_STATUSES = (429, 500, 502, 503, 504)


def _looks_like_csv(text):
    first_line = text.lstrip().split('\n', 1)[0]
    return first_line.count(',') >= 2


def _get_csv(session, url, retries=5, backoff=10.0):
    """CSV text from FIRMS. Errors are re-raised without the key-bearing URL."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise RuntimeError(f'FIRMS request failed: {type(exc).__name__}') from None
            time.sleep(backoff * (attempt + 1))
            continue

        if resp.status_code in _RETRY_STATUSES:
            time.sleep(backoff * (attempt + 1))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f'FIRMS HTTP {resp.status_code}') from None

        text = resp.text
        if not text.strip() or _looks_like_csv(text):
            return text
        # FIRMS reports problems (bad key, transaction limit) as plain-text 200 responses
        if 'limit' in text.lower() and attempt < retries - 1:
            time.sleep(backoff * (attempt + 1) * 6)
            continue
        raise RuntimeError(f'FIRMS error: {text.strip()[:200]}') from None
    raise RuntimeError(f'FIRMS request still failing after {retries} attempts') from None


def data_availability(map_key, session=None):
    """Date coverage of every FIRMS product, as published by the data_availability API."""
    session = session or requests.Session()
    text = _get_csv(session, f'{FIRMS_API_URL}/data_availability/csv/{map_key}/ALL')
    return pd.read_csv(io.StringIO(text))


def archive_end_date(availability, source=SP_SOURCE):
    """Last date covered by `source` in a data_availability table, or None if it cannot be determined."""
    id_col = next((c for c in availability.columns if 'id' in c.lower()), None)
    max_col = next((c for c in availability.columns if 'max' in c.lower()), None)
    if id_col is None or max_col is None:
        return None
    row = availability[availability[id_col].astype(str).str.upper() == source]
    return pd.to_datetime(row[max_col].iloc[0]).date() if len(row) else None


def plan_requests(year, sp_end_date=None, day_range=FIRMS_DAY_RANGE):
    """(start_date, n_days, source) windows covering one calendar year.

    Windows use SP up to `sp_end_date` and NRT after it; a window that straddles the
    boundary is split there. Without a known SP end date, everything is SP.
    """
    plan = []
    current, year_end = date(year, 1, 1), date(year, 12, 31)
    while current <= year_end:
        n = min(day_range, (year_end - current).days + 1)
        window_end = current + timedelta(days=n - 1)
        if sp_end_date is None or window_end <= sp_end_date:
            plan.append((current, n, SP_SOURCE))
        elif current > sp_end_date:
            plan.append((current, n, NRT_SOURCE))
        else:
            n_sp = (sp_end_date - current).days + 1
            plan.append((current, n_sp, SP_SOURCE))
            plan.append((sp_end_date + timedelta(days=1), n - n_sp, NRT_SOURCE))
        current = window_end + timedelta(days=1)
    return plan


def fetch_area(map_key, source, start_date, day_range, bbox=FIRMS_BBOX, session=None):
    """Fire detections in `bbox` for `day_range` days starting at `start_date`."""
    session = session or requests.Session()
    url = f'{FIRMS_API_URL}/area/csv/{map_key}/{source}/{bbox}/{day_range}/{start_date.isoformat()}'
    text = _get_csv(session, url)
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def fetch_year(map_key, year, sp_end_date=None, bbox=FIRMS_BBOX, session=None, pause=0.5):
    """All MODIS fire detections in `bbox` for one calendar year, with a `source` column."""
    session = session or requests.Session()
    frames = []
    for start, n, source in plan_requests(year, sp_end_date):
        df = fetch_area(map_key, source, start, n, bbox, session)
        if len(df):
            df['source'] = source
            frames.append(df)
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['acq_date'] = pd.to_datetime(out['acq_date'])
    return out.sort_values(['acq_date', 'acq_time']).reset_index(drop=True)

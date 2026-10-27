"""
NASA FIRMS Active Fire Downloader for India.

Pulls MODIS Collection 6.1 fire detections through the FIRMS Area API (free MAP_KEY,
at most 5 days per request). Standard-processing (SP) data is the quality-controlled
archive but lags real time by a few months, so dates past the SP archive's end are
filled from near-real-time (NRT) and tagged in a `source` column.

A MAP_KEY allows 5000 transactions per 10-minute window, and a 5-day request over
all of India counts as several. Requests are therefore paced against the key's
transaction status, and a refusal mid-run waits out the window and retries.

FIRMS explains failures in the body of an HTTP 400 ("Invalid MAP_KEY.", "Invalid day
range. Expects [1..5]."), so that text is kept in errors. The MAP_KEY travels inside
the request URL, so no error raised here includes the URL, and the key is scrubbed
from any response text: a failed request must not print the key into notebook output.
"""

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import FIRMS_API_URL, FIRMS_BBOX, FIRMS_DAY_RANGE, FIRMS_LIMIT_WAIT_SECONDS, FIRMS_STATUS_URL

SP_SOURCE = 'MODIS_SP'
NRT_SOURCE = 'MODIS_NRT'
_RETRY_STATUSES = (429, 500, 502, 503, 504)

# Requests that succeeded in this process. Once the key is known to work, a later
# "Invalid MAP_KEY." is treated as a possible quota refusal rather than a bad key.
_successful_requests = 0


class FirmsError(RuntimeError):
    """A FIRMS failure. Its message never contains the MAP_KEY."""

    def __init__(self, message, rate_limited=False):
        super().__init__(message)
        self.rate_limited = rate_limited


def _looks_like_csv(text):
    first_line = text.lstrip().split('\n', 1)[0]
    return first_line.count(',') >= 2


def _scrub(text, secret):
    text = ' '.join(text.split())[:300]
    return text.replace(secret, '***') if secret else text


def _is_rate_limit(text):
    lowered = text.lower()
    return 'limit' in lowered or 'exceed' in lowered


def _get_csv(session, url, secret=None, retries=5, backoff=10.0, limit_wait=FIRMS_LIMIT_WAIT_SECONDS):
    """CSV text from FIRMS, or FirmsError carrying FIRMS's own explanation (key scrubbed)."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=120)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise FirmsError(f'FIRMS request failed: {type(exc).__name__}') from None
            time.sleep(backoff * (attempt + 1))
            continue

        if resp.status_code in _RETRY_STATUSES:
            time.sleep(backoff * (attempt + 1))
            continue

        text = resp.text
        if resp.status_code == 200 and (not text.strip() or _looks_like_csv(text)):
            return text
        if _is_rate_limit(text):
            if attempt < retries - 1:
                time.sleep(limit_wait)
                continue
            raise FirmsError(f'FIRMS transaction limit: {_scrub(text, secret)}', rate_limited=True) from None
        raise FirmsError(f'FIRMS HTTP {resp.status_code}: {_scrub(text, secret)}') from None
    raise FirmsError(f'FIRMS request still failing after {retries} attempts') from None


def data_availability(map_key, session=None):
    """Date coverage of every FIRMS product, as published by the data_availability API."""
    session = session or requests.Session()
    text = _get_csv(session, f'{FIRMS_API_URL}/data_availability/csv/{map_key}/ALL', secret=map_key)
    return pd.read_csv(io.StringIO(text))


def archive_end_date(availability, source=SP_SOURCE):
    """Last date covered by `source` in a data_availability table, or None if it cannot be determined."""
    id_col = next((c for c in availability.columns if 'id' in c.lower()), None)
    max_col = next((c for c in availability.columns if 'max' in c.lower()), None)
    if id_col is None or max_col is None:
        return None
    row = availability[availability[id_col].astype(str).str.upper() == source]
    return pd.to_datetime(row[max_col].iloc[0]).date() if len(row) else None


def transaction_status(map_key, session=None):
    """(used, limit) transactions for the key in the current window, or None if FIRMS doesn't say."""
    session = session or requests.Session()
    try:
        resp = session.get(FIRMS_STATUS_URL, params={'MAP_KEY': map_key}, timeout=60)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        return float(payload['current_transactions']), float(payload['transaction_limit'])
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def wait_for_transactions(map_key, session=None, headroom=0.8, poll_seconds=60,
                          max_wait=FIRMS_LIMIT_WAIT_SECONDS + 120, log=print):
    """Block while the key has used more than `headroom` of its window's transactions."""
    waited, announced = 0, False
    status = transaction_status(map_key, session)
    while status is not None and status[0] >= headroom * status[1] and waited < max_wait:
        if not announced:
            log(f'  FIRMS quota at {status[0]:.0f}/{status[1]:.0f} transactions; pausing for the window to reset')
            announced = True
        time.sleep(poll_seconds)
        waited += max(poll_seconds, 1)
        status = transaction_status(map_key, session)


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
    global _successful_requests
    session = session or requests.Session()
    url = f'{FIRMS_API_URL}/area/csv/{map_key}/{source}/{bbox}/{day_range}/{start_date.isoformat()}'
    text = _get_csv(session, url, secret=map_key)
    _successful_requests += 1
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def _fetch_window(map_key, source, start, n, bbox, session, retries, limit_wait, log):
    for attempt in range(retries + 1):
        try:
            return fetch_area(map_key, source, start, n, bbox, session)
        except FirmsError as err:
            # A key that has already worked can be refused as "Invalid MAP_KEY." once its quota is spent.
            quota_like = err.rate_limited or (_successful_requests > 0 and 'invalid map_key' in str(err).lower())
            if quota_like and attempt < retries:
                log(f'  FIRMS refused {start} ({err}); waiting {limit_wait // 60} min for the transaction window to reset')
                time.sleep(limit_wait)
                continue
            raise FirmsError(f'{err} [window {start} + {n} days, {source}]', err.rate_limited) from None


def fetch_year(map_key, year, sp_end_date=None, bbox=FIRMS_BBOX, session=None, pause=0.5,
               check_every=10, retries=2, limit_wait=FIRMS_LIMIT_WAIT_SECONDS, log=print):
    """All MODIS fire detections in `bbox` for one calendar year, with a `source` column."""
    session = session or requests.Session()
    frames = []
    for i, (start, n, source) in enumerate(plan_requests(year, sp_end_date)):
        if i % check_every == 0:
            wait_for_transactions(map_key, session, log=log)
        df = _fetch_window(map_key, source, start, n, bbox, session, retries, limit_wait, log)
        if len(df):
            df['source'] = source
            frames.append(df)
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['acq_date'] = pd.to_datetime(out['acq_date'])
    return out.sort_values(['acq_date', 'acq_time']).reset_index(drop=True)

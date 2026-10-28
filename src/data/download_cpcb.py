"""
Fetch CPCB station files from the Data Repository.

The repository's web page is behind a CAPTCHA and encrypts its API traffic, but
the file download itself is an ordinary unauthenticated GET:

    /caaqms-common/dataRepository/download-excel-file?file_name=/Raw_data/1Hr/{year}/site_{id}_1Hr.csv

so the data this project needs can be collected without a browser once the
station ids are known. Those ids live in meta/cpcb_manifest.csv, which records
which station-years actually hold readings; see meta/cpcb_stations.tsv for the
full station master the manifest was chosen from.

The repository answers for every year a station has ever been listed under, so a
year it was not yet running still returns a complete file -- a full hourly grid
with every reading NA. Those weigh exactly 823,773 bytes, or 826,029 when the
year is a leap year, whatever the station. Size alone cannot be used more loosely
than that: a station reporting fewer species can be 98% complete in 747 KB.
"""

import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from src.data.cpcb_loader import EMPTY_TEMPLATE_BYTES, HEADER_ONLY_MAX_BYTES

BASE_URL = ("https://airquality.cpcb.gov.in/caaqms-common/dataRepository/"
            "download-excel-file?file_name=/Raw_data/{freq}/{year}/site_{site_id}_{freq}.csv")
USER_AGENT = "Mozilla/5.0 (compatible; DEXTRA research data collection)"
DEFAULT_FREQ = "1Hr"


class CpcbError(RuntimeError):
    """A station-year that could not be fetched, with the reason kept."""


def station_slug(station_name):
    """'Anand Vihar, Delhi - DPCC' -> 'anand_vihar_delhi_dpcc', as the portal names its downloads."""
    return re.sub(r'[^0-9a-z]+', '_', station_name.lower()).strip('_')


def repository_filename(year, site_id, station_name, freq=DEFAULT_FREQ):
    """The name the portal gives a download, which cpcb_loader.parse_filename reads back."""
    return f"Raw_Data_{year}_site_{site_id}_{station_slug(station_name)}_{freq}.csv"


def file_url(year, site_id, freq=DEFAULT_FREQ):
    return BASE_URL.format(freq=freq, year=year, site_id=site_id)


def new_session():
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    return session


def is_placeholder(content):
    """True for the all-NA template the repository returns for a year with no readings."""
    size = len(content)
    return size in EMPTY_TEMPLATE_BYTES or size < HEADER_ONLY_MAX_BYTES


def fetch_station_year(session, year, site_id, freq=DEFAULT_FREQ, retries=3, backoff=4.0):
    """Raw bytes for one station-year, or None where the repository has no readings."""
    last = None
    for attempt in range(retries):
        try:
            response = session.get(file_url(year, site_id, freq), timeout=120)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return None if is_placeholder(response.content) else response.content
        except requests.RequestException as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise CpcbError(f"site_{site_id} {year}: {last}")


def manifest_jobs(manifest, freq=DEFAULT_FREQ):
    """One job per station-year the manifest says holds readings."""
    jobs = []
    for row in manifest.to_dict('records'):
        for year in str(row['years']).split():
            jobs.append({
                'city': row['city'],
                'site_id': int(row['site_id']),
                'station_name': row['station_name'],
                'year': int(year),
                'filename': repository_filename(year, int(row['site_id']), row['station_name'], freq),
            })
    return jobs


def store_path(job, prefix='raw/cpcb'):
    """Where a job's file lives on Hugging Face, grouped by city as notebook 02 expects."""
    return f"{prefix}/{job['city'].lower()}/{job['filename']}"


def _flush(store, city, pending, prefix, log):
    """Commit one batch of a city's files. Returns how many were uploaded."""
    if not pending:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for filename, content in pending:
            (Path(tmp) / filename).write_bytes(content)
        store.upload_folder(tmp, f"{prefix}/{city.lower()}",
                            f"CPCB {city}: {len(pending)} station-years")
    log(f"  committed {len(pending)} files for {city}")
    return len(pending)


def download_manifest(manifest, store, session=None, freq=DEFAULT_FREQ, pause=0.5,
                      prefix='raw/cpcb', batch_size=25, log=print):
    """Fetch every station-year in the manifest and upload it, skipping what is already there.

    Uploads go up a batch at a time, one commit per batch. A commit per file would
    mean 478 of them, which is slow and is the kind of rapid-fire write the Hub
    rate-limits; batching by city keeps it to roughly twenty.

    Resumable by design: the run is long enough that a dropped Colab session is
    likely, and re-running must not re-fetch what landed before it dropped. The
    cost of a drop is bounded by `batch_size` -- whatever was fetched but not yet
    committed is fetched again.
    """
    session = session or new_session()
    jobs = manifest_jobs(manifest, freq)
    have = set(store.files(prefix))
    summary = {'uploaded': 0, 'skipped': 0, 'empty': 0, 'failed': 0, 'commits': 0}

    by_city = {}
    for job in jobs:
        by_city.setdefault(job['city'], []).append(job)

    done = 0
    for city, city_jobs in by_city.items():
        pending = []
        for job in city_jobs:
            done += 1
            if store_path(job, prefix) in have:
                summary['skipped'] += 1
                continue
            try:
                content = fetch_station_year(session, job['year'], job['site_id'], freq)
            except CpcbError as exc:
                summary['failed'] += 1
                log(f"  [{done}/{len(jobs)}] ! {job['filename']}: {exc}")
                continue
            if content is None:
                # The manifest was built from file sizes, so a year can turn out empty
                # after all; recording it is more useful than silently moving on.
                summary['empty'] += 1
                log(f"  [{done}/{len(jobs)}] - {job['filename']}: no readings")
                continue
            pending.append((job['filename'], content))
            if len(pending) >= batch_size:
                summary['uploaded'] += _flush(store, city, pending, prefix, log)
                summary['commits'] += 1
                pending = []
            time.sleep(pause)
        if pending:
            summary['uploaded'] += _flush(store, city, pending, prefix, log)
            summary['commits'] += 1
    return summary

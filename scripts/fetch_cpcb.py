"""
Fetch the CPCB station files on a normal machine, and optionally build the daily
pollutant table from them.

Why this exists rather than a notebook: CPCB drops connections from the cloud IP
ranges Colab runs on. Every request from a notebook times out at connect while
the same URL answers in under a second from an ordinary Indian connection, so
this one stage cannot run where the rest of the pipeline runs.

    python scripts/fetch_cpcb.py                     # download to data/raw/cpcb
    python scripts/fetch_cpcb.py --build             # + build the daily table
    python scripts/fetch_cpcb.py --build --upload    # + send it to Hugging Face

Resumable: anything already in the output folder is skipped, so an interrupted
run costs only the files it had not reached.

--upload signs in with the token `hf auth login` stores, so nothing secret has
to go in the environment or in shell history; HF_TOKEN still works if set. It
sends the daily table, which is a few megabytes. The raw station files stay on
this machine unless --upload-raw is given, since re-hosting a government
dataset is a licensing question of its own and notebook 02 needs only the
daily table.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import daily_cpcb, download_cpcb  # noqa: E402
from src.data.cpcb_loader import load_station_file  # noqa: E402
from src.utils.local_store import LocalStore  # noqa: E402

POLLUTANT_PATH = "processed/pollutant_daily.parquet"


def build_daily(root, manifest):
    """Station files on disk -> one row per city-day, with CPCB AQI."""
    slugs = {city.lower(): city for city in config.TARGET_CITIES}
    frames, empty = [], []
    for slug in sorted(p.name for p in Path(root).iterdir() if p.is_dir()):
        city = slugs.get(slug, slug)
        dailies = []
        for path in sorted((Path(root) / slug).glob("*.csv")):
            daily = daily_cpcb.station_daily(load_station_file(path))
            measured = [c for c in config.POLLUTANTS if c in daily.columns]
            if measured and daily[measured].notna().any().any():
                dailies.append(daily)
            else:
                empty.append(path.name)
        if dailies:
            frames.append(daily_cpcb.city_daily(dailies, city))
        print(f"  {city:<11} {len(dailies):>3} station-years with data, "
              f"{len(list((Path(root) / slug).glob('*.csv'))) - len(dailies)} empty")

    pollutants = daily_cpcb.add_aqi(pd.concat(frames, ignore_index=True))
    pollutants = daily_cpcb.reindex_to_calendar(
        pollutants, config.DATA_START_DATE, config.DATA_END_DATE)
    print(f"\n{len(pollutants):,} city-days, AQI on {pollutants.AQI.notna().mean():.1%} of them, "
          f"{len(empty)} empty station-years skipped")
    return pollutants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "raw" / "cpcb"))
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "meta" / "cpcb_manifest.csv"))
    parser.add_argument("--cities", nargs="*", help="limit to these cities (default: all)")
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--build", action="store_true", help="build the daily pollutant table")
    parser.add_argument("--upload", action="store_true", help="send the daily table to Hugging Face")
    parser.add_argument("--repo", help="dataset repo, e.g. owner/name (default: $HF_DATASET_REPO)")
    parser.add_argument("--upload-raw", action="store_true", help="also send the station files")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.cities:
        manifest = manifest[manifest.city.isin(args.cities)]
    total = sum(len(str(y).split()) for y in manifest["years"])
    print(f"{len(manifest)} stations, {total} station-years -> {args.out}")

    store = LocalStore(args.out)
    summary = download_cpcb.download_manifest(manifest, store, pause=args.pause, prefix="")
    print(f"\ndownloaded {summary['uploaded']}  already had {summary['skipped']}  "
          f"empty {summary['empty']}  failed {summary['failed']}")
    if summary["failed"]:
        print("Re-run to retry the failures; everything already fetched is skipped.")

    if not (args.build or args.upload):
        return
    print("\nBuilding the daily table:")
    pollutants = build_daily(args.out, manifest)
    local = PROJECT_ROOT / "data" / "processed" / "pollutant_daily.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)
    pollutants.to_parquet(local, index=False)
    print(f"wrote {local}")

    if not args.upload:
        return
    # token=None lets huggingface_hub use the login stored by `hf auth login`,
    # which keeps the token out of the environment and out of shell history.
    token = os.environ.get("HF_TOKEN") or None
    repo = args.repo or os.environ.get("HF_DATASET_REPO")
    if not repo:
        print("\n! Pass --repo <owner/name> or set HF_DATASET_REPO.")
        return
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=token).whoami()
    except Exception:
        print("\n! Not signed in to Hugging Face. Run `hf auth login` and paste a token\n"
              "  with write access to the dataset repo, then re-run with --upload.\n"
              f"  The daily table is already written to {local}, so nothing is lost.")
        return
    print(f"\nsigned in to Hugging Face as {who['name']}")
    from src.utils.hf_store import HFStore
    hf = HFStore(repo, token=token)
    hf.upload_file(local, POLLUTANT_PATH, "Daily city pollutants and CPCB AQI")
    print(f"↑ {POLLUTANT_PATH}")
    if args.upload_raw:
        hf.upload_folder(args.out, "raw/cpcb", "CPCB station files")
        print("↑ raw/cpcb/")


if __name__ == "__main__":
    main()

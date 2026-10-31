"""
Build the evening features from the hourly CPCB station files on this machine.

Why this exists rather than a notebook: the station files live only here. CPCB
drops connections from Colab's IP ranges, and re-hosting a government dataset is
a licensing question of its own, so the raw files never went to the dataset
repo. What goes there is the derived table -- one row per city-day, about a
megabyte -- which notebook 07 merges into the daily table.

    python scripts/build_evening_features.py            # writes data/processed/evening_features.parquet
    python scripts/build_evening_features.py --upload   # + sends it to Hugging Face

The ERA5 hourly fields are read from the dataset repo, where notebook 01 put them.
Signing in uses the token `hf auth login` stores; HF_TOKEN still works if set.
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
from src.data import evening  # noqa: E402
from src.data.cpcb_loader import load_station_file, looks_like_empty_template  # noqa: E402
from src.utils.hf_store import HFStore  # noqa: E402

EVENING_PATH = "processed/evening_features.parquet"


def build_pollution(root):
    """Station files on disk -> one frame of evening PM windows per city."""
    frames = []
    for city in config.TARGET_CITIES:
        folder = Path(root) / city.lower()
        paths = sorted(folder.glob("*_1Hr.csv")) if folder.is_dir() else []
        stations = [evening.station_evening(load_station_file(p)) for p in paths
                    if not looks_like_empty_template(p)]
        frame = evening.city_evening(stations, city)
        frames.append(frame)
        covered = frame["pm25_evening"].notna().sum() if len(frame) else 0
        print(f"  {city:<11} {len(paths):>3} station-years, {covered:>5,} days with an evening PM2.5")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpcb", default=str(PROJECT_ROOT / "data" / "raw" / "cpcb"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed" / "evening_features.parquet"))
    parser.add_argument("--upload", action="store_true", help="send the table to Hugging Face")
    parser.add_argument("--repo", help="dataset repo, e.g. owner/name (default: $HF_DATASET_REPO)")
    args = parser.parse_args()

    repo = args.repo or os.environ.get("HF_DATASET_REPO")
    if not repo:
        sys.exit("Give the dataset repo with --repo owner/name (or set HF_DATASET_REPO).")
    store = HFStore(repo, token=os.environ.get("HF_TOKEN"))

    print("Evening PM windows from the hourly station files:")
    pollution = build_pollution(args.cpcb)

    print("Evening weather from the hourly ERA5 fields:")
    surface = pd.concat([store.read_df(f"raw/era5_surface_grid/{city.lower()}.parquet")
                         for city in config.TARGET_CITIES], ignore_index=True)
    patch = store.read_df("raw/era5_blh_patch/2024.parquet")
    weather = evening.era5_evening(surface, patch)
    print(f"  {weather['blh_evening'].notna().sum():,} city-days with an evening boundary layer")

    table = evening.build_table(pollution, weather, config.DATA_START_DATE, config.DATA_END_DATE)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"\n{len(table):,} city-days -> {out}")
    print(table[evening.FEATURES].notna().mean().round(3).rename("share present").to_string())

    if args.upload:
        store.upload_df(table, EVENING_PATH, "Evening features from the hourly CPCB and ERA5 data")
        print(f"↑ {EVENING_PATH}")


if __name__ == "__main__":
    main()

"""
Build the hourly tables and the noise measurements from the CPCB station files on this machine.

The station files live only here: CPCB drops connections from Colab's IP ranges, and
re-hosting a government dataset is a licensing question of its own, so the raw files
never went to the dataset repo. What goes there is derived and city-level:

    processed/city_hourly.parquet       each city's stations combined by median, per hour
    processed/hourly_dynamics.parquet   the shape of each city-day
    processed/noise_samples.parquet     how much two halves of a network disagree, by AQI level
    processed/noise_by_level.csv        the same as a table

    python scripts/build_hourly_tables.py --repo owner/name --upload

Only 2019-2025 is read, so a 2026 file on disk can never reach these tables.
Signing in uses the token `hf auth login` stores; HF_TOKEN still works if set.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import daily_cpcb, hourly  # noqa: E402
from src.data.cpcb_loader import load_station_file, looks_like_empty_template, parse_filename  # noqa: E402
from src.evaluation import noise  # noqa: E402
from src.utils.hf_store import HFStore  # noqa: E402

SPLITS = 10
SAMPLES_PER_LEVEL = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpcb", default=str(PROJECT_ROOT / "data" / "raw" / "cpcb"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--upload", action="store_true", help="send the tables to Hugging Face")
    parser.add_argument("--repo", help="dataset repo, e.g. owner/name (default: $HF_DATASET_REPO)")
    args = parser.parse_args()

    repo = args.repo or os.environ.get("HF_DATASET_REPO")
    if not repo:
        sys.exit("Give the dataset repo with --repo owner/name (or set HF_DATASET_REPO).")
    store = HFStore(repo, token=os.environ.get("HF_TOKEN"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last_year = int(config.DATA_END_DATE[:4])

    tables, shapes, station_days = [], [], []
    for city in config.TARGET_CITIES:
        paths = sorted((Path(args.cpcb) / city.lower()).glob("*_1Hr.csv"))
        frames = [load_station_file(p) for p in paths
                  if parse_filename(p)['year'] <= last_year and not looks_like_empty_template(p)]
        table = hourly.city_hourly(frames, city)
        table = table[table['timestamp'] <= f"{config.DATA_END_DATE} 23:00"]
        tables.append(table)
        shapes.append(hourly.dynamics(table).assign(city=city))
        for frame in frames:
            day = daily_cpcb.station_daily(frame)
            day['city'] = city
            station_days.append(day)
        print(f"  {city:<11} {len(frames):>3} station-years, {len(table):,} hours", flush=True)

    city_hourly = pd.concat(tables, ignore_index=True)
    shape = pd.concat(shapes, ignore_index=True)
    stations = pd.concat(station_days, ignore_index=True)
    stations['date'] = pd.to_datetime(stations['date'])

    daily = store.read_df("processed/city_daily_features.parquet")[['city', 'date', 'AQI']]
    daily['date'] = pd.to_datetime(daily['date']).astype('datetime64[ns]')
    measured = []
    for seed in range(SPLITS):
        halves = noise.split_network(stations, seed=seed)
        halves['date'] = pd.to_datetime(halves['date']).astype('datetime64[ns]')
        measured.append(noise.day_to_day_noise(halves).merge(daily, on=['city', 'date'], how='inner'))
    measured = pd.concat(measured, ignore_index=True).dropna(subset=['AQI'])
    by_level = noise.noise_by_level(measured)
    print("\nNoise of a city's AQI by level (AQI points):")
    print(by_level.round(1).to_string())

    edges = np.asarray(noise.LEVELS, dtype=float)
    measured['level'] = edges[np.clip(np.searchsorted(edges, measured['AQI'], side='right') - 1, 0, len(edges) - 2)]
    samples = pd.concat([part.sample(min(len(part), SAMPLES_PER_LEVEL), random_state=0)[['level', 'noise']]
                         for _, part in measured.groupby('level')], ignore_index=True)

    city_hourly.to_parquet(out / "city_hourly.parquet", index=False)
    shape.to_parquet(out / "hourly_dynamics.parquet", index=False)
    samples.to_parquet(out / "noise_samples.parquet", index=False)
    by_level.reset_index().astype({'AQI': str}).to_csv(out / "noise_by_level.csv", index=False)
    print(f"\n{len(city_hourly):,} city-hours, {len(shape):,} city-days, {len(samples):,} noise samples -> {out}")

    if args.upload:
        store.upload_df(city_hourly, "processed/city_hourly.parquet", "Hourly city series for the final model")
        store.upload_df(shape, "processed/hourly_dynamics.parquet", "Shape of each city-day from the hourly data")
        store.upload_df(samples, "processed/noise_samples.parquet", "Split-network noise samples by AQI level")
        store.upload_df(by_level.reset_index().astype({'AQI': str}), "processed/noise_by_level.csv", "Split-network noise by AQI level")
        print("↑ processed/city_hourly.parquet, hourly_dynamics.parquet, noise_samples.parquet, noise_by_level.csv")


if __name__ == "__main__":
    main()

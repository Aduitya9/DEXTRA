"""
The strongest simple rival: gradient boosting on the daily table, the evening and the hourly shape.

Leave-one-city-out on the same folds as the neural models: fitted on the other fourteen
cities' 2019-2023, scored on the held-out city's 2024-2025. It runs on this machine
rather than in the notebook because boosting wants CPU cores, which a Colab GPU session
has two of. Reads processed/hourly_dynamics.parquet, from build_hourly_tables.py.

    python scripts/gbm_hourly_baseline.py --repo owner/name --upload
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import assemble, evening  # noqa: E402
from src.models import baselines, windows  # noqa: E402
from src.utils.hf_store import HFStore  # noqa: E402

KEYS = ['city', 'date', 'horizon']
COLUMN = 'gbm hourly'


def by_day(frame):
    frame['date'] = pd.to_datetime(frame['date']).astype('datetime64[ns]')
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed" / "gbm_hourly_loco_forecasts.parquet"))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo", help="dataset repo, e.g. owner/name (default: $HF_DATASET_REPO)")
    args = parser.parse_args()
    repo = args.repo or os.environ.get("HF_DATASET_REPO")
    if not repo:
        sys.exit("Give the dataset repo with --repo owner/name (or set HF_DATASET_REPO).")
    store = HFStore(repo, token=os.environ.get("HF_TOKEN"))

    daily = by_day(store.read_df("processed/city_daily_features.parquet")).merge(
        by_day(store.read_df("processed/evening_features.parquet")), on=['city', 'date'], how='left')
    shape = by_day(store.read_df("processed/hourly_dynamics.parquet"))
    shape_columns = [c for c in shape.columns if c.startswith('h_')]
    daily = daily.merge(shape[['city', 'date'] + shape_columns], on=['city', 'date'], how='left')
    samples = windows.usable(windows.make_supervised(
        daily, features=list(config.ALL_FEATURES) + evening.FEATURES + shape_columns)).drop(columns='split')
    samples['split'] = 'x'
    features = windows.feature_matrix(samples)
    samples = samples.drop(columns='split')

    frames = []
    for city in sorted(daily['city'].unique()):
        started = time.time()
        labels = assemble.leave_one_out_split(daily, city)
        by_target = daily[['city', 'date']].assign(split=labels).rename(columns={'date': 'target_date'})
        fold = samples.merge(by_target, on=['city', 'target_date'], how='left')
        booster = baselines.fit_gradient_boosting(fold[fold['split'] == 'train'], features)
        rows = fold[fold['split'].isin(['val', 'test_loco'])].copy()
        rows[COLUMN] = baselines.predict_gradient_boosting(booster, rows, features).to_numpy()
        frames.append(rows[KEYS + [COLUMN]].assign(fold=city))
        print(f"  {city:<11} {time.time() - started:.0f} s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"{len(out):,} rows -> {args.out}")
    if args.upload:
        store.upload_df(out, "processed/gbm_hourly_loco_forecasts.parquet", "Gradient boosting with the hourly shape, leave-one-city-out")
        print("↑ processed/gbm_hourly_loco_forecasts.parquet")


if __name__ == "__main__":
    main()

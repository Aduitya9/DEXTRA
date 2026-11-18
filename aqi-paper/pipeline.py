# %% [markdown]
# # SIGMA-Net — Physics-Guided AQI Forecasting for Indian Cities
#
# **Sub-Index Guided Multi-pollutant Attention Network**, with four comparison models,
# on the DEXTRA city-daily dataset (15 Indian cities, 2019-2025).
#
# One notebook. Fetches data from HuggingFace, cleans it, runs EDA, trains and tunes
# every model, cross-validates, tests, and writes every table and figure the paper needs.
#
# **It is safe to interrupt at any point.** Re-run the notebook top to bottom and it resumes
# from the last completed unit of work. Nothing is recomputed twice.
#
# | | |
# |---|---|
# | Data | `Aduitya9/Aduitya_dextra-aqi-india` (public, no token needed) |
# | Hardware | one NVIDIA L4 (Colab Pro). Runs on CPU too, slowly |
# | Task | forecast AQI at t+1, t+2, t+3 from the last 14 days |
# | Models | persistence, climatology, LightGBM, PatchTST, iTransformer, TSMixer, **SIGMA-Net** |
#
# ---
# ### Run this first: pick a profile in the next cell
#
# | `PROFILE` | Time | What it is for |
# |---|---|---|
# | `"ci"`    | ~2 min   | No tuning, one epoch, three cities. Proves the code runs even with no GPU. |
# | `"smoke"` | ~3-5 min | Every code path, on a reduced problem. **Run this first.** Numbers are meaningless. |
# | `"fast"`  | ~45 min  | Real numbers, reduced trials and folds. Good for iterating. |
# | `"full"`  | ~5 h     | The numbers that go in the paper. |
#
# Each profile writes to its **own** directory, so running `full` never overwrites or
# reuses anything from `smoke`.

# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — the only cell you normally edit
# ═══════════════════════════════════════════════════════════════════════════════

PROFILE = "smoke"          # "smoke" | "fast" | "full"   <-- CHANGE ME
SEED = 42

# --- forecasting task -------------------------------------------------------
LOOKBACK = 14              # L: days of history fed to the model
HORIZONS = (1, 2, 3)       # days ahead to predict, all from one forward pass
EMBARGO_DAYS = 20          # purged from the end of every training block (>= L + max(H))
HAZARD_THRESHOLD = 200.0   # CPCB "Poor" boundary; defines a hazardous day

# --- honesty switches -------------------------------------------------------
PERFECT_MET_ORACLE = False # True = let the model see real future weather (upper bound, NOT a forecast)
USE_INDUSTRIAL_INDEX = True

# --- where state lives ------------------------------------------------------
DRIVE_SUBDIR = "SIGMANet_AQI"   # -> /content/drive/MyDrive/SIGMANet_AQI
LOCAL_FALLBACK = "./sigmanet_run"

# --- profile-dependent budgets ---------------------------------------------
PROFILES = {
    # "ci" skips tuning entirely and trains one epoch on three cities. It exists to
    # prove the code runs on a machine with no GPU at all, in a couple of minutes.
    "ci":    dict(n_trials_deep=0,  n_trials_lgbm=0,  max_epochs=1,   patience=1,
                  cv_folds=1, n_seeds=1, do_loco=True, loco_cities=1, do_ablations=True,
                  subsample_cities=3, batch_size=128),
    "smoke": dict(n_trials_deep=2,  n_trials_lgbm=2,  max_epochs=3,   patience=2,
                  cv_folds=2, n_seeds=1, do_loco=True, loco_cities=2, do_ablations=True,
                  subsample_cities=6, batch_size=256),
    "fast":  dict(n_trials_deep=10, n_trials_lgbm=15, max_epochs=40,  patience=8,
                  cv_folds=3, n_seeds=1, do_loco=True, loco_cities=4, do_ablations=True,
                  subsample_cities=0, batch_size=256),
    "full":  dict(n_trials_deep=30, n_trials_lgbm=40, max_epochs=120, patience=15,
                  cv_folds=5, n_seeds=3, do_loco=True, loco_cities=0, do_ablations=True,
                  subsample_cities=0, batch_size=256),
}
assert PROFILE in PROFILES, f"PROFILE must be one of {list(PROFILES)}"
CFG = dict(PROFILES[PROFILE])
CFG.update(profile=PROFILE, seed=SEED, lookback=LOOKBACK, horizons=list(HORIZONS),
           embargo=EMBARGO_DAYS, hazard=HAZARD_THRESHOLD,
           perfect_met=PERFECT_MET_ORACLE, use_industrial=USE_INDUSTRIAL_INDEX)

H_MAX = max(HORIZONS)
N_H = len(HORIZONS)
print(f"PROFILE = {PROFILE}")
for k, v in CFG.items():
    print(f"  {k:18s} {v}")

# %% [markdown]
# ## 1 · Environment, dependencies, and the resumable run directory

# %%
# ── dependency check (works in Colab and on a plain machine; no shell magics) ──
import importlib, subprocess, sys, os


def _ensure(pairs):
    """pairs: list of (import_name, pip_spec). Installs only what is missing."""
    missing = []
    for mod, spec in pairs:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(spec)
    if missing:
        print("installing:", " ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    else:
        print("all dependencies already present")


_ensure([
    ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
    ("torch", "torch"), ("lightgbm", "lightgbm"), ("optuna", "optuna"),
    ("requests", "requests"),
])

# %%
import json, math, random, shutil, time, warnings, hashlib, gc, traceback
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
import optuna
from sklearn.metrics import (r2_score, average_precision_score, cohen_kappa_score,
                             f1_score, precision_score, recall_score)
from scipy import stats

try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")     # headless; in Colab keep the inline backend so plots show
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_CAPABLE = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

print(f"python      {sys.version.split()[0]}")
print(f"numpy       {np.__version__}")
print(f"pandas      {pd.__version__}")
print(f"torch       {torch.__version__}")
print(f"lightgbm    {lgb.__version__}")
print(f"optuna      {optuna.__version__}")
print(f"in colab    {IN_COLAB}")
print(f"device      {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"bf16 capable {BF16_CAPABLE}  (mixed precision is deliberately not used - see section 6)")

# %%
# ── Google Drive, so that state survives a disconnect ─────────────────────────
if IN_COLAB:
    try:
        _colab_drive.mount("/content/drive")
        RUN_DIR = Path("/content/drive/MyDrive") / DRIVE_SUBDIR
    except Exception as e:
        print(f"!! Drive mount failed ({e}). Falling back to local storage — "
              f"state will NOT survive a disconnect.")
        RUN_DIR = Path(LOCAL_FALLBACK)
else:
    RUN_DIR = Path(LOCAL_FALLBACK)

RUN_DIR = RUN_DIR / PROFILE
DIRS = {k: RUN_DIR / k for k in
        ("data", "results", "ckpt", "studies", "figures", "tables", "logs")}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

print(f"run directory: {RUN_DIR}")
print(f"  (everything below is resumable; delete this folder to start clean)")


# %%
# ── the ledger: one file per completed unit of work, written atomically ───────
def _atomic_write_text(path: Path, text: str):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _safe_key(key: str) -> str:
    """Filesystem-safe name for a work-unit key."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in key)
    return out[:180] if len(out) <= 180 else out[:150] + "_" + hashlib.md5(
        key.encode()).hexdigest()[:12]


def result_path(key: str) -> Path:
    return DIRS["results"] / (_safe_key(key) + ".json")


def is_done(key: str) -> bool:
    p = result_path(key)
    if not p.exists():
        return False
    try:
        json.loads(p.read_text(encoding="utf-8"))
        return True
    except Exception:
        print(f"  (corrupt result for {key}; will recompute)")
        p.unlink(missing_ok=True)
        return False


def save_result(key: str, payload: dict):
    payload = dict(payload)
    payload["_key"] = key
    payload["_saved_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_text(result_path(key), json.dumps(payload, indent=1, default=_jsonable))


def load_result(key: str) -> dict:
    return json.loads(result_path(key).read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_unit(key: str, fn, force: bool = False):
    """Run fn() once, ever. Returns the stored result on later calls."""
    if not force and is_done(key):
        print(f"  [skip] {key}")
        return load_result(key)
    t0 = time.time()
    print(f"  [run ] {key}", flush=True)
    out = fn()
    out = dict(out)
    out["_seconds"] = round(time.time() - t0, 2)
    save_result(key, out)
    print(f"  [done] {key}  ({out['_seconds']}s)")
    return out


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
print(f"results so far: {len(list(DIRS['results'].glob('*.json')))} completed units")

# %% [markdown]
# ## 2 · Data: download, validate, and establish the facts
#
# The parquet is fetched once into the run directory. Everything after this is offline.

# %%
HF_REPO = "Aduitya9/Aduitya_dextra-aqi-india"
HF_FILE = "processed/city_daily_features.parquet"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"
RAW_PARQUET = DIRS["data"] / "city_daily_features.parquet"


def fetch_dataset() -> Path:
    if RAW_PARQUET.exists() and RAW_PARQUET.stat().st_size > 1_000_000:
        print(f"  already downloaded: {RAW_PARQUET} "
              f"({RAW_PARQUET.stat().st_size/1e6:.1f} MB)")
        return RAW_PARQUET
    print(f"  downloading {HF_URL}")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
        shutil.copy(p, RAW_PARQUET)
    except Exception as e:
        print(f"  huggingface_hub route failed ({type(e).__name__}); using direct HTTP")
        import requests
        with requests.get(HF_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = Path(str(RAW_PARQUET) + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, RAW_PARQUET)
    print(f"  saved {RAW_PARQUET} ({RAW_PARQUET.stat().st_size/1e6:.1f} MB)")
    return RAW_PARQUET


fetch_dataset()
raw = pd.read_parquet(RAW_PARQUET)
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values(["city", "date"]).reset_index(drop=True)
print(f"\nloaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")
print(f"cities: {raw['city'].nunique()}   dates: {raw['date'].min().date()} -> {raw['date'].max().date()}")

# %%
# ── structural validation: fail loudly now rather than silently later ────────
POLLUTANTS_ALL = ["PM2.5", "PM10", "NO2", "SO2", "NH3", "CO", "O3"]
# NH3 is excluded from the sub-index maximum: its highest observed value anywhere is
# ~208 ug/m3, which maps to sub-index ~52, so it can never be the maximum. Keeping it
# in the physics head would feed it a channel that is 17.7% imputed for no benefit.
POLLUTANTS_AQI = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
MET_COLS = ["blh", "blh_min", "blh_anomaly", "temp2m", "wind_speed", "wind_dir",
            "precip", "inv_idx"]
FIRE_COLS = ["fire_count_100km", "fire_count_500km", "frp_500km"]
QUAL_COLS = ["n_stations", "n_stations_pm25", "hours"]
SPLITS = ["train", "val", "test_seen", "test_unseen", "support_unseen", "zeroshot_unseen"]


def validate_raw(df: pd.DataFrame):
    problems = []
    need = ["city", "date", "AQI", "split", "archetype_id", "coastal_flag",
            "industrial_index", "inversion_level_hpa"] + POLLUTANTS_ALL + MET_COLS + FIRE_COLS + QUAL_COLS
    for c in need:
        if c not in df.columns:
            problems.append(f"missing column {c!r}")
    if problems:
        raise RuntimeError("dataset does not match expectations:\n  " + "\n  ".join(problems))

    # one row per city-day, no gaps
    for city, sub in df.groupby("city", sort=True):
        d = sub["date"]
        if d.duplicated().any():
            problems.append(f"{city}: duplicate dates")
        expected = pd.date_range(d.min(), d.max(), freq="D")
        if len(d) != len(expected):
            problems.append(f"{city}: {len(expected)-len(d)} missing calendar days")
    unknown = set(df["split"].unique()) - set(SPLITS)
    if unknown:
        problems.append(f"unexpected split labels: {unknown}")
    if problems:
        raise RuntimeError("dataset validation failed:\n  " + "\n  ".join(problems))
    print("  structural validation passed: one row per city-day, no gaps, no duplicates")
    print(f"  AQI missing: {df['AQI'].isna().sum()} rows "
          f"({df['AQI'].isna().mean()*100:.2f}%) -- dropped as targets, kept in windows with a mask")
    dup = (df["n_stations"] - df["station_density"]).abs().max() if "station_density" in df else None
    if dup is not None and dup == 0:
        print("  note: 'station_density' is identical to 'n_stations'; the duplicate is dropped")


validate_raw(raw)

# %%
# ── the CPCB National AQI operator, in numpy, and its fidelity on this data ───
# Concentration breakpoints (contiguous form) and the index scale they map onto.
# PM2.5/PM10/NO2/SO2/NH3/O3 in ug/m3, CO in mg/m3.
CPCB_BREAKS = {
    "PM2.5": [0, 30, 60, 90, 120, 250, 380, 500],
    "PM10":  [0, 50, 100, 250, 350, 430, 510, 590],
    "NO2":   [0, 40, 80, 180, 280, 400, 520, 640],
    "SO2":   [0, 40, 80, 380, 800, 1600, 2400, 3200],
    "NH3":   [0, 200, 400, 800, 1200, 1800, 2400, 3000],
    "CO":    [0, 1, 2, 10, 17, 34, 51, 68],
    "O3":    [0, 50, 100, 168, 208, 748, 1288, 1828],
}
CPCB_INDEX = [0, 50, 100, 200, 300, 400, 500, 500]
AQI_CAP = 500.0
CATEGORY_EDGES = [50, 100, 200, 300, 400]      # -> 6 ordinal classes 0..5
CATEGORY_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def sub_index_np(pollutant: str, conc):
    """Exact CPCB sub-index as a sum of clamped ramps. Vectorised, NaN-preserving."""
    c = np.asarray(conc, dtype=np.float64)
    edges = np.asarray(CPCB_BREAKS[pollutant], dtype=np.float64)
    idx = np.asarray(CPCB_INDEX, dtype=np.float64)
    out = np.full(c.shape, idx[0], dtype=np.float64)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        slope = idx[k + 1] - idx[k]
        if hi <= lo or slope == 0:
            continue
        out = out + slope * np.clip((c - lo) / (hi - lo), 0.0, 1.0)
    return np.where(np.isfinite(c), np.minimum(out, AQI_CAP), np.nan)


def naqi_np(df: pd.DataFrame, pollutants=POLLUTANTS_AQI):
    """CPCB NAQI: max sub-index, needing >=3 pollutants of which one is PM2.5 or PM10."""
    S = np.column_stack([sub_index_np(p, df[p].to_numpy()) for p in pollutants])
    n_valid = np.isfinite(S).sum(axis=1)
    pm_ok = np.isfinite(S[:, [pollutants.index("PM2.5"), pollutants.index("PM10")]]).any(axis=1)
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
    mx = np.where(np.isneginf(mx), np.nan, mx)
    aqi = np.where((n_valid >= 3) & pm_ok, np.minimum(mx, AQI_CAP), np.nan)
    driver = np.full(len(df), -1, dtype=np.int64)
    ok = np.isfinite(aqi)
    if ok.any():
        driver[ok] = np.nanargmax(np.where(np.isfinite(S[ok]), S[ok], -np.inf), axis=1)
    return aqi, driver, S


def aqi_to_category(aqi):
    return np.digitize(np.asarray(aqi, dtype=float), CATEGORY_EDGES, right=True)


_recon, _driver, _S = naqi_np(raw)
_m = np.isfinite(_recon) & raw["AQI"].notna().to_numpy()
_err = np.abs(_recon[_m] - raw["AQI"].to_numpy()[_m])
DSI_FIDELITY = dict(n=int(_m.sum()), mae=float(_err.mean()), median=float(np.median(_err)),
                    p99=float(np.quantile(_err, 0.99)), max=float(_err.max()),
                    within_2=float((_err < 2).mean()), within_5=float((_err < 5).mean()))
print("CPCB operator vs the dataset's own AQI column")
print(f"  compared on        {DSI_FIDELITY['n']:,} city-days")
print(f"  MAE                {DSI_FIDELITY['mae']:.3f} AQI points")
print(f"  median abs error   {DSI_FIDELITY['median']:.3f}")
print(f"  within +/-2        {DSI_FIDELITY['within_2']*100:.2f}%")
print(f"  within +/-5        {DSI_FIDELITY['within_5']*100:.2f}%")
print(f"  worst              {DSI_FIDELITY['max']:.1f}")
print("\n  The residual gap is expected and explainable: the released AQI is the median of")
print("  per-station AQI, whereas this operator takes the AQI of the median concentration.")
print("  SIGMA-Net's learned residual term exists to absorb exactly this.")
save_result("dsi_fidelity", DSI_FIDELITY)

# %% [markdown]
# ## 3 - Features: strictly causal, with train-only statistics
#
# Every per-timestep channel comes from a day at or before the window's last day `t`.
# The only future-dated inputs are calendar variables at the target date, which are
# genuinely knowable in advance. Meteorology is ERA5 *reanalysis* (observed weather), so
# letting the model see it after `t` would be using the future; `PERFECT_MET_ORACLE`
# exists only to quantify that upper bound and is off by default.

# %%
CITIES_ALL = sorted(raw["city"].unique().tolist())
TRAIN_CITIES = sorted(raw.loc[raw["split"] == "train", "city"].unique().tolist())
UNSEEN_CITIES = sorted(raw.loc[raw["split"] == "test_unseen", "city"].unique().tolist())

if CFG["subsample_cities"]:
    # Always keep at least one training city and one held-out city, so that no split
    # can come out empty however small the subsample is.
    keep_tr = [c for c in ["Delhi", "Lucknow", "Kolkata", "Chennai"] if c in TRAIN_CITIES]
    keep_un = [c for c in ["Patna", "Guwahati"] if c in UNSEEN_CITIES]
    n = int(CFG["subsample_cities"])
    n_un = max(1, n // 3)
    n_tr = max(1, n - n_un)
    CITIES = sorted(keep_tr[:n_tr] + keep_un[:n_un])
    print(f"!! {PROFILE} profile: restricted to {len(CITIES)} cities {CITIES}")
else:
    CITIES = CITIES_ALL

raw = raw[raw["city"].isin(CITIES)].sort_values(["city", "date"]).reset_index(drop=True)
TRAIN_CITIES = [c for c in TRAIN_CITIES if c in CITIES]
UNSEEN_CITIES = [c for c in UNSEEN_CITIES if c in CITIES]
CITY_IDX = {c: i for i, c in enumerate(CITIES)}
print(f"cities in use      {len(CITIES)}")
print(f"  training cities  {TRAIN_CITIES}")
print(f"  held-out cities  {UNSEEN_CITIES}")

TRAIN_MASK_ROWS = (raw["split"] == "train").to_numpy()
TRAIN_END = raw.loc[raw["split"] == "train", "date"].max()
VAL_START = raw.loc[raw["split"] == "val", "date"].min()
print(f"train ends         {TRAIN_END.date()}   val starts {VAL_START.date()}")


# %%
def build_features(df):
    """Returns (feature frame, channel names, fitted stats). Train-only statistics."""
    df = df.copy()
    tr = df["split"].to_numpy() == "train"

    # ---- masks FIRST, from the untouched NaN pattern -----------------------
    masks = {}
    for p in POLLUTANTS_ALL:
        masks[f"m_{p}"] = df[p].notna().to_numpy(np.float32)
    masks["m_AQI"] = df["AQI"].notna().to_numpy(np.float32)
    masks["m_blh_anomaly"] = df["blh_anomaly"].notna().to_numpy(np.float32)

    # ---- causal fill: forward only, within a city, limited to 3 days -------
    fill_cols = POLLUTANTS_ALL + ["AQI", "blh_anomaly"]
    for c in fill_cols:
        df[c] = df.groupby("city", sort=False)[c].ffill(limit=3)

    # ---- remaining gaps: the TRAIN-split global median. Global rather than
    #      per-city, so a city with no training history is handled identically.
    med = {}
    for c in fill_cols:
        v = df.loc[tr, c]
        m = float(v.median()) if v.notna().any() else 0.0
        med[c] = 0.0 if not np.isfinite(m) else m
        df[c] = df[c].fillna(med[c])

    # ---- transforms --------------------------------------------------------
    out = pd.DataFrame(index=df.index)
    for p in POLLUTANTS_ALL:
        out[f"log_{p}"] = np.log1p(df[p].clip(lower=0)).to_numpy(np.float32)
    for k, v in masks.items():
        out[k] = v
    out["aqi_hist"] = (df["AQI"].to_numpy(np.float64) / 100.0).astype(np.float32)
    for c in ["blh", "blh_min", "blh_anomaly", "temp2m", "wind_speed", "precip", "inv_idx"]:
        out[c] = df[c].to_numpy(np.float32)
    wd = np.deg2rad(df["wind_dir"].to_numpy(np.float64))
    out["wd_sin"] = np.sin(wd).astype(np.float32)
    out["wd_cos"] = np.cos(wd).astype(np.float32)
    for c in FIRE_COLS:
        out[f"log_{c}"] = np.log1p(df[c].clip(lower=0)).to_numpy(np.float32)
    for c in QUAL_COLS:
        out[c] = df[c].to_numpy(np.float32)

    # ---- robust scaling, median/IQR fitted on train rows only -------------
    stats_ = {}
    no_scale = set(masks.keys())
    for c in out.columns:
        if c in no_scale:
            stats_[c] = (0.0, 1.0)
            continue
        v = out.loc[tr, c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            stats_[c] = (0.0, 1.0)
            continue
        c_med = float(np.median(v))
        iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
        scale = iqr / 1.349 if iqr > 1e-9 else float(v.std())
        if (not np.isfinite(scale)) or scale < 1e-9:
            scale = 1.0
        stats_[c] = (c_med, scale)
        out[c] = np.clip((out[c].to_numpy(np.float32) - c_med) / scale, -8.0, 8.0).astype(np.float32)

    return out, list(out.columns), dict(medians=med, scalers=stats_), df


FEAT, DYN_COLS, FIT_STATS, FILLED = build_features(raw)
C_DYN = len(DYN_COLS)
print(f"dynamic channels: {C_DYN}")
print("  " + ", ".join(DYN_COLS))

# channel groups the models address by index
VENT_CHANNELS = ["blh", "blh_min", "blh_anomaly", "wind_speed", "inv_idx", "precip"]
VENT_IDX = [DYN_COLS.index(c) for c in VENT_CHANNELS]
POL_LOG_IDX = [DYN_COLS.index(f"log_{p}") for p in POLLUTANTS_ALL]
POL_AQI_LOG_IDX = [DYN_COLS.index(f"log_{p}") for p in POLLUTANTS_AQI]
assert all(0 <= i < C_DYN for i in VENT_IDX + POL_LOG_IDX)

# standardisation of the AQI-pollutant log channels, needed to invert the physics head
POL_MU = np.array([FIT_STATS["scalers"][f"log_{p}"][0] for p in POLLUTANTS_AQI], dtype=np.float32)
POL_SD = np.array([FIT_STATS["scalers"][f"log_{p}"][1] for p in POLLUTANTS_AQI], dtype=np.float32)
POL_CAP = np.array([float(np.nanmax(raw[p].to_numpy()) * 3.0 + 10.0) for p in POLLUTANTS_AQI],
                   dtype=np.float32)

# Constants needed to read the AQI on day t back out of a scaled window. Every deep model
# is anchored on that value and predicts a *change* from it (see section 6).
AQI_CH = DYN_COLS.index("aqi_hist")
AQI_MED_S, AQI_SC_S = FIT_STATS["scalers"]["aqi_hist"]
print("\nphysics-head inversion constants (log1p space, train-fitted)")
for p, m, s, cp in zip(POLLUTANTS_AQI, POL_MU, POL_SD, POL_CAP):
    print(f"  {p:6s} mu={m:7.3f} sd={s:6.3f} cap={cp:9.1f}")

# %% [markdown]
# ## 4 - Windowing: one sample per (city, t), three horizons from one forward pass
#
# A sample is anchored at the last observed day `t`. Its window is `[t-L+1, t]` and its
# targets are the AQI on `t+1, t+2, t+3`. A sample is kept only if all three target dates
# fall in the **same** split, which removes every boundary-straddling case. Training
# samples are additionally held back by an embargo, so no training window can overlap a
# validation window.

# %%
CAL_PER_H = 8  # doy sin/cos, month sin/cos, dow sin/cos, is_diwali, is_stubble
MET_ORACLE_CHANNELS = ("blh", "blh_min", "blh_anomaly", "temp2m", "wind_speed",
                       "precip", "inv_idx", "wd_sin", "wd_cos")


def calendar_block(sub):
    doy = sub["day_of_year"].to_numpy(np.float64)
    mon = sub["month"].to_numpy(np.float64)
    dow = sub["day_of_week"].to_numpy(np.float64)
    return np.column_stack([
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * mon / 12.0), np.cos(2 * np.pi * mon / 12.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        sub["is_diwali"].to_numpy(np.float64),
        sub["is_stubble_season"].to_numpy(np.float64),
    ]).astype(np.float32)


def build_samples():
    X, Ycov, Yarch = [], [], []
    Yaqi, Yaqi_m, Ypol, Ypol_m = [], [], [], []
    meta_city, meta_t, meta_split, meta_persist = [], [], [], []

    inv_all = raw["inversion_level_hpa"].to_numpy(np.float64)
    inv_mu = float(np.median(inv_all[TRAIN_MASK_ROWS])) if TRAIN_MASK_ROWS.any() else 900.0
    inv_sd = float(np.std(inv_all[TRAIN_MASK_ROWS]))
    inv_sd = inv_sd if inv_sd > 1e-9 else 1.0
    oracle_idx = [DYN_COLS.index(c) for c in MET_ORACLE_CHANNELS if c in DYN_COLS]

    for city in CITIES:
        sel = (raw["city"] == city).to_numpy()
        sub = raw.loc[sel].reset_index(drop=True)
        arr = FEAT.loc[sel].to_numpy(np.float32)
        cal = calendar_block(sub)
        T = len(sub)

        # `raw` is untouched, so its AQI is NaN exactly where the value was never observed.
        # `FILLED` carries the causally forward-filled AQI, used only for persistence.
        aqi_target = sub["AQI"].to_numpy(np.float64)      # NaN where unobserved -> not a target
        aqi_persist = FILLED.loc[sel, "AQI"].to_numpy(np.float64)
        split_arr = sub["split"].to_numpy()
        date_arr = sub["date"].to_numpy()
        pol_obs = np.column_stack([sub[p].to_numpy(np.float64) for p in POLLUTANTS_AQI])
        pol_std = ((np.log1p(np.clip(pol_obs, 0, None)) - POL_MU) / POL_SD).astype(np.float32)
        pol_valid = np.isfinite(pol_obs)

        statics = np.array([
            float(sub["coastal_flag"].iloc[0]),
            float(np.log1p(sub["industrial_index"].iloc[0])) if CFG["use_industrial"] else 0.0,
            (float(sub["inversion_level_hpa"].iloc[0]) - inv_mu) / inv_sd,
        ], dtype=np.float32)
        arch = int(sub["archetype_id"].iloc[0])
        embargo_cut = TRAIN_END - pd.Timedelta(days=CFG["embargo"])

        for t in range(LOOKBACK - 1, T - H_MAX):
            tgt = [t + h for h in HORIZONS]
            sps = {split_arr[j] for j in tgt}
            if len(sps) != 1:
                continue                                   # straddles a split boundary
            sp = sps.pop()
            if sp in ("support_unseen", "zeroshot_unseen"):
                continue                                   # deliberately unused
            ys = aqi_target[tgt]
            ym = np.isfinite(ys).astype(np.float32)
            if ym.sum() == 0:
                continue                                   # no usable target
            if sp == "train" and pd.Timestamp(date_arr[tgt[-1]]) > embargo_cut:
                continue                                   # embargo

            w = arr[t - LOOKBACK + 1: t + 1]
            if CFG["perfect_met"]:
                w = w.copy()
                w[-1, oracle_idx] = arr[t + 1, oracle_idx]

            X.append(w)
            Ycov.append(np.concatenate([cal[tgt].reshape(-1), statics]))
            Yarch.append(arch)
            Yaqi.append(np.nan_to_num(ys, nan=0.0))
            Yaqi_m.append(ym)
            Ypol.append(np.nan_to_num(pol_std[tgt], nan=0.0))
            Ypol_m.append(pol_valid[tgt].astype(np.float32))
            meta_city.append(CITY_IDX[city])
            meta_t.append(date_arr[t])
            meta_split.append(sp)
            meta_persist.append(aqi_persist[t])

    return dict(
        X=np.asarray(X, dtype=np.float32),
        cov=np.asarray(Ycov, dtype=np.float32),
        arch=np.asarray(Yarch, dtype=np.int64),
        y=np.asarray(Yaqi, dtype=np.float32),
        ym=np.asarray(Yaqi_m, dtype=np.float32),
        ypol=np.asarray(Ypol, dtype=np.float32),
        ypolm=np.asarray(Ypol_m, dtype=np.float32),
        city=np.asarray(meta_city, dtype=np.int64),
        t=pd.to_datetime(np.asarray(meta_t)),
        split=np.asarray(meta_split),
        persist=np.asarray(meta_persist, dtype=np.float32),
    )


SAMPLES = build_samples()
N_COV = SAMPLES["cov"].shape[1]
N_ARCH = int(raw["archetype_id"].max()) + 1
print(f"samples: {SAMPLES['X'].shape[0]:,}   window {SAMPLES['X'].shape[1]}x{SAMPLES['X'].shape[2]}"
      f"   covariates {N_COV}   archetypes {N_ARCH}")
print(f"memory:  {SAMPLES['X'].nbytes/1e6:.1f} MB")
print("\nsamples per split:")
for sp in ["train", "val", "test_seen", "test_unseen"]:
    m = SAMPLES["split"] == sp
    if m.sum():
        print(f"  {sp:12s} {int(m.sum()):6,}  "
              f"{pd.Timestamp(SAMPLES['t'][m].min()).date()} -> "
              f"{pd.Timestamp(SAMPLES['t'][m].max()).date()}  "
              f"cities={len(set(SAMPLES['city'][m].tolist()))}")

IDX = {sp: np.where(SAMPLES["split"] == sp)[0] for sp in
       ["train", "val", "test_seen", "test_unseen"]}
for sp, ix in IDX.items():
    assert len(ix) > 0, f"split {sp} produced no samples - check the city subsample"


# %%
# ---- leakage self-tests. Assertions, not prints: the notebook stops if any fail. ----
def leakage_selftests():
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    tr_t = SAMPLES["t"][IDX["train"]]
    va_t = SAMPLES["t"][IDX["val"]]
    ts_t = SAMPLES["t"][IDX["test_seen"]]

    chk("train anchors precede val anchors", tr_t.max() < va_t.min(),
        f"{tr_t.max().date()} < {va_t.min().date()}")
    chk("val anchors precede test anchors", va_t.max() < ts_t.min(),
        f"{va_t.max().date()} < {ts_t.min().date()}")
    gap = (va_t.min() - tr_t.max()).days
    chk(f"train/val gap >= L+Hmax ({LOOKBACK + H_MAX}d)", gap >= LOOKBACK + H_MAX,
        f"actual {gap}d")
    chk("last training window ends before val period", tr_t.max() < VAL_START,
        f"{tr_t.max().date()} < {VAL_START.date()}")

    unseen_ids = {CITY_IDX[c] for c in UNSEEN_CITIES}
    fit_cities = (set(SAMPLES["city"][IDX["train"]].tolist())
                  | set(SAMPLES["city"][IDX["val"]].tolist()))
    chk("no held-out city in train or val", len(fit_cities & unseen_ids) == 0,
        f"overlap={sorted(CITIES[i] for i in (fit_cities & unseen_ids))}")
    chk("test_unseen is only held-out cities",
        set(SAMPLES["city"][IDX["test_unseen"]].tolist()) <= unseen_ids)
    chk("test_seen has no held-out city",
        len(set(SAMPLES["city"][IDX["test_seen"]].tolist()) & unseen_ids) == 0)

    chk("scaler rows are the train split only",
        bool(TRAIN_MASK_ROWS.sum()) and
        set(raw.loc[TRAIN_MASK_ROWS, "split"].unique()) == {"train"})
    chk("AQI_category excluded from inputs", "AQI_category" not in DYN_COLS)
    chk("station_density duplicate excluded", "station_density" not in DYN_COLS)
    chk("raw AQI excluded from inputs (only AQI/100 history)", "AQI" not in DYN_COLS)

    for k in ("X", "cov", "y", "ym", "ypol", "ypolm", "persist"):
        chk(f"{k} is finite", bool(np.isfinite(SAMPLES[k]).all()))

    chk("perfect-met oracle off" if not CFG["perfect_met"] else "oracle ON (declared)",
        True, "strictly causal" if not CFG["perfect_met"] else "upper bound, not a forecast")

    width = max(len(n) for n, _, _ in checks)
    print("leakage self-tests")
    bad = [n for n, ok, _ in checks if not ok]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    if bad:
        raise AssertionError("leakage self-tests failed: " + "; ".join(bad))
    print(f"  all {len(checks)} checks passed")
    return [dict(check=n, passed=o, detail=d) for n, o, d in checks]


LEAK_CHECKS = leakage_selftests()
save_result("leakage_selftests", dict(checks=LEAK_CHECKS, n=len(LEAK_CHECKS)))

# %% [markdown]
# ## 5 - Exploratory analysis
#
# This section produces Table I and Figures 1-2 of the paper. Everything is written to
# `tables/` and `figures/` as CSV and as 300-dpi PNG + PDF.

# %%
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(DIRS["figures"] / f"{name}.{ext}", bbox_inches="tight")
    print(f"  wrote figures/{name}.png + .pdf")
    if IN_COLAB:
        try:
            from IPython.display import display
            display(fig)          # also show it in the notebook, not only on disk
        except Exception:
            pass
    plt.close(fig)


def save_table(df, name, float_fmt="%.3f"):
    df.to_csv(DIRS["tables"] / f"{name}.csv", index=True, float_format=float_fmt)
    print(f"  wrote tables/{name}.csv  ({df.shape[0]}x{df.shape[1]})")
    return df


# %%
# ---- Table I: split summary -------------------------------------------------
obs = raw.dropna(subset=["AQI"])
rows = []
for sp in ["train", "val", "test_seen", "test_unseen", "support_unseen", "zeroshot_unseen"]:
    g = obs[obs["split"] == sp]
    if not len(g):
        continue
    rows.append(dict(
        split=sp, city_days=len(g), cities=g["city"].nunique(),
        start=str(g["date"].min().date()), end=str(g["date"].max().date()),
        AQI_mean=g["AQI"].mean(), AQI_median=g["AQI"].median(),
        pct_above_200=100 * (g["AQI"] > HAZARD_THRESHOLD).mean(),
        pct_above_300=100 * (g["AQI"] > 300).mean(),
        used_here=sp in ("train", "val", "test_seen", "test_unseen"),
    ))
TABLE_I = save_table(pd.DataFrame(rows).set_index("split"), "table1_splits")
print(TABLE_I.to_string())

# %%
# ---- per-city summary -------------------------------------------------------
rows = []
for c in CITIES:
    g = obs[obs["city"] == c]
    gt = g[g["split"].isin(["train", "val"])]
    g25 = g[g["split"].isin(["test_seen", "test_unseen"])]
    rows.append(dict(
        city=c, role="held-out" if c in UNSEEN_CITIES else "training",
        archetype=int(g["archetype_id"].iloc[0]),
        coastal=int(g["coastal_flag"].iloc[0]),
        n_obs=len(g), AQI_mean=g["AQI"].mean(), AQI_p95=g["AQI"].quantile(0.95),
        AQI_max=g["AQI"].max(),
        hazard_rate_pct=100 * (g["AQI"] > HAZARD_THRESHOLD).mean(),
        hazard_days_2025=int((g25["AQI"] > HAZARD_THRESHOLD).sum()) if len(g25) else 0,
        mean_stations=g["n_stations"].mean(),
    ))
TABLE_CITY = save_table(pd.DataFrame(rows).set_index("city").sort_values(
    ["role", "hazard_rate_pct"], ascending=[True, False]), "table_cities")
print(TABLE_CITY.to_string())
print("\nNote the tail: cities with very few hazardous days in 2025 cannot support a")
print("per-city episode metric. Headline episode numbers are pooled, never per city.")

# %%
# ---- missingness ------------------------------------------------------------
miss = (raw.groupby("city")[POLLUTANTS_ALL + ["AQI"]]
        .apply(lambda g: g.isna().mean() * 100).round(1))
save_table(miss, "table_missingness", float_fmt="%.1f")
print(miss.to_string())

runs = []
for p in POLLUTANTS_ALL + ["AQI"]:
    worst, wcity = 0, None
    for c, sub in raw.sort_values("date").groupby("city"):
        isna = sub[p].isna().to_numpy().astype(np.int8)
        if not isna.any():
            continue
        # longest run of True, vectorised
        d = np.diff(np.concatenate([[0], isna, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        mx = int((ends - starts).max()) if len(starts) else 0
        if mx > worst:
            worst, wcity = mx, c
    runs.append(dict(variable=p, longest_gap_days=worst, city=wcity))
TABLE_GAPS = save_table(pd.DataFrame(runs).set_index("variable"), "table_gaps", float_fmt="%.0f")
print("\nlongest consecutive-missing run per variable")
print(TABLE_GAPS.to_string())
print("\nThis is why every pollutant carries a missingness mask channel: a 900-day PM10")
print("gap cannot be imputed away, and the model must be told when a value is invented.")

# %%
# ---- Figure 2: AQI distribution and seasonality -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

ax = axes[0, 0]
order = TABLE_CITY.sort_values("AQI_mean", ascending=False).index.tolist()
data = [obs.loc[obs["city"] == c, "AQI"].to_numpy() for c in order]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=90, fontsize=7)
for patch, c in zip(bp["boxes"], order):
    patch.set_facecolor("#c0392b" if c in UNSEEN_CITIES else "#2c7fb8")
    patch.set_alpha(0.75)
ax.axhline(HAZARD_THRESHOLD, color="k", ls="--", lw=1)
ax.text(0.99, HAZARD_THRESHOLD + 8, "AQI 200 (hazardous)", ha="right", fontsize=7,
        transform=ax.get_yaxis_transform())
ax.set_ylabel("AQI")
ax.set_title("AQI by city (red = held out from training)", fontsize=9)

ax = axes[0, 1]
piv = (obs.assign(mon=obs["date"].dt.month)
          .pivot_table(index="city", columns="mon", values="AQI", aggfunc="mean")
          .reindex(order))
im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(12))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=7)
ax.set_yticks(range(len(piv)))
ax.set_yticklabels(piv.index, fontsize=7)
ax.set_title("Mean AQI by month", fontsize=9)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04)

ax = axes[1, 0]
for sp, col in [("train", "#2c7fb8"), ("test_seen", "#31a354"), ("test_unseen", "#c0392b")]:
    v = obs.loc[obs["split"] == sp, "AQI"]
    if len(v):
        ax.hist(v, bins=60, range=(0, 500), density=True, histtype="step", lw=1.6,
                label=f"{sp} (n={len(v):,})", color=col)
ax.axvline(HAZARD_THRESHOLD, color="k", ls="--", lw=1)
ax.set_xlabel("AQI"); ax.set_ylabel("density")
ax.set_title("Target distribution by split", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
lags = np.arange(1, 31)
for c in (["Delhi", "Patna"] if "Delhi" in CITIES else CITIES[:2]):
    s = raw.loc[raw["city"] == c].set_index("date")["AQI"].astype(float)
    ac = [s.autocorr(lag=int(k)) for k in lags]
    ax.plot(lags, ac, marker="o", ms=2.5, lw=1.2, label=c)
for h in HORIZONS:
    ax.axvline(h, color="grey", ls=":", lw=0.8)
ax.set_xlabel("lag (days)"); ax.set_ylabel("autocorrelation")
ax.set_title("AQI autocorrelation (dotted = our horizons)", fontsize=9)
ax.legend(fontsize=7)

save_fig(fig, "fig2_eda")

# %%
# ---- Figure: which pollutant drives the AQI ---------------------------------
drv_names = np.array(POLLUTANTS_AQI + ["none"])
# recomputed on the (possibly city-subsampled) frame so the index cannot drift
_, _driver_now, _ = naqi_np(raw)
_dr = _driver_now.copy()
_dr[_dr < 0] = len(POLLUTANTS_AQI)
raw_drv = pd.DataFrame(dict(city=raw["city"].to_numpy(),
                            month=raw["date"].dt.month.to_numpy(),
                            driver=drv_names[_dr]))
raw_drv = raw_drv[raw_drv["driver"] != "none"]
share = (raw_drv.pivot_table(index="city", columns="driver", aggfunc="size", fill_value=0)
         .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100).reindex(order))
save_table(share, "table_aqi_driver_share", float_fmt="%.1f")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
bottom = np.zeros(len(share))
for p in [c for c in POLLUTANTS_AQI if c in share.columns]:
    ax.bar(share.index, share[p].to_numpy(), bottom=bottom, label=p, width=0.78)
    bottom += share[p].to_numpy()
ax.set_ylabel("% of city-days")
ax.set_title("Which pollutant attains the maximum CPCB sub-index", fontsize=9)
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.45))
ax.set_ylim(0, 100)
save_fig(fig, "fig_driver_share")
print("\nThis figure is the motivation for the physics-guided head: the AQI is driven by")
print("different pollutants in different cities, and a direct AQI regressor never learns that.")

# %% [markdown]
# ## 6 - Models
#
# All four deep comparison models are implemented from scratch in plain PyTorch. That is a
# deliberate choice: no `mamba-ssm`, no time-series framework, no CUDA extension that can
# fail to compile on a Colab runtime. The only non-stock dependencies in the whole notebook
# are LightGBM and Optuna.
#
# | Model | Idea | Reference |
# |---|---|---|
# | PatchTST | patch the series, attend over patches, one channel at a time | Nie et al., ICLR 2023 |
# | iTransformer | invert the problem: each *variate* is a token, attend across variates | Liu et al., ICLR 2024 |
# | TSMixer | drop attention, mix alternately along time and feature axes | Chen et al., TMLR 2023 |
# | **SIGMA-Net** | **patch + variate attention, ventilation-gated, with a differentiable CPCB head** | **ours** |
#
# ### One design decision applied to all four deep models
#
# Day-to-day AQI is strongly autocorrelated, which makes **persistence** (today's value,
# carried forward) a hard baseline to beat. So every deep model here is **anchored**: it
# receives today's AQI and predicts the *change* from it, and the final layer of every head
# is zero-initialised, so each model *starts* exactly at persistence and has to earn any
# departure from it. SIGMA-Net is anchored one level deeper - its head predicts a change in
# each pollutant *concentration*, so at initialisation it emits today's concentrations
# through the exact CPCB formula.
#
# This is deliberately the harder setting for our own model, not an easier one: it hands
# every baseline a strong starting point. It is also what PatchTST's instance normalisation
# is for, so leaving it out would have made that baseline weaker than its paper.

# %%
from contextlib import nullcontext

# Mixed precision is deliberately OFF. The physics head works in physical units
# (concentrations up to ~700 ug/m3, sub-indices up to 500) and bf16 carries about three
# decimal digits, which would inject one to two AQI points of noise into the very
# quantity being measured. These models are small; fp32 on an L4 costs nothing.
AMP_ENABLED = False

# ---- stage every tensor on the device once; the dataset is small enough -----
T_X = torch.as_tensor(SAMPLES["X"]).to(DEVICE)
T_COV = torch.as_tensor(SAMPLES["cov"]).to(DEVICE)
T_ARCH = torch.as_tensor(SAMPLES["arch"]).to(DEVICE)
T_Y = torch.as_tensor(SAMPLES["y"]).to(DEVICE)
T_YM = torch.as_tensor(SAMPLES["ym"]).to(DEVICE)
T_YPOL = torch.as_tensor(SAMPLES["ypol"]).to(DEVICE)
T_YPOLM = torch.as_tensor(SAMPLES["ypolm"]).to(DEVICE)
N_POL_AQI = len(POLLUTANTS_AQI)
print(f"staged {T_X.numel()*4/1e6:.1f} MB of windows on {DEVICE}")


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  C1 - the differentiable CPCB sub-index head
# ═══════════════════════════════════════════════════════════════════════════════
class CPCBSubIndex(nn.Module):
    """The exact Indian NAQI operator, as a differentiable layer.

    A monotone piecewise-linear function is exactly a sum of clamped ramps:

        I(c) = I_0 + sum_k (I_{k+1} - I_k) * clip((c - c_k) / (c_{k+1} - c_k), 0, 1)

    so no gather, no branching, and a well-defined gradient almost everywhere. The
    maximum across pollutants is replaced by a temperature-controlled smooth maximum,
    tau * logsumexp(I / tau), whose temperature is learned. At small tau this recovers
    the true maximum; the overshoot is bounded above by tau * ln(P).
    """

    def __init__(self, pollutants, tau_init=2.0):
        super().__init__()
        self.pollutants = list(pollutants)
        P, K = len(pollutants), len(CPCB_INDEX) - 1
        c_lo = torch.zeros(P, K)
        c_hi = torch.zeros(P, K)
        slope = torch.zeros(P, K)
        for i, p in enumerate(pollutants):
            e = CPCB_BREAKS[p]
            for k in range(K):
                c_lo[i, k] = float(e[k])
                c_hi[i, k] = float(e[k + 1])
                slope[i, k] = float(CPCB_INDEX[k + 1] - CPCB_INDEX[k])
        self.register_buffer("c_lo", c_lo)
        self.register_buffer("c_hi", c_hi)
        self.register_buffer("slope", slope)
        self.i0 = float(CPCB_INDEX[0])
        # tau = softplus(raw) + 0.5, initialised at tau_init
        raw0 = math.log(math.expm1(max(tau_init - 0.5, 1e-3)))
        self.raw_tau = nn.Parameter(torch.tensor(float(raw0)))

    @property
    def tau(self):
        return F.softplus(self.raw_tau) + 0.5

    def forward(self, conc):
        """conc: (B, H, P) in physical units -> (aqi (B,H), sub_index (B,H,P))."""
        c = conc.unsqueeze(-1)                                  # B,H,P,1
        denom = (self.c_hi - self.c_lo).clamp(min=1e-6)
        ramp = ((c - self.c_lo) / denom).clamp(0.0, 1.0)         # B,H,P,K
        sub = (self.i0 + (ramp * self.slope).sum(-1)).clamp(max=AQI_CAP)
        tau = self.tau
        aqi = tau * torch.logsumexp(sub / tau, dim=-1)
        return aqi, sub


class PollutantDecoder(nn.Module):
    """Maps a standardised log1p prediction back to a bounded physical concentration."""

    def __init__(self, mu, sd, cap):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.register_buffer("log_cap", torch.log1p(torch.as_tensor(cap)))

    def forward(self, z):
        log_c = (z * self.sd + self.mu).clamp(min=0.0)
        log_c = torch.minimum(log_c, self.log_cap)
        return torch.expm1(log_c)


# ═══════════════════════════════════════════════════════════════════════════════
#  C2 - ventilation-gated meteorological modulation
# ═══════════════════════════════════════════════════════════════════════════════
class VentilationFiLM(nn.Module):
    """Box-model physics says concentration ~ emissions / (BLH * wind). A quotient
    cannot be expressed by concatenating boundary-layer height as one more input
    channel, so meteorology instead produces a multiplicative scale and an additive
    shift for the pollutant representation. Zero-initialised, so the module starts as
    the identity and the network only departs from it if the data pay for it."""

    def __init__(self, n_vent, d, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vent, max(16, d // 2)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max(16, d // 2), 2 * d),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, vent):
        gamma, beta = self.net(vent).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  shared pieces
# ═══════════════════════════════════════════════════════════════════════════════
def anchor_from_window(x):
    """The observed AQI on day t, recovered from the scaled window. (B,) in AQI units."""
    return (x[:, -1, AQI_CH] * AQI_SC_S + AQI_MED_S) * 100.0


def zero_init_last(module):
    """Zero the final Linear of a head, so a model starts exactly at persistence and has
    to learn its way away from it. Applied to every deep model, so no model is favoured."""
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


_a = anchor_from_window(T_X).cpu().numpy()
_d = np.abs(_a - SAMPLES["persist"]).max()
assert _d < 0.05, (f"the anchor recovered from the scaled window differs from the observed "
                   f"AQI by up to {_d:.4f} - the scaling is not invertible, so delta "
                   f"learning would be built on a wrong baseline")
print(f"anchor self-test: window-recovered AQI(t) matches the observed value "
      f"to {_d:.2e} AQI points")
del _a, _d


class CovEncoder(nn.Module):
    """Known-future calendar covariates + city statics + an archetype embedding."""

    def __init__(self, n_cov, n_arch, d_arch, d_out, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_arch, d_arch)
        self.net = nn.Sequential(nn.Linear(n_cov + d_arch, d_out), nn.GELU(), nn.Dropout(dropout))
        self.d_out = d_out

    def forward(self, cov, arch):
        return self.net(torch.cat([cov, self.emb(arch)], dim=-1))


def encoder_stack(d_model, n_heads, d_ff, dropout, n_layers):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


def n_patches(L, patch_len, stride):
    assert patch_len <= L and stride >= 1
    return (L - patch_len) // stride + 1


class PatchEmbed(nn.Module):
    """Channel-independent patching: (B,L,C) -> (B, C, n_patch, d)."""

    def __init__(self, L, patch_len, stride, d_model):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patch = n_patches(L, patch_len, stride)
        self.proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        z = x.permute(0, 2, 1)                                      # B,C,L
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(z)                                         # B,C,n_patch,d


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  M4  PatchTST
# ═══════════════════════════════════════════════════════════════════════════════
class PatchTST(nn.Module):
    def __init__(self, C, L, H, n_cov, n_arch, d_model=64, n_heads=4, n_layers=2,
                 patch_len=4, stride=2, d_ff=128, dropout=0.1, d_head=16, d_arch=8, **_):
        super().__init__()
        self.patch = PatchEmbed(L, patch_len, stride, d_model)
        self.enc = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers)
        self.flat = nn.Linear(self.patch.n_patch * d_model, d_head)
        self.cov = CovEncoder(n_cov, n_arch, d_arch, d_ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(C * d_head + d_ff, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, H))
        zero_init_last(self.head)
        self.C, self.d_model = C, d_model

    def forward(self, x, cov, arch):
        B = x.shape[0]
        z = self.patch(x)                                           # B,C,np,d
        z = z.reshape(B * self.C, self.patch.n_patch, self.d_model) + self.patch.pos
        z = self.enc(z).reshape(B, self.C, -1)
        z = self.flat(z).reshape(B, -1)
        h = torch.cat([z, self.cov(cov, arch)], dim=-1)
        return anchor_from_window(x).unsqueeze(-1) + self.head(h) * 100.0, None


# ═══════════════════════════════════════════════════════════════════════════════
#  M5  iTransformer
# ═══════════════════════════════════════════════════════════════════════════════
class ITransformer(nn.Module):
    def __init__(self, C, L, H, n_cov, n_arch, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1, d_arch=8, **_):
        super().__init__()
        self.embed = nn.Linear(L, d_model)                          # one token per variate
        self.enc = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cov = CovEncoder(n_cov, n_arch, d_arch, d_ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(C * d_model + d_ff, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, H))
        zero_init_last(self.head)

    def forward(self, x, cov, arch):
        B = x.shape[0]
        z = self.embed(x.permute(0, 2, 1))                          # B,C,d
        z = self.norm(self.enc(z)).reshape(B, -1)                   # attention across variates
        h = torch.cat([z, self.cov(cov, arch)], dim=-1)
        return anchor_from_window(x).unsqueeze(-1) + self.head(h) * 100.0, None


# ═══════════════════════════════════════════════════════════════════════════════
#  M6  TSMixer
# ═══════════════════════════════════════════════════════════════════════════════
class MixerBlock(nn.Module):
    def __init__(self, L, C, d_time, d_feat, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(C)
        self.time = nn.Sequential(nn.Linear(L, d_time), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_time, L), nn.Dropout(dropout))
        self.n2 = nn.LayerNorm(C)
        self.feat = nn.Sequential(nn.Linear(C, d_feat), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_feat, C), nn.Dropout(dropout))

    def forward(self, x):
        z = self.n1(x).permute(0, 2, 1)
        x = x + self.time(z).permute(0, 2, 1)
        return x + self.feat(self.n2(x))


class TSMixer(nn.Module):
    def __init__(self, C, L, H, n_cov, n_arch, n_blocks=3, d_time=64, d_feat=64,
                 d_ff=128, dropout=0.1, d_arch=8, **_):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(L, C, d_time, d_feat, dropout)
                                     for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(C)
        self.cov = CovEncoder(n_cov, n_arch, d_arch, d_ff, dropout)
        self.head = nn.Sequential(
            nn.Linear(L * C + d_ff, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, H))
        zero_init_last(self.head)

    def forward(self, x, cov, arch):
        anchor = anchor_from_window(x)
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.norm(z).reshape(x.shape[0], -1)
        h = torch.cat([z, self.cov(cov, arch)], dim=-1)
        return anchor.unsqueeze(-1) + self.head(h) * 100.0, None


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  M7  SIGMA-Net  (ours)
# ═══════════════════════════════════════════════════════════════════════════════
class SigmaNet(nn.Module):
    """Patch + variate-attention backbone, ventilation-gated, with the CPCB operator
    as the output layer. `use_dsi` and `use_vgm` are the ablation switches."""

    def __init__(self, C, L, H, n_cov, n_arch, vent_idx, n_pol,
                 d_model=64, n_heads=4, n_layers=2, n_var_layers=1,
                 patch_len=4, stride=2, d_ff=128, d_hidden=128, dropout=0.1,
                 d_arch=8, use_dsi=True, use_vgm=True, tau_init=2.0,
                 pol_mu=None, pol_sd=None, pol_cap=None, **_):
        super().__init__()
        self.use_dsi, self.use_vgm = use_dsi, use_vgm
        self.H, self.n_pol, self.C, self.d_model = H, n_pol, C, d_model
        self.register_buffer("vent_idx", torch.as_tensor(list(vent_idx), dtype=torch.long))

        self.patch = PatchEmbed(L, patch_len, stride, d_model)
        self.temporal = encoder_stack(d_model, n_heads, d_ff, dropout, n_layers)
        self.variate = encoder_stack(d_model, n_heads, d_ff, dropout, n_var_layers)
        self.pool_norm = nn.LayerNorm(d_model)
        self.reduce = nn.Sequential(nn.Linear(C * d_model, d_hidden), nn.GELU(),
                                    nn.Dropout(dropout))

        if use_vgm:
            self.film = VentilationFiLM(2 * len(vent_idx), d_hidden, dropout)

        self.cov = CovEncoder(n_cov, n_arch, d_arch, d_hidden // 2, dropout)
        self.trunk = nn.Sequential(
            nn.Linear(d_hidden + d_hidden // 2, d_hidden), nn.GELU(), nn.Dropout(dropout))

        self.register_buffer("pol_ch", torch.as_tensor(POL_AQI_LOG_IDX, dtype=torch.long))
        if use_dsi:
            # The head predicts a CHANGE from today's observed concentrations, so at
            # initialisation the model emits today's concentrations through the exact
            # CPCB formula - i.e. concentration persistence - and learns from there.
            self.pol_head = nn.Linear(d_hidden, H * n_pol)
            nn.init.zeros_(self.pol_head.weight)
            nn.init.zeros_(self.pol_head.bias)
            self.decoder = PollutantDecoder(pol_mu, pol_sd, pol_cap)
            self.dsi = CPCBSubIndex(POLLUTANTS_AQI, tau_init=tau_init)
            self.res_head = nn.Linear(d_hidden, H)
            nn.init.zeros_(self.res_head.weight)    # start as pure physics
            nn.init.zeros_(self.res_head.bias)
            self.res_scale = 50.0
        else:
            self.direct = nn.Linear(d_hidden, H)
            nn.init.zeros_(self.direct.weight)
            nn.init.zeros_(self.direct.bias)

    def _vent_summary(self, x):
        v = x.index_select(dim=2, index=self.vent_idx)               # B,L,V
        return torch.cat([v[:, -1, :], v.mean(dim=1)], dim=-1)       # B,2V

    def forward(self, x, cov, arch):
        B = x.shape[0]
        z = self.patch(x)                                            # B,C,np,d
        z = z.reshape(B * self.C, self.patch.n_patch, self.d_model) + self.patch.pos
        z = self.temporal(z)
        z = self.pool_norm(z.mean(dim=1)).reshape(B, self.C, self.d_model)
        z = self.variate(z)                                          # inter-pollutant coupling
        h = self.reduce(z.reshape(B, -1))
        if self.use_vgm:
            h = self.film(h, self._vent_summary(x))
        h = self.trunk(torch.cat([h, self.cov(cov, arch)], dim=-1))

        if not self.use_dsi:
            return anchor_from_window(x).unsqueeze(-1) + self.direct(h) * 100.0, None

        # today's standardised log1p concentrations, the anchor for the physics head
        z_last = x[:, -1, :].index_select(dim=1, index=self.pol_ch)          # B,P
        zpol = z_last.unsqueeze(1) + self.pol_head(h).reshape(B, self.H, self.n_pol)
        conc = self.decoder(zpol)
        aqi, _ = self.dsi(conc)
        aqi = aqi + self.res_head(h) * self.res_scale
        return aqi, zpol


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  C3 - episode-focal asymmetric loss
# ═══════════════════════════════════════════════════════════════════════════════
def masked_huber_aqi(pred, true, mask, delta=1.0):
    l = F.huber_loss(pred / 100.0, true / 100.0, reduction="none", delta=delta)
    return (l * mask).sum() / mask.sum().clamp(min=1.0)


def efa_loss(pred, true, mask, alpha=1.0, beta=0.5, soft=40.0,
             thr=HAZARD_THRESHOLD, delta=1.0):
    """Huber, re-weighted around the regulatory hazard boundary, plus an asymmetric
    penalty: under-forecasting a hazardous day costs more than over-forecasting it,
    because the first fails to issue a health advisory and the second merely annoys."""
    gate = torch.sigmoid((true - thr) / soft)
    w = 1.0 + alpha * gate
    base = F.huber_loss(pred / 100.0, true / 100.0, reduction="none", delta=delta)
    under = F.relu((true - pred) / 100.0) * gate
    l = w * base + beta * under
    return (l * mask).sum() / mask.sum().clamp(min=1.0)


def pol_aux_loss(zpred, ztrue, pmask, delta=1.0):
    """Keeps the physics head physical: the predicted concentrations are supervised
    against the observed ones wherever they exist, in standardised log1p space."""
    l = F.huber_loss(zpred, ztrue, reduction="none", delta=delta)
    return (l * pmask).sum() / pmask.sum().clamp(min=1.0)


# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  model registry
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(name, hp):
    common = dict(C=C_DYN, L=LOOKBACK, H=N_H, n_cov=N_COV, n_arch=N_ARCH)
    name = name.lower()
    if name == "patchtst":
        return PatchTST(**common, **hp)
    if name == "itransformer":
        return ITransformer(**common, **hp)
    if name == "tsmixer":
        return TSMixer(**common, **hp)
    if name.startswith("sigmanet"):
        return SigmaNet(**common, vent_idx=VENT_IDX, n_pol=N_POL_AQI,
                        pol_mu=POL_MU, pol_sd=POL_SD, pol_cap=POL_CAP, **hp)
    raise KeyError(f"unknown model {name!r}")


DEEP_MODELS = ["patchtst", "itransformer", "tsmixer", "sigmanet"]

DEFAULT_HP = {
    "patchtst":     dict(d_model=64, n_heads=4, n_layers=2, patch_len=4, stride=2,
                         d_ff=128, dropout=0.1, d_head=16, lr=1e-3, weight_decay=1e-4),
    "itransformer": dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1,
                         lr=1e-3, weight_decay=1e-4),
    "tsmixer":      dict(n_blocks=3, d_time=64, d_feat=64, d_ff=128, dropout=0.1,
                         lr=1e-3, weight_decay=1e-4),
    "sigmanet":     dict(d_model=64, n_heads=4, n_layers=2, n_var_layers=1, patch_len=4,
                         stride=2, d_ff=128, d_hidden=128, dropout=0.1, tau_init=2.0,
                         use_dsi=True, use_vgm=True, lr=1e-3, weight_decay=1e-4,
                         efa_alpha=1.0, efa_beta=0.5, lambda_aux=0.3, use_efa=True),
}
# keys that configure optimisation or the loss, not the architecture
NON_ARCH_KEYS = {"lr", "weight_decay", "efa_alpha", "efa_beta", "lambda_aux", "use_efa"}


def split_hp(hp):
    arch = {k: v for k, v in hp.items() if k not in NON_ARCH_KEYS}
    opt = {k: v for k, v in hp.items() if k in NON_ARCH_KEYS}
    return arch, opt


def suggest_hp(name, trial):
    """Optuna search spaces. `patch_len`/`stride` are constrained so that the patch
    count stays valid for L=14."""
    if name == "patchtst":
        pl = trial.suggest_categorical("patch_len", [2, 4, 7])
        return dict(
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            patch_len=pl, stride=trial.suggest_categorical("stride", [1, 2]),
            d_ff=trial.suggest_categorical("d_ff", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            d_head=trial.suggest_categorical("d_head", [8, 16, 32]),
            lr=trial.suggest_float("lr", 2e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    if name == "itransformer":
        return dict(
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            d_ff=trial.suggest_categorical("d_ff", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 2e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    if name == "tsmixer":
        return dict(
            n_blocks=trial.suggest_int("n_blocks", 1, 4),
            d_time=trial.suggest_categorical("d_time", [32, 64, 128]),
            d_feat=trial.suggest_categorical("d_feat", [32, 64, 128]),
            d_ff=trial.suggest_categorical("d_ff", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 2e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    if name == "sigmanet":
        return dict(
            d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4]),
            n_layers=trial.suggest_int("n_layers", 1, 3),
            n_var_layers=trial.suggest_int("n_var_layers", 1, 2),
            patch_len=trial.suggest_categorical("patch_len", [2, 4, 7]),
            stride=trial.suggest_categorical("stride", [1, 2]),
            d_ff=trial.suggest_categorical("d_ff", [64, 128, 256]),
            d_hidden=trial.suggest_categorical("d_hidden", [64, 128, 256]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            tau_init=trial.suggest_float("tau_init", 1.0, 8.0),
            lr=trial.suggest_float("lr", 2e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            efa_alpha=trial.suggest_float("efa_alpha", 0.0, 3.0),
            efa_beta=trial.suggest_float("efa_beta", 0.0, 2.0),
            lambda_aux=trial.suggest_float("lambda_aux", 0.0, 1.0),
            use_dsi=True, use_vgm=True, use_efa=True)
    raise KeyError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# %% [markdown]
# ### 6.1 The trainer
#
# Early stopping is always on. Every epoch writes a checkpoint holding the model, the
# optimiser, the scheduler, the best weights so far and the patience counter, so an
# interrupted fit resumes mid-run rather than restarting. Per-epoch shuffling is seeded
# from `(seed, epoch)`, which makes a resumed run bit-identical to an uninterrupted one
# without having to serialise RNG state.

# %%
def _batches(n, bs, seed, epoch, shuffle=True):
    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(seed) * 100_003 + int(epoch))
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for i in range(0, n, bs):
        yield order[i: i + bs]


@torch.no_grad()
def predict_torch(model, idx, batch_size=1024):
    """Returns AQI predictions (n, H) in float32, on CPU."""
    model.eval()
    ti = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=DEVICE)
    out = []
    for i in range(0, len(ti), batch_size):
        b = ti[i: i + batch_size]
        aqi, _ = model(T_X[b], T_COV[b], T_ARCH[b])
        out.append(aqi.float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, N_H), dtype=np.float32)


def make_loss_fn(name, opt_hp):
    use_efa = bool(opt_hp.get("use_efa", False)) and name.startswith("sigmanet")
    a = float(opt_hp.get("efa_alpha", 1.0))
    b = float(opt_hp.get("efa_beta", 0.5))
    lam = float(opt_hp.get("lambda_aux", 0.0))

    def loss_fn(pred_aqi, zpol, y, ym, ypol, ypolm):
        main = (efa_loss(pred_aqi, y, ym, alpha=a, beta=b) if use_efa
                else masked_huber_aqi(pred_aqi, y, ym))
        if zpol is not None and lam > 0:
            main = main + lam * pol_aux_loss(zpol, ypol, ypolm)
        return main

    return loss_fn


def train_torch(name, hp, train_idx, val_idx, *, seed=SEED, max_epochs=None,
                patience=None, batch_size=None, ckpt_key=None, verbose=False,
                trial=None):
    """Fits one model. Returns (model, history dict). Resumes from ckpt_key if present.
    If `trial` is given, the validation MAE is reported to Optuna each epoch so that
    hopeless configurations are pruned early."""
    max_epochs = max_epochs or CFG["max_epochs"]
    patience = patience or CFG["patience"]
    batch_size = batch_size or CFG["batch_size"]
    arch_hp, opt_hp = split_hp(hp)

    set_seed(seed)
    model = build_model(name, arch_hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(opt_hp.get("lr", 1e-3)),
                            weight_decay=float(opt_hp.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_epochs, 1))
    loss_fn = make_loss_fn(name, opt_hp)

    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=DEVICE)
    if len(tr) == 0:
        raise ValueError(f"{name}: empty training index")
    if not (SAMPLES["ym"][np.asarray(val_idx)] > 0).any():
        raise ValueError(f"{name}: validation fold has no observed target")
    start_epoch, best_val, best_state, bad = 0, float("inf"), None, 0
    hist = []

    ckpt_path = DIRS["ckpt"] / f"{_safe_key(ckpt_key)}.pt" if ckpt_key else None
    if ckpt_path is not None and ckpt_path.exists():
        try:
            # weights_only=True: the checkpoint holds only tensors and primitives, and
            # refusing to unpickle anything else costs nothing here.
            try:
                ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            except TypeError:                      # torch < 1.13 has no such argument
                ck = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch = int(ck["epoch"]) + 1
            best_val = float(ck["best_val"])
            best_state = ck["best_state"]
            bad = int(ck["bad"])
            hist = list(ck.get("hist", []))
            print(f"      resumed {ckpt_key} at epoch {start_epoch} (best val MAE {best_val:.3f})")
        except Exception as e:
            print(f"      checkpoint unreadable ({type(e).__name__}); restarting this fit")
            start_epoch, best_val, best_state, bad, hist = 0, float("inf"), None, 0, []

    n = len(tr)
    for epoch in range(start_epoch, max_epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in _batches(n, batch_size, seed, epoch):
            bi = tr[b.to(DEVICE)]
            opt.zero_grad(set_to_none=True)
            ctx = (torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16)
                   if AMP_ENABLED else nullcontext())
            with ctx:
                aqi, zpol = model(T_X[bi], T_COV[bi], T_ARCH[bi])
                loss = loss_fn(aqi.float(), None if zpol is None else zpol.float(),
                               T_Y[bi], T_YM[bi], T_YPOL[bi], T_YPOLM[bi])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()

        vp = predict_torch(model, val_idx)
        vm = SAMPLES["ym"][val_idx] > 0
        vy = SAMPLES["y"][val_idx]
        val_mae = float(np.abs(vp[vm] - vy[vm]).mean())
        hist.append(dict(epoch=epoch, train_loss=tot / max(nb, 1), val_mae=val_mae))

        if val_mae < best_val - 1e-6:
            best_val, bad = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if verbose:
            print(f"      epoch {epoch:3d}  train {tot/max(nb,1):.4f}  "
                  f"val MAE {val_mae:7.3f}  best {best_val:7.3f}  bad {bad}")

        if ckpt_path is not None:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save(dict(epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), best_val=best_val,
                            best_state=best_state, bad=bad, hist=hist), tmp)
            os.replace(tmp, ckpt_path)

        if trial is not None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if bad >= patience:
            if verbose:
                print(f"      early stop at epoch {epoch} (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_val_mae=best_val, epochs_run=len(hist),
                       history=hist, params=count_params(model))

# %% [markdown]
# ## 7 - Metrics and significance tests
#
# Everything the paper reports comes from one continuous AQI forecast. Categories are
# obtained by binning it and hazardous days by thresholding it, so the classification and
# regression numbers can never contradict each other.
#
# Significance is assessed two ways, because neither alone is quite right here:
#
# * a **paired block bootstrap** over 7-day date blocks, resampling whole blocks with every
#   city inside them, which respects both serial and cross-city dependence. This is the
#   primary test.
# * a **Diebold-Mariano** test on the city-averaged daily loss differential with a
#   Newey-West variance, which is the standard forecast-comparison statistic and is what a
#   reviewer will look for.

# %%
AQI_MIN, AQI_MAX = 0.0, 500.0


def clip_aqi(p):
    return np.clip(np.asarray(p, dtype=np.float64), AQI_MIN, AQI_MAX)


def score_point(yt, yp):
    """Regression + episode + category metrics for one horizon."""
    yt = np.asarray(yt, dtype=np.float64)
    yp = clip_aqi(yp)
    n = len(yt)
    if n == 0:
        return {}
    err = yp - yt
    out = dict(
        n=int(n),
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err ** 2).mean())),
        bias=float(err.mean()),
        r2=float(r2_score(yt, yp)) if n > 1 and np.var(yt) > 0 else float("nan"),
        smape=float(np.mean(200.0 * np.abs(err) / np.clip(np.abs(yt) + np.abs(yp), 1e-9, None))),
    )
    # hazardous-day detection, from the continuous forecast
    tt = yt > HAZARD_THRESHOLD
    pp = yp > HAZARD_THRESHOLD
    tp = int((tt & pp).sum()); fp = int((~tt & pp).sum()); fn = int((tt & ~pp).sum())
    out.update(
        hazard_n=int(tt.sum()),
        hazard_precision=float(precision_score(tt, pp, zero_division=0)),
        hazard_recall=float(recall_score(tt, pp, zero_division=0)),
        hazard_f1=float(f1_score(tt, pp, zero_division=0)),
        hazard_csi=float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else float("nan"),
        hazard_prauc=float(average_precision_score(tt, yp)) if 0 < tt.sum() < n else float("nan"),
    )
    ct, cp = aqi_to_category(yt), aqi_to_category(yp)
    labels = list(range(len(CATEGORY_NAMES)))
    out.update(
        cat_accuracy=float((ct == cp).mean()),
        cat_macro_f1=float(f1_score(ct, cp, average="macro", labels=labels, zero_division=0)),
        cat_qwk=float(cohen_kappa_score(ct, cp, weights="quadratic", labels=labels))
        if len(set(ct.tolist()) | set(cp.tolist())) > 1 else float("nan"),
    )
    return out


def evaluate(yhat, idx):
    """yhat: (n, H) predictions aligned to idx. Returns {'h1':..,'h2':..,'h3':..,'pooled':..}."""
    idx = np.asarray(idx)
    y, m = SAMPLES["y"][idx], SAMPLES["ym"][idx] > 0
    yhat = np.asarray(yhat, dtype=np.float64).reshape(len(idx), N_H)
    res = {}
    for hi, h in enumerate(HORIZONS):
        sel = m[:, hi]
        res[f"h{h}"] = score_point(y[sel, hi], yhat[sel, hi])
    res["pooled"] = score_point(y[m], yhat[m])
    return res


def residuals(yhat, idx):
    """Per-sample absolute errors and their dates, for the significance tests."""
    idx = np.asarray(idx)
    y, m = SAMPLES["y"][idx], SAMPLES["ym"][idx] > 0
    yhat = clip_aqi(np.asarray(yhat).reshape(len(idx), N_H))
    anchor = SAMPLES["t"][idx]
    out = {}
    for hi, h in enumerate(HORIZONS):
        sel = m[:, hi]
        out[h] = dict(ae=np.abs(yhat[sel, hi] - y[sel, hi]),
                      date=(anchor[sel] + pd.Timedelta(days=int(h))).to_numpy())
    return out


# %%
def block_bootstrap_test(ae_a, ae_b, dates, n_boot=2000, block_days=7, seed=0):
    """Paired block bootstrap on MAE difference (a - b). Negative => a is better."""
    ae_a, ae_b = np.asarray(ae_a, float), np.asarray(ae_b, float)
    d = pd.to_datetime(pd.Series(dates))
    if len(ae_a) == 0 or len(ae_a) != len(ae_b):
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"))
    blk = ((d - d.min()).dt.days // block_days).to_numpy()
    uniq = np.unique(blk)
    groups = {b: np.where(blk == b)[0] for b in uniq}
    obs = float(ae_a.mean() - ae_b.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([groups[b] for b in pick])
        draws[i] = ae_a[sel].mean() - ae_b[sel].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    centred = draws - draws.mean()
    p = float((np.abs(centred) >= abs(obs)).mean())
    return dict(delta=obs, lo=float(lo), hi=float(hi), p=min(max(p, 1.0 / n_boot), 1.0))


def diebold_mariano(ae_a, ae_b, dates, horizon):
    """DM on the daily mean loss differential, Newey-West with lag = horizon - 1."""
    df = pd.DataFrame(dict(d=np.asarray(ae_a, float) - np.asarray(ae_b, float),
                           date=pd.to_datetime(pd.Series(dates))))
    s = df.groupby("date")["d"].mean().sort_index().to_numpy()
    n = len(s)
    if n < 10:
        return dict(dm=float("nan"), p=float("nan"), n_days=int(n))
    dbar = float(s.mean())
    dev = s - dbar
    lag = max(int(horizon) - 1, 0)
    var = float((dev ** 2).mean())
    for k in range(1, lag + 1):
        ck = float((dev[k:] * dev[:-k]).mean())
        var += 2.0 * (1.0 - k / (lag + 1.0)) * ck
    if not np.isfinite(var) or var <= 0:
        return dict(dm=float("nan"), p=float("nan"), n_days=int(n))
    dm = dbar / math.sqrt(var / n)
    return dict(dm=float(dm), p=float(2.0 * (1.0 - stats.norm.cdf(abs(dm)))), n_days=int(n))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        running = max(running, adj)
        out[i] = min(running, 1.0)
    return out


# %% [markdown]
# ## 8 - Reference models and LightGBM

# %%
def predict_persistence(idx):
    """Today's AQI, carried forward to every horizon."""
    p = SAMPLES["persist"][np.asarray(idx)]
    return np.repeat(p.reshape(-1, 1), N_H, axis=1)


def _build_climatology():
    """Smoothed day-of-year AQI climatology from TRAIN rows only, with fallbacks for
    cities that have no training history at all."""
    tr = raw[(raw["split"] == "train") & raw["AQI"].notna()].copy()
    tr["doy"] = tr["date"].dt.day_of_year.clip(upper=365)
    grid = np.arange(1, 366)

    def smooth(series_by_doy):
        v = series_by_doy.reindex(grid).to_numpy(dtype=np.float64)
        if np.all(np.isnan(v)):
            return None
        # circular +/-7 day mean, NaN-tolerant
        pad = np.concatenate([v[-7:], v, v[:7]])
        out = np.empty(365)
        for i in range(365):
            w = pad[i: i + 15]
            out[i] = np.nanmean(w) if np.any(np.isfinite(w)) else np.nan
        if np.any(np.isnan(out)):
            out = np.where(np.isnan(out), np.nanmean(out), out)
        return out

    per_city = {c: smooth(g.groupby("doy")["AQI"].mean())
                for c, g in tr.groupby("city")}
    per_arch = {int(a): smooth(g.groupby("doy")["AQI"].mean())
                for a, g in tr.groupby("archetype_id")}
    glob = smooth(tr.groupby("doy")["AQI"].mean())
    return per_city, per_arch, glob


CLIM_CITY, CLIM_ARCH, CLIM_GLOBAL = _build_climatology()


def predict_climatology(idx):
    idx = np.asarray(idx)
    anchor = SAMPLES["t"][idx]
    city = SAMPLES["city"][idx]
    arch = SAMPLES["arch"][idx]
    out = np.empty((len(idx), N_H), dtype=np.float64)
    for hi, h in enumerate(HORIZONS):
        doy = (anchor + pd.Timedelta(days=int(h))).day_of_year.to_numpy().clip(1, 365)
        for j in range(len(idx)):
            curve = CLIM_CITY.get(CITIES[city[j]])
            if curve is None:
                curve = CLIM_ARCH.get(int(arch[j]))
            if curve is None:
                curve = CLIM_GLOBAL
            out[j, hi] = curve[doy[j] - 1]
    return out


# %%
def build_lgbm_matrix():
    """Window summaries + explicit lags + known-future covariates. Exactly the same
    information the deep models receive, in a representation a GBDT can use."""
    X = SAMPLES["X"]
    parts, names = [], []

    for tag, arr in (("last", X[:, -1, :]), ("mean", X.mean(axis=1)),
                     ("std", X.std(axis=1)), ("min", X.min(axis=1)), ("max", X.max(axis=1))):
        parts.append(arr)
        names += [f"{tag}__{c}" for c in DYN_COLS]

    for cname, lags in (("aqi_hist", [0, 1, 2, 3, 6, 13]),
                        ("log_PM2.5", [0, 1, 2, 6]),
                        ("log_PM10", [0, 1, 2, 6]),
                        ("blh", [0, 1, 2]),
                        ("wind_speed", [0, 1, 2])):
        ci = DYN_COLS.index(cname)
        for lg in lags:
            if lg < LOOKBACK:
                parts.append(X[:, -1 - lg, ci: ci + 1])
                names.append(f"lag{lg}__{cname}")

    # AQI trend over the window
    ai = DYN_COLS.index("aqi_hist")
    parts.append((X[:, -1, ai] - X[:, 0, ai]).reshape(-1, 1)); names.append("aqi_window_delta")
    parts.append((X[:, -1, ai] - X[:, -4:, ai].mean(axis=1)).reshape(-1, 1))
    names.append("aqi_dev_from_3d")

    cal_names = ["doy_sin", "doy_cos", "mon_sin", "mon_cos", "dow_sin", "dow_cos",
                 "is_diwali", "is_stubble"]
    parts.append(SAMPLES["cov"])
    names += [f"fut_h{h}__{c}" for h in HORIZONS for c in cal_names] + \
             ["static__coastal", "static__log_industrial", "static__inv_level"]
    parts.append(SAMPLES["arch"].reshape(-1, 1).astype(np.float32))
    names.append("static__archetype")

    M = np.concatenate(parts, axis=1).astype(np.float32)
    assert M.shape[1] == len(names), f"{M.shape[1]} columns vs {len(names)} names"
    assert np.isfinite(M).all(), "non-finite value in the LightGBM design matrix"
    return M, names


LGBM_X, LGBM_NAMES = build_lgbm_matrix()
print(f"LightGBM design matrix: {LGBM_X.shape[0]:,} x {LGBM_X.shape[1]} "
      f"({LGBM_X.nbytes/1e6:.1f} MB)")

LGBM_DEFAULT_HP = dict(num_leaves=63, learning_rate=0.05, min_child_samples=20,
                       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                       lambda_l2=1.0, n_estimators=2000)


def suggest_lgbm_hp(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        n_estimators=2000)


def train_lgbm(hp, train_idx, val_idx, n_estimators_cap=None):
    """One booster per horizon (direct multi-horizon). Early stopping on the fold's
    validation block, which is what fixes the tree count."""
    hp = dict(hp)
    hp["n_estimators"] = int(n_estimators_cap or hp.get("n_estimators", 2000))
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    boosters, best_iters = [], []
    for hi, h in enumerate(HORIZONS):
        mtr = SAMPLES["ym"][tr][:, hi] > 0
        mva = SAMPLES["ym"][va][:, hi] > 0
        if mtr.sum() < 50 or mva.sum() < 10:
            boosters.append(None); best_iters.append(0)
            continue
        model = lgb.LGBMRegressor(objective="regression_l1", verbosity=-1,
                                  random_state=SEED, n_jobs=-1, **hp)
        model.fit(
            LGBM_X[tr][mtr], SAMPLES["y"][tr][mtr, hi],
            eval_set=[(LGBM_X[va][mva], SAMPLES["y"][va][mva, hi])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        boosters.append(model)
        best_iters.append(int(getattr(model, "best_iteration_", 0) or hp["n_estimators"]))
    return dict(boosters=boosters, best_iters=best_iters)


def predict_lgbm(fit, idx):
    idx = np.asarray(idx)
    out = np.zeros((len(idx), N_H), dtype=np.float64)
    fallback = SAMPLES["persist"][idx]
    for hi, b in enumerate(fit["boosters"]):
        out[:, hi] = fallback if b is None else b.predict(LGBM_X[idx])
    return out

# %% [markdown]
# ## 9 - Hyperparameter search (Optuna, resumable)
#
# The objective is **validation-2024 MAE only**. Neither test set is touched until
# every hyperparameter and every early-stopping decision is frozen.
#
# The study lives in a SQLite file inside the run directory, so an interrupted search
# resumes at the trial it had reached rather than starting over.

# %%
def pred_path(key):
    return DIRS["results"] / (_safe_key(key) + ".npy")


def save_pred(key, arr):
    p = pred_path(key)
    tmp = Path(str(p) + ".tmp.npy")
    np.save(tmp, np.asarray(arr, dtype=np.float32))
    os.replace(tmp, p)


def load_pred(key):
    return np.load(pred_path(key))


def make_storage():
    db = DIRS["studies"] / "optuna.db"
    url = "sqlite:///" + Path(db).as_posix()
    try:
        st = optuna.storages.RDBStorage(
            url=url, engine_kwargs={"connect_args": {"timeout": 90}})
        # Write-probe using only the public API, so this does not depend on which
        # Optuna version is installed. A version-specific probe that happened to fail
        # would silently cost us tuning resumability, which is the whole point.
        optuna.create_study(study_name="__write_probe__", storage=st, load_if_exists=True)
        print(f"  Optuna studies persist to {db}")
        return st
    except Exception as e:
        print(f"  !! Drive-backed Optuna storage unavailable ({type(e).__name__}: {e}).")
        print(f"     Falling back to an in-memory study: tuning will NOT resume after a")
        print(f"     disconnect, though every finished fit still will.")
        return None


OPTUNA_STORAGE = make_storage()
# torch.cuda.OutOfMemoryError does not exist in every build; fall back to RuntimeError
OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
HPO_EPOCHS = max(8, min(CFG["max_epochs"], 40))
HPO_PATIENCE = max(3, min(CFG["patience"], 8))


def hpo_deep(name):
    """Returns the best hyperparameters for one deep model."""
    key = f"hpo__{name}"
    n_trials = CFG["n_trials_deep"]

    def _objective(trial):
        hp = suggest_hp(name, trial)
        try:
            _, info = train_torch(name, hp, IDX["train"], IDX["val"], seed=SEED,
                                  max_epochs=HPO_EPOCHS, patience=HPO_PATIENCE,
                                  ckpt_key=None, trial=trial)
        except optuna.TrialPruned:
            raise
        except (FloatingPointError, OOM_ERROR) as e:
            print(f"      trial {trial.number} failed ({type(e).__name__}); pruned")
            raise optuna.TrialPruned()
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        return info["best_val_mae"]

    def _run():
        study = optuna.create_study(
            study_name=f"{PROFILE}__{name}", direction="minimize",
            storage=OPTUNA_STORAGE, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(3, n_trials // 5),
                                               n_warmup_steps=3))
        done = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE)
        todo = max(n_trials - done, 0)
        if done:
            print(f"      {done} trials already in the study; running {todo} more")
        if todo:
            study.optimize(_objective, n_trials=todo, gc_after_trial=True,
                           show_progress_bar=False)
        best = dict(DEFAULT_HP[name])
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            best.update(study.best_params)
            bv = float(study.best_value)
        else:
            print("      no trial completed; falling back to the default hyperparameters")
            bv = float("nan")
        if name == "sigmanet":
            best.update(use_dsi=True, use_vgm=True, use_efa=True)
        return dict(best_params=best, best_val_mae=bv,
                    n_trials_complete=sum(1 for t in study.trials
                                          if t.state == optuna.trial.TrialState.COMPLETE))

    return run_unit(key, _run)["best_params"]


def hpo_lgbm():
    key = "hpo__lgbm"

    def _objective(trial):
        hp = suggest_lgbm_hp(trial)
        fit = train_lgbm(hp, IDX["train"], IDX["val"])
        p = predict_lgbm(fit, IDX["val"])
        return evaluate(p, IDX["val"])["pooled"]["mae"]

    def _run():
        study = optuna.create_study(
            study_name=f"{PROFILE}__lgbm", direction="minimize",
            storage=OPTUNA_STORAGE, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
        done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        todo = max(CFG["n_trials_lgbm"] - done, 0)
        if done:
            print(f"      {done} trials already in the study; running {todo} more")
        if todo:
            study.optimize(_objective, n_trials=todo, gc_after_trial=True,
                           show_progress_bar=False)
        best = dict(LGBM_DEFAULT_HP)
        bv = float("nan")
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            best.update(study.best_params)
            best["n_estimators"] = 2000
            bv = float(study.best_value)
        return dict(best_params=best, best_val_mae=bv)

    return run_unit(key, _run)["best_params"]


print("hyperparameter search")
BEST_HP = {}
for _m in DEEP_MODELS:
    print(f"  --- {_m} ---")
    BEST_HP[_m] = hpo_deep(_m)
print("  --- lgbm ---")
BEST_HP["lgbm"] = hpo_lgbm()

print("\nselected hyperparameters")
for k, v in BEST_HP.items():
    print(f"  {k}: " + ", ".join(f"{a}={b}" for a, b in sorted(v.items())))
save_result("best_hyperparameters", dict(hp=BEST_HP))

# %% [markdown]
# ## 10 - Blocked expanding-window cross-validation
#
# Five folds inside 2019-2023. Each fold's training block ends an embargo before its
# validation block starts, so no training window can overlap a validation window.

# %%
def blocked_folds(n_folds):
    tr_all = IDX["train"]
    dates = SAMPLES["t"][tr_all]
    dmin, dmax = dates.min(), dates.max()
    span = (dmax - dmin).days
    seg = span // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        v0 = dmin + pd.Timedelta(days=seg * (k + 1))
        v1 = v0 + pd.Timedelta(days=seg)
        t_end = v0 - pd.Timedelta(days=CFG["embargo"])
        tm = dates <= t_end
        vm = (dates > v0) & (dates <= v1)
        if tm.sum() < 200 or vm.sum() < 50:
            continue
        folds.append(dict(fold=k, train=tr_all[tm], val=tr_all[vm],
                          train_end=str(t_end.date()),
                          val_range=f"{v0.date()}..{v1.date()}"))
    return folds


FOLDS = blocked_folds(CFG["cv_folds"])
print(f"{len(FOLDS)} usable folds")
for f in FOLDS:
    print(f"  fold {f['fold']}: train n={len(f['train']):6,} (to {f['train_end']})  "
          f"val n={len(f['val']):5,} ({f['val_range']})")
assert FOLDS, "no usable CV fold; reduce cv_folds or the embargo"

ALL_MODELS = ["persistence", "climatology", "lgbm"] + DEEP_MODELS
MODEL_LABEL = {"persistence": "Persistence", "climatology": "Climatology",
               "lgbm": "LightGBM", "patchtst": "PatchTST",
               "itransformer": "iTransformer", "tsmixer": "TSMixer",
               "sigmanet": "SIGMA-Net (ours)"}


def fit_and_predict(name, hp, train_idx, val_idx, eval_sets, ckpt_key=None, seed=SEED):
    """One fit, then predictions on each named index set. Returns (preds, info)."""
    if name == "persistence":
        return {k: predict_persistence(v) for k, v in eval_sets.items()}, {}
    if name == "climatology":
        return {k: predict_climatology(v) for k, v in eval_sets.items()}, {}
    if name == "lgbm":
        fit = train_lgbm(hp, train_idx, val_idx)
        return ({k: predict_lgbm(fit, v) for k, v in eval_sets.items()},
                dict(best_iters=fit["best_iters"]))
    model, info = train_torch(name, hp, train_idx, val_idx, seed=seed, ckpt_key=ckpt_key)
    preds = {k: predict_torch(model, v) for k, v in eval_sets.items()}
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return preds, {k: v for k, v in info.items() if k != "history"}


def run_cv():
    rows = []
    for name in ALL_MODELS:
        for f in FOLDS:
            key = f"cv__{name}__fold{f['fold']}"

            def _run(name=name, f=f, key=key):
                preds, info = fit_and_predict(
                    name, BEST_HP.get(name, {}), f["train"], f["val"],
                    {"val": f["val"]}, ckpt_key=key + "__ck")
                sc = evaluate(preds["val"], f["val"])
                return dict(model=name, fold=f["fold"], scores=sc, info=info)

            r = run_unit(key, _run)
            for hk, hv in r["scores"].items():
                rows.append(dict(model=name, fold=r["fold"], horizon=hk, **hv))
    return pd.DataFrame(rows)


print("\nblocked cross-validation")
CV_DF = run_cv()
CV_SUMMARY = (CV_DF[CV_DF["horizon"] == "pooled"]
              .groupby("model")[["mae", "rmse", "hazard_f1", "cat_qwk"]]
              .agg(["mean", "std"]).round(3))
save_table(CV_SUMMARY, "table_cv_blocked")
print(CV_SUMMARY.to_string())
print("\nCaveat, stated so it cannot be mistaken for a result: the climatology reference")
print("is fitted on the whole training split, so its cross-validation row is mildly")
print("optimistic. Its test-set numbers, which are the ones the paper reports, are built")
print("from 2019-2023 only and evaluated on 2025, so they are clean. Persistence needs")
print("no fitting at all and is unaffected either way.")

# %% [markdown]
# ## 11 - Final fits and the frozen test sets
#
# Trained on 2019-2023, early-stopped on 2024, then evaluated **once** on 2025 for the
# ten training cities (`test_seen`) and the five cities with no training days at all
# (`test_unseen`). Deep models are fitted with several seeds; tables report mean and
# standard deviation, and the significance tests use the seed-averaged forecast.

# %%
EVAL_SETS = {"val": IDX["val"], "test_seen": IDX["test_seen"],
             "test_unseen": IDX["test_unseen"]}


def run_final():
    rows, store = [], {}
    for name in ALL_MODELS:
        deterministic = name in ("persistence", "climatology")
        seeds = [SEED] if deterministic else [SEED + i for i in range(CFG["n_seeds"])]
        per_seed = {k: [] for k in EVAL_SETS}
        for sd in seeds:
            key = f"final__{name}__seed{sd}"

            def _run(name=name, sd=sd, key=key):
                preds, info = fit_and_predict(
                    name, BEST_HP.get(name, {}), IDX["train"], IDX["val"],
                    EVAL_SETS, ckpt_key=key + "__ck", seed=sd)
                for k, v in preds.items():
                    save_pred(f"{key}__{k}", v)
                return dict(model=name, seed=sd, info=info,
                            scores={k: evaluate(v, EVAL_SETS[k]) for k, v in preds.items()})

            r = run_unit(key, _run)
            for setname in EVAL_SETS:
                per_seed[setname].append(load_pred(f"{key}__{setname}"))
                for hk, hv in r["scores"][setname].items():
                    rows.append(dict(model=name, seed=sd, eval_set=setname,
                                     horizon=hk, **hv))
        for setname in EVAL_SETS:
            store[(name, setname)] = np.mean(np.stack(per_seed[setname], 0), axis=0)
    return pd.DataFrame(rows), store


print("final fits")
FINAL_DF, ENSEMBLE_PRED = run_final()
FINAL_DF.to_csv(DIRS["tables"] / "raw_final_scores.csv", index=False)
print(f"  wrote tables/raw_final_scores.csv ({len(FINAL_DF)} rows)")


def headline(eval_set, metrics=("mae", "rmse", "r2", "hazard_recall", "hazard_f1",
                                "hazard_csi", "cat_accuracy", "cat_qwk")):
    d = FINAL_DF[(FINAL_DF["eval_set"] == eval_set)]
    out = []
    for name in ALL_MODELS:
        for hk in [f"h{h}" for h in HORIZONS] + ["pooled"]:
            g = d[(d["model"] == name) & (d["horizon"] == hk)]
            if not len(g):
                continue
            row = dict(model=MODEL_LABEL[name], horizon=hk)
            for mt in metrics:
                row[mt] = g[mt].mean()
                if len(g) > 1:
                    row[mt + "_sd"] = g[mt].std(ddof=0)
            out.append(row)
    return pd.DataFrame(out).set_index(["model", "horizon"])


TABLE_II = save_table(headline("test_seen").round(3), "table2_test_seen")
TABLE_III = save_table(headline("test_unseen").round(3), "table3_test_unseen")

for nm, tb in (("test_seen (10 training cities, 2025)", TABLE_II),
               ("test_unseen (5 cities with NO training days, 2025)", TABLE_III)):
    print(f"\n=== {nm} ===")
    cols = [c for c in ["mae", "rmse", "hazard_recall", "hazard_f1", "cat_qwk"] if c in tb.columns]
    print(tb.xs("pooled", level="horizon")[cols].round(3).to_string())

# %% [markdown]
# ## 12 - Significance tests
#
# SIGMA-Net against every other model, per horizon, on both test sets.

# %%
def run_significance():
    rows = []
    for setname in ("test_seen", "test_unseen"):
        idx = EVAL_SETS[setname]
        ref = residuals(ENSEMBLE_PRED[("sigmanet", setname)], idx)
        for name in ALL_MODELS:
            if name == "sigmanet":
                continue
            other = residuals(ENSEMBLE_PRED[(name, setname)], idx)
            for h in HORIZONS:
                a, b = ref[h]["ae"], other[h]["ae"]
                bs = block_bootstrap_test(a, b, ref[h]["date"],
                                          n_boot=400 if PROFILE == "smoke" else 2000,
                                          seed=SEED)
                dm = diebold_mariano(a, b, ref[h]["date"], h)
                rows.append(dict(eval_set=setname, baseline=MODEL_LABEL[name], horizon=h,
                                 mae_sigma=float(a.mean()), mae_base=float(b.mean()),
                                 delta=bs["delta"], ci_lo=bs["lo"], ci_hi=bs["hi"],
                                 p_bootstrap=bs["p"], dm=dm["dm"], p_dm=dm["p"]))
    df = pd.DataFrame(rows)
    for setname, g in df.groupby("eval_set"):
        df.loc[g.index, "p_bootstrap_holm"] = holm(g["p_bootstrap"].to_numpy())
        df.loc[g.index, "p_dm_holm"] = holm(g["p_dm"].to_numpy())
    df["sigma_better"] = df["delta"] < 0
    return df


SIG_DF = save_table(run_significance().set_index(["eval_set", "baseline", "horizon"]),
                    "table5_significance")
print(SIG_DF.round(4).to_string())
print("\ndelta = MAE(SIGMA-Net) - MAE(baseline). Negative means SIGMA-Net is better.")

# %% [markdown]
# ## 13 - Ablations
#
# Each of the three contributions is removed in turn, and then all three at once.

# %%
ABLATIONS = {
    "sigmanet_full":   dict(use_dsi=True,  use_vgm=True,  use_efa=True),
    "sigmanet_no_dsi": dict(use_dsi=False, use_vgm=True,  use_efa=True),
    "sigmanet_no_vgm": dict(use_dsi=True,  use_vgm=False, use_efa=True),
    "sigmanet_no_efa": dict(use_dsi=True,  use_vgm=True,  use_efa=False),
    "sigmanet_plain":  dict(use_dsi=False, use_vgm=False, use_efa=False),
}


def run_ablations():
    rows = []
    if not CFG["do_ablations"]:
        print("  ablations disabled for this profile")
        return pd.DataFrame(rows)
    for tag, flags in ABLATIONS.items():
        hp = dict(BEST_HP["sigmanet"])
        hp.update(flags)
        for sd in [SEED + i for i in range(CFG["n_seeds"])]:
            key = f"abl__{tag}__seed{sd}"

            def _run(hp=hp, sd=sd, key=key, tag=tag):
                preds, info = fit_and_predict(
                    "sigmanet", hp, IDX["train"], IDX["val"], EVAL_SETS,
                    ckpt_key=key + "__ck", seed=sd)
                return dict(variant=tag, seed=sd, info=info,
                            scores={k: evaluate(v, EVAL_SETS[k]) for k, v in preds.items()})

            r = run_unit(key, _run)
            for setname in ("test_seen", "test_unseen"):
                sc = r["scores"][setname]["pooled"]
                rows.append(dict(variant=tag, seed=sd, eval_set=setname, **sc))
    return pd.DataFrame(rows)


print("ablations")
ABL_DF = run_ablations()
if len(ABL_DF):
    ABL_SUMMARY = (ABL_DF.groupby(["eval_set", "variant"])
                   [["mae", "rmse", "hazard_recall", "hazard_f1", "hazard_csi", "cat_qwk"]]
                   .agg(["mean", "std"]).round(3))
    save_table(ABL_SUMMARY, "table4_ablation")
    print(ABL_SUMMARY.to_string())

# %% [markdown]
# ## 14 - Leave-one-city-out
#
# For each training city: fit on the other cities only, early-stop on the other cities
# only, and score on the held-out city's 2024 block. The held-out city contributes
# nothing to fitting or to model selection, which makes this the in-sample analogue of
# the unseen-city test.

# %%
def run_loco():
    rows = []
    if not CFG["do_loco"]:
        print("  LOCO disabled for this profile")
        return pd.DataFrame(rows)
    cities = TRAIN_CITIES if not CFG["loco_cities"] else TRAIN_CITIES[: CFG["loco_cities"]]
    models = ["persistence", "lgbm"] + DEEP_MODELS
    for city in cities:
        cid = CITY_IDX[city]
        tr = IDX["train"][SAMPLES["city"][IDX["train"]] != cid]
        va_other = IDX["val"][SAMPLES["city"][IDX["val"]] != cid]
        held = IDX["val"][SAMPLES["city"][IDX["val"]] == cid]
        if len(held) < 30 or len(tr) < 200:
            print(f"  skipping {city}: not enough samples")
            continue
        for name in models:
            key = f"loco__{name}__{_safe_key(city)}"

            def _run(name=name, tr=tr, va_other=va_other, held=held, city=city, key=key):
                preds, info = fit_and_predict(name, BEST_HP.get(name, {}), tr, va_other,
                                              {"held": held}, ckpt_key=key + "__ck")
                return dict(model=name, city=city, info=info,
                            scores=evaluate(preds["held"], held))

            r = run_unit(key, _run)
            rows.append(dict(model=name, city=city, **r["scores"]["pooled"]))
    return pd.DataFrame(rows)


print("leave-one-city-out")
LOCO_DF = run_loco()
if len(LOCO_DF):
    LOCO_SUMMARY = (LOCO_DF.pivot_table(index="city", columns="model", values="mae")
                    .round(2))
    LOCO_SUMMARY.loc["MEAN"] = LOCO_SUMMARY.mean()
    LOCO_SUMMARY.loc["STD"] = LOCO_SUMMARY.iloc[:-1].std()
    save_table(LOCO_SUMMARY, "table6_loco_mae")
    print(LOCO_SUMMARY.to_string())

# %% [markdown]
# ## 15 - Result figures

# %%
def _pooled(df, setname, model, metric="mae"):
    g = df[(df.eval_set == setname) & (df.model == model) & (df.horizon == "pooled")]
    return float(g[metric].mean()) if len(g) else float("nan")


# ---- Figure: MAE by horizon, both test sets --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, setname, title in zip(
        axes, ("test_seen", "test_unseen"),
        ("Seen cities (2025)", "Unseen cities (2025, no training days)")):
    for name in ALL_MODELS:
        ys = []
        for h in HORIZONS:
            g = FINAL_DF[(FINAL_DF.eval_set == setname) & (FINAL_DF.model == name)
                         & (FINAL_DF.horizon == f"h{h}")]
            ys.append(g["mae"].mean() if len(g) else np.nan)
        ax.plot(list(HORIZONS), ys, marker="o", ms=4,
                lw=2.2 if name == "sigmanet" else 1.2,
                ls="-" if name == "sigmanet" else "--",
                label=MODEL_LABEL[name], zorder=5 if name == "sigmanet" else 2)
    ax.set_xlabel("horizon (days ahead)")
    ax.set_xticks(list(HORIZONS))
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel("MAE (AQI points)")
axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save_fig(fig, "fig3_mae_by_horizon")

# %%
# ---- Figure: hazardous-day detection, precision-recall ---------------------
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, setname, title in zip(
        axes, ("test_seen", "test_unseen"), ("Seen cities", "Unseen cities")):
    idx = EVAL_SETS[setname]
    y, m = SAMPLES["y"][idx], SAMPLES["ym"][idx] > 0
    for name in ALL_MODELS:
        p = clip_aqi(ENSEMBLE_PRED[(name, setname)])
        yt = (y[m] > HAZARD_THRESHOLD).astype(int)
        ys = p[m]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        pr, rc, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ax.plot(rc, pr, lw=2.2 if name == "sigmanet" else 1.1,
                ls="-" if name == "sigmanet" else "--",
                label=f"{MODEL_LABEL[name]} (AP={ap:.3f})",
                zorder=5 if name == "sigmanet" else 2)
    base = (y[m] > HAZARD_THRESHOLD).mean()
    ax.axhline(base, color="k", ls=":", lw=1)
    ax.text(0.02, base + 0.02, f"base rate {base:.3f}", fontsize=7)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{title}: days above AQI {HAZARD_THRESHOLD:.0f}", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(0, 1.02)
save_fig(fig, "fig4_hazard_pr")

# %%
# ---- Figure: an actual episode, traced ------------------------------------
def episode_panel(city, setname, horizon=1):
    idx = EVAL_SETS[setname]
    cid = CITY_IDX.get(city)
    if cid is None:
        return None
    sel = SAMPLES["city"][idx] == cid
    if sel.sum() < 30:
        return None
    hi = list(HORIZONS).index(horizon)
    # ENSEMBLE_PRED rows are aligned to the WHOLE split index, so the city filter has
    # to be applied to the predictions in exactly the same order: select, then sort.
    order = np.argsort(SAMPLES["t"][idx[sel]].to_numpy())
    sub = idx[sel][order]
    dates = (SAMPLES["t"][sub] + pd.Timedelta(days=horizon)).to_numpy()
    mask = SAMPLES["ym"][sub][:, hi] > 0
    truth = SAMPLES["y"][sub][:, hi]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(dates[mask], truth[mask], color="k", lw=1.8, label="observed", zorder=6)
    for name in ["persistence", "lgbm", "itransformer", "sigmanet"]:
        if (name, setname) not in ENSEMBLE_PRED:
            continue
        p = clip_aqi(ENSEMBLE_PRED[(name, setname)])[sel][order][:, hi]
        ax.plot(dates[mask], p[mask], lw=1.9 if name == "sigmanet" else 1.0,
                ls="-" if name == "sigmanet" else "--", alpha=0.95,
                label=MODEL_LABEL[name], zorder=5 if name == "sigmanet" else 2)
    ax.axhline(HAZARD_THRESHOLD, color="r", ls=":", lw=1)
    ax.set_ylabel("AQI")
    ax.set_title(f"{city} - {horizon}-day-ahead forecast, {setname}", fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc="upper center")
    return fig


for _city, _set in [("Delhi", "test_seen"), ("Patna", "test_unseen"),
                    ("Lucknow", "test_seen"), ("Guwahati", "test_unseen")]:
    _f = episode_panel(_city, _set)
    if _f is not None:
        save_fig(_f, f"fig5_trace_{_city.lower()}")

# %%
# ---- Figure: ablation ------------------------------------------------------
if len(ABL_DF):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    nice = {"sigmanet_full": "full", "sigmanet_no_dsi": "-DSI",
            "sigmanet_no_vgm": "-VGM", "sigmanet_no_efa": "-EFA",
            "sigmanet_plain": "plain"}
    order = list(ABLATIONS.keys())
    for ax, metric, lab in zip(axes, ("mae", "hazard_recall", "cat_qwk"),
                               ("MAE (lower better)", "hazardous-day recall", "category QWK")):
        w = 0.38
        xs = np.arange(len(order))
        for off, setname, col in ((-w / 2, "test_seen", "#2c7fb8"),
                                  (w / 2, "test_unseen", "#c0392b")):
            g = ABL_DF[ABL_DF.eval_set == setname]
            mu = [g[g.variant == v][metric].mean() for v in order]
            sd = [g[g.variant == v][metric].std(ddof=0) for v in order]
            ax.bar(xs + off, mu, width=w, yerr=sd, capsize=2.5, label=setname, color=col,
                   alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels([nice[v] for v in order], rotation=30, fontsize=7.5)
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7)
    save_fig(fig, "fig6_ablation")

# %%
# ---- Figure: what the physics head says is driving the AQI ----------------
def dsi_attribution(setname="test_unseen", seed=SEED):
    """Loads the trained SIGMA-Net and reads its sub-index decomposition. This is the
    interpretability payoff of the physics head: it names the responsible pollutant."""
    ck = DIRS["ckpt"] / f"{_safe_key('final__sigmanet__seed' + str(seed) + '__ck')}.pt"
    if not ck.exists():
        print("  no SIGMA-Net checkpoint found; skipping the attribution figure")
        return None
    hp = dict(BEST_HP["sigmanet"])
    arch_hp, _ = split_hp(hp)
    model = build_model("sigmanet", arch_hp).to(DEVICE)
    try:
        state = torch.load(ck, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ck, map_location=DEVICE)
    model.load_state_dict(state["best_state"] or state["model"])
    model.eval()
    if not getattr(model, "use_dsi", False):
        return None

    idx = EVAL_SETS[setname]
    ti = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
    subs, aqis = [], []
    with torch.no_grad():
        for i in range(0, len(ti), 1024):
            b = ti[i: i + 1024]
            aqi, zpol = model(T_X[b], T_COV[b], T_ARCH[b])
            conc = model.decoder(zpol)
            _, sub = model.dsi(conc)
            subs.append(sub.float().cpu().numpy())
            aqis.append(aqi.float().cpu().numpy())
    sub = np.concatenate(subs)           # (n, H, P)
    pred_driver = sub[:, 0, :].argmax(axis=1)

    _, true_driver_all, _ = naqi_np(raw)
    key = {(c, pd.Timestamp(d)): i for i, (c, d)
           in enumerate(zip(raw["city"].to_numpy(), raw["date"].to_numpy()))}
    tgt_dates = SAMPLES["t"][idx] + pd.Timedelta(days=int(HORIZONS[0]))
    true_driver = np.full(len(idx), -1)
    for j, (cid, dt) in enumerate(zip(SAMPLES["city"][idx], tgt_dates)):
        row = key.get((CITIES[cid], pd.Timestamp(dt)))
        if row is not None:                       # never silently fall back to row 0
            true_driver[j] = true_driver_all[row]

    ok = true_driver >= 0
    names = POLLUTANTS_AQI
    cm = np.zeros((len(names), len(names)), dtype=float)
    for t_, p_ in zip(true_driver[ok], pred_driver[ok]):
        cm[t_, p_] += 1
    cm_n = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    agree = float((true_driver[ok] == pred_driver[ok]).mean())

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("pollutant SIGMA-Net blames"); ax.set_ylabel("pollutant actually responsible")
    ax.set_title(f"Driving-pollutant attribution, {setname}\nagreement {agree*100:.1f}%",
                 fontsize=9)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm_n[i, j] > 0.01:
                ax.text(j, i, f"{cm_n[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if cm_n[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_fig(fig, f"fig7_driver_attribution_{setname}")
    save_result(f"dsi_attribution__{setname}",
                dict(agreement=agree, n=int(ok.sum()),
                     learned_tau=float(model.dsi.tau.detach().cpu())))
    print(f"  driving-pollutant agreement on {setname}: {agree*100:.1f}%  "
          f"(learned tau = {float(model.dsi.tau.detach().cpu()):.2f})")
    del model
    gc.collect()
    return agree


for _s in ("test_seen", "test_unseen"):
    dsi_attribution(_s)

# %%
# ---- Figure: LightGBM feature importance (a sanity check on the features) --
def lgbm_importance_figure():
    key = f"final__lgbm__seed{SEED}"
    if not is_done(key):
        return
    fit = train_lgbm(BEST_HP["lgbm"], IDX["train"], IDX["val"])
    b = fit["boosters"][0]
    if b is None:
        return
    imp = pd.Series(b.feature_importances_, index=LGBM_NAMES).sort_values(ascending=False)
    save_table(imp.head(40).to_frame("split_importance"), "table_lgbm_importance",
               float_fmt="%.0f")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    top = imp.head(25)[::-1]
    ax.barh(range(len(top)), top.to_numpy(), color="#2c7fb8")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=6.5)
    ax.set_xlabel("LightGBM split importance")
    ax.set_title("Most used features, 1-day horizon", fontsize=9)
    save_fig(fig, "fig_lgbm_importance")


lgbm_importance_figure()

# %% [markdown]
# ## 16 - Paper bundle and run status

# %%
def write_paper_summary():
    lines = []
    A = lines.append
    A("# SIGMA-Net run summary")
    A("")
    A(f"- profile: `{PROFILE}`   generated: {datetime.now().isoformat(timespec='seconds')}")
    A(f"- device: {DEVICE}   torch {torch.__version__}")
    A(f"- cities: {len(CITIES)} ({len(TRAIN_CITIES)} training, {len(UNSEEN_CITIES)} held out)")
    A(f"- samples: train {len(IDX['train']):,} / val {len(IDX['val']):,} / "
      f"test_seen {len(IDX['test_seen']):,} / test_unseen {len(IDX['test_unseen']):,}")
    A(f"- lookback {LOOKBACK} d, horizons {list(HORIZONS)}, embargo {CFG['embargo']} d")
    A(f"- strictly causal features: {not CFG['perfect_met']}")
    A("")
    A("## CPCB operator fidelity (justifies the physics head)")
    A(f"- reproduces the released AQI to MAE **{DSI_FIDELITY['mae']:.3f}** points on "
      f"{DSI_FIDELITY['n']:,} city-days; {DSI_FIDELITY['within_2']*100:.2f}% within +/-2")
    A("")
    A("## Headline, pooled over horizons")
    for setname, label in (("test_seen", "Seen cities, 2025"),
                           ("test_unseen", "Unseen cities, 2025")):
        A("")
        A(f"### {label}")
        A("")
        A("| model | MAE | RMSE | hazard recall | hazard F1 | QWK |")
        A("|---|---|---|---|---|---|")
        for name in ALL_MODELS:
            g = FINAL_DF[(FINAL_DF.eval_set == setname) & (FINAL_DF.model == name)
                         & (FINAL_DF.horizon == "pooled")]
            if not len(g):
                continue
            A(f"| {MODEL_LABEL[name]} | {g['mae'].mean():.2f} | {g['rmse'].mean():.2f} "
              f"| {g['hazard_recall'].mean():.3f} | {g['hazard_f1'].mean():.3f} "
              f"| {g['cat_qwk'].mean():.3f} |")
    if len(ABL_DF):
        A("")
        A("## Ablation (pooled MAE)")
        A("")
        A("| variant | test_seen | test_unseen |")
        A("|---|---|---|")
        for v in ABLATIONS:
            a = ABL_DF[(ABL_DF.variant == v) & (ABL_DF.eval_set == "test_seen")]["mae"].mean()
            b = ABL_DF[(ABL_DF.variant == v) & (ABL_DF.eval_set == "test_unseen")]["mae"].mean()
            A(f"| {v} | {a:.2f} | {b:.2f} |")
    A("")
    A("## Files")
    A(f"- tables: `{DIRS['tables']}`")
    A(f"- figures: `{DIRS['figures']}`")
    A("")
    A("## Attribution required at submission")
    A("- CPCB pollutant data: download is permitted; other use needs approval "
      "(`ccb.cpcb@nic.in`). Request before camera-ready.")
    A("- ERA5 / Copernicus C3S (CC-BY), Open-Meteo (CC-BY-4.0), NASA FIRMS - cite all three.")
    txt = "\n".join(lines)
    _atomic_write_text(RUN_DIR / "RUN_SUMMARY.md", txt)
    print(txt)


write_paper_summary()

# %%
# ---- what is finished, what is left --------------------------------------
def status_report():
    expected = []
    expected += [f"hpo__{m}" for m in DEEP_MODELS] + ["hpo__lgbm"]
    expected += [f"cv__{m}__fold{f['fold']}" for m in ALL_MODELS for f in FOLDS]
    for m in ALL_MODELS:
        seeds = [SEED] if m in ("persistence", "climatology") else \
            [SEED + i for i in range(CFG["n_seeds"])]
        expected += [f"final__{m}__seed{s}" for s in seeds]
    if CFG["do_ablations"]:
        expected += [f"abl__{t}__seed{SEED+i}" for t in ABLATIONS
                     for i in range(CFG["n_seeds"])]
    done = [k for k in expected if is_done(k)]
    todo = [k for k in expected if not is_done(k)]
    print(f"work units: {len(done)}/{len(expected)} complete")
    if todo:
        print("  outstanding:")
        for k in todo[:25]:
            print(f"    - {k}")
        if len(todo) > 25:
            print(f"    ... and {len(todo)-25} more")
        print("\n  Re-run the notebook top to bottom; finished units are skipped.")
    else:
        print("  Everything finished. Tables and figures are in the run directory.")
    ck = list(DIRS["ckpt"].glob("*.pt"))
    print(f"\ncheckpoints on disk: {len(ck)} "
          f"({sum(p.stat().st_size for p in ck)/1e6:.1f} MB)")
    print(f"run directory: {RUN_DIR}")
    return dict(done=len(done), total=len(expected), outstanding=todo)


STATUS = status_report()
save_result("run_status", STATUS)
print("\nDone.")

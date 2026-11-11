"""Build notebooks/08_final_model.ipynb.

The notebook is generated rather than written by hand so that every code cell is
compiled before it reaches Colab. Run from anywhere: python tools/build_nb08.py
"""
import io
import json
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")
PROJECT = Path(__file__).resolve().parents[1]

nb04 = json.load(io.open(PROJECT / "notebooks" / "04_itransformer.ipynb", encoding="utf-8"))
code04 = [c for c in nb04["cells"] if c["cell_type"] == "code"]
secrets_cell = "".join(next(c for c in code04 if "userdata" in "".join(c["source"]))["source"])

md = nbformat.v4.new_markdown_cell
py = nbformat.v4.new_code_cell

SETTINGS = r'''# Leave empty to read from the HF_DATASET_REPO Colab secret.
HF_DATASET_REPO = ""
GITHUB_REPO_URL = "https://github.com/Aduitya9/DEXTRA.git"

SEEDS = [42, 43, 44]               # notebooks 06 and 07's, so every model has the same three seeds

# Architecture and training: the all-token head of notebooks 06 and 07, unchanged.
D_MODEL, N_HEADS, N_LAYERS, DROPOUT = 128, 8, 2, 0.3
EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE = 30, 256, 3e-4, 1e-3, 10

LOCO_CITIES = None                 # None holds out all fifteen
BOOTSTRAP_DRAWS = 1000
NOISE_DRAWS = 200

# Safety
TIME_BUDGET_HOURS = 10.5

SAVE_TO_HF = True
'''

SETUP = r'''import hashlib
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_DIR = "/content/DEXTRA"
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", GITHUB_REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd
import torch
from huggingface_hub.utils import disable_progress_bars
from src import config
from src.data import assemble, evening, hourly
from src.evaluation import alarms, gap, noise
from src.models import deep_baselines, itransformer, multires, sequences, training
from src.utils.hf_store import HFStore

disable_progress_bars()
device = itransformer.pick_device()
store = HFStore(HF_DATASET_REPO, token=HF_TOKEN)
needed = ["processed/city_daily_features.parquet", "processed/evening_features.parquet",
          "processed/city_hourly.parquet", "processed/evening_loco_forecasts.parquet",
          "processed/gbm_hourly_loco_forecasts.parquet", "processed/noise_samples.parquet"]
missing = [p for p in needed if not store.exists(p)]
if missing:
    raise RuntimeError(f"Missing from the dataset repo: {missing}. The hourly tables and noise samples come from "
                       "scripts/build_hourly_tables.py --upload, the boosting forecasts from scripts/gbm_hourly_baseline.py --upload.")


def by_day(frame):
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame


ANCHOR_COLUMNS = ["anchor1", "anchor2", "anchor3"]
# What a model may read: the 26 daily variates; those and the evening; those and the nightly anchors.
FEATURES = {"daily": list(config.ALL_FEATURES)}
FEATURES["evening"] = FEATURES["daily"] + evening.FEATURES
FEATURES["final"] = FEATURES["evening"] + ANCHOR_COLUMNS
KEYS = ["city", "date", "horizon"]
LINE = config.RARE_AQI_THRESHOLD

hourly_table = store.read_df("processed/city_hourly.parquet")
hourly_table["timestamp"] = pd.to_datetime(hourly_table["timestamp"])
daily = by_day(store.read_df("processed/city_daily_features.parquet"))
daily = daily.merge(by_day(store.read_df("processed/evening_features.parquet")), on=["city", "date"], how="left")
daily = hourly.add_night_anchors(daily, hourly_table)

# 2026 is sealed until the model is frozen: see PREREGISTRATION.md.
SEAL = pd.Timestamp("2026-01-01")
if daily.date.max() >= SEAL or hourly_table.timestamp.max() >= SEAL:
    raise RuntimeError("2026 data reached this notebook. It is sealed for the final test.")

# The 12-hour ceiling counts from when this VM booted, not from this cell.
try:
    session_age = float(open("/proc/uptime").read().split()[0])
except (OSError, ValueError, IndexError):
    session_age = 0.0
if not 0.0 <= session_age < 12 * 3600:
    session_age = 0.0
deadline = time.monotonic() + max(0.0, TIME_BUDGET_HOURS * 3600 - session_age)

registry = training.RunRegistry(store, "runs/08")
cities = sorted(daily.city.unique()) if LOCO_CITIES is None else list(LOCO_CITIES)
models_run = lambda city: "loco-" + city.replace(" ", "_") + "-models"
mechanism_run = lambda city: "loco-" + city.replace(" ", "_") + "-mechanism"
baseline_run = lambda city: "loco-" + city.replace(" ", "_") + "-baselines"
planned = [run(c) for run in (models_run, mechanism_run, baseline_run) for c in cities]
finished = [name for name in planned if registry.done(name)]
seeded = lambda name: [f"{name}_seed{s}" for s in SEEDS]

print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
if device.type != "cuda":
    print("! No GPU: this will take many hours. Runtime -> Change runtime type -> T4 GPU, then Run all.")
print(f"{len(daily):,} city-days, {daily.city.nunique()} cities, {len(hourly_table):,} city-hours, data to {daily.date.max().date()}")
print(f"{len(finished)} of {len(planned)} pieces already finished" + (" -- resuming where the last session stopped" if finished else ""))
print(f"time budget left this session: {(deadline - time.monotonic()) / 3600:.1f} h")
'''

MODELS = r'''LOOKBACK, HORIZON = config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
variates = {kind: sequences.feature_names(daily, columns) for kind, columns in FEATURES.items()}
target = {kind: names.index(sequences.TARGET) for kind, names in variates.items()}
anchor_indices = [variates["final"].index(c) for c in ANCHOR_COLUMNS]
n_hourly = len(multires.HOURLY_CHANNELS)
size = dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT)


def all_token(kind, anchors=False, hours=False, mixture=multires.MIXTURE):
    """The all-token head on one feature set, with the anchored forecast and the 72-hour tokens optional."""
    return multires.FinalModel(len(variates[kind]), LOOKBACK, HORIZON, target[kind],
                               anchor_indices=anchor_indices if anchors else None, mixture=mixture,
                               n_channels=n_hourly if hours else 0, **size)


# name -> (the feature set it reads, whether it also reads the 72 hours, how it is built)
BUILDERS = {
    "backbone": ("daily", False, lambda: itransformer.ITransformer(len(variates["daily"]), LOOKBACK, HORIZON, target["daily"], **size)),
    "head": ("daily", False, lambda: all_token("daily")),
    "control": ("evening", False, lambda: all_token("evening")),
    "anchor_inputs": ("final", False, lambda: all_token("final")),
    "anchor_token": ("final", False, lambda: all_token("final", anchors=True, mixture=("mean",))),
    "mean_today": ("final", False, lambda: all_token("final", anchors=True, mixture=("mean", "today"))),
    "anchored": ("final", False, lambda: all_token("final", anchors=True)),
    "multires": ("evening", True, lambda: all_token("evening", hours=True)),
    "dextra": ("final", True, lambda: all_token("final", anchors=True, hours=True)),
    "lstm": ("evening", False, lambda: deep_baselines.LSTMForecaster(len(variates["evening"]), HORIZON, target["evening"], dropout=DROPOUT)),
    "dlinear": ("daily", False, lambda: deep_baselines.DLinear(LOOKBACK, HORIZON, target["daily"])),
    "patchtst": ("daily", False, lambda: deep_baselines.PatchTST(LOOKBACK, HORIZON, target["daily"], dropout=DROPOUT)),
}
MODEL_FAMILY = ["backbone", "head", "control", "anchor_inputs", "anchored", "multires", "dextra"]
MECHANISM_FAMILY = ["anchor_token", "mean_today"]                               # the rungs between anchor_inputs and anchored
BASELINE_FAMILY = ["lstm", "dlinear", "patchtst"]
CANDIDATES = ["control", "anchor_inputs", "anchored", "multires", "dextra"]     # one of these becomes the final model
KEEP_WEIGHTS = ["head"] + CANDIDATES                                            # frozen for the final test
ANCHORED_MODELS = {"anchored", "dextra"}

rows = [{"model": name, "reads": kind + (" + 72 hours" if hours else ""), "parameters": itransformer.count_parameters(build())}
        for name, (kind, hours, build) in BUILDERS.items()]
display(pd.DataFrame(rows).set_index("model"))
print(f"Variates: {', '.join(f'{kind} {len(names)}' for kind, names in variates.items())}")
print(f"Share of hourly values present: {hourly_table[multires.HOURLY_CHANNELS].notna().mean().round(3).to_dict()}")
sample = daily.dropna(subset=["AQI", "anchor1"])
print(f"Nightly anchor on every city-day, against the next day's AQI: "
      f"correlation {sample.anchor1.corr(sample.groupby('city').AQI.shift(-1)):.3f}, today's AQI {sample.AQI.corr(sample.groupby('city').AQI.shift(-1)):.3f}")


def save_run(name, result, frames=None, weights=None):
    """Record a finished piece with its forecasts (and its weights, as one file) in one commit."""
    if not SAVE_TO_HF:
        return
    with tempfile.TemporaryDirectory() as tmp:
        attachments = {}
        for filename, frame in (frames or {}).items():
            attachments[filename] = Path(tmp) / filename
            frame.to_parquet(attachments[filename], index=False)
        if weights:
            attachments[f"{name}.pt"] = Path(tmp) / f"{name}.pt"
            torch.save(weights, attachments[f"{name}.pt"])
            # The fingerprint the frozen model is later checked against.
            result = {**result, "weights_sha256": hashlib.sha256(attachments[f"{name}.pt"].read_bytes()).hexdigest()}
        registry.record(name, result, attachments=attachments)


def load_frame(name):
    return by_day(pd.read_parquet(io.BytesIO(store.read_bytes(registry.attachment(f"{name}.parquet")))))


def stop_here(message="time budget reached"):
    print(f"⏸ {message} -- Runtime -> Disconnect and delete runtime, then Run all")
    return True


def best_epoch(history):
    return min(history, key=lambda h: h["val_rmse"])["epoch"] if history else 0


def fold_windows(city, names):
    """What a fold's models read, cut once per feature set, with the 72 hours beside it where a model wants them."""
    labelled = daily.assign(split=assemble.leave_one_out_split(daily, city))
    windows = {}
    for kind in dict.fromkeys(BUILDERS[name][0] for name in names):
        X, y, index = sequences.build_sequences(labelled, features=FEATURES[kind])
        windows[kind] = {"X": X, "y": y, "index": index}
        if any(BUILDERS[name][1] for name in names if BUILDERS[name][0] == kind):
            windows[kind]["packed"] = multires.pack(X, multires.attach_hourly(index, hourly_table))
    return windows


def run_family(names, windows, keep_weights=()):
    """Train each model with every seed. Returns (frame, runs, weights); the frame is None if time ran out."""
    runs, weights, frames = [], {}, []
    for kind, cut in windows.items():
        X, y, index = cut["X"], cut["y"], cut["index"]
        train_rows, val_rows, test_rows = ((index.split == s).to_numpy() for s in ("train", "val", "test_loco"))
        keep = val_rows | test_rows
        part = {}
        for name in [n for n in names if BUILDERS[n][0] == kind]:
            inputs = cut["packed"] if BUILDERS[name][1] else X
            trust = []
            val_loader = itransformer.make_loader(inputs[val_rows], y[val_rows], BATCH_SIZE * 2)
            for seed in SEEDS:
                itransformer.set_seed(seed)
                model, history, outcome = training.train(
                    BUILDERS[name][2](), itransformer.make_loader(inputs[train_rows], y[train_rows], BATCH_SIZE, shuffle=True),
                    val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                    deadline=deadline, save_finished=False, log=lambda *a: None)
                if outcome == "out_of_time":
                    return None, None, None
                part[f"{name}_seed{seed}"] = itransformer.predict(model, inputs[keep], device)
                if name in ANCHORED_MODELS:
                    trust.append(model.anchor_weights(inputs[keep], device))
                runs.append({"model": name, "seed": seed, "outcome": outcome, "best_epoch": best_epoch(history),
                             "epochs_run": len(history), "val_rmse": min(h["val_rmse"] for h in history)})
                if name in keep_weights:
                    weights[f"{name}_seed{seed}"] = {k: v.cpu() for k, v in model.state_dict().items()}
            if trust:
                part[f"{name}-weight-today"] = np.mean(trust, axis=0)[:, :, 1]
                part[f"{name}-weight-night"] = np.mean(trust, axis=0)[:, :, 2]
        frames.append(gap.long_frame(index[keep], y[keep], X[keep, -1, target[kind]], part))
    frame = frames[0]
    for other in frames[1:]:
        frame = frame.merge(other.drop(columns=["split", "y", "persistence"]), on=KEYS, how="inner")
    return frame, runs, weights
'''

MODEL_LOOP = r'''stopped = False
for city in cities:
    run = models_run(city)
    if registry.done(run):
        print(f"✓ {run} already finished")
        continue
    if stopped or time.monotonic() > deadline:
        stopped = stop_here()
        break
    print(f"\n▶ {run}")
    started = time.monotonic()
    frame, runs, weights = run_family(MODEL_FAMILY, fold_windows(city, MODEL_FAMILY), keep_weights=KEEP_WEIGHTS)
    if frame is None:
        stopped = stop_here("time budget reached mid-fold; this fold starts again next session")
        break
    minutes = round((time.monotonic() - started) / 60, 2)
    save_run(run, {"city": city, "runs": runs, "windows": len(frame) // HORIZON, "minutes": minutes},
             frames={f"{run}.parquet": frame}, weights=weights)
    summary = pd.DataFrame(runs).groupby("model").val_rmse.mean().round(2).to_dict()
    print(f"  validation RMSE {summary};  ↑ recorded, {minutes} min")
'''

MECHANISM_LOOP = r'''for city in cities:
    run = mechanism_run(city)
    if registry.done(run):
        print(f"✓ {run} already finished")
        continue
    if stopped or time.monotonic() > deadline:
        stopped = stop_here()
        break
    print(f"\n▶ {run}")
    started = time.monotonic()
    frame, runs, _ = run_family(MECHANISM_FAMILY, fold_windows(city, MECHANISM_FAMILY))
    if frame is None:
        stopped = stop_here("time budget reached mid-fold; this fold starts again next session")
        break
    minutes = round((time.monotonic() - started) / 60, 2)
    save_run(run, {"city": city, "runs": runs, "windows": len(frame) // HORIZON, "minutes": minutes},
             frames={f"{run}.parquet": frame})
    summary = pd.DataFrame(runs).groupby("model").val_rmse.mean().round(2).to_dict()
    print(f"  validation RMSE {summary};  ↑ recorded, {minutes} min")
'''

BASELINE_LOOP = r'''for city in cities:
    run = baseline_run(city)
    if registry.done(run):
        print(f"✓ {run} already finished")
        continue
    if stopped or time.monotonic() > deadline:
        stopped = stop_here()
        break
    print(f"\n▶ {run}")
    started = time.monotonic()
    frame, runs, _ = run_family(BASELINE_FAMILY, fold_windows(city, BASELINE_FAMILY))
    if frame is None:
        stopped = stop_here("time budget reached mid-fold; this fold starts again next session")
        break
    minutes = round((time.monotonic() - started) / 60, 2)
    save_run(run, {"city": city, "runs": runs, "windows": len(frame) // HORIZON, "minutes": minutes},
             frames={f"{run}.parquet": frame})
    print(f"  ↑ recorded, {minutes} min")
'''

GATHER = r'''before = by_day(store.read_df("processed/evening_loco_forecasts.parquet"))
gbm_hourly = by_day(store.read_df("processed/gbm_hourly_loco_forecasts.parquet"))
# From notebooks 06 and 07, on the same rows: the forecasters that are not retrained here,
# and the two that are (the head, and the head with the evening), kept as a check on this notebook.
SAVED = ["gbm", "gbm evening", "chronos-bolt median", "chronos-2 median"] + seeded("plain") + seeded("evening")
night_long = daily.melt(id_vars=["city", "date"], value_vars=ANCHOR_COLUMNS, value_name="nightly anchor")
night_long["horizon"] = night_long["variable"].str[-1].astype(int)
frames = {}
for city in cities:
    if not registry.done(models_run(city)):
        continue
    frame = load_frame(models_run(city))
    frame = frame.merge(before.loc[before.fold == city, KEYS + SAVED], on=KEYS, how="inner")
    frame = frame.merge(gbm_hourly.loc[gbm_hourly.fold == city, KEYS + ["gbm hourly"]], on=KEYS, how="left")
    frame = frame.merge(night_long[KEYS + ["nightly anchor"]], on=KEYS, how="left")
    for extra in (mechanism_run(city), baseline_run(city)):
        if registry.done(extra):
            frame = frame.merge(load_frame(extra).drop(columns=["split", "y", "persistence"]), on=KEYS, how="left")
    frames[city] = frame
print(f"{len(frames)} of {len(cities)} folds have their models")

FINAL_NAME, choice = "control", pd.DataFrame()
if frames:
    rows = []
    for name in CANDIDATES:
        scores = [(gap.rmse(f[f.split == "val"], seeded(name)), gap.average_precision(f[f.split == "val"], seeded(name)))
                  for f in frames.values()]
        rows.append({"model": name, "validation RMSE": np.mean([s[0] for s in scores]), "validation AP": np.nanmean([s[1] for s in scores])})
    choice = pd.DataFrame(rows).set_index("model")
    display(choice.round(3))
    allowed = choice[choice["validation AP"] >= choice.loc["control", "validation AP"] - 0.002]
    FINAL_NAME = allowed["validation RMSE"].idxmin()
    note = "" if len(frames) == len(cities) else "  (provisional: not every fold has finished)"
    print(f"The rule picks: {FINAL_NAME}{note}")
FINAL_COLUMNS = seeded(FINAL_NAME)
for frame in frames.values():
    frame["final"] = frame[FINAL_COLUMNS].mean(axis=1)
'''

TABLE = r'''LINEUP = [("Persistence", ["persistence"]),
          ("Nightly anchor, no training", ["nightly anchor"]),
          ("Gradient boosting", ["gbm"]),
          ("Gradient boosting + evening", ["gbm evening"]),
          ("Gradient boosting + hourly shape", ["gbm hourly"]),
          ("DLinear", seeded("dlinear")),
          ("PatchTST", seeded("patchtst")),
          ("LSTM + evening", seeded("lstm")),
          ("Chronos-Bolt, zero-shot", ["chronos-bolt median"]),
          ("Chronos-2, zero-shot", ["chronos-2 median"]),
          ("iTransformer", seeded("backbone")),
          ("All-token head", seeded("head")),
          ("+ evening (control)", seeded("control")),
          ("+ anchors as inputs only", seeded("anchor_inputs")),
          ("+ anchor token, mean start (RevIN)", seeded("anchor_token")),
          ("+ learned mix of mean and today", seeded("mean_today")),
          ("+ anchored forecast", seeded("anchored")),
          ("+ 72-hour tokens", seeded("multires")),
          ("+ anchored forecast and 72-hour tokens", seeded("dextra"))]
REPLICATED = [("All-token head", seeded("head"), seeded("plain")), ("+ evening (control)", seeded("control"), seeded("evening"))]
available = [(n, c) for n, c in LINEUP if frames and all(all(x in f for x in c) for f in frames.values())]
compared = list(dict.fromkeys([c for _, cols in available for c in cols] + SAVED))
loco_frames = {c: f.dropna(subset=compared) for c, f in frames.items()}
tested = (pd.concat([f[f.split == "test_loco"] for f in loco_frames.values()], ignore_index=True) if frames else pd.DataFrame())
onsets = lambda frame: frame[frame.persistence <= LINE]


def onset_hits(column):
    hits = 0
    for f in loco_frames.values():
        level = alarms.tune_alarm(f.loc[f.split == "val", "y"], f.loc[f.split == "val", column])
        test = onsets(f[f.split == "test_loco"])
        hits += int(((test.y > LINE) & (test[column] > level)).sum())
    return hits


table, resamples = pd.DataFrame(), {}
if len(tested):
    next_day = tested[tested.horizon == 1]
    target_year = (tested.date + pd.to_timedelta(tested.horizon, unit="D")).dt.year
    # Every forecaster scored once on the same resamples of whole held-out cities; sections F and G read from these.
    started = time.monotonic()
    for label, part in (("all", tested), ("onsets", onsets(tested)), ("next day", next_day), ("next-day onsets", onsets(next_day))):
        resamples[label] = gap.Resamples(part, compared, draws=BOOTSTRAP_DRAWS)
    rows = []
    for name, columns in available:
        rows.append({"forecaster": name,
                     "AP": resamples["all"].value(columns),
                     "AP, no Delhi": gap.average_precision(tested[tested.city != "Delhi"], columns),
                     "AP, 2025 only": gap.average_precision(tested[target_year == 2025], columns),
                     "onset AP": resamples["onsets"].value(columns),
                     "next-day AP": resamples["next day"].value(columns),
                     "next-day onset AP": resamples["next-day onsets"].value(columns),
                     "RMSE": resamples["all"].value(columns, "rmse"),
                     "next-day RMSE": resamples["next day"].value(columns, "rmse"),
                     "alarm F1": float(np.mean([gap.score_folds(loco_frames, c)[0]["f1"] for c in columns])),
                     "onsets caught": float(np.mean([onset_hits(c) for c in columns]))})
    table = pd.DataFrame(rows).set_index("forecaster")
    print(f"Held-out cities, 2024-25: {len(tested):,} rows, {int((tested.y > LINE).sum())} severe targets, "
          f"{int((onsets(tested).y > LINE).sum())} of them onsets  (scored in {(time.monotonic() - started) / 60:.1f} min)")
    display(table.round(3))
    for name, here, there in REPLICATED:
        print(f"Check, {name}: here AP {resamples['all'].value(here):.3f}, RMSE {resamples['all'].value(here, 'rmse'):.2f}; "
              f"as trained in notebooks 06-07, AP {resamples['all'].value(there):.3f}, RMSE {resamples['all'].value(there, 'rmse'):.2f}")
else:
    print("No fold has finished yet. Run all again to continue.")
'''

ABLATION = r'''def compare(rows_label, with_columns, without_columns):
    """AP, onset AP and RMSE of one forecaster minus another, on the resamples section E drew."""
    onset_label = "onsets" if rows_label == "all" else "next-day onsets"
    d, low, high, p = resamples[rows_label].difference(with_columns, without_columns)
    od, olow, ohigh, op = resamples[onset_label].difference(with_columns, without_columns)
    r, rlow, rhigh, rp = resamples[rows_label].difference(with_columns, without_columns, "rmse")
    return {"AP difference": d, "AP 95%": f"[{low:+.3f}, {high:+.3f}]", "AP p": p,
            "onset AP difference": od, "onset 95%": f"[{olow:+.3f}, {ohigh:+.3f}]", "onset p": op,
            "RMSE difference": r, "RMSE 95%": f"[{rlow:+.3f}, {rhigh:+.3f}]", "RMSE p": rp}


def with_holm(table):
    for column in ("AP p", "onset p", "RMSE p"):
        table.insert(table.columns.get_loc(column) + 1, column + " (Holm)", gap.holm(table[column]))
    return table


STEPS = [("nightly anchor, no training: against persistence", ["nightly anchor"], ["persistence"]),
         ("evening features: against the head", seeded("control"), seeded("head")),
         ("anchors as inputs only: against control", seeded("anchor_inputs"), seeded("control")),
         ("anchor token, mean start: against anchors as inputs only", seeded("anchor_token"), seeded("anchor_inputs")),
         ("learned mix of mean and today: against mean start", seeded("mean_today"), seeded("anchor_token")),
         ("nightly anchor in the mix: against mean and today", seeded("anchored"), seeded("mean_today")),
         ("anchored forecast: against anchors as inputs only", seeded("anchored"), seeded("anchor_inputs")),
         ("anchored forecast: against control", seeded("anchored"), seeded("control")),
         ("72-hour tokens: against control", seeded("multires"), seeded("control")),
         ("both: against control", seeded("dextra"), seeded("control")),
         ("both: against the anchored forecast alone", seeded("dextra"), seeded("anchored")),
         ("both: against the 72-hour tokens alone", seeded("dextra"), seeded("multires"))]
ablation = {}
if len(tested):
    for label in ("all", "next day"):
        # A step whose models have not finished in every fold is left out, not guessed.
        rows = [{"step": step, **compare(label, with_, without)} for step, with_, without in STEPS
                if all(c in compared for c in with_ + without)]
        ablation[label] = with_holm(pd.DataFrame(rows).set_index("step"))
        print(f"\nWhat each part adds, {'all three horizons' if label == 'all' else 'next day only'} "
              "(first minus second; positive AP and negative RMSE are improvements):")
        display(ablation[label].round(4))
'''

VERDICT = r'''verdict = pd.DataFrame()
if len(tested):
    rows = [{"final against": name, **compare("all", FINAL_COLUMNS, columns)} for name, columns in available if columns != FINAL_COLUMNS]
    verdict = with_holm(pd.DataFrame(rows).set_index("final against"))
    print(f"The final model ({FINAL_NAME}) against every other, all three horizons:")
    display(verdict.round(4))
'''

PER_CITY = r'''per_city = pd.DataFrame()
anchor_table = pd.DataFrame()
if len(tested):
    rows = []
    for city, part in tested.groupby("city"):
        severe = int((part.y > LINE).sum())
        if severe < 10:
            continue
        row = {"city": city, "severe": severe, "onsets": int((onsets(part).y > LINE).sum())}
        for label, columns in (("persistence", ["persistence"]), ("nightly anchor", ["nightly anchor"]), ("gbm hourly", ["gbm hourly"]),
                               ("chronos-2", ["chronos-2 median"]), ("control", seeded("control")), ("final", FINAL_COLUMNS)):
            row[f"AP {label}"] = gap.average_precision(part, columns)
        rows.append(row)
    per_city = pd.DataFrame(rows).set_index("city").sort_values("severe", ascending=False)
    print("Per held-out city with at least ten severe targets:")
    display(per_city.round(3))
    rows = []
    for name in ("anchored", "dextra"):
        today_column, night_column = f"{name}-weight-today", f"{name}-weight-night"
        for horizon, part in tested.groupby("horizon"):
            first = (part.y > LINE) & (part.persistence <= LINE)
            for label, chosen in (("every day", part.index == part.index), ("onsets", first), ("other days", ~first)):
                today, night = part.loc[chosen, today_column].mean(), part.loc[chosen, night_column].mean()
                rows.append({"model": name, "horizon": horizon, "days": label, "fortnight mean": 1 - today - night, "today": today, "nightly anchor": night})
    anchor_table = pd.DataFrame(rows).set_index(["model", "horizon", "days"])
    print("How much the forecast starts from each anchor, by horizon:")
    display(anchor_table.round(3))
'''

ALARM = r'''alarm_tables = {}
if len(tested):
    def alarm_table(horizons):
        counts = {}
        steps = []
        def add(rule, city, raised, truth):
            bad = truth > LINE
            counts.setdefault(rule, []).append((city, int((raised & bad).sum()), int(bad.sum()), int(raised.sum())))
        for city, frame in loco_frames.items():
            for h in horizons:
                part = frame[frame.horizon == h].sort_values(["city", "date"])
                val, test = part[part.split == "val"], part[part.split == "test_loco"]
                f, y = test["final"].to_numpy(), test["y"].to_numpy()
                tuned = alarms.tune_alarm(val.y, val["final"])
                start = alarms.recall_level(val.y, val["final"])
                step = alarms.choose_step(val.city.to_numpy(), val.y.to_numpy(), val["final"].to_numpy(), start, delay=int(h))
                steps.append(step)
                add("official line, 300", city, f > LINE, y)
                add("level tuned for F1 on other cities", city, f > tuned, y)
                add("level for 70% recall on other cities", city, f > start, y)
                add("city-adaptive, from that same level", city, alarms.adaptive_alarm(y, f, start, step, delay=int(h)), y)
        rows = []
        for rule, record in counts.items():
            c = pd.DataFrame(record, columns=["city", "hits", "severe", "raised"]).groupby("city").sum()
            active = c[c.severe >= 15]
            recall = active.hits / active.severe
            rows.append({"rule": rule, "recall": c.hits.sum() / c.severe.sum(), "false alarms": int((c.raised - c.hits).sum()),
                         "F1": 2 * c.hits.sum() / (c.severe.sum() + c.raised.sum()), "mean |recall - 0.7|": float((recall - 0.7).abs().mean()),
                         "lowest city recall": float(recall.min()), "highest city recall": float(recall.max()), "cities": len(active)})
        return pd.DataFrame(rows).set_index("rule"), float(np.median(steps))

    for label, horizons in (("next day", [1]), ("all horizons", [1, 2, 3])):
        alarm_tables[label], step = alarm_table(horizons)
        print(f"\n{label}, target recall 70%, median step chosen on validation {step:.0f}:")
        display(alarm_tables[label].round(3))
'''

NOISE = r'''noise_table = pd.DataFrame()
if len(tested):
    pool = store.read_df("processed/noise_samples.parquet")
    limit = noise.ceilings(next_day.y, next_day.persistence, pool, draws=NOISE_DRAWS)
    floor = limit["rmse_floor"]
    reference = np.mean((next_day.persistence - next_day.y) ** 2)
    rows = [{"forecaster": "noise ceiling (no forecaster can pass this)", "next-day RMSE": floor, "next-day AP": limit["ap_ceiling"],
             "next-day onset AP": limit["onset_ap_ceiling"], "share of squared error that is noise": 1.0, "share of reducible error removed": 1.0}]
    for name, columns in available:
        mse = np.mean([np.mean((next_day[c] - next_day.y) ** 2) for c in columns])
        rows.append({"forecaster": name, "next-day RMSE": float(np.sqrt(mse)), "next-day AP": resamples["next day"].value(columns),
                     "next-day onset AP": resamples["next-day onsets"].value(columns),
                     "share of squared error that is noise": floor ** 2 / mse,
                     "share of reducible error removed": noise.reducible_share(mse, reference, floor ** 2)})
    noise_table = pd.DataFrame(rows).set_index("forecaster")
    display(noise_table.round(3))
    if store.exists("processed/noise_by_level.csv"):
        print(store.read_df("processed/noise_by_level.csv").round(1).to_string(index=False))
    first = onsets(next_day)
    first = first[first.y > LINE]
    print(f"\nOf {len(first)} next-day onsets, {(first.y <= LINE + 10).mean():.0%} end within 10 points of the line and {(first.y <= LINE + 20).mean():.0%} within 20.")
'''

SAVE = r'''if SAVE_TO_HF and frames:
    keep = KEYS + ["split", "y", "persistence", "nightly anchor", "gbm", "gbm evening", "gbm hourly", "chronos-2 median", "chronos-bolt median", "final"]
    columns = keep + [c for name in MODEL_FAMILY + MECHANISM_FAMILY + BASELINE_FAMILY for c in seeded(name)]
    columns += [c for name in sorted(ANCHORED_MODELS) for c in (f"{name}-weight-today", f"{name}-weight-night")]
    store.upload_df(pd.concat([f[[c for c in columns if c in f]].assign(fold=c2) for c2, f in frames.items()], ignore_index=True),
                    "processed/final_loco_forecasts.parquet", "Notebook 08: every forecaster on the held-out cities")
    tables = {"final_selection": choice, "final_scores": table, "final_verdict": verdict, "final_per_city": per_city,
              "final_anchor_weights": anchor_table, "final_noise_ceiling": noise_table}
    tables.update({f"final_ablation_{label.replace(' ', '_')}": frame for label, frame in ablation.items()})
    tables.update({f"final_alarm_{label.replace(' ', '_')}": frame for label, frame in alarm_tables.items()})
    for name, frame in tables.items():
        if len(frame):
            store.upload_df(frame.reset_index(), f"processed/{name}.csv", f"Notebook 08: {name.replace('_', ' ')}")
    fingerprints = {city: registry.load(models_run(city)).get("weights_sha256") for city in frames}
    store.upload_json({"final_model": FINAL_NAME, "folds": len(frames), "of": len(cities), "seeds": SEEDS, "weights_sha256": fingerprints},
                      "processed/final_choice.json", "Notebook 08: which model the rule chose")
    print("↑ processed/final_loco_forecasts.parquet and the paper's tables")
if len(frames) == len(cities):
    print(f"The rule picked: {FINAL_NAME}. Notebook 09 freezes it and runs the sealed 2026 test.")
else:
    print(f"{len(frames)} of {len(cities)} folds finished. Reconnect and Run all to continue.")
'''

cells = [
    md("""# DEXTRA · 08 — The final model, its rivals, and what each part adds

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aduitya9/DEXTRA/blob/main/notebooks/08_final_model.ipynb)

Notebook 07 showed that a little sub-daily data helps: nine summaries of the evening lifted the all-token head on cities it had never seen. Tests on saved forecasts then showed that the hourly shape of recent days holds much more than those summaries, at every horizon. This notebook builds that into the model, in two new parts, and measures each on its own.

### The two new parts of the model
- **Anchored forecast.** Reversible normalisation restores every forecast from the window's mean, which knows nothing of tonight. Here the forecast starts from a choice of three anchors, weighed window by window and horizon by horizon: the fortnight's mean, today's value, and the **nightly anchor** — today's mean moved towards what tonight's last hours imply for that day, from the ratio that has lately held in this city. The nightly anchor needs no training and no severe day from the city, and the model stays scale-equivariant, which is what lets it serve cities it has never seen. Its one setting was chosen on the training years.
- **Hourly tokens.** In an inverted transformer a token is a variate's whole series, so a series at another resolution is one more token with its own embedding. Each hourly channel's last 72 hours becomes a token beside the 14-day daily ones.

### What is run, and why each model is there
Fifteen leave-one-city-out folds, three seeds each. Every step changes one thing, so a gain can be traced to its cause:

| Model | Reads | Asks |
|---|---|---|
| iTransformer | 26 daily variates | the starting point |
| All-token head | the same | notebook 05's finding, retrained here |
| + evening (**control**) | + 9 evening summaries | notebook 07's model, retrained here |
| + anchors as inputs only | + the 3 nightly anchors as ordinary variates | is the anchor just a useful feature? |
| + anchor token, mean start | the same, and a token saying where the anchors sit; the forecast still starts from the mean, as in RevIN | does telling the model where the anchors are help? |
| + learned mix of mean and today | the same; the forecast starts from a learned mix of RevIN's start and NLinear's | is mixing the two known starts enough? |
| + anchored forecast | the same; the nightly anchor joins the mix | does the new anchor add to the mechanism itself? |
| + 72-hour tokens | control + 8 hourly tokens | do hourly tokens help? |
| + both | everything | do the two parts combine? |

Reversible normalisation (RevIN) and NLinear's last-value start are the two corners of the anchored forecast that already exist, which is why they are rungs of this ladder: a gain that a mix of those two already gives is not a gain from the new anchor.

Then LSTM, DLinear and PatchTST. Gradient boosting, Chronos-2 and Chronos-Bolt come from earlier notebooks' saved forecasts on the same rows, and the nightly anchor is also scored on its own as a forecaster that was never trained.

### Which model is final — fixed before any result is read
Read on **validation rows only** (the other cities' 2024): among the five models from the control down, the one with the lowest validation RMSE whose validation average precision is no more than 0.002 below the control's. Its weights are saved here, with a fingerprint, for the final test. See `PREREGISTRATION.md`.

### 2026 stays sealed
No 2026 value reaches this notebook, and it stops if one does. These 2024-25 results have been explored before, so they choose and describe the model; the sealed 2026 test in notebook 09 is what confirms it.

### Runtime
**GPU, about 5 to 6 hours on a T4**, about half that on an A100. Every piece is saved when it finishes."""),
    py(SETTINGS),
    md("""### If Colab disconnects
Each piece — a fold's seven models, then its two mechanism rungs, then its three baselines — is saved in one commit when it finishes, and re-running skips everything finished. A piece cut off part-way starts again, at most about 15 minutes on a T4. If you are disconnected: reconnect and **Run all**. Near Colab's 12-hour limit: **Runtime → Disconnect and delete runtime**, then **Run all**. Keep this tab open in a window you can see, and your computer awake, while it runs. If it stops early, the tables below are still printed for the folds that finished.

Files named `loco-…-final` in `runs/08` are from an earlier four-model version of this notebook and are not read."""),
    md("## Secrets\n`HF_TOKEN` and `HF_DATASET_REPO`, as in the earlier notebooks."),
    py(secrets_cell),
    py('%pip install -q "huggingface_hub>=0.23.0" pyarrow'),
    py(SETUP),
    md("## A · The models\nParameters of everything trained here, how much of the hourly record exists, and how the nightly anchor relates to tomorrow."),
    py(MODELS),
    md("## B · The model and its parts, fold by fold\nEach city held out in turn; seven models, three seeds each. The weights of the head and of the five candidates are kept."),
    py(MODEL_LOOP),
    md("## B2 · Inside the anchored forecast\nThe two rungs between *anchors as inputs only* and the *anchored forecast*: the anchor token with RevIN's start from the mean, and a learned mix of the mean and today. They show whether the nightly anchor adds to the mechanism, or whether mixing the two known starts already does the work."),
    py(MECHANISM_LOOP),
    md("## C · The deep baselines\nLSTM reads the daily and evening variates; DLinear and PatchTST read the target's own series, as they are usually run."),
    py(BASELINE_LOOP),
    md("## D · Which model is final\nThe rule fixed at the top, on validation rows only."),
    py(GATHER),
    md("## E · Every forecaster, held-out cities, 2024-25\n**Average precision** (AP) asks how well a forecast ranks the severe days, with no alarm level involved. **Onsets** are severe days that were not severe on the day the forecast was made. *2025 only* leaves out the year the other cities' validation rows share. The check at the end compares the two models retrained here with the same models as notebooks 06 and 07 trained them."),
    py(TABLE),
    md("## F · What each part adds\nEach row is one step of the table at the top: the first model minus the second. The interval and p come from resampling whole held-out cities, and Holm corrects across the rows."),
    py(ABLATION),
    md("## G · The final model against every other"),
    py(VERDICT),
    md("## H · City by city, and how the anchors are used"),
    py(PER_CITY),
    md("## I · The city-adaptive alarm\nA city's alarm starts from a level fitted on other cities, and moves only when an outcome becomes known: a severe day missed lowers it, one caught raises it a little, so each city settles near the target recall whatever its forecasts' bias. The step is chosen on validation rows."),
    py(ALARM),
    md("## J · The noise ceiling\nEach city's stations were split into two fixed halves (`scripts/build_hourly_tables.py`, `src/evaluation/noise.py`). Their day-to-day disagreement is noise that no forecaster reading city-level inputs can predict, which sets a floor on RMSE and a ceiling on AP. *Share of reducible error removed* is how much of the error above that floor a forecaster has taken off persistence's."),
    py(NOISE),
    md("## K · Save\nEvery forecast and table, for the paper."),
    py(SAVE),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"},
               "colab": {"provenance": [], "toc_visible": True},
               "accelerator": "GPU"}
for cell in nb.cells:
    if cell.cell_type == "code" and not cell.source.lstrip().startswith(("%", "!")):
        compile(cell.source, "<cell>", "exec")
nbformat.validate(nb)
out = PROJECT / "notebooks" / "08_final_model.ipynb"
nbformat.write(nb, out)
print(f"wrote {out}: {len(nb.cells)} cells, all code cells compile")

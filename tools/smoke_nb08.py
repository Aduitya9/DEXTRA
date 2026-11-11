"""Notebook 08 end to end, offline, through a simulated Colab disconnect.

Each phase is its own Python process; the store is a local folder holding copies of
the real tables from the dataset repo:
  1  a time budget so small the session drops inside the first fold; every scoring
     cell must still run on nothing
  2  re-run: everything left trains
  3  re-run again: every piece is skipped; selection, tables, ablation, verdict,
     alarms, noise ceiling and save all run

Tiny settings: one seed, two epochs, three held-out cities out of five, a small model.
Usage: python tools/smoke_nb08.py <phase>
"""
import importlib.machinery
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import nbformat
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars

PHASE = int(sys.argv[1])
STORE = Path(tempfile.gettempdir()) / "dextra_smoke08"
REPO = os.environ.get("HF_DATASET_REPO", "Aduitya9/Aduitya_dextra-aqi-india")
if PHASE == 1:
    disable_progress_bars()
    shutil.rmtree(STORE, ignore_errors=True)
    (STORE / "processed").mkdir(parents=True)
    for name in ("city_daily_features.parquet", "evening_features.parquet", "city_hourly.parquet", "evening_loco_forecasts.parquet",
                 "gbm_hourly_loco_forecasts.parquet", "noise_samples.parquet", "noise_by_level.csv"):
        shutil.copy(hf_hub_download(REPO, f"processed/{name}", repo_type="dataset"), STORE / "processed" / name)

# Colab's secrets module, faked; accelerate checks __spec__, so each fake needs one.
userdata = types.ModuleType("google.colab.userdata")
userdata.get = lambda name: {"HF_DATASET_REPO": "local"}.get(name, "x")
colab = types.ModuleType("google.colab")
colab.userdata = userdata
google = types.ModuleType("google")
google.colab = colab
for module in (google, colab, userdata):
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
sys.modules.update({"google": google, "google.colab": colab, "google.colab.userdata": userdata})

import src.utils.hf_store as hf_store
from src.utils.local_store import LocalStore

hf_store.HFStore = lambda *a, **k: LocalStore(STORE)
import subprocess

subprocess.run = lambda *a, **k: None

nb = nbformat.read(PROJECT / "notebooks" / "08_final_model.ipynb", as_version=4)
code = [c.source.replace("/content/DEXTRA", str(PROJECT).replace("\\", "/")) for c in nb.cells if c.cell_type == "code"]
(settings, secrets, _pip, setup, models, model_loop, mechanism_loop, baseline_loop, gather, table, ablation, verdict,
 per_city, alarm, noise_cell, save) = code
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

LABELS = {1: "PHASE 1: Colab drops inside the first fold", 2: "PHASE 2: a fresh VM -- reconnect and Run all",
          3: "PHASE 3: another fresh VM -- Run all once more"}
print(f"\n{'=' * 20} {LABELS[PHASE]} {'=' * 20}")
print(f"process {os.getpid()}, hash('Delhi') = {hash('Delhi')}  (differs per process)")
t0 = time.time()
ns = {"display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
exec(settings, ns)
ns.update(SEEDS=[42], EPOCHS=2, PATIENCE=2, D_MODEL=32, N_HEADS=2, N_LAYERS=1, BOOTSTRAP_DRAWS=20, NOISE_DRAWS=5,
          LOCO_CITIES=["Delhi", "Patna", "Bengaluru"], TIME_BUDGET_HOURS=15 / 3600 if PHASE == 1 else 10.5)
exec(secrets, ns)
exec(setup, ns)
KEEP = ["Delhi", "Patna", "Bengaluru", "Lucknow", "Guwahati"]
ns["daily"] = ns["daily"][ns["daily"].city.isin(KEEP)].reset_index(drop=True)
ns["hourly_table"] = ns["hourly_table"][ns["hourly_table"].city.isin(KEEP)].reset_index(drop=True)
for cell in (models, model_loop, mechanism_loop, baseline_loop, gather, table, ablation, verdict, per_city, alarm, noise_cell,
             save):
    exec(cell, ns)
runs = STORE / "runs" / "08"
print(f"\nrecorded pieces: {sorted(p.stem for p in runs.glob('*.json')) if runs.exists() else []}")
print(f"weight files kept: {len(list(runs.glob('*.pt'))) if runs.exists() else 0}")
print(f"saved: {sorted(p.name for p in (STORE / 'processed').glob('final_*'))}")
print(f"phase {PHASE}: {time.time() - t0:.0f}s")

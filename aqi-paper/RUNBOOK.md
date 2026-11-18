# Runbook — exact order of operations

Do these in order. Do not skip step 3; it is what guarantees you never hit an error in
the long run.

---

## Step 0 — What you have

| File | What it is |
|---|---|
| `SIGMANet_AQI.ipynb` | **The one notebook.** Everything happens here. |
| `pipeline.py` | The same code as a plain script. The notebook is generated from it, so the two can never drift. Useful for `git diff`. |
| `to_ipynb.py` | Regenerates the `.ipynb` from `pipeline.py`. Run it if you edit the script. |
| `RESEARCH_PLAN.md` | The research design, the novelty claims, and the honest weaknesses. Read before writing the paper. |
| `RUNBOOK.md` | This file. |

You do **not** need to download any data by hand. The notebook pulls
`processed/city_daily_features.parquet` from your public HuggingFace dataset on its
first run and caches it in your Drive.

---

## Step 1 — Set up Colab (5 minutes, once)

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook** → pick `SIGMANet_AQI.ipynb`.
3. **Runtime → Change runtime type**:
   - Hardware accelerator: **GPU**
   - GPU type: **L4**
   - Click **Save**.
4. Confirm you are on Colab **Pro** (bottom-right of the runtime menu shows your
   compute units). The `full` profile needs roughly 5 GPU-hours.

> **Why L4 and not T4.** The notebook runs on a T4 as well, roughly 2–3× slower. It also
> runs on CPU, far slower. Nothing will break; only the clock changes.

---

## Step 2 — First launch: let it mount Drive

Run **only the first four cells** (`Shift+Enter` through them):

1. the configuration cell,
2. the dependency check,
3. the imports,
4. the Drive mount.

When cell 4 runs, Colab shows a permission pop-up. Click through it and allow access to
Google Drive. The cell then prints something like:

```
run directory: /content/drive/MyDrive/SIGMANet_AQI/smoke
```

**If you see `!! Drive mount failed`, stop and fix it before going further.** Without
Drive, nothing survives a disconnect, which defeats the whole point.

---

## Step 3 — Smoke run (3–5 minutes). Do not skip this.

The configuration cell already says `PROFILE = "smoke"`. Leave it.

> There is also a `PROFILE = "ci"` that skips tuning entirely, trains one epoch on three
> cities, and finishes in about two minutes even with **no GPU at all**. It is what I used
> to verify the code on your laptop. Use it if Colab gives you no GPU and you just want to
> confirm the notebook still runs after an edit. For anything else, use `smoke`.

**Runtime → Run all.**

This executes every single line of the notebook on a reduced problem: 6 cities,
2 Optuna trials, 3 epochs, 2 folds. The numbers it prints are meaningless. The point is
that it proves the whole pipeline runs — data fetch, cleaning, EDA, all five model
families, tuning, cross-validation, the test evaluation, the significance tests, the
ablations, leave-one-city-out, and every figure and table.

**What success looks like.** The last cell prints:

```
work units: NN/NN complete
  Everything finished. Tables and figures are in the run directory.
```

Along the way you should also see:

```
  all 19 checks passed          <- the leakage self-tests
```

If anything raises, the notebook stops at that cell. Send me the traceback and I will fix
it; do not move to step 4 with a failing smoke run.

---

## Step 4 — Real run, in two stages

### 4a. `fast` (about 45 minutes) — sanity-check the science

In the configuration cell change one line:

```python
PROFILE = "fast"
```

**Runtime → Run all.**

This uses all 15 cities and gives real, if under-tuned, numbers. Look at the two headline
tables it prints. You are checking that the ordering is sane:

- every learned model should beat **persistence** by a clear margin at h=1;
- MAE should **increase** with horizon (h3 worse than h1);
- `test_unseen` should be **worse** than `test_seen`.

If any of those three is violated, something is wrong with the setup, not with the models.
Tell me before burning five GPU-hours.

### 4b. `full` (about 5 hours) — the numbers for the paper

```python
PROFILE = "full"
```

**Runtime → Run all.** Then leave it alone.

`smoke`, `fast` and `full` write to **separate** directories, so running `full` does not
destroy your earlier results and does not reuse their tuning.

---

## Step 5 — When Colab disconnects (it will)

This is already handled. Do exactly this:

1. Reopen the notebook.
2. **Runtime → Run all.**

That is the entire recovery procedure. Everything already finished is skipped — you will
see lines like:

```
  [skip] hpo__patchtst
  [skip] final__sigmanet__seed42
```

and a fit that was interrupted part-way resumes from its last epoch:

```
      resumed final__sigmanet__seed43__ck at epoch 37 (best val MAE 18.412)
```

**Do not change `PROFILE` between a disconnect and the resume**, or it will start a fresh
run directory and begin again.

Two habits that make long runs survive:
- keep the browser tab open and the laptop awake;
- re-run as soon as you notice a disconnect, because Colab reclaims idle runtimes.

To see what is outstanding without running anything, run the **last cell** on its own.

---

## Step 6 — Collect the outputs

Everything lands in `MyDrive/SIGMANet_AQI/full/`:

| Folder | Contents |
|---|---|
| `tables/` | Every table as CSV. `table2_test_seen.csv` and `table3_test_unseen.csv` are the headline results; `table4_ablation.csv` is the novelty evidence; `table5_significance.csv` holds the p-values. |
| `figures/` | Every figure as 300-dpi PNG **and** PDF. Use the **PDF** in LaTeX — it is vector, so it stays sharp. |
| `RUN_SUMMARY.md` | A one-page digest: headline numbers, the CPCB-operator fidelity number, and the attribution statements you owe. |
| `results/` | One JSON per completed unit of work. This is the audit trail; keep it. |
| `ckpt/` | Model checkpoints. Safe to delete once `full` has finished, to reclaim Drive space. |
| `studies/optuna.db` | The tuning history. Keep it — a reviewer may ask how the search was run. |

---

## Step 7 — Writing the paper

1. Start from `RESEARCH_PLAN.md` §9, which is already a section-by-section outline.
2. Download the IEEE conference template (`IEEEtran`, two-column, 10 pt) and use Overleaf.
3. Map the artefacts straight in:

| Paper element | File |
|---|---|
| Table I — dataset and splits | `tables/table1_splits.csv` |
| Table II — results, seen cities | `tables/table2_test_seen.csv` |
| Table III — results, unseen cities | `tables/table3_test_unseen.csv` |
| Table IV — ablation | `tables/table4_ablation.csv` |
| Table V — significance | `tables/table5_significance.csv` |
| Table VI — leave-one-city-out | `tables/table6_loco_mae.csv` |
| Fig. 1 — data and seasonality | `figures/fig2_eda.pdf` |
| Fig. 2 — motivation for the physics head | `figures/fig_driver_share.pdf` |
| Fig. 3 — accuracy vs horizon | `figures/fig3_mae_by_horizon.pdf` |
| Fig. 4 — hazardous-day detection | `figures/fig4_hazard_pr.pdf` |
| Fig. 5 — a traced episode | `figures/fig5_trace_delhi.pdf`, `fig5_trace_patna.pdf` |
| Fig. 6 — ablation | `figures/fig6_ablation.pdf` |
| Fig. 7 — interpretability | `figures/fig7_driver_attribution_test_unseen.pdf` |

4. The architecture diagram (§5.1 of the research plan) is the one figure you have to draw
   by hand. The ASCII sketch there has the right boxes and arrows.

---

## Step 8 — Two things to do before submitting

1. **CPCB permission.** The pollutant data come from CPCB, whose copyright policy allows
   downloading but requires approval for any other use. Email `ccb.cpcb@nic.in` and ask
   for permission to publish analysis derived from CAAQMS data. Do this **early** — it is
   the one item with an external turnaround you cannot control.
2. **Attribution.** The acknowledgements must credit ERA5 / Copernicus C3S (CC-BY),
   Open-Meteo (CC-BY-4.0) and NASA FIRMS. The exact wording is in the dataset card and is
   repeated in `RUN_SUMMARY.md`.

---

## Troubleshooting

| Symptom | What it means | Do this |
|---|---|---|
| `!! Drive mount failed` | Colab could not reach Drive | Re-run the cell and complete the pop-up. Do not continue without it. |
| `Everything finished` but a table is missing | that unit was skipped in this profile | Check the last cell's outstanding list; `smoke` deliberately runs fewer things |
| `CUDA out of memory` | another notebook holds the GPU, or `d_model` got large | Runtime → Restart, then Run all. Resume makes this cheap. If it repeats, lower `batch_size` to 128 in the config cell |
| Tuning seems to start over | you changed `PROFILE`, or the Optuna DB was deleted | Profiles are independent by design; put `PROFILE` back |
| `database is locked` | two copies of the notebook are running against one Drive DB | Close one. Only ever run one copy per profile |
| Resume says `checkpoint unreadable` | a write was cut off mid-disconnect | Harmless; that one fit restarts. Everything else is kept |
| A run is far slower than the table says | you are on a T4 or CPU, not an L4 | Runtime → Change runtime type → L4 |
| `No GPU available` | Pro quota exhausted for now | Wait, or run `fast` on CPU overnight |

---

## Honest expectations

- `full` takes about five hours on an L4 and may span two or three Colab sessions. That is
  normal and the resume logic is built for it.
- **SIGMA-Net may not win every metric.** If the ablation shows that the differentiable
  CPCB head does not help, that is the experiment's answer, and `RESEARCH_PLAN.md` §2 and
  §10 already explain how to write the paper around it. Report what you measure. A paper
  that reports a well-motivated prior failing, and explains why, is publishable; one that
  quietly tunes until the proposed method wins is not, and reviewers are good at spotting
  the difference.

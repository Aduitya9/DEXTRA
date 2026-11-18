# Research Plan — SIGMA-Net
### Physics-Guided Multi-Pollutant Forecasting of the Indian National AQI, with Evaluation on Unseen Cities

**Target venue:** IEEE conference, 6 pages, `IEEEtran` two-column
(good fits: IEEE INDICON, IEEE TENCON, IEEE CONECCT, IEEE ICCCNT, IEEE Big Data workshops)

**Status:** plan locked 2026-11-16. One notebook: `DEXTRA_AQI_SIGMANet.ipynb`.

---

## 1. One-paragraph summary

India's National Air Quality Index (NAQI) is, by definition, the **maximum of piecewise-linear
sub-indices** computed from individual pollutant concentrations. Every published deep-learning
AQI forecaster we are aware of ignores this: it either regresses the AQI scalar directly, or
forecasts pollutants and applies the formula *afterwards*, outside the computational graph.
We instead make the NAQI definition a **differentiable layer inside the network**. The model
forecasts six pollutant concentrations, maps each through the exact CPCB breakpoint function,
and combines them with a temperature-controlled smooth-maximum, plus a learned residual that
absorbs aggregation-convention mismatch. Two further components address the physics and the
economics of the problem: meteorology modulates the pollutant representation **multiplicatively**
(a ventilation prior, because concentration ≈ emissions ÷ ventilation, not emissions *minus*
ventilation), and the loss is **anchored on the regulatory AQI-200 hazard boundary** with an
asymmetric penalty, because under-forecasting a hazardous day costs more than over-forecasting
one. We evaluate 1–3 day-ahead forecasts against a persistence reference, LightGBM, PatchTST,
iTransformer and TSMixer, on ten cities the models trained on and on **five cities with zero
training days**.

---

## 2. Why this is publishable (the honest version)

**What is genuinely novel**

| | Contribution | Why a reviewer should accept it as new |
|---|---|---|
| **C1** | **Differentiable NAQI sub-index head (DSI)** | The NAQI operator is known, exact, and non-smooth. Baking it in as an inductive bias — so that the AQI loss back-propagates through *which pollutant dominates* — has not been done. It is also the only one of our components that yields free interpretability: the arg-max sub-index tells a city which pollutant drove the forecast. We verify the operator reproduces the dataset's own AQI column to **MAE 0.57 (99.3 % of days within ±2 AQI points)**, which is what licenses using it as a hard prior. |
| **C2** | **Ventilation-gated meteorological modulation (VGM)** | Box-model physics says C ∝ E/(BLH·u). Concatenating boundary-layer height as an extra input channel cannot express a quotient. We generate FiLM scale/shift parameters from the ventilation variables and modulate the pollutant representation multiplicatively. |
| **C3** | **Episode-focal asymmetric loss (EFA)** | Threshold-anchored focal *regression* — focal weight centred on the AQI-200 regulatory boundary, plus an asymmetric under-prediction penalty — aligns the objective with the operational cost of a missed health advisory. |

**What is not novel, and we will not claim it is**
- The dataset. It already exists (DEXTRA, 38,355 city-days). Our paper is a *modelling* paper on
  an existing resource; the data contribution is cited, not re-claimed.
- The backbone. Patch embedding and variate attention are from PatchTST and iTransformer. We say so.
- The unseen-city split design. It ships with the dataset. We use it; we did not invent it.

### 2.1 Overlap with work already in the DEXTRA repository — read this before writing

This plan was written before the `Aduitya9/DEXTRA` repository was examined properly. The
repository turned out to contain notebooks 01–08, `src/models/`, 18 test files and a
**pre-registration**. Checking C1–C3 against it changes the honest claim:

| | Status against the existing repo |
|---|---|
| **C1 — differentiable sub-index head** | **Survives.** Verified: no model in `src/models/` imports `src/utils/aqi_calculator.py`. The CPCB formula is used only to *build data*, never inside a network's forward pass or loss. This is the one genuinely new thing here. |
| **C2 — ventilation-gated modulation** | **Substantially pre-existing.** `src/models/met_moe.py` is a meteorological mixture-of-experts whose router reads meteorological gate features and produces mixture weights. The mechanism differs (FiLM scale/shift vs expert routing) but the *idea* — meteorology gates the model — is already yours. Demote it from "contribution" to "component", and cite `met_moe` explicitly. |
| **C3 — episode-focal asymmetric loss** | **Substantially pre-existing, and theirs is better.** `src/models/rare_days.py` already has `WeightedSquaredError` (severe targets upweighted), `oversampling_loader` (tail oversampling), and `pinball` / `QuantileForecaster` — and the pinball loss is the principled version of the asymmetric under-forecast penalty I hand-rolled. **Prefer the existing pinball loss over EFA**, or justify EFA against it in the ablation. |

Two further corrections to advice given earlier in this plan:

- **Anchoring is not a discovery, it is yours.** §5.0 presents anchoring as a fix found during
  development. Notebook 08 already has a pre-registered ablation ladder over exactly this —
  "anchors as inputs only", "anchored forecast", plus explicit **RevIN** and **NLinear** rungs.
  Cite notebook 08; do not present anchoring as new.
- **Foundation models are not out of scope.** §2 said to argue them away. That is wrong here:
  `src/models/foundation.py` already wraps **Chronos**, and notebook 06 already compares against
  it on the severe-day gap. Including that comparison is nearly free and removes the single most
  likely reviewer objection. Do it.

**The bigger strategic question, stated plainly.** The pre-registration commits to a
**leave-one-city-out test over all 15 cities on unread 2026 CPCB data**. That is a far stronger
piece of evidence than this plan's 2025 `test_seen` / `test_unseen` split, and pre-registered
hold-out tests are rare in this literature. If you have the time, the stronger paper is the
pre-registered one with the differentiable sub-index head added as its new component. This
notebook is the fast, self-contained, low-risk route — which is what was asked for — but it is
**not** the most publishable thing you already own. Decide deliberately, not by default.

**The weakest point, stated up front.** Each component is individually a modest idea. The paper
stands on (a) the physics-guided head being a clean, verifiable, *exact* prior rather than a
hand-wave, (b) a genuinely rigorous protocol, and (c) the unseen-city result. If the ablation
shows DSI does not help, we report that — a negative result on a well-motivated prior, with a
working explanation, is still a publishable conference paper, and it is the honest outcome.

**Two things you must do yourself before submitting.**

1. **Verify the novelty claim against the literature.** I am asserting that no published work
   uses the NAQI operator as a *differentiable output layer* in an AQI forecaster, and the paper
   says "to our knowledge". I cannot check every paper, and that claim is the spine of the
   contribution. Search Scopus / IEEE Xplore / Google Scholar for *differentiable air quality
   index*, *physics-guided AQI forecasting*, *sub-index neural network*, *AQI breakpoint layer*
   before you commit to the framing. If someone has done it, the fallback framing is still
   sound: the combination with the ventilation gate and the episode-focal loss, evaluated on
   unseen cities, is new even if the head is not.
2. **Decide how to handle time-series foundation models.** A 2026 reviewer may well ask why
   there is no zero-shot Chronos / TimesFM / Moirai baseline. This is **deliberately out of
   scope**: those models are univariate, cannot ingest the meteorological and fire covariates
   that carry most of the signal here, and would add a heavyweight dependency to a notebook that
   must not break. Say that in Related Work in one sentence, rather than leaving the gap
   unexplained. Adding one is about one extra cell if a reviewer insists.

---

## 3. Data

**Source:** `Aduitya9/Aduitya_dextra-aqi-india` on HuggingFace (public — no token needed).
**File used:** `processed/city_daily_features.parquet` — 38,355 rows × 36 columns.

One row = one city on one Indian-Standard-Time calendar day, 15 cities, 2019-01-01 → 2025-12-31.
**Verified: every city has all 2,557 days, with no missing dates and no duplicates.** A regular,
gap-free daily grid, which makes windowing safe.

### 3.1 Splits (ship with the dataset; we use them unchanged)

| Split | Rows | Cities | Dates | Our use |
|---|---|---|---|---|
| `train` | 18,260 | 10 | 2019–2023 | Fitting, CV, HPO search space |
| `val` | 3,660 | 10 | 2024 | Early stopping, model selection, Optuna objective |
| `test_seen` | 3,650 | 10 | 2025 | **Headline test 1** — seen cities, unseen year |
| `test_unseen` | 1,825 | 5 | 2025 | **Headline test 2** — cities with zero training days |
| `support_unseen` | 155 | 5 | Dec 2024 | Not used for fitting. Noted as the origin of one static feature |
| `zeroshot_unseen` | 10,805 | 5 | 2019 → Nov 2024 | **Deliberately unused.** Training on it would destroy the unseen-city claim |

### 3.2 Facts we established by probing the data (and that correct the dataset card)

These matter, and several contradict or refine the published card:

1. **The card says six pollutants; there are seven.** `NH3` is present.
2. **`NH3` is missing for all 2,557 days in Pune** and is 17.7 % missing across `train`.
   Its maximum observed value anywhere is 208 µg/m³ → sub-index 52, so **NH3 can never attain
   the maximum sub-index** in this dataset. Decision: NH3 is kept as an *input* feature with a
   missingness mask, but **excluded from the sub-index maximum**, so the physics head is never fed
   a channel that is almost entirely imputed. Documented in the paper.
3. **`PM10` has a 921-consecutive-day gap in Kanpur**, and is 8.1 % missing in `train`.
   Mask channels are therefore not optional — they are load-bearing.
4. **`station_density` is numerically identical to `n_stations`.** We drop the duplicate.
5. **`inversion_level_hpa` is constant per city** (850–925 hPa). It is a city static (an elevation
   proxy), not a time-varying feature, and is treated as one.
6. **`AQI` has 915 nulls**, concentrated in Bhopal (299), Pune (239) and Mumbai (164).
   Those rows are dropped **as targets** but may still appear inside an input window (with a mask).
7. **`AQI_category` is an exact function of `AQI`.** It is excluded from the inputs — it is the
   target in disguise.
8. **The card's hazard rates are misleading for our test year.** The card reports 20.4 % of
   held-out-city days above AQI 200; that is across all years, dominated by `zeroshot_unseen`.
   **In 2025 the true rates are 6.9 % (`test_seen`) and 7.5 % (`test_unseen`).** We use the real
   numbers.
9. **Per-city episode counts in 2025 are tiny** — Varanasi has 2 days above AQI 200, Kanpur 9.
   Per-city episode metrics are therefore reported as *indicative only*; headline episode metrics
   are **pooled** across the five unseen cities (137 hazardous days of 1,823).
10. `industrial_index` for held-out cities is derived from the 31-day December-2024 support
    window, and ranges 0.22 → 5.79 across cities. It is `log1p`-compressed and the dependence is
    disclosed; an ablation drops it.

### 3.3 Feature construction — all strictly causal

Per-timestep channels inside a lookback window ending at day *t* (every value from day ≤ *t*):

| Group | Channels |
|---|---|
| Pollutants | `log1p` of PM2.5, PM10, NO2, SO2, NH3, CO, O3 |
| Missingness | one binary mask per pollutant, and one for AQI |
| Target history | AQI/100 |
| Meteorology | `blh`, `blh_min`, `blh_anomaly`, `temp2m`, `wind_speed`, `sin/cos(wind_dir)`, `precip`, `inv_idx` |
| Fire | `log1p` of `fire_count_100km`, `fire_count_500km`, `frp_500km` |
| Observation quality | `n_stations`, `n_stations_pm25`, `hours` |

Known-future and static covariates, attached per sample (legitimately knowable at forecast time):

| Group | Channels |
|---|---|
| Calendar **at the target date t+h** | `sin/cos(day_of_year)`, `sin/cos(month)`, `sin/cos(day_of_week)`, `is_diwali`, `is_stubble_season` |
| City statics | `archetype_id` (embedding), `coastal_flag`, `log1p(industrial_index)`, `inversion_level_hpa` |

**Deliberate exclusion.** Meteorology is ERA5 *reanalysis* — i.e. observed weather. Using the
actual met on day *t+h* would be using the future. The main setting is therefore **strictly causal:
no meteorological value after day t enters any feature.** Calendar variables at *t+h* are the only
future-dated inputs, and those are genuinely known in advance. A `PERFECT_MET_ORACLE` config flag
exists to reproduce the common (and in our view over-optimistic) "perfect prognosis" setting as an
upper-bound ablation; it is **off** by default.

### 3.4 Leakage controls

| Risk | Control |
|---|---|
| Future met | Strictly causal windowing (§3.3) |
| Scaler fit on test | Robust median/IQR statistics computed on `train` rows only, then frozen |
| Imputation fit on test | Causal forward-fill within a city (limit 3 days), then the **train-split global median**; never a test statistic. Global, not per-city, so it also works for cities with no training history |
| Window straddling a split boundary | A sample belongs to the split of its **target date**; an **embargo of L + H days** is purged at every boundary |
| Shuffled time | No shuffling across time anywhere; CV is blocked |
| Target in the inputs | `AQI_category` dropped; AQI only ever enters from days ≤ t |
| HPO touching the test set | Optuna optimises **val-2024 MAE only**. `test_seen` and `test_unseen` are read **once**, after all hyperparameters and all early-stopping decisions are frozen |

---

## 4. Task

Given the last **L = 14** days of all channels for a city, predict **AQI at t+1, t+2 and t+3**
(direct multi-horizon — one forward pass emits all three, no recursive feeding).

Everything else is derived from that single continuous forecast, never modelled separately:
- the 6-class CPCB category, by binning the predicted AQI;
- the hazardous-day flag, by thresholding it at AQI 200.

This is deliberate: a separate classifier would be a second model to tune and would let the
classification and regression heads disagree with each other.

---

## 5. Models

| # | Model | Family | Reference |
|---|---|---|---|
| M1 | Persistence (AQI at t, carried forward) | trivial reference | — |
| M2 | Seasonal-climatology (city × day-of-year mean, train only) | trivial reference | — |
| M3 | LightGBM, direct per-horizon, on lag + rolling features | GBDT | Ke et al., NeurIPS 2017 |
| M4 | PatchTST | patch Transformer, channel-independent | Nie et al., ICLR 2023 |
| M5 | iTransformer | inverted attention over variates | Liu et al., ICLR 2024 |
| M6 | TSMixer | all-MLP time/feature mixing | Chen et al., TMLR 2023 |
| **M7** | **SIGMA-Net (ours)** | **DSI + VGM + EFA** | **this work** |

All four deep baselines are **implemented from scratch in plain PyTorch inside the notebook**.
This is a deliberate engineering choice: no `mamba-ssm`, no `tslib`, no exotic CUDA extension that
could fail to compile on Colab. The only non-stock dependencies are `lightgbm` and `optuna`, both
pure-pip and reliable.

### 5.0 Anchoring — one decision applied identically to every deep model

> **Attribution, per §2.1:** anchoring is *already yours*. Notebook 08 carries a
> pre-registered ablation ladder over exactly this idea, with explicit RevIN and NLinear
> rungs. What follows is a re-derivation of it, reached independently during development.
> Cite notebook 08; do not present it as new.

Daily city AQI is strongly autocorrelated (lag-1 correlation around 0.8), which makes plain
**persistence a hard baseline**. An early run confirmed it: untrained-to-barely-trained deep
models all lost to persistence outright. So every deep model is **anchored** — it is given
today's AQI and predicts the *change* from it, and the final layer of every head is
zero-initialised, so each model *begins* exactly at persistence and must earn any departure.

SIGMA-Net is anchored one level deeper: its head predicts a change in each pollutant
*concentration*, so at initialisation it emits today's concentrations through the exact CPCB
formula. (A small positive bias of up to `tau·ln P` ≈ 3.6 AQI points comes from the smooth
maximum at initialisation; the learned residual removes it.)

Three things to note, because a reviewer will:

- This makes the comparison **harder** for our model, not easier — it hands every baseline a
  strong starting point and raises the bar that SIGMA-Net has to clear.
- It is not a trick we invented. PatchTST uses instance normalisation for exactly this reason,
  so omitting it would have made that baseline *weaker than its own paper*.
- It must be stated in the paper. An unanchored comparison would have produced a flattering
  but meaningless table in which every deep model loses to a one-line heuristic.

### 5.1 SIGMA-Net architecture

```
             ┌──────────────── pollutant channels (7 + masks) ───────────────┐
 window      │  patch-embed (channel-independent)  →  temporal encoder       │
 L×C  ───────┤                                                               ├──→ z_pol
             │  variate attention across pollutants (inter-pollutant coupling)│
             └───────────────────────────────────────────────────────────────┘
                                        │
   met/ventilation ──→ FiLM(γ, β) ──────┤  z_pol ← γ ⊙ z_pol + β      [C2 VGM]
   (blh, blh_min, blh_anomaly,          │
    wind_speed, inv_idx, precip)        │
                                        ▼
        static + known-future calendar ──→ concat ──→ MLP trunk
                                        │
                     ┌──────────────────┴────────────────────┐
                     ▼                                       ▼
        6 pollutant concentrations                  residual correction r
             at t+h  (ĉ)                                     │
                     │                                        │
      differentiable CPCB sub-index                            │
        I_p = piecewise_linear(ĉ_p)        [C1 DSI]            │
                     │                                        │
      smooth-max over p:  τ·logΣexp(I_p/τ)                     │
                     │                                        │
                     └──────────────→  ÂQI  ←─────────────────┘
                                        │
                        episode-focal asymmetric loss  [C3 EFA]
```

- **DSI** uses the exact CPCB breakpoints over 6 pollutants (NH3 excluded, §3.2.2), implemented as
  a vectorised piecewise-linear interpolation that is differentiable almost everywhere. τ is
  learned, initialised small so the smooth-max starts close to a true max.
- **Auxiliary supervision:** the predicted pollutant concentrations are *also* supervised against
  observed concentrations where available. This is what keeps the head physically meaningful
  rather than letting it find an arbitrary 6-vector that happens to max to the right AQI.
- **EFA:** `Huber(ÂQI/100, AQI/100)` × `[1 + α·σ((AQI−200)/s)]` + `β·relu(AQI − ÂQI)·σ((AQI−200)/s)`.

---

## 6. Protocol

**Validation.** Three independent axes, all reported:

1. **Blocked expanding-window CV** — 5 folds inside 2019–2023, with an L+H embargo between
   each train block and its validation block. Gives mean ± std over time.
2. **Leave-one-city-out CV** — 10 folds over the training cities. Gives cross-city variance and
   is the in-sample analogue of the unseen-city test.
3. **The two frozen 2025 test sets** — read exactly once.

**Hyperparameter optimisation.** Optuna TPE, `MedianPruner`, 30 trials per deep model and 40 for
LightGBM, objective = **val-2024 MAE averaged over the three horizons**. The study lives in a
**SQLite file on Google Drive**, so an interrupted search resumes at the trial it reached.

**Early stopping.** Always on, patience 15 epochs on val MAE, max 120 epochs, best-weights restore.

**Seeds.** Final models trained with 3 seeds; tables report mean ± std. CV folds use 1 seed to
keep the compute budget honest.

**Significance.** Diebold–Mariano test on the per-day loss differential, per horizon, SIGMA-Net
against every baseline, plus paired block-bootstrap 95 % confidence intervals (block = 7 days, to
respect autocorrelation). Reported with a Holm correction across the comparison family.

### 6.1 Metrics

- **Regression:** MAE, RMSE, R², sMAPE — per horizon and pooled; plus skill score against persistence.
- **Hazardous-day detection (AQI > 200), thresholded from the continuous forecast:** precision,
  recall, F1, **Critical Success Index**, and PR-AUC.
- **6-class CPCB category:** accuracy, macro-F1, and **quadratic weighted kappa** (the categories
  are ordinal, so QWK is the right agreement statistic).

### 6.2 Ablations

`SIGMA-Net − DSI` · `− VGM` · `− EFA` · `− all` (plain backbone) · `− industrial_index`,
each × 3 seeds, scored on both test sets.

---

## 7. Compute budget (single L4, Colab Pro)

| Stage | Runs | Estimate |
|---|---|---|
| Data prep + EDA | 1 | ~3 min |
| Trivial references | 2 | seconds |
| Optuna HPO | 4 deep × 30 + LGBM × 40 | ~2.0 h |
| Blocked CV | 5 models × 5 folds | ~1.0 h |
| Final fits | 5 models × 3 seeds | ~25 min |
| Leave-one-city-out | 5 models × 10 cities | ~1.3 h |
| Ablations | 5 variants × 3 seeds | ~25 min |
| Evaluation, tests, figures | 1 | ~5 min |
| **Total** | | **≈ 5 h** |

The notebook has three profiles: **`smoke`** (≈2 min, proves the whole pipeline executes),
**`fast`** (≈45 min, reduced trials and folds, real numbers), **`full`** (≈5 h, the paper's numbers).

## 8. Resumability

Every unit of work — each Optuna study, each (model, fold, seed) fit, each evaluation — writes an
atomic result file to Drive and is recorded in a ledger. Re-running the notebook top-to-bottom
after a disconnect skips everything already finished. Within a single fit, per-epoch checkpoints
carry model, optimiser, scheduler and RNG state, so even one interrupted training resumes
mid-run rather than restarting.

---

## 9. Paper outline (6 pages)

| § | Content | Artefacts |
|---|---|---|
| I | Introduction — NAQI is a max of sub-indices; forecasters ignore it; most Indian cities have no monitoring history | — |
| II | Related work — AQI forecasting in India; current time-series architectures; physics-guided ML | — |
| III | Data and protocol | Table I (splits), Fig. 1 (pipeline), Fig. 2 (AQI distribution by city) |
| IV | SIGMA-Net | Fig. 3 (architecture), the DSI fidelity number (MAE 0.57) |
| V | Experiments | Tables II–III (seen / unseen), Table IV (ablation), Table V (DM tests), Fig. 4 (episode PR curves), Fig. 5 (Delhi + Patna winter episode traces) |
| VI | Interpretability | Fig. 6 (driving-pollutant attribution from the DSI head) |
| VII | Limitations — thin station coverage; 2 hazardous days in Varanasi; reanalysis not forecast met; CPCB licence | — |
| VIII | Conclusion | — |

**Licence note for submission.** The pollutant tables derive from CPCB, whose copyright policy
permits download but requires approval for other use. Request approval at `ccb.cpcb@nic.in`
**before** camera-ready. ERA5 (CC-BY), Open-Meteo (CC-BY-4.0) and NASA FIRMS need attribution
statements in the acknowledgements — exact wording is in the dataset card.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| **EFA lowers MAE performance while raising hazard recall** — already visible in the one-epoch verification run, where `−EFA` had the *best* MAE and the *full* model the best hazardous-day F1 | This is the loss doing exactly what it was designed to do, and it is the most likely shape of the final result. **Plan for it:** lead the paper with hazardous-day detection (recall, F1, CSI) as the headline metric and report MAE alongside, which is the right framing for an air-quality paper anyway — the operational purpose is catching bad days, and a forecast that is 2 AQI points better on average but misses a Severe day is worse, not better. State the trade-off explicitly rather than burying it; a reviewer who finds it themselves will assume you hid it |
| **LightGBM beats the transformers** — also plausible on 18k rows, and consistent with the tabular-vs-deep literature | Report it plainly. It is a legitimate, useful finding for a conference audience, and it does not weaken the SIGMA-Net contributions, which are about the *output layer and the loss* and can be bolted onto any backbone. Do not tune the baselines less to avoid this |
| DSI does not beat direct regression | Report it. The ablation is the experiment; a well-motivated negative result with an explanation is still publishable |
| Unseen-city numbers are dominated by 2 cities having almost no episodes | Pool episode metrics; report per-city as indicative; say so in Limitations |
| Reviewer: "reanalysis met is not a forecast" | Pre-empted in §3.3 and in Limitations; the oracle flag quantifies the gap |
| Reviewer: "iTransformer was already run on this data" | Prior DEXTRA outputs are ignored; every model here is retrained from scratch under one protocol |
| Colab disconnects | §8 |

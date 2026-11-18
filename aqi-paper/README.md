# SIGMA-Net — AQI forecasting for an IEEE conference paper

Forecast India's National AQI one to three days ahead for 15 cities, compare four current
models against a novel physics-guided one, and evaluate on five cities with **no training
days at all**.

## Read in this order

| | |
|---|---|
| 1. [RUNBOOK.md](RUNBOOK.md) | **Start here.** Exactly what to do, in order, to get results. |
| 2. [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | The research design, the three novelty claims, the honest weaknesses, and the two things you must check yourself before submitting. |
| 3. [SIGMANet_AQI.ipynb](SIGMANet_AQI.ipynb) | The one notebook. Upload to Colab, pick an L4, Run all. |

## The other files

| | |
|---|---|
| `pipeline.py` | The notebook as a plain script. The `.ipynb` is **generated** from it, so the two cannot drift. This is the file to edit and to `git diff`. |
| `to_ipynb.py` | Regenerates the notebook: `python to_ipynb.py pipeline.py SIGMANet_AQI.ipynb` |

## What the notebook does

Fetches the data from HuggingFace, cleans it, runs EDA, then trains, tunes, cross-validates
and tests **seven** models:

| | Model | |
|---|---|---|
| M1 | Persistence | trivial reference |
| M2 | Day-of-year climatology | trivial reference |
| M3 | LightGBM | GBDT on window summaries and lags |
| M4 | PatchTST | ICLR 2023 |
| M5 | iTransformer | ICLR 2024 |
| M6 | TSMixer | TMLR 2023 |
| **M7** | **SIGMA-Net** | **ours** |

SIGMA-Net's three components, each separately ablated. **Only the first is new relative to
the work already in this repository** — see `RESEARCH_PLAN.md` §2.1, which checks all three
against `src/models/` and is blunt about what is a re-derivation:

- **DSI** — the exact CPCB sub-index formula as a *differentiable output layer*, so the AQI
  loss back-propagates through which pollutant dominates. Verified to reproduce the dataset's
  own AQI column to **MAE 0.57** points.
- **VGM** — meteorology modulates the pollutant representation *multiplicatively*, because
  box-model physics says concentration ≈ emissions ÷ ventilation, and concatenation cannot
  express a quotient. *Overlaps `src/models/met_moe.py`; a component, not a contribution.*
- **EFA** — a loss anchored on the regulatory AQI-200 boundary, penalising under-forecasts of
  hazardous days more than over-forecasts. *Overlaps `src/models/rare_days.py`, whose pinball
  loss is a cleaner form of the same asymmetry. Prefer that one, or beat it in the ablation.*

## Verification status

The code was executed end-to-end on a machine with **no GPU** before being handed over:
every stage ran to completion (exit code 0, all work units complete), producing 20 figures
and 13 tables. The notebook itself — not just the equivalent script — was executed with a
real Jupyter kernel. Twenty leakage self-tests and an anchor-invertibility test run inside
the notebook on every launch and **stop it** if they fail.

## Guarantees and non-guarantees

- **Resumable.** Interrupt it anywhere; re-run top to bottom and it skips what finished and
  resumes an interrupted fit from its last epoch.
- **Leak-free by construction**, and it checks itself. See `RESEARCH_PLAN.md` §3.4.
- **Not guaranteed: that SIGMA-Net wins.** The ablation is a real experiment with a real
  possibility of a negative result. `RESEARCH_PLAN.md` §2 and §10 say how to write the paper
  either way.
- **Not guaranteed: novelty against the whole literature.** You must check that yourself —
  `RESEARCH_PLAN.md` §2 says exactly what to search for.

## Before you submit

The pollutant data derive from CPCB, whose copyright policy permits downloading but requires
approval for other use. **Email `ccb.cpcb@nic.in` early** — it is the one blocker with an
external turnaround you do not control. ERA5/Copernicus, Open-Meteo and NASA FIRMS all need
attribution in the acknowledgements.

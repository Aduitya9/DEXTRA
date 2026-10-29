---
license: other
license_name: dextra-mixed-sources
license_link: https://huggingface.co/datasets/Aduitya9/Aduitya_dextra-aqi-india#licence-and-attribution
pretty_name: DEXTRA city-daily air quality, meteorology and fire, India 2019-2025
task_categories:
- time-series-forecasting
tags:
- air-quality
- aqi
- pollution
- india
- era5
- nasa-firms
- cpcb
size_categories:
- 10K<n<100K
language:
- en
---

# DEXTRA: city-daily air quality, meteorology and fire for 15 Indian cities, 2019–2025

Data for **DEXTRA** (Diffusion-Enhanced Cross-City Transfer with Rare-Event Augmentation), a research project on forecasting India's National Air Quality Index one to three days ahead. The emphasis is on hazardous days and on cities a model has never been trained on.

Each row is one city on one Indian calendar day. It holds six measured pollutants, the CPCB AQI calculated from them, eight meteorological variables from ERA5, three fire-smoke variables from NASA FIRMS, and calendar and city descriptors. There are **38,355 city-days across 15 cities, from 1 January 2019 to 31 December 2025.**

> **Before reusing the pollutant tables, read [Licence and attribution](#licence-and-attribution).** The meteorology and fire data are openly licensed. The pollutant data are derived from CPCB, whose terms require its approval for any use other than downloading.

Code, notebooks and tests are at **https://github.com/Aduitya9/DEXTRA**.

## Contents

| Path | Files | What it is | From |
|---|---|---|---|
| `processed/city_daily_features.parquet` | 1 | **The modelling table.** 38,355 city-days, 26 variates, AQI and a split label | all sources |
| `processed/city_daily_scaled.parquet` | 1 | The same table, scaled with statistics fitted on the training split only | all sources |
| `processed/pollutant_daily.parquet` | 1 | Daily city pollutants and AQI | CPCB |
| `processed/meteo_daily.parquet` | 1 | Daily meteorology | ERA5 |
| `processed/fire_daily.parquet` | 1 | Daily fire counts and radiative power near each city | NASA FIRMS |
| `processed/extreme_episodes.parquet` | 1 | 720 runs of AQI above 200 lasting at least two days | derived |
| `processed/scaler_params.json` | 1 | Scaling parameters, fitted on the training split only | derived |
| `processed/baseline_*`, `processed/itransformer_*` | 4 | Forecasts and scores from the baselines and the iTransformer backbone | derived |
| `models/itransformer.pt`, `models/itransformer.json` | 2 | Trained iTransformer weights and the architecture needed to load them | derived |
| `raw/era5_surface_grid/` | 15 | Hourly ERA5 surface fields, one file per city, raw grid cell | ERA5 via Open-Meteo |
| `raw/era5_pressure/` | 7 | Temperature at 850–950 hPa at 00 and 03 UTC, one file per year | ERA5 via CDS |
| `raw/era5_blh_patch/` | 1 | Boundary-layer height for Jan–Jul 2024, which the Open-Meteo archive lacks | ERA5 via CDS |
| `raw/firms/` | 7 | MODIS Collection 6.1 active-fire detections, one file per year | NASA FIRMS |
| `raw/era5_surface/`, `raw/era5_inversion/` | 17 | Superseded earlier pulls, kept as an audit trail and **not used** | ERA5 |
| `meta/` | 4 | City list, static city features, Diwali dates, ECMWF job log | — |

Raw CPCB station files are **not** included. See [Pollutants — CPCB](#pollutants--cpcb).

## The modelling table

### Splits

The `split` column assigns every city-day to one of six roles. Ten cities are used for training and five are held out entirely.

| Split | City-days | Cities | Dates | Used for |
|---|---|---|---|---|
| `train` | 18,260 | 10 | 2019–2023 | Fitting |
| `val` | 3,660 | 10 | 2024 | Early stopping and model selection |
| `test_seen` | 3,650 | 10 | 2025 | Test on the training cities |
| `support_unseen` | 155 | 5 | Dec 2024 | The 31 days a few-shot method may adapt on |
| `test_unseen` | 1,825 | 5 | 2025 | Test on the held-out cities, the same year as `test_seen` |
| `zeroshot_unseen` | 10,805 | 5 | Jan 2019 – Nov 2024 | Optional zero-shot evaluation. It precedes the support window, so nothing can have adapted on it |

Splits are by date and never shuffled. Every held-out city gets the same 31 support days, and both test splits cover the same year, so the gap between them measures what a new city costs rather than which year was dirtier.

### Variates

| Group | Variates | Notes |
|---|---|---|
| Pollutants (6) | `PM2.5`, `PM10`, `NO2`, `SO2`, `O3` in µg/m³; `CO` in mg/m³ | Daily values follow CPCB rules: a 24-hour mean, except CO and O3, which take the highest 8-hour running mean. At least 16 valid hours are required. The city value is the median across its stations |
| Meteorology (8) | `blh`, `blh_min`, `blh_anomaly` (m); `temp2m` (°C); `wind_speed` (m/s); `wind_dir` (°); `precip` (mm); `inv_idx` (°C) | `blh_min` is the daily minimum boundary-layer height. `blh_anomaly` is measured against the city's own trailing 30-day mean. `wind_dir` is a circular mean. `inv_idx` is the temperature at a level about 75 hPa above the city's surface, minus 2 m temperature, taking the stronger of the 00 and 03 UTC readings |
| Fire (4) | `fire_count_100km`, `fire_count_500km`, `frp_500km` (MW), `is_stubble_season` | Vegetation fires with confidence 70 or above. The stubble season runs from 15 October to 30 November |
| Calendar (4) | `day_of_week`, `month`, `day_of_year`, `is_diwali` | `is_diwali` covers three days either side of Lakshmi Puja |
| City (4) | `archetype_id`, `station_density`, `coastal_flag`, `industrial_index` | `station_density` is the number of stations reporting that day. `industrial_index` is the median SO2/NO2 ratio, computed from training years for training cities and from the support window only for held-out cities |

**Target:** `AQI`, the CPCB National Air Quality Index. It is the highest sub-index across at least three pollutants, one of which must be PM2.5 or PM10, and it is capped at 500.

### Processing that matters when you use it

- **All days are Indian Standard Time calendar days.** CPCB timestamps are local time. ERA5 and FIRMS are UTC and are shifted by +5 h 30 min before grouping.
- **The 2024 boundary-layer gap is patched.** Open-Meteo's ERA5 archive has no boundary-layer height for January to June 2024 in any city. Those hours were refetched from CDS, and on an overlap month the two sources agree to about 1.3 m.
- **The inversion level is chosen per city** (850, 875, 900 or 925 hPa). A fixed 925 hPa level lies below the ground in Bengaluru, where mean surface pressure is 917 hPa.
- **Contradictory PM readings are removed.** On any station-day where PM2.5 exceeds PM10, or is under 5% of it, both PM channels are discarded before the city median. This removed 12 city-days in seven years, 11 of them in Guwahati.
- **Days without fires are zeros, not missing values.**

## Cities

| City | State | Archetype | Coastal | Role | Stations used |
|---|---|---|---|---|---|
| Delhi | Delhi | IGP-Biomass | no | training | 6 |
| Mumbai | Maharashtra | Coastal-Mixed | yes | training | 6 |
| Kolkata | West Bengal | Industrial | no | training | 6 |
| Bengaluru | Karnataka | Traffic-Urban | no | training | 6 |
| Chennai | Tamil Nadu | Coastal-Mixed | yes | training | 6 |
| Hyderabad | Telangana | Traffic-Urban | no | training | 6 |
| Ahmedabad | Gujarat | Industrial | no | training | 6 |
| Pune | Maharashtra | Traffic-Urban | no | training | 5 |
| Lucknow | Uttar Pradesh | IGP-Biomass | no | training | 6 |
| Jaipur | Rajasthan | Traffic-Urban | no | training | 3 |
| Kanpur | Uttar Pradesh | Industrial | no | held out | 4 |
| Patna | Bihar | IGP-Biomass | no | held out | 6 |
| Bhopal | Madhya Pradesh | Traffic-Urban | no | held out | 2 |
| Varanasi | Uttar Pradesh | IGP-Biomass | no | held out | 4 |
| Guwahati | Assam | Traffic-Urban | no | held out | 4 |

Stations were chosen for coverage first, then spread apart, because a city value is a median across stations. All four archetypes appear among the training cities. Coastal-Mixed appears only among them.

## Licence and attribution

The sources carry different terms, so there is no single licence for the whole repository. Credit every source you use.

### Meteorology — ERA5

ECMWF ERA5 reanalysis, published by the Copernicus Climate Change Service (C3S) under a **CC-BY** licence.

- Hersbach, H. et al. *ERA5 hourly data on single levels from 1940 to present.* C3S Climate Data Store. doi:[10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47)
- Hersbach, H. et al. *ERA5 hourly data on pressure levels from 1940 to present.* C3S Climate Data Store. doi:[10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6)
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. doi:[10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

> Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

### Meteorology — Open-Meteo

The hourly surface fields were retrieved through the [Open-Meteo](https://open-meteo.com/) Historical Weather API (`models=era5`, raw grid cell, no elevation correction) and then aggregated, so changes were made. Open-Meteo API data are offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Weather data by [Open-Meteo.com](https://open-meteo.com/).
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API.* Zenodo. doi:[10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649)

### Fire — NASA FIRMS

MODIS Collection 6.1 active-fire detections, standard-processing archive, from NASA's Fire Information for Resource Management System. NASA data are openly available without restriction.

> We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://www.earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) and NASA's Earth Science Data and Information System (ESDIS).

- Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41. doi:[10.1016/j.rse.2016.02.054](https://doi.org/10.1016/j.rse.2016.02.054)

FIRMS data are provided as is, and users bear responsibility for their use. If you pass FIRMS data on, NASA asks that you include or link its [disclaimer](https://www.earthdata.nasa.gov/data/tools/firms).

### Pollutants — CPCB

Hourly station readings from the Central Pollution Control Board's Continuous Ambient Air Quality Monitoring Stations (CAAQMS), obtained through the [CPCB data repository](https://airquality.cpcb.gov.in/ccr/#/repository/data).

- **Terms.** CPCB's [copyright policy](https://cpcb.gov.in/copyright-policy/) allows its material to be downloaded without prior permission. Any other use needs the approval of CPCB's competent authority, which can be requested at `ccb.cpcb@nic.in`.
- **What is here.** This repository holds only daily city-level aggregates derived from the station data, plus forecasts and model outputs built from them. It does not hold the raw station files.
- **Reproducing from source.** `scripts/fetch_cpcb.py` in the GitHub repository rebuilds the pollutant tables directly from CPCB, using the station list in `meta/cpcb_manifest.csv`.
- **Before reusing** the pollutant-derived tables (`pollutant_daily`, `city_daily_features`, `city_daily_scaled`, `extreme_episodes`, and the forecast files, which contain observed AQI), confirm your use with CPCB.

### This project's own contributions

The derived features, split design, episode list, scaling parameters and model outputs are the work of the DEXTRA project. Please cite the project when you reuse them. They remain subject to the source terms above.

## Known limitations

- **Some held-out and training cities have thin station coverage.** Bhopal has two stations and usually only one reporting, and Jaipur has three. A city-day there can rest on a single sensor. `station_density` records how many stations reported.
- **Three training cities have almost no hazardous days.** Hyderabad never exceeds AQI 200 in seven years, Bengaluru does so once, and Chennai on 1% of days. Rare-event metrics for them are undefined or very noisy.
- **The held-out cities are dirtier than the training cities.** 20.4% of their days exceed AQI 200, against 11.0% for the training cities.
- **Per-city rare-event counts in 2025 can be very small.** Varanasi has only six days above AQI 200 in `test_unseen`, so its recall is a single coin-flip's worth of evidence.
- **CPCB does not state the time zone of its timestamps.** They are treated as IST.
- **A median across at most six stations smooths very local peaks.** The city value describes the city, not its worst street.

## Citation

A paper describing DEXTRA is in preparation. Until it is published, cite the repository:

```
DEXTRA: Diffusion-Enhanced Cross-City Transfer with Rare-Event Augmentation
for AQI Forecasting in Indian Cities. https://github.com/Aduitya9/DEXTRA
```

Please cite the underlying sources listed above as well.

"""
DEXTRA Project Configuration
Central configuration for target cities, feature sets, paths, and hyperparameters.

Reconciled against the three specification documents on 2026-10-24:
  - Research Gap Analysis And Project Blueprint.pdf
  - Dextra Dataset And Workflow.pdf
  - Dexta_Detailed Plan.pdf        (authoritative for architecture)

Where those documents disagreed, the resolution is recorded inline next to the
affected constant. Four such decisions were taken: claimed novelty count,
project timeline, unseen test city split, and AQI event thresholds.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 15 Target Indian Cities with Geolocation & Archetypes
#
# DECISION 3 (test split): Kanpur is held out as an unseen test city and Jaipur
# moves into training. The Blueprint and Detailed Plan both name Kanpur; the
# Workflow named Jaipur. Kanpur is preferred because it maximises archetype
# breadth in the test set: holding out Kanpur covers 3 of the 4 clusters
# (Industrial, IGP-Biomass, Traffic-Urban), whereas holding out Jaipur covers
# only 2 and leaves no Industrial city in test. Coastal-Mixed is necessarily
# train-only -- Mumbai and Chennai are the only two genuinely coastal cities
# here and both are data-rich, so neither fits the data-sparse target premise.
# Kanpur correlates strongly with Delhi/Lucknow in the IGP belt, so report
# per-city transfer gain rather than the mean -- Guwahati is the hard case.
#
# 'archetype' is the descriptive emission profile (read by the mock generator).
# 'archetype_cluster' is the expected K=4 cluster, used only to sanity-check the
# k-means assignment produced by Module 1; the real mapping comes from
# data/processed/city_archetypes.json. This also resolves the Workflow listing
# Kanpur under two clusters at once.
TARGET_CITIES = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi', 'archetype': 'IGP-Biomass', 'archetype_cluster': 'IGP-Biomass', 'stations': 40, 'coastal': False, 'is_training': True},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra', 'archetype': 'Coastal-Mixed', 'archetype_cluster': 'Coastal-Mixed', 'stations': 25, 'coastal': True, 'is_training': True},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal', 'archetype': 'Industrial', 'archetype_cluster': 'Industrial', 'stations': 15, 'coastal': False, 'is_training': True},
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka', 'archetype': 'Traffic-Urban', 'archetype_cluster': 'Traffic-Urban', 'stations': 12, 'coastal': False, 'is_training': True},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu', 'archetype': 'Coastal-Mixed', 'archetype_cluster': 'Coastal-Mixed', 'stations': 10, 'coastal': True, 'is_training': True},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana', 'archetype': 'Traffic-Urban', 'archetype_cluster': 'Traffic-Urban', 'stations': 10, 'coastal': False, 'is_training': True},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat', 'archetype': 'Industrial', 'archetype_cluster': 'Industrial', 'stations': 8, 'coastal': False, 'is_training': True},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra', 'archetype': 'Traffic-Urban', 'archetype_cluster': 'Traffic-Urban', 'stations': 8, 'coastal': False, 'is_training': True},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462, 'state': 'Uttar Pradesh', 'archetype': 'IGP-Biomass', 'archetype_cluster': 'IGP-Biomass', 'stations': 6, 'coastal': False, 'is_training': True},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873, 'state': 'Rajasthan', 'archetype': 'Dust-Traffic', 'archetype_cluster': 'Traffic-Urban', 'stations': 5, 'coastal': False, 'is_training': True},
    # Unseen Test Cities (Few-Shot Cross-City Evaluation)
    'Kanpur': {'lat': 26.4499, 'lon': 80.3319, 'state': 'Uttar Pradesh', 'archetype': 'Industrial-Biomass', 'archetype_cluster': 'Industrial', 'stations': 5, 'coastal': False, 'is_training': False},
    'Patna': {'lat': 25.5941, 'lon': 85.1376, 'state': 'Bihar', 'archetype': 'IGP-Biomass', 'archetype_cluster': 'IGP-Biomass', 'stations': 4, 'coastal': False, 'is_training': False},
    'Bhopal': {'lat': 23.2599, 'lon': 77.4126, 'state': 'Madhya Pradesh', 'archetype': 'Traffic-Urban', 'archetype_cluster': 'Traffic-Urban', 'stations': 4, 'coastal': False, 'is_training': False},
    'Varanasi': {'lat': 25.3176, 'lon': 82.9739, 'state': 'Uttar Pradesh', 'archetype': 'IGP-Biomass', 'archetype_cluster': 'IGP-Biomass', 'stations': 3, 'coastal': False, 'is_training': False},
    'Guwahati': {'lat': 26.1445, 'lon': 91.7362, 'state': 'Assam', 'archetype': 'Valley-Traffic', 'archetype_cluster': 'Traffic-Urban', 'stations': 3, 'coastal': False, 'is_training': False},
}

TRAIN_CITIES = [c for c, meta in TARGET_CITIES.items() if meta['is_training']]
TEST_UNSEEN_CITIES = [c for c, meta in TARGET_CITIES.items() if not meta['is_training']]

# 'coastal' is the authoritative source for the coastal_flag city-meta feature
# (CPCB station within ~50 km of open coast). Only Mumbai and Chennai qualify;
# Kolkata sits ~100 km up the Hooghly and does not.
#
# Guwahati was previously labelled 'Coastal-Mixed' in both the Workflow document
# and this config, but it lies ~450 km inland in the Brahmaputra valley. It is
# reassigned to the Traffic-Urban cluster: the archetype taxonomy is an EMISSION
# profile, and Guwahati's dominant sources are vehicular and construction dust,
# with low 500 km fire exposure and a low SO2/NO2 industrial index. Its defining
# trait -- severe winter valley inversion -- is an atmospheric regime, which the
# Met-MoE gate already models through blh, inv_idx and blh_anomaly. Keeping
# emission profile and dispersion regime in separate mechanisms is deliberate.
_coastal_cluster = {c for c, m in TARGET_CITIES.items() if m['archetype_cluster'] == 'Coastal-Mixed'}
_coastal_flagged = {c for c, m in TARGET_CITIES.items() if m['coastal']}
assert _coastal_cluster == _coastal_flagged, \
    f"Coastal archetype/flag disagree for: {_coastal_cluster ^ _coastal_flagged}"

ARCHETYPE_CLUSTERS = ['IGP-Biomass', 'Coastal-Mixed', 'Industrial', 'Traffic-Urban']
N_ARCHETYPES = len(ARCHETYPE_CLUSTERS)

# Pollutant & Feature Definitions
# The 25 model variates: 6 pollutants + 7 meteorological + 4 fire + 4 calendar
# + 4 city metadata. CITY_META_FEATURES were missing from the original config,
# which would have produced a shape mismatch against the 25-variate backbone.
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
METEO_FEATURES = ['blh', 'temp2m', 'wind_speed', 'wind_dir', 'precip', 'inv_idx', 'blh_anomaly']
FIRE_FEATURES = ['fire_count_100km', 'fire_count_500km', 'frp_500km', 'is_stubble_season']
TEMPORAL_FEATURES = ['day_of_week', 'month', 'day_of_year', 'is_diwali']
CITY_META_FEATURES = ['archetype_id', 'station_density', 'coastal_flag', 'industrial_index']

ALL_FEATURES = POLLUTANTS + METEO_FEATURES + FIRE_FEATURES + TEMPORAL_FEATURES + CITY_META_FEATURES

N_VARIATES = 25
assert len(ALL_FEATURES) == N_VARIATES, f"Expected {N_VARIATES} variates, got {len(ALL_FEATURES)}"

# Meteorological gating signals routed to the MoE router (Novelty 2)
GATE_FEATURES = ['blh', 'wind_speed', 'inv_idx', 'fire_count_500km', 'blh_anomaly', 'is_stubble_season']

# Time Series Window Parameters
LOOKBACK_WINDOW = 14  # 14 days input lookback
FORECAST_HORIZON = 3  # 3 days ahead forecasting (t+1, t+2, t+3)

# Data Coverage
DATA_START_DATE = "2019-01-01"
DATA_END_DATE = "2025-12-31"

# Temporal split boundaries (no leakage: splits are by date, not shuffled).
#
# Three whole years end the record, so validation and test are one calendar year
# each: 2019-2023 train, 2024 validate, 2025 test. Ending validation mid-2024
# would leave it without an October, November or December, and a model chosen on
# a window that never contains the stubble-burning season is chosen on a window
# that never contains the thing this project forecasts. Measured on the CPCB
# data: that half-year holds 121 days above AQI 200, against 280 for the full
# year, and 600 winter-month days against 1,210.
TRAIN_END_DATE = "2023-12-31"
VAL_END_DATE = "2024-12-31"

# ---------------------------------------------------------------------------
# External data retrieval
# ---------------------------------------------------------------------------
# ERA5 surface variables come as hourly point series from the Open-Meteo ERA5
# archive (15 grid cells, not the India grid). Temperature on pressure levels is
# not served there, so it comes from the Copernicus CDS. Open-Meteo is also missing
# boundary-layer height for all of Jan-Jun 2024 (the validation split), which is
# patched from CDS.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CDS_URL = "https://cds.climate.copernicus.eu/api"

# CDS bounding box [N, W, S, E]: the 15 cities with ~1 degree margin.
CDS_AREA = [30.0, 71.5, 12.0, 93.0]

# Inversion strength is measured against the pressure level ~75 hPa above each
# city's mean surface pressure, NOT a fixed 925 hPa: Bengaluru sits at ~914 hPa,
# so 925 hPa is below ground there. Levels span every city's choice.
ERA5_PRESSURE_LEVELS = [850, 875, 900, 925, 950]
INVERSION_OFFSET_HPA = 75.0
# 00 and 03 UTC = 05:30 and 08:30 IST. Inversions are nocturnal/early-morning;
# a daily mean would blend them with afternoon mixing and wash them out.
INVERSION_HOURS_UTC = ["00:00", "03:00"]

# NASA FIRMS active fires. Bounding box W,S,E,N covers India plus the Pakistani
# Punjab and Bangladesh, both inside the 500 km transport radius of target cities.
FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api"
FIRMS_STATUS_URL = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"
FIRMS_BBOX = "68.1,6.5,97.4,35.5"
FIRMS_DAY_RANGE = 5  # API maximum per request
FIRMS_MIN_CONFIDENCE = 70
# A MAP_KEY allows 5000 transactions per 10-minute window, and a multi-day request
# over all of India counts as several; after a refusal, wait out the whole window.
FIRMS_LIMIT_WAIT_SECONDS = 600

# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------
# CPCB timestamps are local Indian Standard Time; ERA5 and FIRMS are UTC. Everything
# is shifted to IST before being grouped into days, so a "day" means the same span
# in every source and matches CPCB's own daily AQI.
IST_OFFSET = "5h30min"

# CPCB National AQI methodology: PM2.5, PM10, NO2, SO2 and NH3 use the 24-hour mean;
# CO and O3 use the highest 8-hour running mean of the day.
POLLUTANTS_24H = ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3']
POLLUTANTS_8H_MAX = ['CO', 'O3']
# PM2.5 is part of PM10, so it can neither exceed it nor be a negligible share of
# it. Measured across 34,832 city-days the ratio has a median of 0.462 and a 1st
# percentile of 0.208, so 0.05 sits far below any real dust event and flags only
# instrument faults: 12 days in seven years, 11 of them Guwahati.
MIN_PM25_PM10_RATIO = 0.05

MIN_HOURS_FOR_DAILY = 16   # CPCB: a 24-hour average needs at least 16 hourly values
MIN_HOURS_FOR_8H = 6       # an 8-hour running mean needs at least 6 of its 8 hours

# Hourly readings outside these bounds are instrument faults, not air quality.
PHYSICAL_LIMITS = {
    'PM2.5': (0.0, 999.0), 'PM10': (0.0, 1500.0), 'NO2': (0.0, 600.0),
    'SO2': (0.0, 1000.0), 'CO': (0.0, 50.0), 'O3': (0.0, 500.0), 'NH3': (0.0, 1000.0),
}

# FIRMS detections kept for the fire features. type 0 is "presumed vegetation fire";
# excluding the other types keeps persistent gas flares and volcanoes from showing up
# as a constant fire count for industrial cities.
FIRMS_VEGETATION_TYPE = 0
FIRE_RADII_KM = {'fire_count_100km': 100.0, 'fire_count_500km': 500.0}
FRP_RADIUS_KM = 500.0

# ---------------------------------------------------------------------------
# AQI Event Thresholds
#
# DECISION 4 (thresholds): two thresholds only, both CPCB category boundaries.
# The specs variously used 200, 250 and 300 for overlapping purposes. The ad-hoc
# AQI > 250 cut for diffusion training is dropped -- the generator now trains on
# the same episode set the loss penalises, which also keeps the scarce extreme
# window count as high as possible (~800-1500 windows is the binding constraint).
# Severity is steered at sampling time instead, via DIFFUSION_SEVERITY_CLASSES.
# ---------------------------------------------------------------------------
HAZARDOUS_AQI_THRESHOLD = 200.0  # CPCB 'Poor' and above; GRAP action begins here.
                                 # Used by: episode extraction, L_extreme penalty,
                                 # MAE_extreme / RMSE_extreme, diffusion training set.
RARE_AQI_THRESHOLD = 300.0       # CPCB 'Very Poor' and above; < 8% of timestamps.
                                 # Used by: rare-event recall / F1, augmentation target.

MIN_EPISODE_DURATION_DAYS = 2    # contiguous days above HAZARDOUS to count as an episode
SYNTHETIC_AQI_VALID_RANGE = (200.0, 500.0)  # generated windows outside this are rejected; CPCB AQI caps at 500
TARGET_TAIL_RATIO = 0.25         # post-augmentation share of AQI > 300 samples (from ~0.08)

# Severity class conditioned on during diffusion sampling (CPCB-aligned)
DIFFUSION_SEVERITY_CLASSES = {0: 'Poor', 1: 'Very Poor', 2: 'Severe'}

# ---------------------------------------------------------------------------
# Claimed Novelties
#
# DECISION 1 (scope): claim 5 novelties, not the 7 in the Detailed Plan. Each
# claimed novelty maps to exactly one ablation row, so every claim is
# independently validated. The two dropped items are real deliverables but
# neither is ablatable, and in an 8-page paper unvalidatable claims read as
# padding. See DEFERRED_CONTRIBUTIONS.
# ---------------------------------------------------------------------------
NOVELTY_CLAIMS = {
    'N1': 'C-STDA: city-conditioned diffusion augmenter for rare AQI episodes',
    'N2': 'Met-MoE: meteorologically gated dual-regime Mixture-of-Experts',
    'N3': 'TS-JEPA pretrained city pollution archetype embeddings',
    'N4': 'LoRA few-shot cross-city adapter (30-day support set)',
    'N5': 'On-device MC-Dropout epistemic uncertainty calibration',
}

# Ablation variant -> the novelty it isolates. Mirrors the paper's ablation table.
ABLATION_VARIANTS = {
    'w/o Diffusion Aug': 'N1',
    'w/o MoE': 'N2',
    'w/o Met Gating': 'N2',  # data-driven gate substituted for the physics gate
    'w/o JEPA Pretrain': 'N3',
    'w/o Cross-City Adapt': 'N4',
}

# Shipped, but described as system contributions rather than numbered novelties.
# The advisory engine is a CPCB rule table, NOT a language model -- the Detailed
# Plan claimed quantized Gemma-2B while specifying a 4 ms SQLite lookup inside a
# 3.5 MB bundle, which cannot contain an SLM. Claim what actually ships.
DEFERRED_CONTRIBUTIONS = {
    'advisory_engine': 'CPCB rule-based offline health advisory (system feature)',
    'cpcb_benchmark': 'First 15-city Indian CPCB multi-archetype benchmark',
    'compute_envelope': 'Full pipeline trainable on free Colab T4, Android-deployable',
    'mcp_a2a': 'MCP tool server + A2A federated sync (future work)',
}

# ---------------------------------------------------------------------------
# Project Timeline
#
# DECISION 2 (timeline): keep the Detailed Plan's 12 weeks rather than the
# 24-week Blueprint/Workflow schedule, but take the full Android application off
# the critical path. The 12-week plan was only unrealistic because it ran ML and
# Android in parallel. No claim in the paper depends on the Compose UI, Room
# persistence or the advisory screens -- only on the ONNX export and a measured
# on-device latency, which a minimal harness provides.
#
# Scheduling risk is data acquisition, not compute (total training is ~7.2 GPU
# hours). Submit the ERA5 CDS requests on day one and clean CPCB while they
# queue; FIRMS additionally needs Earthdata registration and archive approval.
# ---------------------------------------------------------------------------
PROJECT_TIMELINE = {
    'W1-2': 'Data ingestion (CPCB, ERA5, FIRMS), cleaning, EDA, splits',
    'W3': 'Module 1 - TS-JEPA city embedding pretraining',
    'W4': 'Module 2 - C-STDA conditional diffusion augmenter',
    'W5': 'Synthetic hazardous episode generation + tail rebalancing',
    'W6': 'Module 3 - iTransformer backbone + Met-MoE training',
    'W7': 'Module 4 - LoRA cross-city adapter + MC-Dropout calibration',
    'W8': 'Baselines, ablation study, ONNX INT8 export',
    'W9': 'Minimal Android inference harness (on-device latency benchmark)',
    'W10-12': 'IEEE paper writing, figures, submission',
}

MILESTONES = {
    'M1': ('W2', '15-city benchmark dataset with train/val/test splits'),
    'M2': ('W5', 'Diffusion generates physically consistent AQI > 300 episodes'),
    'M3': ('W7', 'DEXTRA-MoE beats vanilla iTransformer by > 25% on severe episodes'),
    'M4': ('W8', 'INT8 ONNX exported, < 4 MB, < 150 ms inference'),
    'M5': ('W9', 'On-device latency verified on physical Android hardware'),
    'M6': ('W12', '8-page double-column IEEE paper finalized'),
}

# Deliberately after submission -- not required by any claim in the paper.
POST_SUBMISSION_WORK = 'Full Android app: Compose UI, Room, advisory screens, MCP/A2A'

# Model Hyperparameters (Colab T4 Friendly)
MODEL_CONFIG = {
    'd_model': 128,
    'n_heads': 8,
    'e_layers': 4,
    'd_ff': 256,
    'dropout': 0.15,  # MC-Dropout: kept ACTIVE at inference for Novelty 5
    'num_experts': 2,
    'gate_input_dim': len(GATE_FEATURES),
    'lora_rank': 4,
    'lora_alpha': 8,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 100
}

# Novelty 5 - on-device Bayesian uncertainty
MC_DROPOUT_CONFIG = {
    'n_passes': 5,
    'uncertainty_flag_threshold': 35.0,  # sigma_aqi above this -> 'UNSTABLE_ATMOSPHERE'
}

# Novelty 1 - conditional DDPM augmenter
DIFFUSION_CONFIG = {
    'timesteps': 200,
    'noise_schedule': 'cosine',
    'train_aqi_threshold': HAZARDOUS_AQI_THRESHOLD,  # see DECISION 4 above
    'batch_size': 32,
    'learning_rate': 2e-4,
    'epochs': 100,
}

# Novelty 3 - self-supervised city embedding pretraining
JEPA_CONFIG = {
    'encoder_layers': 3,
    'mask_ratio': 0.30,
    'ema_momentum': 0.996,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 50,
}

# Novelty 4 - few-shot adaptation to an unseen city
FEW_SHOT_CONFIG = {
    'support_days': 30,
    'epochs': 25,
    'learning_rate': 5e-4,
}

# Multi-objective loss weights: L_fMAE + l1 * L_extreme + l2 * L_balance
LOSS_WEIGHTS = {
    'lambda_extreme': 0.5,
    'lambda_balance': 0.01,
}

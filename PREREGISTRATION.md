# Pre-registration: the final test on 2026

First committed on 3 November 2026, before any 2026 pollution value was read. CPCB's 2026
files have only been checked for dates and counts of readings, on 3 November 2026. No
2026 reading has reached a model, a feature or a table. Amendments are dated below.

## The model

One of five candidates, all the all-token inverted transformer (`src/models/multires.py`),
each trained in notebook 08:

1. **control**: reads the daily and evening variates
2. **anchors as inputs only**: also reads the three nightly anchors as ordinary variates
3. **anchored forecast**: the same inputs, and the forecast starts from a weighted choice
   of the fortnight's mean, today's value and the nightly anchor
4. **72-hour tokens**: the control with each hourly channel's last 72 hours as a token
5. **both**: the anchored forecast with the 72-hour tokens

Which is final is decided in notebook 08 by a rule that reads validation rows only: the
candidate with the lowest mean validation RMSE, among those whose validation average
precision is no more than 0.002 below the control's.

Settings: 14-day lookback, 3-day horizon, d_model 128, 8 heads, 2 layers, dropout 0.3,
lr 3e-4, weight decay 1e-3, batch 256, at most 30 epochs, patience 10, seeds 42, 43, 44.
The nightly anchor's one setting (weight 0.25) was chosen on 2019-2023 alone.

## The test

- **Protocol:** leave one city out. For each of the 15 cities, the three models trained on
  the other 14 cities' 2019-2023 and early-stopped on their 2024, with no retraining. The
  weights are those notebook 08 saved, checked against the fingerprints it recorded.
- **Test rows:** the held-out city's target days from 1 January 2026 until the last day CPCB
  has published when the test is run. That last day is written into this file and committed
  before any of those days is read. Target days from 1 January to 31 October 2026 were
  already published but unread at the first commit; later days had not been published.
- **Inputs:** the same features, built by the same code. Hourly and daily station data from
  the stations in `meta/cpcb_manifest.csv`, ERA5 and FIRMS as in notebooks 01 and 02.
- **Severe:** AQI above 300.

## What is reported, whatever it shows

Primary: pooled average precision for severe days, final model against persistence,
with a 95% interval and p-value from resampling whole held-out cities.

Secondary, Holm-corrected together, on average precision: the final model against
gradient boosting with evening features, against Chronos-2, and against the all-token
head without evening features; and the nightly anchor on its own, with no training,
against persistence on next-day forecasts.

Reported beside them: each step of notebook 08's ablation, RMSE, next-day onsets,
per-city results, and the adaptive alarm against the fixed one (recall, false alarms and
the spread of recall across cities, target recall 70%, step chosen on validation rows).

## What will not happen

No setting, feature, seed or model changes after the first 2026 score is read. If a bug is
found afterwards, the fix and both sets of numbers are reported.

## Amendments

**11 November 2026.** Made before notebook 08 had finished and before any 2026 value was
read. Notebook 08's first run stopped after one of fifteen folds; of that fold, only
validation scores were looked at.

- The candidates were four (the model of notebook 07, with 72-hour tokens, with the
  anchored forecast, with both). They are now the five above, all trained by one code
  path in notebook 08. The second candidate is new: it separates what the anchored
  forecast adds from what the nightly anchor adds as a plain input. The 72-hour
  candidate no longer reads the anchors, so that it changes one thing only.
- The all-token head without evening features is retrained in notebook 08 too, and its
  weights are saved, so every neural model in the final test is frozen there.
- The nightly anchor against persistence was added to the secondary comparisons.
- The first version said the model would be fixed before Diwali on 8 November 2026. It
  was not: the candidates changed on 11 November. No model here can be shaped by 2026
  either way, because training data ends in 2023 and validation in 2024, and CPCB had
  published nothing after 31 October 2026 when this was written.

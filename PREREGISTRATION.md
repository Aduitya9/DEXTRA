# Pre-registration: the final test on 2026

Committed before any 2026 pollution value was read, and before Diwali on 8 November 2026.
As of this commit, CPCB's 2026 files have only been checked for dates and counts of
readings, on 3 November 2026. No 2026 reading has reached a model, a feature or a table.

## The model

One of four candidates, all the all-token inverted transformer reading the daily and
evening variates (`src/models/met_moe.py` with `router='none'`, and
`src/models/multires.py`): as it stands, with 72-hour tokens added, with the anchored
forecast added (mean, today, nightly anchor, weighed window by window), or with both.
Which is final is decided in notebook 08 by a rule that reads validation rows only: the
candidate with the lowest mean validation RMSE, among those whose validation average
precision is no more than 0.002 below the first candidate's.

Settings: 14-day lookback, 3-day horizon, d_model 128, 8 heads, 2 layers, dropout 0.3,
lr 3e-4, weight decay 1e-3, batch 256, at most 30 epochs, patience 10, seeds 42, 43, 44.

## The test

- **Protocol:** leave one city out. For each of the 15 cities, the three models trained on
  the other 14 cities' 2019-2023 and early-stopped on their 2024, with no retraining.
- **Test rows:** the held-out city's target days from 1 January 2026 until the last day CPCB
  has published when the test is run. That last day is written into this file and committed
  before any of those days is read. Target days from 1 January to 31 October 2026 were
  already published but unread at this commit; later days had not been published.
- **Inputs:** the same features, built by the same code. Hourly and daily station data from
  the stations in `meta/cpcb_manifest.csv`, ERA5 and FIRMS as in notebooks 01 and 02.
- **Severe:** AQI above 300.

## What is reported, whatever it shows

Primary: pooled average precision for severe days, final model against persistence,
with a 95% interval and p-value from resampling whole held-out cities.

Secondary, Holm-corrected together: final model against gradient boosting with evening
features, Chronos-2 and the all-token head without evening features, on average
precision. Reported beside them: RMSE, next-day onsets, per-city results, and the
adaptive alarm against the fixed one (recall, false alarms and the spread of recall
across cities, target recall 70%, step chosen on validation rows).

## What will not happen

No setting, feature, seed or model changes after the first 2026 score is read. If a bug is
found afterwards, the fix and both sets of numbers are reported.

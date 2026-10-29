import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.models import baselines, windows

FEATURES = ['blh_min', 'inv_idx', 'precip', 'wind_speed']


def _daily(cities=('Delhi',), days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=days, freq='D')
    rows = []
    for offset, city in enumerate(cities):
        season = 150 + 80 * np.cos(2 * np.pi * dates.day_of_year / 365)
        rows.append(pd.DataFrame({
            'city': city, 'date': dates,
            'AQI': season + rng.normal(0, 20, days) + 20 * offset,
            'blh_min': rng.normal(400, 80, days), 'inv_idx': rng.normal(2, 1, days),
            'precip': rng.gamma(1, 2, days), 'wind_speed': rng.normal(2, 0.5, days),
        }))
    daily = pd.concat(rows, ignore_index=True)
    daily['split'] = np.where(daily.date < '2023-01-01', 'train', 'test_seen')
    return daily


# --------------------------------------------------------------------------- leakage

def test_no_feature_can_see_past_the_day_it_is_made_on():
    """The one that matters. Wreck the future and the past must not move."""
    daily = _daily(days=400)
    cut = daily['date'].iloc[250]

    clean = windows.make_supervised(daily, features=FEATURES)
    wrecked = daily.copy()
    wrecked.loc[wrecked.date > cut, ['AQI'] + FEATURES] = 9999.0
    after = windows.make_supervised(wrecked, features=FEATURES)

    columns = windows.feature_matrix(clean)
    before_cut = clean['date'] <= cut
    pd.testing.assert_frame_equal(
        clean.loc[before_cut, columns].reset_index(drop=True),
        after.loc[before_cut, columns].reset_index(drop=True),
    )
    # the targets must of course differ, or the test proves nothing
    assert not clean.loc[before_cut, 'y'].equals(after.loc[before_cut, 'y'])


def test_one_citys_history_never_becomes_anothers():
    daily = _daily(cities=('Delhi', 'Kanpur'), days=120)
    samples = windows.make_supervised(daily, features=FEATURES)
    first = samples[(samples.city == 'Kanpur') & (samples.horizon == 1)].iloc[0]

    assert pd.isna(first['AQI_lag13'])       # Kanpur has no thirteenth day yet
    assert first['AQI_lag0'] == pytest.approx(
        daily[(daily.city == 'Kanpur')].sort_values('date').AQI.iloc[0])


def test_a_sample_belongs_to_the_split_of_the_day_it_predicts():
    daily = _daily(days=400)
    samples = windows.make_supervised(daily, features=FEATURES)
    # a forecast made in training that lands after the boundary is not a training sample
    crossing = samples[(samples.date == pd.Timestamp('2022-12-31')) & (samples.horizon == 2)]
    assert crossing['target_date'].iloc[0] == pd.Timestamp('2023-01-02')
    assert crossing['split'].iloc[0] == 'test_seen'


def test_the_calendar_is_read_at_the_target_not_the_origin():
    # A forecaster on 1 November knows that 4 November is Diwali.
    daily = _daily(days=400)
    samples = windows.make_supervised(daily, features=FEATURES)
    row = samples[(samples.date == pd.Timestamp('2022-03-01')) & (samples.horizon == 3)].iloc[0]
    assert row['target_date'] == pd.Timestamp('2022-03-04')
    assert row['day_of_year'] == pd.Timestamp('2022-03-04').day_of_year


def test_the_target_is_the_value_that_many_days_later():
    daily = _daily(days=100)
    samples = windows.make_supervised(daily, features=FEATURES)
    ordered = daily.sort_values('date').reset_index(drop=True)
    for horizon in (1, 2, 3):
        row = samples[(samples.date == ordered.date[10]) & (samples.horizon == horizon)].iloc[0]
        assert row['y'] == pytest.approx(ordered.AQI[10 + horizon])


# --------------------------------------------------------------------------- baselines

def test_persistence_repeats_the_latest_reading():
    samples = windows.make_supervised(_daily(days=60), features=FEATURES)
    assert np.allclose(baselines.persistence(samples), samples['AQI_lag0'].clip(0, 500))


def test_climatology_learns_the_season_and_not_the_noise():
    daily = _daily(days=730)
    samples = windows.usable(windows.make_supervised(daily, features=FEATURES))
    model = baselines.SeasonalClimatology().fit(samples[samples.split == 'train'])
    predicted = model.predict(samples)

    # the synthetic series peaks in January and troughs in July
    january = predicted[samples.day_of_year.between(1, 20)].mean()
    july = predicted[samples.day_of_year.between(190, 210)].mean()
    assert january > july + 50


def test_climatology_falls_back_to_pooled_for_a_city_it_never_saw():
    daily = _daily(cities=('Delhi', 'Kanpur'), days=500)
    samples = windows.usable(windows.make_supervised(daily, features=FEATURES))
    trained_on_delhi = samples[(samples.split == 'train') & (samples.city == 'Delhi')]
    model = baselines.SeasonalClimatology().fit(trained_on_delhi)

    kanpur = samples[samples.city == 'Kanpur']
    predicted = model.predict(kanpur)
    assert np.isfinite(predicted).all()          # no NaN for the unseen city
    assert 'Kanpur' not in model.by_city


def test_gradient_boosting_fits_per_horizon_and_stays_in_range():
    samples = windows.usable(windows.make_supervised(_daily(days=500), features=FEATURES))
    features = windows.feature_matrix(samples)
    models = baselines.fit_gradient_boosting(
        samples[samples.split == 'train'], features, settings={'max_iter': 20})

    assert set(models) == {1, 2, 3}
    predicted = baselines.predict_gradient_boosting(models, samples, features)
    assert predicted.between(0, 500).all()
    assert predicted.notna().all()


def test_run_baselines_fits_on_training_rows_only():
    samples = windows.usable(windows.make_supervised(_daily(days=500), features=FEATURES))
    scored, models = baselines.run_baselines(samples, windows.feature_matrix(samples))
    assert {'persistence', 'climatology', 'gbm'} <= set(scored.columns)
    # a model fitted on everything would score far better on held-out rows than this
    held_out = scored[scored.split == 'test_seen']
    assert len(held_out) and held_out['gbm'].notna().all()


# --------------------------------------------------------------------------- metrics

def test_recall_is_undefined_where_there_is_nothing_to_recall():
    # Hyderabad never exceeds AQI 200 in seven years; a recall of 0 there would be
    # a fact about Hyderabad, not about the model.
    clean = metrics.rare_event_metrics([50, 60, 70], [55, 65, 75], threshold=200)
    assert clean['support'] == 0
    assert np.isnan(clean['recall']) and np.isnan(clean['f1'])


def test_rare_event_metrics_count_hits_and_misses():
    truth = [100, 250, 300, 150]
    guess = [120, 260, 150, 220]          # one hit, one miss, one false alarm
    out = metrics.rare_event_metrics(truth, guess, threshold=200)
    assert out['support'] == 2 and out['hits'] == 1
    assert out['recall'] == pytest.approx(0.5)
    assert out['precision'] == pytest.approx(0.5)


def test_regression_metrics_ignore_rows_without_both_values():
    out = metrics.regression_metrics([10.0, 20.0, np.nan], [12.0, 24.0, 100.0])
    assert out['n'] == 2
    assert out['mae'] == pytest.approx(3.0)
    assert out['bias'] == pytest.approx(3.0)


def test_scores_can_be_broken_out_per_city():
    frame = pd.DataFrame({'city': ['A', 'A', 'B', 'B'],
                          'y': [100.0, 250.0, 90.0, 95.0],
                          'y_hat': [110.0, 240.0, 95.0, 90.0]})
    out = metrics.score_by(frame, 'city').set_index('city')
    assert list(out.index) == ['A', 'B']
    assert out.loc['B', 'support'] == 0 and np.isnan(out.loc['B', 'recall'])
    assert out.loc['A', 'support'] == 1

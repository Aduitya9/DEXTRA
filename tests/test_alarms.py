import numpy as np
import pandas as pd
import pytest

from src.evaluation import alarms


def test_an_alarm_counts_hits_misses_and_false_alarms():
    truth = [350, 320, 250, 100, 310]
    forecast = [340, 280, 305, 90, np.nan]                # the last row has no forecast
    assert alarms.alarm_scores(truth, forecast) == {
        'hits': 1, 'severe': 2, 'false_alarms': 1, 'recall': 0.5, 'precision': 0.5, 'f1': 0.5}


def test_f1_is_zero_when_nothing_is_raised_and_undefined_when_nothing_happened():
    assert alarms.alarm_scores([350, 320], [100, 100])['f1'] == 0.0
    assert np.isnan(alarms.alarm_scores([100, 120], [100, 100])['f1'])
    assert alarms.alarm_scores([100, 120], [310, 100])['f1'] == 0.0      # a false alarm, nothing to find


def test_a_forecast_biased_low_gets_a_lower_alarm_from_validation():
    rng = np.random.default_rng(0)
    truth = np.concatenate([rng.uniform(320, 420, 50), rng.uniform(50, 250, 450)])
    forecast = truth - 40                                 # ranks every day right, sits 40 below
    level = alarms.tune_alarm(truth, forecast)
    assert 238 <= level <= 280
    assert alarms.alarm_scores(truth, forecast, level)['f1'] == 1.0
    assert alarms.alarm_scores(truth, forecast)['f1'] < 1.0               # at 300 it misses some


def test_a_forecast_biased_high_gets_a_higher_alarm_from_validation():
    # A 90th-percentile forecast sits above the truth on purpose; capping the
    # alarm near 300 would handicap exactly the forecasts built to catch peaks.
    rng = np.random.default_rng(1)
    truth = np.concatenate([rng.uniform(320, 420, 50), rng.uniform(50, 250, 450)])
    forecast = truth + 60
    level = alarms.tune_alarm(truth, forecast)
    assert 310 <= level <= 380
    assert alarms.alarm_scores(truth, forecast, level)['f1'] == 1.0


def test_a_tie_goes_to_the_official_line():
    truth, forecast = [100, 400, 120, 450], [100, 400, 120, 450]          # every level 120..398 is perfect
    assert alarms.tune_alarm(truth, forecast) == 300.0


def test_severe_bias_reads_the_severe_days_alone():
    assert alarms.severe_bias([400, 100, 350], [330, 300, 300]) == pytest.approx(-60.0)
    assert np.isnan(alarms.severe_bias([100, 200], [500, 500]))


def test_return_to_normal_tables_the_forecasts_issued_on_severe_days():
    frame = pd.DataFrame({'horizon': [1, 2, 1, 2], 'persistence': [350, 350, 100, 100],
                          'y': [360, 340, 90, 95], 'model': [300, 250, 100, 100]})
    table = alarms.return_to_normal(frame, ['model'])
    assert table['forecasts'].tolist() == [1, 1]          # only the day that was severe
    assert table['model'].tolist() == [300, 250] and table['y'].tolist() == [360, 340]


def test_pooled_f1_and_a_city_bootstrap_around_it():
    counts = pd.DataFrame({'city': list('ABCDEF'), 'hits': [10, 0, 5, 8, 2, 0],
                           'severe': [20, 0, 10, 10, 6, 1], 'raised': [12, 3, 6, 9, 2, 0]})
    point = alarms.pooled_f1(counts)
    assert point == pytest.approx(2 * 25 / (47 + 32))
    low, high = alarms.bootstrap_groups(counts, draws=500)
    assert low <= point <= high and low < high
    assert alarms.bootstrap_groups(counts, draws=500) == (low, high)      # seeded: the same twice

import numpy as np
import pandas as pd

from src.evaluation.physics import EXPECTED_SIGN, failures, physics_report, season_of

RNG = np.random.default_rng(0)


def _year(years=3):
    dates = pd.date_range('2021-01-01', periods=365 * years, freq='D')
    return pd.DataFrame({'date': dates, 'AQI': RNG.normal(150, 40, len(dates))})


def _report(frame, feature, expected_sign):
    return physics_report(frame, expectations={feature: (expected_sign, 'because')}).iloc[0]


def test_seasons_follow_the_indian_calendar():
    dates = pd.Series(pd.to_datetime(['2023-01-15', '2023-04-10', '2023-07-20', '2023-11-05']))
    assert list(season_of(dates)) == ['winter', 'pre-monsoon', 'monsoon', 'post-monsoon']


def test_a_feature_that_always_moves_with_aqi_is_consistent():
    frame = _year()
    frame['f'] = frame['AQI'] * 0.5 + RNG.normal(0, 10, len(frame))
    row = _report(frame, 'f', +1)
    assert row['verdict'] == 'consistent'
    assert row['winter'] > 0 and row['monsoon'] > 0


def test_a_feature_that_reverses_is_reported_as_seasonal_and_names_the_season():
    # This is blh_anomaly: a lid in winter, a symptom of dust-raising heat in summer.
    frame = _year()
    summer = frame['date'].dt.month.isin([3, 4, 5])
    frame['f'] = np.where(summer, frame['AQI'], -frame['AQI']) + RNG.normal(0, 5, len(frame))
    row = _report(frame, 'f', -1)

    assert row['verdict'] == 'seasonal'          # not a failure
    assert 'pre-monsoon' in row['note']
    assert row['winter'] < 0 < row['pre-monsoon']


def test_a_feature_that_contradicts_the_physics_all_year_is_a_real_failure():
    frame = _year()
    frame['f'] = frame['AQI'] * 0.5 + RNG.normal(0, 10, len(frame))
    row = _report(frame, 'f', -1)                # claim it should fall with AQI; it rises
    assert row['verdict'] == 'unexpected'
    assert len(failures(physics_report(frame, expectations={'f': (-1, 'because')}))) == 1


def test_noise_is_called_weak_rather_than_given_a_sign():
    frame = _year()
    frame['f'] = RNG.normal(0, 1, len(frame))
    assert _report(frame, 'f', +1)['verdict'] == 'weak'


def test_the_real_feature_set_is_all_declared():
    # Anything the gate or the model reads should have a stated expectation, or the
    # report silently skips it and a broken feature passes unexamined.
    from src.config import GATE_FEATURES
    undeclared = [f for f in GATE_FEATURES if f not in EXPECTED_SIGN
                  and not f.startswith('is_')]
    assert undeclared == []

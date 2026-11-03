import numpy as np
import pandas as pd
import pytest

from src.evaluation import alarms, noise


def _network(sites=4, days=4000, truth=60.0, sd=5.0, offset=0.0, seed=0):
    """A city whose stations all see the same air, each with its own independent error."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2019-01-01', periods=days)
    rows = []
    for site in range(sites):
        rows.append(pd.DataFrame({'city': 'X', 'date': dates, 'site_id': site,
                                  'PM2.5': truth + site * offset + rng.normal(0, sd, days),
                                  'NO2': 10.0, 'SO2': 5.0}))      # AQI needs three pollutants; PM2.5 decides it
    return pd.concat(rows, ignore_index=True)


def _aqi_slope_at(pm25):
    """Between 60 and 90 ug/m3 of PM2.5 the CPCB index climbs 99 points per 29.9 ug/m3."""
    return 99 / 29.9


@pytest.mark.parametrize('sites', [2, 4, 6])
def test_the_estimate_recovers_a_known_noise(sites):
    """Even halves: the full network's mean error has SD sd / sqrt(k), and that is what is estimated."""
    sd = 3.0
    stations = _network(sites=sites, truth=75.0, sd=sd)                  # stays inside one AQI band
    halves = noise.split_network(stations, seed=1)
    measured = noise.day_to_day_noise(halves)
    full = (stations.groupby('date')['PM2.5'].median() - 75.0) * _aqi_slope_at(75.0)
    assert measured['noise'].std() == pytest.approx(full.std(), rel=0.12)


def test_a_lasting_difference_between_stations_is_not_noise():
    """One station always dirtier is something a forecaster can learn, so it must not count."""
    quiet = noise.day_to_day_noise(noise.split_network(_network(truth=70.0, sd=2.0), seed=2))
    offset = noise.day_to_day_noise(noise.split_network(_network(truth=70.0, sd=2.0, offset=2.0), seed=2))
    assert offset['noise'].std() == pytest.approx(quiet['noise'].std(), rel=0.1)


def test_halves_are_fixed_over_time():
    """A split redrawn each day would count permanent differences between stations as noise."""
    stations = _network(sites=4, days=100, truth=70.0, sd=1.0, offset=5.0)
    halves = noise.split_network(stations, seed=3)
    difference = halves['A'] - halves['B']
    assert difference.std() < 0.5 * difference.abs().mean()              # steady, not flipping sign


def test_a_single_station_city_is_left_out():
    assert len(noise.split_network(_network(sites=1, days=10))) == 0


def test_noise_is_read_at_its_own_level():
    pool = pd.DataFrame({'level': [0.0] * 50 + [300.0] * 50, 'noise': [1.0] * 50 + [9.0] * 50})
    out = noise.draw_noise([50.0, 310.0], pool, np.random.default_rng(0))
    assert out.tolist() == [1.0, 9.0]


def test_ceilings_fall_as_noise_grows():
    y = np.concatenate([np.full(900, 100.0), np.linspace(250, 350, 100)])
    today = np.full(1000, 100.0)
    small = pd.DataFrame({'level': np.repeat(noise.LEVELS[:-1], 200), 'noise': np.tile(np.random.default_rng(0).normal(0, 2, 200), len(noise.LEVELS) - 1)})
    large = small.assign(noise=small.noise * 20)
    a, b = noise.ceilings(y, today, small, draws=20), noise.ceilings(y, today, large, draws=20)
    assert a['rmse_floor'] < b['rmse_floor'] and a['ap_ceiling'] > b['ap_ceiling']


def test_reducible_share():
    assert noise.reducible_share(mse_model=500, mse_reference=800, mse_floor=200) == pytest.approx(0.5)


def test_recall_level_flags_the_share_asked_for():
    truth = np.array([400.0] * 100)
    forecast = np.arange(100.0)
    level = alarms.recall_level(truth, forecast, miss_rate=0.3)
    assert (forecast > level).mean() == pytest.approx(0.7, abs=0.02)


def test_the_adaptive_alarm_reaches_its_target_recall_whatever_the_bias():
    rng = np.random.default_rng(0)
    truth = np.where(rng.random(4000) < 0.2, 350.0, 100.0)
    for bias in (-80.0, 0.0, 80.0):
        forecast = truth + bias + rng.normal(0, 30, 4000)
        flags = alarms.adaptive_alarm(truth, forecast, start=300.0, step=10.0, miss_rate=0.3)
        assert flags[truth > 300].mean() == pytest.approx(0.7, abs=0.03)


def test_the_adaptive_alarm_uses_no_outcome_before_it_is_known():
    truth = np.array([400.0, 100.0, 100.0, 100.0, 100.0])
    forecast = np.array([100.0, 299.0, 299.0, 299.0, 299.0])
    # The miss on day 0 lowers the level by 7 only once it is known, `delay` days later.
    flags = alarms.adaptive_alarm(truth, forecast, start=300.0, step=10.0, miss_rate=0.3, delay=3)
    assert flags.tolist() == [False, False, False, True, True]


def test_a_city_with_no_severe_days_keeps_its_starting_level():
    flags = alarms.adaptive_alarm(np.full(100, 80.0), np.full(100, 299.0), start=300.0, step=40.0)
    assert not flags.any()


def test_choose_step_prefers_the_smallest_that_is_good_enough():
    rng = np.random.default_rng(1)
    truth = np.where(rng.random(3000) < 0.2, 350.0, 100.0)
    forecast = truth + rng.normal(0, 20, 3000)
    cities = np.repeat(['a', 'b', 'c'], 1000)
    assert alarms.choose_step(cities, truth, forecast, start=300.0) == 5.0


def test_choose_step_falls_back_to_the_closest_when_none_is_good_enough():
    truth = np.array([350.0, 100.0] * 20)
    forecast = np.array([100.0, 100.0] * 20)
    cities = np.repeat(['a'], 40)
    assert alarms.choose_step(cities, truth, forecast, start=400.0, tolerance=0.0) in alarms.ADAPTIVE_STEPS

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import hourly
from src.models import deep_baselines, multires


def _table(city='Delhi', start='2024-01-01', days=10, value=None):
    stamps = pd.date_range(start, periods=days * 24, freq='h')
    data = {c: np.arange(len(stamps), dtype=float) + i * 1000 for i, c in enumerate(multires.HOURLY_CHANNELS)}
    frame = pd.DataFrame(data)
    frame.insert(0, 'timestamp', stamps)
    frame.insert(0, 'city', city)
    return frame


def test_a_window_ends_at_the_last_hour_of_its_own_day_and_never_later():
    table = _table()
    index = pd.DataFrame({'city': ['Delhi'], 'date': pd.to_datetime(['2024-01-05'])})
    out = multires.attach_hourly(index, table)
    assert out.shape == (1, multires.HOURS, len(multires.HOURLY_CHANNELS))
    last_hour = (pd.Timestamp('2024-01-05 23:00') - pd.Timestamp('2024-01-01')).components.days * 24 + 23
    assert out[0, -1, 0] == last_hour                                 # 23:00 on the day of issue
    assert out[0, 0, 0] == last_hour - multires.HOURS + 1             # exactly 72 hours back


def test_a_window_without_its_whole_span_stays_empty():
    table = _table()
    index = pd.DataFrame({'city': ['Delhi', 'Delhi', 'Patna'],
                          'date': pd.to_datetime(['2024-01-02', '2024-02-01', '2024-01-05'])})
    out = multires.attach_hourly(index, table)
    assert np.isnan(out).all()                                         # too early, past the table, no such city


def test_cities_never_share_hours():
    table = pd.concat([_table('Delhi'), _table('Patna').assign(**{c: 5.0 for c in multires.HOURLY_CHANNELS})])
    index = pd.DataFrame({'city': ['Patna', 'Delhi'], 'date': pd.to_datetime(['2024-01-05', '2024-01-05'])})
    out = multires.attach_hourly(index, table)
    assert (out[0] == 5.0).all() and (out[1] != 5.0).any()


def test_a_missing_channel_is_absent_rather_than_an_error():
    table = _table().drop(columns='site_wind_speed')
    index = pd.DataFrame({'city': ['Delhi'], 'date': pd.to_datetime(['2024-01-05'])})
    out = multires.attach_hourly(index, table)
    assert np.isnan(out[0, :, multires.HOURLY_CHANNELS.index('site_wind_speed')]).all()
    assert np.isfinite(out[0, :, 0]).all()


def test_hourly_standardising_ignores_gaps():
    hours = torch.tensor([[1.0], [2.0], [3.0], [float('nan')]]).unsqueeze(0)     # (1, 4, 1)
    out = multires.standardise_hourly(hours)
    known = torch.tensor([1.0, 2.0, 3.0])
    expected = (known - known.mean()) / (known.std(unbiased=False) + 1e-5)
    assert out[0, :3, 0].numpy() == pytest.approx(expected.numpy(), abs=1e-4)
    assert out[0, 3, 0] == 0                                           # a gap reads as the window's mean


def test_a_constant_or_absent_channel_gives_a_token_of_zeros():
    constant = torch.full((1, 4, 1), 7.0)
    absent = torch.full((1, 4, 1), float('nan'))
    assert (multires.standardise_hourly(constant) == 0).all()
    assert (multires.standardise_hourly(absent) == 0).all()


def _model(anchors=None, channels=2, **kw):
    return multires.FinalModel(5, 14, 3, 0, anchor_indices=anchors, n_channels=channels, hours=6, d_model=16, n_heads=2,
                               n_layers=1, dropout=0.0, **kw)


def _batch(n=4, nan=False):
    X, H = np.random.default_rng(0).random((n, 14, 5), dtype=np.float32) * 100, np.random.default_rng(1).random((n, 6, 2), dtype=np.float32) * 100
    if nan:
        H[0, :, 1] = np.nan
        H[1, 3:, 0] = np.nan
    return torch.from_numpy(multires.pack(X, H))


def test_forward_gives_one_forecast_per_horizon_and_survives_gaps():
    model = _model()
    out = model(_batch(nan=True))
    assert out.shape == (4, 3) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_the_hourly_channels_reach_the_forecast():
    torch.manual_seed(0)
    model = _model().eval()
    base = _batch(1)
    changed = base.clone()
    changed[0, 14 * 5:] = torch.flip(changed[0, 14 * 5:], dims=[0])
    assert not torch.allclose(model(base), model(changed))


def test_rescaling_the_target_rescales_the_forecast():
    """Per-window normalisation: a city ten times dirtier gets a forecast ten times higher."""
    torch.manual_seed(0)
    model = _model().eval()
    flat = _batch(1)
    bigger = flat.clone()
    bigger[0, :14 * 5:5] *= 10                                          # the target's own series
    assert model(bigger).detach().numpy() == pytest.approx(10 * model(flat).detach().numpy(), rel=1e-2)


def test_the_anchored_forecast_is_as_scale_equivariant_as_the_plain_one():
    """Anchors scale with the city and the weights read standardised tokens: nothing else changes."""
    torch.manual_seed(0)
    model = _model(anchors=[2, 3, 4], channels=0).eval()
    window = torch.rand(1, 14, 5) * 100
    bigger = window.clone()
    bigger[:, :, [0, 2, 3, 4]] *= 10                                    # the target and its anchors
    assert model(bigger).detach().numpy() == pytest.approx(10 * model(window).detach().numpy(), rel=1e-2)


def test_anchor_weights_start_on_the_mean_and_sum_to_one():
    model = _model(anchors=[2, 3, 4], channels=0).eval()
    weights = model.anchor_weights(torch.rand(4, 14, 5).numpy() * 100)
    assert weights.shape == (4, 3, 3)
    assert weights.sum(axis=-1) == pytest.approx(np.ones((4, 3)), abs=1e-5)
    assert (weights[:, :, 0] > 0.85).all()                               # starts as reversible normalisation


def test_each_horizon_reads_its_own_nightly_anchor_from_the_last_day():
    model = _model(anchors=[2, 3, 4], channels=0)
    window = torch.zeros(1, 14, 5)
    window[0, -1, 2:5] = torch.tensor([10.0, 20.0, 30.0])
    window[0, :-1, 2:5] = 999.0                                          # earlier days must not be read
    window[0, :, 0] = torch.arange(14.0)
    anchors = model.anchors(window)
    assert anchors[0, :, 2].tolist() == [10.0, 20.0, 30.0]
    assert anchors[0, :, 1].tolist() == [13.0, 13.0, 13.0]               # today
    assert anchors[0, :, 0].tolist() == pytest.approx([6.5, 6.5, 6.5])   # the fortnight's mean


def test_the_anchors_reach_the_forecast():
    torch.manual_seed(0)
    model = _model(anchors=[2, 3, 4], channels=0).eval()
    window = torch.rand(2, 14, 5) * 100
    moved = window.clone()
    moved[:, -1, 2] += 200
    assert not torch.allclose(model(window)[:, 0], model(moved)[:, 0])


def test_hourly_tokens_and_anchors_work_together():
    model = _model(anchors=[2, 3, 4], channels=2)
    out = model(_batch(nan=True))
    assert out.shape == (4, 3) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_one_anchor_is_needed_per_horizon():
    with pytest.raises(ValueError):
        _model(anchors=[2, 3], channels=0)


def _pm_table(values, start='2024-01-01'):
    """Hourly PM2.5 with PM10 at twice it, from a function of the hour index."""
    n = len(values)
    return pd.DataFrame({'city': 'X', 'timestamp': pd.date_range(start, periods=n, freq='h'),
                         'PM2.5': np.asarray(values, dtype=float), 'PM10': 2 * np.asarray(values, dtype=float)})


def test_the_nightly_anchor_applies_the_recent_ratio_to_tonight():
    """Each night reads half of the next day's mean, so the anchor is twice tonight."""
    days = 30
    hours = np.arange(days * 24)
    level = 20.0
    values = np.where(hours % 24 >= 21, level, 2 * level * 24 / 21 - level * 3 / 21)     # daily mean = 2 * night
    series = _pm_table(values).set_index('timestamp')['PM2.5']
    late, mean = hourly._night_and_day(series)
    assert mean.iloc[0] == pytest.approx(2 * late.iloc[0], rel=1e-6)
    anchor = hourly.night_anchor(series, 1)
    assert anchor.iloc[20] == pytest.approx(2 * late.iloc[20], rel=1e-6)


def test_the_nightly_anchor_knows_nothing_after_the_day_it_is_issued():
    rng = np.random.default_rng(0)
    values = rng.uniform(20, 80, 40 * 24)
    series = _pm_table(values).set_index('timestamp')['PM2.5']
    changed = values.copy()
    changed[25 * 24:] = 900.0                                            # everything after day 24
    series2 = _pm_table(changed).set_index('timestamp')['PM2.5']
    for horizon in (1, 2, 3):
        a, b = hourly.night_anchor(series, horizon), hourly.night_anchor(series2, horizon)
        pd.testing.assert_series_equal(a.iloc[:25], b.iloc[:25])
        assert not np.allclose(a.iloc[25:].to_numpy(dtype=float), b.iloc[25:].to_numpy(dtype=float), equal_nan=True)


def test_anchors_fall_back_on_today_without_history_and_respect_another_pollutant():
    table = _pm_table(np.full(30 * 24, 50.0))
    dates = pd.date_range('2024-01-01', periods=30)
    daily = pd.DataFrame({'city': 'X', 'date': dates, 'PM2.5': 50.0, 'PM10': 100.0, 'AQI': 100.0})
    daily.loc[20, 'AQI'] = 250.0                                         # ozone, say, set this day's AQI
    out = hourly.add_night_anchors(daily, table)
    assert len(out) == 30 and {'anchor1', 'anchor2', 'anchor3'} <= set(out.columns)
    assert out['anchor1'].iloc[0] == 100.0                               # no ratio yet: persistence
    assert out['anchor1'].iloc[20] == 250.0                              # never below another pollutant's index
    assert out['anchor1'].iloc[15] == pytest.approx(100.0, abs=1.0)      # a steady city projects itself


@pytest.mark.parametrize('model', [
    deep_baselines.LSTMForecaster(5, 3, 0, hidden=16, layers=2, dropout=0.0),
    deep_baselines.DLinear(14, 3, 0),
    deep_baselines.PatchTST(14, 3, 0, d_model=16, n_heads=2, n_layers=1, dropout=0.0),
])
def test_baselines_forecast_in_aqi_points_and_learn(model):
    torch.manual_seed(0)
    window = torch.rand(32, 14, 5) * 50 + 100
    answer = window[:, -3:, 0].clone()
    first = model(window)
    assert first.shape == (32, 3)
    assert 50 < first.mean() < 250                                      # back on the AQI scale, not standardised
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = None
    for _ in range(60):
        optimiser.zero_grad()
        loss = ((model(window) - answer) ** 2).mean()
        loss.backward()
        optimiser.step()
    assert loss < ((first.detach() - answer) ** 2).mean()


def test_target_only_baselines_ignore_the_other_variates():
    window = torch.rand(2, 14, 5)
    other = window.clone()
    other[:, :, 1:] = torch.rand(2, 14, 4)
    for model in (deep_baselines.DLinear(14, 3, 0), deep_baselines.PatchTST(14, 3, 0, d_model=16, n_heads=2, n_layers=1, dropout=0.0).eval()):
        assert torch.allclose(model(window), model(other))


def test_dlinear_needs_an_odd_kernel():
    with pytest.raises(ValueError):
        deep_baselines.DLinear(14, 3, 0, kernel=4)


def _hourly_frame(day='2024-11-01', days=2, value=lambda h: 50.0):
    stamps = pd.date_range(day, periods=24 * days, freq='h')
    return pd.DataFrame({'timestamp': stamps, 'site_id': 1, 'station': 's',
                         'PM2.5': [value(i) for i in range(len(stamps))], 'PM10': [2 * value(i) for i in range(len(stamps))]})


def test_city_hourly_takes_the_median_of_stations_on_an_unbroken_grid():
    a = _hourly_frame(value=lambda h: 10.0)
    b = _hourly_frame(value=lambda h: 20.0)
    c = _hourly_frame(value=lambda h: 90.0).iloc[:24]
    out = hourly.city_hourly([a, b, c], 'Delhi')
    assert len(out) == 48 and (out.city == 'Delhi').all()
    assert out['PM2.5'].iloc[0] == pytest.approx(20.0) and out['PM2.5'].iloc[-1] == pytest.approx(15.0)
    assert out['NO2'].isna().all()                                      # a channel nobody reports stays NaN


def test_city_hourly_blanks_a_day_whose_pm_channels_contradict():
    bad = _hourly_frame(days=1, value=lambda h: 100.0)
    bad['PM10'] = 20.0                                                  # PM2.5 above PM10 is impossible
    out = hourly.city_hourly([bad], 'Delhi')
    assert out['PM2.5'].isna().all() and out['PM10'].isna().all()


def test_dynamics_reads_each_part_of_the_day():
    frame = hourly.city_hourly([_hourly_frame(days=1, value=lambda h: 100.0 if h >= 18 else 10.0)], 'Delhi')
    out = hourly.dynamics(frame).iloc[0]
    assert out['h_pm25_evening'] == pytest.approx(100.0) and out['h_pm25_morning'] == pytest.approx(10.0)
    assert out['h_pm25_last'] == pytest.approx(100.0)
    assert out['h_pm25_peakhour'] == 18
    assert out['h_pm25_hours_over90'] == 6


def test_dynamics_of_a_day_uses_only_that_day_and_earlier():
    series = lambda spike: _hourly_frame(days=3, value=lambda h: 500.0 if spike and h >= 48 else 10.0)
    plain = hourly.dynamics(hourly.city_hourly([series(False)], 'Delhi'))
    spiked = hourly.dynamics(hourly.city_hourly([series(True)], 'Delhi'))
    columns = [c for c in plain.columns if c.startswith('h_')]
    pd.testing.assert_frame_equal(plain[columns].iloc[:2], spiked[columns].iloc[:2])

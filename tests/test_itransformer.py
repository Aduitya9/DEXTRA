import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.models import itransformer, sequences  # noqa: E402

FEATURES = ['blh_min', 'inv_idx', 'precip']
LOOKBACK, HORIZON = 14, 3


def _daily(cities=('Delhi',), days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=days, freq='D')
    rows = []
    for offset, city in enumerate(cities):
        rows.append(pd.DataFrame({
            'city': city, 'date': dates,
            'AQI': 150 + 50 * np.sin(np.arange(days) / 9) + rng.normal(0, 5, days) + 40 * offset,
            'blh_min': rng.normal(400, 60, days), 'inv_idx': rng.normal(2, 1, days),
            'precip': rng.gamma(1, 2, days),
        }))
    daily = pd.concat(rows, ignore_index=True)
    daily['split'] = np.where(daily.date < '2022-05-01', 'train', 'val')
    return daily


def _build(daily):
    return sequences.build_sequences(daily, LOOKBACK, HORIZON, features=FEATURES)


# --------------------------------------------------------------------------- windows

def test_a_window_ends_at_t_and_the_answers_start_at_t_plus_one():
    daily = _daily(days=60)
    X, y, index = _build(daily)
    ordered = daily.sort_values('date').reset_index(drop=True)
    names = sequences.feature_names(daily, FEATURES)

    sample = 0
    origin = index.loc[sample, 'date']
    position = int(ordered.index[ordered.date == origin][0])
    assert X[sample].shape == (LOOKBACK, len(names))
    # the last row of the window is day t itself
    assert X[sample][-1, names.index('AQI')] == pytest.approx(ordered.AQI[position], abs=1e-4)
    # the first answer is the very next day, with no overlap
    assert y[sample][0] == pytest.approx(ordered.AQI[position + 1], abs=1e-4)
    assert y[sample][-1] == pytest.approx(ordered.AQI[position + HORIZON], abs=1e-4)


def test_no_window_spans_two_cities():
    daily = _daily(cities=('Delhi', 'Kanpur'), days=40)
    X, y, index = _build(daily)
    names = sequences.feature_names(daily, FEATURES)
    aqi = names.index('AQI')

    for city in ('Delhi', 'Kanpur'):
        own = daily[daily.city == city].AQI
        low, high = own.min() - 1, own.max() + 1
        rows = index.index[index.city == city]
        assert ((X[rows][:, :, aqi] >= low) & (X[rows][:, :, aqi] <= high)).all()


def test_a_sample_belongs_to_the_split_of_its_furthest_target():
    daily = _daily(days=200)
    _, _, index = _build(daily)
    boundary = pd.Timestamp('2022-05-01')
    crossing = index[(index.date < boundary) & (index.last_target_date >= boundary)]

    assert len(crossing)
    assert set(crossing['split']) == {'val'}     # not train, despite the origin


def test_nothing_after_the_window_reaches_into_it():
    daily = _daily(days=200)
    cut = pd.Timestamp('2022-04-01')
    clean_X, _, clean_index = _build(daily)

    wrecked = daily.copy()
    wrecked.loc[wrecked.date > cut, ['AQI'] + FEATURES] = 9999.0
    after_X, _, after_index = _build(wrecked)

    before = clean_index['date'] <= cut
    assert before.any()
    assert np.allclose(clean_X[before.to_numpy()], after_X[before.to_numpy()])


def test_answers_come_from_the_original_column_not_the_filled_one():
    daily = _daily(days=60)
    missing = daily.index[30]
    daily.loc[missing, 'AQI'] = np.nan
    X, y, index = _build(daily)

    gap_day = daily.loc[missing, 'date']
    # no sample is scored on a day whose answer was never measured
    assert not (index['last_target_date'] == gap_day).any() or \
        np.isfinite(y[(index['last_target_date'] == gap_day).to_numpy()]).all() is np.False_


def test_gaps_are_filled_forward_and_then_from_the_training_median():
    daily = _daily(days=60)
    daily.loc[daily.index[:3], 'blh_min'] = np.nan      # leading gap, nothing to carry
    daily.loc[daily.index[40], 'blh_min'] = np.nan      # interior gap
    medians = sequences.pooled_training_medians(daily, FEATURES)
    filled = sequences.fill_gaps(daily, FEATURES, medians)

    assert filled['blh_min'].notna().all()
    assert filled.loc[40, 'blh_min'] == pytest.approx(daily.loc[39, 'blh_min'])   # carried
    assert filled.loc[0, 'blh_min'] == pytest.approx(medians['blh_min'])          # median


# --------------------------------------------------------------------------- model

def _model(n_variates, target_index):
    return itransformer.ITransformer(n_variates, LOOKBACK, HORIZON, target_index,
                                     d_model=32, n_heads=4, n_layers=1, dropout=0.0)


def test_the_model_returns_one_forecast_per_horizon():
    model = _model(4, 3).eval()
    out = model(torch.randn(7, LOOKBACK, 4))
    assert out.shape == (7, HORIZON)
    assert torch.isfinite(out).all()


def test_shifting_the_target_history_shifts_the_forecast_by_the_same_amount():
    """The window normalisation has to be exactly reversible, or the forecast is on
    a different scale from the city it was made for."""
    torch.manual_seed(0)
    model = _model(4, 3).eval()
    window = torch.randn(5, LOOKBACK, 4)

    with torch.no_grad():
        plain = model(window)
        lifted = window.clone()
        lifted[:, :, 3] += 100.0
        assert torch.allclose(model(lifted), plain + 100.0, atol=1e-3)

        stretched = window.clone()
        stretched[:, :, 3] *= 3.0
        assert torch.allclose(model(stretched), plain * 3.0, atol=1e-3)


def test_the_other_variates_actually_change_the_forecast():
    torch.manual_seed(0)
    model = _model(4, 3).eval()
    window = torch.randn(5, LOOKBACK, 4)
    meddled = window.clone()
    meddled[:, :, 0] = meddled[:, :, 0].flip(-1)     # same values, different shape

    with torch.no_grad():
        assert not torch.allclose(model(meddled), model(window), atol=1e-4)


def test_a_level_shift_in_another_variate_is_deliberately_ignored():
    # Standardising inside the window is what lets one model serve Delhi and
    # Chennai, and the price is that a variate's absolute level is discarded.
    torch.manual_seed(0)
    model = _model(4, 3).eval()
    window = torch.randn(5, LOOKBACK, 4)
    lifted = window.clone()
    lifted[:, :, 0] += 5.0

    with torch.no_grad():
        assert torch.allclose(model(lifted), model(window), atol=1e-4)


def test_a_variate_that_never_moves_still_reaches_the_model():
    """archetype_id, coastal_flag and industrial_index are constant for a city, so
    standardising them would leave exactly zero and say nothing about which city
    this is. The same would hide a fortnight wholly inside the stubble season."""
    torch.manual_seed(0)
    model = _model(4, 3).eval()
    window = torch.randn(5, LOOKBACK, 4)
    window[:, :, 0] = 2.0                 # one archetype
    other = window.clone()
    other[:, :, 0] = 3.0                  # a different one

    with torch.no_grad():
        assert not torch.allclose(model(other), model(window), atol=1e-4)

    in_season, out_of_season = window.clone(), window.clone()
    in_season[:, :, 1] = 1.0
    out_of_season[:, :, 1] = 0.0
    with torch.no_grad():
        assert not torch.allclose(model(in_season), model(out_of_season), atol=1e-4)


def test_d_model_must_divide_by_the_head_count():
    with pytest.raises(ValueError, match='must divide'):
        itransformer.ITransformer(4, LOOKBACK, HORIZON, 3, d_model=30, n_heads=4)


def test_training_improves_the_validation_error():
    itransformer.set_seed(0)
    daily = _daily(days=400)
    X, y, index = _build(daily)
    train_rows, val_rows = (index.split == 'train').to_numpy(), (index.split == 'val').to_numpy()

    model = _model(X.shape[2], sequences.feature_names(daily, FEATURES).index('AQI'))
    device = torch.device('cpu')
    _, history = itransformer.train(
        model,
        itransformer.make_loader(X[train_rows], y[train_rows], batch_size=64, shuffle=True),
        itransformer.make_loader(X[val_rows], y[val_rows], batch_size=128),
        device, epochs=8, patience=8, log=lambda *a: None)

    assert history[-1]['val_rmse'] < history[0]['val_rmse']
    assert all(np.isfinite(h['val_rmse']) for h in history)


def test_predict_matches_a_direct_forward_pass():
    itransformer.set_seed(0)
    model = _model(4, 3).eval()
    X = np.random.default_rng(0).normal(size=(10, LOOKBACK, 4)).astype(np.float32)
    with torch.no_grad():
        direct = model(torch.from_numpy(X)).numpy()
    assert np.allclose(itransformer.predict(model, X), direct, atol=1e-5)

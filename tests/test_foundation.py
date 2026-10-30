import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.models import foundation  # noqa: E402

DAYS = pd.date_range('2024-01-01', periods=10, freq='D')


def _table():
    return pd.concat([pd.DataFrame({'city': city, 'date': DAYS, 'AQI': offset + np.arange(10.0),
                                    'wind': -np.arange(10.0)})
                      for city, offset in (('A', 0.0), ('B', 100.0))], ignore_index=True)


def test_a_history_ends_on_the_day_the_forecast_is_issued_and_pads_on_the_left():
    origins = pd.DataFrame({'city': ['A', 'B', 'A'], 'date': [DAYS[9], DAYS[2], DAYS[0]]})
    windows = foundation.history_windows(_table(), origins, 'AQI', 4)
    assert windows[0].tolist() == [6, 7, 8, 9]
    assert np.isnan(windows[1, 0]) and windows[1, 1:].tolist() == [100, 101, 102]
    assert np.isnan(windows[2, :3]).all() and windows[2, 3] == 0.0
    assert windows.dtype == np.float32


def test_a_forecast_day_missing_from_the_table_is_refused():
    origins = pd.DataFrame({'city': ['A'], 'date': [pd.Timestamp('2030-01-01')]})
    with pytest.raises(ValueError, match='missing'):
        foundation.history_windows(_table(), origins, 'AQI', 4)


class Chronos2Pipeline:
    """Named like the real class, which is how predict_quantiles tells the families apart."""

    def __init__(self):
        self.seen = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, batch_size):
        self.seen.extend(inputs)
        return ([torch.full((1, prediction_length, len(quantile_levels)), float(np.nanmean(i['target'])))
                 for i in inputs], None)


class ChronosBoltPipeline:
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        mean = torch.nanmean(inputs, dim=1)
        return mean[:, None, None].expand(-1, prediction_length, len(quantile_levels)).clone(), None


def test_both_families_come_back_in_one_shape():
    target = np.array([[1.0, 2.0, 3.0], [np.nan, 4.0, 6.0]], dtype=np.float32)
    for pipeline in (Chronos2Pipeline(), ChronosBoltPipeline()):
        out = foundation.predict_quantiles(pipeline, target, horizon=3, batch_size=1)
        assert out.shape == (2, 3, 3)
        assert out[:, 0, 0].tolist() == [2.0, 5.0]


def test_covariates_reach_chronos_2_as_past_covariates_and_bolt_refuses_them():
    target = np.ones((2, 3), np.float32)
    covariates = {'wind': np.zeros((2, 3), np.float32)}
    pipeline = Chronos2Pipeline()
    foundation.predict_quantiles(pipeline, target, covariates)
    assert set(pipeline.seen[0]) == {'target', 'past_covariates'}
    assert pipeline.seen[0]['past_covariates']['wind'].shape == (3,)
    with pytest.raises(ValueError, match='Chronos-2'):
        foundation.predict_quantiles(ChronosBoltPipeline(), target, covariates)


def test_forecasting_in_chunks_gives_what_one_pass_gives():
    table = _table()
    origins = table[['city', 'date']].iloc[[3, 15, 9, 19, 0]]
    whole = foundation.forecast(Chronos2Pipeline(), table, origins, 'AQI', ['wind'], length=5,
                                chunk=100, log=lambda *a: None)
    chunked = foundation.forecast(Chronos2Pipeline(), table, origins, 'AQI', ['wind'], length=5,
                                  chunk=2, log=lambda *a: None)
    assert whole.shape == (5, 3, 3)
    assert np.allclose(whole, chunked)

import numpy as np
import pandas as pd
import pytest

from src.evaluation import gap


def test_a_seeded_column_names_its_forecaster_and_seed():
    assert gap.forecaster('plain_seed42') == ('plain', 42)
    assert gap.forecaster('chronos-2-weather q90') == ('chronos-2-weather q90', None)


def test_the_long_frame_has_a_row_per_origin_and_horizon():
    rows = pd.DataFrame({'city': ['A', 'B'], 'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
                         'split': ['val', 'test_loco']})
    answers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    frame = gap.long_frame(rows, answers, [10.0, 20.0], {'m_seed1': answers + 1})
    assert len(frame) == 6
    first = frame[(frame.city == 'A') & (frame.horizon == 3)].iloc[0]
    assert (first.y, first.persistence, first.m_seed1) == (3.0, 10.0, 4.0)


def _frame(bias):
    """Two cities, validation and test, a forecast that sits `bias` off the truth."""
    rng = np.random.default_rng(0)
    truth = np.concatenate([rng.uniform(320, 400, 40), rng.uniform(50, 250, 160)])
    return pd.DataFrame({'city': np.tile(['A', 'B'], 100), 'y': truth, 'f': truth + bias,
                         'split': np.repeat(['val', 'test'], 100)[rng.permutation(200)]})


def test_the_alarm_is_tuned_on_one_set_of_rows_and_read_on_another():
    frame = _frame(-50.0)
    summary, counts = gap.score_column(frame, 'f', frame.split == 'val', frame.split == 'test')
    assert summary['alarm level'] < 300                           # moved down for a low forecast
    assert summary['f1'] > 0.9 > summary['f1 at 300']             # and it pays on rows it never saw
    assert summary['bias on severe days'] == pytest.approx(-50.0)
    assert counts.city.tolist() == ['A', 'B']
    assert counts.severe.sum() == summary['severe'] == (frame[frame.split == 'test'].y > 300).sum()


def test_a_fixed_alarm_level_is_used_as_given():
    frame = _frame(-50.0)
    summary, _ = gap.score_column(frame, 'f', frame.split == 'val', frame.split == 'test', level=300.0)
    assert summary['alarm level'] == 300.0
    assert summary['f1'] == summary['f1 at 300']


def test_folds_tune_their_own_alarms_and_pool_their_counts():
    a, b = _frame(-50.0), _frame(-10.0)
    folds = {'A': a.assign(split=a.split.map({'val': 'val', 'test': 'test_loco'})),
             'B': b.assign(split=b.split.map({'val': 'val', 'test': 'test_loco'}))}
    summary, counts = gap.score_folds(folds, 'f')
    assert summary['severe'] == int(counts.severe.sum())
    assert summary['f1'] == pytest.approx(2 * counts.hits.sum() / (counts.severe.sum() + counts.raised.sum()))
    assert len(counts) == 4                                       # two cities in each fold's test rows


def test_seeds_are_averaged_and_their_spread_kept():
    counts = lambda hits: pd.DataFrame({'city': ['A'], 'hits': [hits], 'severe': [10.0], 'raised': [10.0]})
    results = {'plain_seed1': ({'f1': 0.4, 'rmse': 30.0}, counts(4.0)),
               'plain_seed2': ({'f1': 0.6, 'rmse': 32.0}, counts(6.0)),
               'persistence': ({'f1': 0.5, 'rmse': 35.0}, counts(5.0))}
    table, pooled = gap.combine_seeds(results)
    assert table.loc['plain', 'f1'] == pytest.approx(0.5) and table.loc['plain', 'runs'] == 2
    assert table.loc['plain', 'f1 sd'] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert table.loc['persistence', 'f1 sd'] == 0.0
    assert pooled['plain'].hits.tolist() == [5.0]


def _ranked(n_cities=6, seed=0):
    """Cities whose severe days a sharp forecast ranks perfectly and a blurred one only roughly."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(n_cities):
        y = np.concatenate([rng.uniform(310, 400, 15), rng.uniform(50, 290, 185)])
        parts.append(pd.DataFrame({'city': f'c{c}', 'y': y, 'sharp_seed1': y + rng.normal(0, 5, 200),
                                   'sharp_seed2': y + rng.normal(0, 5, 200), 'blurred': y + rng.normal(0, 90, 200)}))
    return pd.concat(parts, ignore_index=True)


def test_average_precision_ranks_without_any_alarm_level():
    frame = _ranked()
    sharp = gap.average_precision(frame, ['sharp_seed1', 'sharp_seed2'])
    assert sharp == pytest.approx(1.0, abs=0.02)
    assert gap.average_precision(frame, ['blurred']) < sharp
    assert gap.average_precision(frame.assign(sharp_seed1=frame.sharp_seed1 - 200), ['sharp_seed1']) == \
        pytest.approx(gap.average_precision(frame, ['sharp_seed1']))      # a shift changes nothing
    assert np.isnan(gap.average_precision(frame.assign(y=100.0), ['blurred']))


def test_a_paired_ap_difference_has_an_interval_and_a_p_value():
    frame = _ranked()
    point, low, high, p = gap.paired_ap_difference(frame, ['sharp_seed1', 'sharp_seed2'], ['blurred'], draws=300)
    assert point > 0 and 0 < low <= point <= high and p < 0.01
    assert p == pytest.approx(2 / 301)                  # every resample agrees, yet p is never 0
    same = gap.paired_ap_difference(frame, ['blurred'], ['blurred'], draws=100)
    assert same[0] == 0.0 and same[3] == 1.0


def test_holm_scales_the_smallest_p_most_and_never_reorders():
    assert gap.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert gap.holm([0.5]).tolist() == [0.5]


def test_a_paired_difference_resamples_the_same_cities_for_both():
    a = pd.DataFrame({'city': list('ABCD'), 'hits': [8.0, 6, 9, 7], 'severe': [10.0] * 4, 'raised': [9.0] * 4})
    b = a.assign(hits=a.hits - 3)
    point, low, high = gap.paired_difference(a, b, draws=500)
    assert point > 0 and low > 0 and low <= point <= high
    same, low, high = gap.paired_difference(a, a, draws=200)
    assert same == 0.0 and low == high == 0.0

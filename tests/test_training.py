import io
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.models import itransformer, training  # noqa: E402
from src.utils.local_store import LocalStore  # noqa: E402

LOOKBACK, HORIZON, VARIATES = 14, 3, 4
QUIET = dict(log=lambda *a: None)


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, LOOKBACK, VARIATES)).astype(np.float32)
    X[:, :, -1] = 100 + 20 * np.sin(np.arange(LOOKBACK) / 3) + rng.normal(0, 3, (n, LOOKBACK))
    y = (X[:, -1:, -1] + rng.normal(0, 2, (n, HORIZON))).astype(np.float32)
    return X, y


def _loaders(data_seed=0):
    X, y = _data(seed=data_seed)
    return (itransformer.make_loader(X[:220], y[:220], batch_size=32, shuffle=True),
            itransformer.make_loader(X[220:], y[220:], batch_size=64))


def _model():
    return itransformer.ITransformer(VARIATES, LOOKBACK, HORIZON, VARIATES - 1,
                                     d_model=16, n_heads=2, n_layers=1, dropout=0.0)


def _checkpointer(store):
    return training.Checkpointer(store, 'ckpt/run.pt', every_seconds=0, **QUIET)


def _fit(store=None, epochs=6, deadline=None, seed=7, lr=3e-4, data_seed=0):
    itransformer.set_seed(seed)
    model = _model()
    train_loader, val_loader = _loaders(data_seed)
    return training.train(model, train_loader, val_loader, torch.device('cpu'), epochs=epochs,
                          lr=lr, patience=50, deadline=deadline,
                          checkpointer=_checkpointer(store) if store else None, **QUIET)


class _SessionDropped(Exception):
    """Stands in for Colab reclaiming the runtime."""


def _fit_until_dropped(store, after_epochs, epochs=6):
    """Train with checkpoints, and let the session die partway through an epoch."""
    validated = {'n': 0}
    original = training._validation_rmse

    def failing(*args, **kwargs):
        validated['n'] += 1
        if validated['n'] > after_epochs:
            raise _SessionDropped
        return original(*args, **kwargs)

    training._validation_rmse = failing
    try:
        _fit(store, epochs=epochs)
    except _SessionDropped:
        pass
    finally:
        training._validation_rmse = original


# --------------------------------------------------------------------------- resume

def test_a_run_dropped_midway_and_resumed_is_the_same_run_as_one_never_dropped(tmp_path):
    """The requirement. A session lost after three epochs, picked up in a fresh one
    with a different seed and fresh weights, must end on the identical curve."""
    _, uninterrupted, _ = _fit(epochs=6)

    store = LocalStore(tmp_path)
    _fit_until_dropped(store, after_epochs=3)
    assert store.exists('ckpt/run.pt')

    _, resumed, outcome = _fit(store, epochs=6, seed=999)     # a brand-new session
    assert outcome == 'completed'
    assert [h['epoch'] for h in resumed] == [1, 2, 3, 4, 5, 6]
    assert [h['val_rmse'] for h in resumed] == pytest.approx(
        [h['val_rmse'] for h in uninterrupted], abs=1e-5)


def test_a_run_stopped_by_its_time_budget_resumes_from_the_saved_point(tmp_path):
    store = LocalStore(tmp_path)
    _, history, outcome = _fit(store, deadline=time.monotonic() - 1)
    assert outcome == 'out_of_time'
    assert history == []
    assert store.exists('ckpt/run.pt')

    _, uninterrupted, _ = _fit(epochs=6)
    _, resumed, outcome = _fit(store, epochs=6)
    assert outcome == 'completed'
    assert [h['val_rmse'] for h in resumed] == pytest.approx(
        [h['val_rmse'] for h in uninterrupted], abs=1e-5)


def test_a_finished_run_is_returned_without_training_again(tmp_path):
    store = LocalStore(tmp_path)
    _fit(store, epochs=4)

    validated = {'n': 0}
    original = training._validation_rmse

    def counting(*args, **kwargs):
        validated['n'] += 1
        return original(*args, **kwargs)

    training._validation_rmse = counting
    try:
        _, history, outcome = _fit(store, epochs=4)
    finally:
        training._validation_rmse = original
    assert validated['n'] == 0
    assert outcome == 'completed' and len(history) == 4


@pytest.mark.parametrize('change', [{'lr': 1e-3}, {'epochs': 8}, {'data_seed': 1}],
                         ids=['learning rate', 'epochs', 'data'])
def test_a_checkpoint_from_other_settings_or_data_is_set_aside_not_resumed(tmp_path, change):
    """Resumed, it would carry the old run on under the new one's name: the
    optimiser restores its own learning rate, the schedule its own length, and
    the weights were fitted to other windows. It must start afresh instead."""
    store = LocalStore(tmp_path)
    _fit_until_dropped(store, after_epochs=3)

    _, fresh, _ = _fit(**change)
    _, rerun, outcome = _fit(store, **change)
    assert outcome == 'completed'
    assert [h['epoch'] for h in rerun] == [h['epoch'] for h in fresh]
    assert [h['val_rmse'] for h in rerun] == pytest.approx([h['val_rmse'] for h in fresh], abs=1e-5)


def test_checkpoints_load_without_unpickling_arbitrary_objects(tmp_path):
    store = LocalStore(tmp_path)
    _fit(store, epochs=2)
    state = torch.load(io.BytesIO(store.read_bytes('ckpt/run.pt')), map_location='cpu',
                       weights_only=True)
    assert {'model', 'optimiser', 'schedule', 'best', 'history', 'rng'} <= set(state)


def test_random_state_survives_a_round_trip():
    np.random.seed(3)
    torch.manual_seed(3)
    saved = training.capture_rng()
    expected = (np.random.rand(3), torch.rand(3))
    np.random.seed(0)
    torch.manual_seed(0)
    training.restore_rng(saved)
    assert np.allclose(np.random.rand(3), expected[0])
    assert torch.allclose(torch.rand(3), expected[1])


# --------------------------------------------------------------------------- guards

class _RunsAway(torch.nn.Module):
    """Fine for a while, then produces NaN -- a learning rate that has diverged."""

    def __init__(self):
        super().__init__()
        self.inner = _model()
        self.calls = 0

    def forward(self, window):
        self.calls += 1
        out = self.inner(window)
        return out * float('nan') if self.calls > 20 else out


def test_a_diverging_run_stops_on_its_best_weights_instead_of_saving_nan():
    itransformer.set_seed(0)
    train_loader, val_loader = _loaders()
    model, history, outcome = training.train(_RunsAway(), train_loader, val_loader,
                                             torch.device('cpu'), epochs=10, patience=50, **QUIET)
    assert outcome == 'diverged'
    assert history                                   # it did train before it ran away
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_early_stopping_hands_back_the_best_epoch_not_the_last():
    itransformer.set_seed(0)
    train_loader, val_loader = _loaders()
    model, history, _ = training.train(_model(), train_loader, val_loader, torch.device('cpu'),
                                       epochs=40, patience=3, lr=5e-2, **QUIET)
    best = min(h['val_rmse'] for h in history)
    assert training._validation_rmse(model, val_loader, torch.device('cpu')) == pytest.approx(best, abs=1e-4)


# --------------------------------------------------------------------------- registry

def test_the_registry_knows_which_runs_have_finished(tmp_path):
    registry = training.RunRegistry(LocalStore(tmp_path), 'runs/05')
    assert not registry.done('meteorological_seed0')
    registry.record('meteorological_seed0', {'val_rmse': 31.2})
    registry.record('learned_seed0', {'val_rmse': 31.9})
    assert registry.done('meteorological_seed0')
    assert registry.finished() == ['learned_seed0', 'meteorological_seed0']
    assert registry.load('meteorological_seed0') == {'val_rmse': 31.2}


def test_a_result_and_its_weights_are_recorded_together(tmp_path):
    store = LocalStore(tmp_path)
    registry = training.RunRegistry(store, 'runs/05')
    weights = tmp_path / 'w.pt'
    torch.save({'x': torch.ones(2)}, weights)
    registry.record('meteorological_seed42', {'val_rmse': 31.4},
                    attachments={'meteorological_seed42.pt': weights})
    assert registry.done('meteorological_seed42')
    assert store.exists(registry.attachment('meteorological_seed42.pt'))
    assert registry.load('meteorological_seed42') == {'val_rmse': 31.4}


def test_a_caller_that_records_runs_itself_leaves_no_finished_checkpoint(tmp_path):
    store = LocalStore(tmp_path)
    itransformer.set_seed(7)
    train_loader, val_loader = _loaders()
    training.train(_model(), train_loader, val_loader, torch.device('cpu'), epochs=3, patience=50,
                   checkpointer=training.Checkpointer(store, 'ckpt/run.pt', every_seconds=1e9, **QUIET),
                   save_finished=False, **QUIET)
    assert not store.exists('ckpt/run.pt')

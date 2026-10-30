import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.models import itransformer, rare_days, training  # noqa: E402
from src.utils.local_store import LocalStore  # noqa: E402

QUIET = dict(log=lambda *a: None)


def test_a_window_is_severe_when_any_of_its_targets_is():
    y = np.array([[100, 200, 250], [310, 100, 90], [299, 300, 301]])
    assert rare_days.severe_windows(y).tolist() == [False, True, True]


def test_oversampling_makes_severe_windows_a_quarter_of_every_epoch():
    rng = np.random.default_rng(0)
    y = np.full((1000, 3), 100.0, dtype=np.float32)
    y[:50, 1] = 350.0                                     # 5% severe, as in training
    X = rng.normal(size=(1000, 14, 4)).astype(np.float32)
    torch.manual_seed(0)
    loader = rare_days.oversampling_loader(X, y, batch_size=100)
    drawn = np.concatenate([answer.numpy() for _ in range(20) for _, answer in loader])
    assert len(drawn) == 20 * 1000                        # an epoch is as long as before
    assert rare_days.severe_windows(drawn).mean() == pytest.approx(0.25, abs=0.02)


def test_with_nothing_to_rebalance_it_is_an_ordinary_loader():
    X, y = np.zeros((40, 14, 4), np.float32), np.full((40, 3), 100.0, np.float32)
    assert sum(len(answer) for _, answer in rare_days.oversampling_loader(X, y, 8)) == 40


def test_the_severe_weight_gives_severe_targets_a_quarter_of_the_loss():
    y = np.full((100, 3), 100.0)
    y[:10, 0] = 400.0                                     # 10 of 300 targets
    weight = rare_days.severe_weight(y)
    assert weight * 10 / (weight * 10 + 290) == pytest.approx(0.25)
    assert rare_days.severe_weight(np.full((5, 3), 100.0)) == 1.0


class _Constant(torch.nn.Module):
    """Forecasts one learnable number for every window and horizon."""

    horizon = 3

    def __init__(self, value=0.0):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, window):
        return self.value.expand(window.shape[0], self.horizon)


def test_the_weighted_loss_counts_a_severe_miss_weight_times():
    model = rare_days.WeightedSquaredError(_Constant(100.0), weight=4.0)
    window = torch.zeros(2, 14, 4)
    answer = torch.tensor([[100.0, 100.0, 400.0], [110.0, 100.0, 100.0]])
    expected = (4 * 300 ** 2 + 10 ** 2) / (4 + 5)
    assert float(model.training_loss(window, answer).detach()) == pytest.approx(expected)
    assert torch.equal(model(window), model.inner(window))          # forecasting is unchanged


def test_a_quantile_forecaster_learns_the_ninetieth_percentile():
    model = rare_days.QuantileForecaster(_Constant(50.0), quantile=0.9)
    answer = torch.linspace(0, 100, 1001).reshape(-1, 1).expand(-1, 3)
    window = torch.zeros(len(answer), 14, 4)
    optimiser = torch.optim.Adam(model.parameters(), lr=1.0)
    for _ in range(400):
        optimiser.zero_grad()
        model.training_loss(window, answer).backward()
        optimiser.step()
    assert float(model.inner.value.detach()) == pytest.approx(90.0, abs=1.5)


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 14, 4)).astype(np.float32)
    X[:, :, -1] = 150 + 40 * np.sin(np.arange(14) / 3) + rng.normal(0, 5, (n, 14))
    y = (X[:, -1:, -1] + rng.normal(0, 10, (n, 3))).astype(np.float32)
    return (itransformer.make_loader(X[:180], y[:180], 32, shuffle=True),
            itransformer.make_loader(X[180:], y[180:], 60))


def _quantile_model():
    return rare_days.QuantileForecaster(itransformer.ITransformer(4, 14, 3, 3, d_model=16, n_heads=2,
                                                                  n_layers=1, dropout=0.0))


def test_a_quantile_forecaster_early_stops_on_its_own_validation_loss():
    itransformer.set_seed(0)
    train_loader, val_loader = _data()
    score = rare_days.pinball_score(0.9)
    model, history, _ = training.train(_quantile_model(), train_loader, val_loader, torch.device('cpu'),
                                       epochs=4, lr=3e-3, patience=10, score=score, **QUIET)
    assert all({'train_loss', 'val_score'} <= set(h) and 'val_rmse' not in h for h in history)
    assert min(h['val_score'] for h in history) == pytest.approx(
        score(model, val_loader, torch.device('cpu')), abs=1e-4)


def test_a_run_scored_another_way_is_a_different_run(tmp_path):
    """A finished run early-stopped on RMSE must not be handed back as one stopped on pinball loss."""
    store = LocalStore(tmp_path)
    checkpointer = lambda: training.Checkpointer(store, 'ckpt/q.pt', every_seconds=0, **QUIET)
    itransformer.set_seed(0)
    train_loader, val_loader = _data()
    training.train(_quantile_model(), train_loader, val_loader, torch.device('cpu'), epochs=2,
                   checkpointer=checkpointer(), **QUIET)
    itransformer.set_seed(0)
    _, history, _ = training.train(_quantile_model(), train_loader, val_loader, torch.device('cpu'),
                                   epochs=2, checkpointer=checkpointer(),
                                   score=rare_days.pinball_score(0.9), **QUIET)
    assert all('val_score' in h for h in history)

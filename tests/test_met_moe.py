import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.models import itransformer, met_moe, training  # noqa: E402

LOOKBACK, HORIZON = 14, 3
VARIATES = ['inv_idx', 'fire_count_500km', 'wind_speed', 'PM2.5', 'month', 'AQI']
GATE = ['inv_idx', 'fire_count_500km', 'wind_speed']
TARGET = VARIATES.index('AQI')


def _windows(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, LOOKBACK, len(VARIATES))).astype(np.float32)
    X[:, :, VARIATES.index('fire_count_500km')] = rng.poisson(40, (n, LOOKBACK))
    X[:, :, TARGET] = 120 + rng.normal(0, 15, (n, LOOKBACK))
    return X


def _model(router='meteorological', X=None, seed=0, objective='cooperative'):
    torch.manual_seed(seed)
    indices, logged = met_moe.gate_layout(VARIATES, GATE)
    centre, scale = met_moe.gate_statistics(_windows() if X is None else X, indices, logged)
    return met_moe.MetMoE(len(VARIATES), LOOKBACK, HORIZON, TARGET, indices, logged,
                          router=router, d_model=16, n_heads=2, n_layers=1, dropout=0.0,
                          gate_centre=centre, gate_scale=scale, objective=objective).eval()


# --------------------------------------------------------------------------- shape and routing

@pytest.mark.parametrize('router', met_moe.ROUTERS)
def test_one_forecast_per_horizon_from_either_router(router):
    out = _model(router)(torch.from_numpy(_windows(8)))
    assert out.shape == (8, HORIZON) and torch.isfinite(out).all()


@pytest.mark.parametrize('router', ['meteorological', 'learned'])
def test_mixture_weights_are_a_proper_split_for_every_window(router):
    weights = _model(router).gate_weights(_windows(30))
    assert weights.shape == (30, 2)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert (weights >= 0).all()


def test_the_physics_router_sees_how_strong_the_inversion_is_and_the_learned_one_cannot():
    """The whole case for meteorological gating in one comparison. The trunk
    standardises every variate inside its window -- which is what lets one model
    serve Delhi and Chennai -- so a stronger inversion everywhere in the fortnight
    looks identical to it. The physics router reads the absolute value."""
    X = _windows(16)
    stronger = X.copy()
    stronger[:, :, VARIATES.index('inv_idx')] += 5.0          # 5 C warmer lid, all fortnight

    physics, learned = _model('meteorological'), _model('learned')
    assert not np.allclose(physics.gate_weights(stronger), physics.gate_weights(X), atol=1e-4)
    assert np.allclose(learned.gate_weights(stronger), learned.gate_weights(X), atol=1e-5)


def test_the_control_has_one_expert_and_no_router_to_learn():
    model = _model('none')
    assert model.n_experts == 1 and model.router is None
    assert np.allclose(model.gate_weights(_windows(5)), 1.0)
    assert float(model.auxiliary_loss()) == 0.0


def test_the_learned_router_gets_more_inputs_and_parameters_than_the_physics_one():
    # If physics still wins, it is not for want of capacity or information.
    physics, learned = _model('meteorological'), _model('learned')
    count = lambda m: sum(p.numel() for p in m.router.parameters())
    assert learned.router[0].in_features > physics.router[0].in_features
    assert count(learned) > count(physics)


def test_fire_counts_are_logged_before_the_router_sees_them():
    indices, logged = met_moe.gate_layout(VARIATES, GATE)
    assert logged == [False, True, False]
    window = torch.from_numpy(_windows(4))
    features = met_moe.gate_inputs(window, indices, logged)
    fire_on_day_t = window[:, -1, VARIATES.index('fire_count_500km')]
    assert torch.allclose(features[:, 1], torch.log1p(fire_on_day_t))


def test_gate_features_missing_from_the_windows_are_refused():
    with pytest.raises(ValueError, match='blh_min'):
        met_moe.gate_layout(VARIATES, ['blh_min', 'inv_idx'])


@pytest.mark.parametrize('kwargs, message', [({'router': 'random'}, 'router'),
                                             ({'n_experts': 1}, 'two experts'),
                                             ({'objective': 'greedy'}, 'objective'),
                                             ({'router': 'none', 'objective': 'competitive'}, 'compete')])
def test_a_misconfigured_mixture_is_refused(kwargs, message):
    indices, logged = met_moe.gate_layout(VARIATES, GATE)
    with pytest.raises(ValueError, match=message):
        met_moe.MetMoE(len(VARIATES), LOOKBACK, HORIZON, TARGET, indices, logged,
                       d_model=16, n_heads=2, **kwargs)


# --------------------------------------------------------------------------- balance

def test_the_balance_penalty_is_zero_when_shared_and_largest_when_one_expert_takes_all():
    model = _model()
    model._mean_weights = torch.tensor([0.5, 0.5])
    assert float(model.auxiliary_loss()) == pytest.approx(0.0, abs=1e-6)
    model._mean_weights = torch.tensor([1.0, 0.0])
    assert float(model.auxiliary_loss()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- objectives

def test_the_cooperative_objective_is_exactly_the_squared_error_training_always_used():
    # Notebook 05's original variants must train bit for bit as they did before
    # the competitive objective existed.
    model = _model()
    window = torch.from_numpy(_windows(8))
    answer = 120 + 10 * torch.randn(8, HORIZON)
    assert not model.uses_own_loss
    assert torch.equal(model.training_loss(window, answer), torch.nn.MSELoss()(model(window), answer))


def _trust(model, expert):
    """Point the router firmly at one expert, whatever the window."""
    with torch.no_grad():
        model.router[-1].weight.zero_()
        model.router[-1].bias.copy_(torch.tensor([10.0, -10.0] if expert == 0 else [-10.0, 10.0]))


def test_the_competitive_objective_scores_each_expert_on_its_own_forecast():
    """-log sum_k g_k N(y; e_k, s^2): when one expert has a window exactly right,
    trusting it costs only the spread term, and trusting the other costs more."""
    model = _model(objective='competitive')
    with torch.no_grad():
        model.experts[1].bias += 2.0                 # expert 1 forecasts ~30 AQI higher
    X = _windows(8)
    window, answer = torch.from_numpy(X), torch.from_numpy(model.expert_forecasts(X)[:, 0, :])

    _trust(model, 0)
    right = float(model.training_loss(window, answer).detach())
    _trust(model, 1)
    wrong = float(model.training_loss(window, answer).detach())
    assert model.uses_own_loss
    assert right == pytest.approx(HORIZON * math.log(met_moe.COMPETITIVE_SIGMA_INIT), abs=1e-3)
    assert wrong > right + 0.5
    # and the balance penalty still sees where the weight went
    assert float(model.auxiliary_loss().detach()) == pytest.approx(1.0, abs=1e-6)


def test_each_expert_can_be_read_on_its_own_and_the_mixture_is_their_weighted_mean():
    model = _model()
    X = _windows(12)
    alone, weights = model.expert_forecasts(X), model.gate_weights(X)
    assert alone.shape == (12, 2, HORIZON)
    assert np.allclose((weights[:, :, None] * alone).sum(axis=1), itransformer.predict(model, X), atol=1e-3)


def test_competitive_and_cooperative_runs_can_never_resume_each_other():
    loader = itransformer.make_loader(_windows(16), np.zeros((16, HORIZON), np.float32), 8)
    signatures = [training.run_signature(_model(objective=o), loader, loader) for o in met_moe.OBJECTIVES]
    assert signatures[0] != signatures[1]


def test_a_competitive_mixture_trains_on_its_own_loss_and_is_judged_on_rmse():
    X, y, _ = _regime_task(n=200)
    model = _model(X=X[:160], objective='competitive').train()
    spread = float(model.log_sigma.detach())
    val_loader = itransformer.make_loader(X[160:], y[160:], 40)
    _, history, outcome = training.train(
        model, itransformer.make_loader(X[:160], y[:160], 32, shuffle=True), val_loader,
        torch.device('cpu'), epochs=3, lr=3e-3, patience=10, aux_weight=0.01, log=lambda *a: None)
    assert outcome == 'completed'
    assert all('train_loss' in h and 'train_rmse' not in h for h in history)
    assert float(model.log_sigma.detach()) != spread                       # the spread is learned too
    assert min(h['val_rmse'] for h in history) == pytest.approx(
        training._validation_rmse(model, val_loader, torch.device('cpu')), abs=1e-4)


# --------------------------------------------------------------------------- the trunk's guarantees still hold

def test_shifting_the_target_history_still_shifts_the_forecast_by_the_same_amount():
    model = _model()
    X = torch.from_numpy(_windows(6))
    lifted = X.clone()
    lifted[:, :, TARGET] += 100.0
    with torch.no_grad():
        # The router reads inv_idx, fires and wind, none of which moved, so the mix
        # is unchanged and the forecast must move by exactly the shift.
        assert torch.allclose(model(lifted), model(X) + 100.0, atol=1e-3)


def test_a_saved_mixture_rebuilds_from_its_spec_and_weights_alone():
    model = _model()
    indices, logged = met_moe.gate_layout(VARIATES, GATE)
    rebuilt = met_moe.MetMoE(len(VARIATES), LOOKBACK, HORIZON, TARGET, indices, logged,
                             d_model=16, n_heads=2, n_layers=1, dropout=0.0)    # no statistics given
    rebuilt.load_state_dict(model.state_dict(), strict=True)                  # they travel as buffers
    X = _windows(10)
    assert np.allclose(itransformer.predict(rebuilt.eval(), X), itransformer.predict(model, X), atol=1e-6)


def test_the_trunk_can_start_from_a_notebook_04_backbone():
    backbone = itransformer.ITransformer(len(VARIATES), LOOKBACK, HORIZON, TARGET,
                                         d_model=16, n_heads=2, n_layers=1)
    model = _model()
    result = model.load_trunk(backbone.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(model.trunk.embed.weight, backbone.embed.weight)


# --------------------------------------------------------------------------- it learns regimes

def _regime_task(seed=1, n=600):
    """A target whose relationship with a driver reverses with a regime flag --
    the seasonal reversals notebook 02 measured, in miniature. The flag lives in
    the absolute level of inv_idx, which the trunk standardises away."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, LOOKBACK, len(VARIATES))).astype(np.float32)
    regime = rng.integers(0, 2, n)
    X[:, :, VARIATES.index('inv_idx')] = (regime * 4.0)[:, None] + rng.normal(0, 0.3, (n, LOOKBACK))
    X[:, :, TARGET] = 120 + rng.normal(0, 10, (n, LOOKBACK))
    driver = X[:, -1, VARIATES.index('PM2.5')]
    sign = np.where(regime == 1, 1.0, -1.0)
    y = (X[:, -1, TARGET] + 25 * sign * driver)[:, None].repeat(HORIZON, axis=1).astype(np.float32)
    return X, y, regime


def _regime_model(router, objective='cooperative'):
    X, y, regime = _regime_task()
    model = _model(router, X=X[:480], objective=objective).train()
    itransformer.set_seed(0)
    torch.manual_seed(0)
    training.train(model, itransformer.make_loader(X[:480], y[:480], 64, shuffle=True),
                   itransformer.make_loader(X[480:], y[480:], 120), torch.device('cpu'),
                   epochs=60, lr=3e-3, patience=60, aux_weight=0.01, log=lambda *a: None)
    return model, X, y, regime


def _regime_split(router, objective='cooperative'):
    model, X, _, regime = _regime_model(router, objective)
    weights = model.gate_weights(X)
    return abs(weights[regime == 1, 0].mean() - weights[regime == 0, 0].mean())


@pytest.mark.parametrize('objective', met_moe.OBJECTIVES)
def test_given_two_regimes_the_physics_router_sends_them_to_different_experts(objective):
    split = _regime_split('meteorological', objective)
    assert split > 0.8, f"physics router did not separate the regimes (split {split:.2f})"


@pytest.mark.parametrize('objective', met_moe.OBJECTIVES)
def test_the_learned_router_cannot_find_a_regime_the_trunk_never_sees(objective):
    """The 'w/o Met Gating' ablation, in miniature: same experts, same training,
    but routing on the trunk's own standardised encoding. The regime is in how
    strong the inversion is, and that level never reaches the encoding -- under
    either objective, since no loss can route on what the router cannot see."""
    split = _regime_split('learned', objective)
    assert split < 0.2, f"learned router separated a regime it should not see (split {split:.2f})"


def test_under_competition_each_regime_gets_a_whole_forecaster_of_its_own():
    """What the competitive objective is for: the expert a regime is routed to
    forecasts that regime about as well on its own as the whole mixture does."""
    model, X, y, regime = _regime_model('meteorological', 'competitive')
    held_out = slice(480, None)
    X, y, regime = X[held_out], y[held_out], regime[held_out]
    weights, alone = model.gate_weights(X), model.expert_forecasts(X)
    mixture = itransformer.predict(model, X)
    for value in (0, 1):
        days = regime == value
        mine = int(np.argmax(weights[days].mean(axis=0)))
        rmse = lambda forecast: float(np.sqrt(np.mean((forecast[days] - y[days]) ** 2)))
        assert rmse(alone[:, mine]) <= 1.1 * rmse(mixture) + 0.5, (value, rmse(alone[:, mine]), rmse(mixture))

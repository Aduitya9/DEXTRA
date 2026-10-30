"""
Met-MoE: a dual-regime mixture of experts, routed on the weather (Novelty 2).

Notebook 02 measured four of the twenty-six variates reversing their relationship
with AQI between India's seasons. blh_anomaly is -0.17 in winter, when a shallow
layer is a lid, and +0.21 in the pre-monsoon, when a deep layer is a symptom of
the heat that lifts dust; fire_count_500km is -0.10 in winter and +0.33 before
the monsoon. A single forecasting head has to average those regimes into one
mapping. Several heads, mixed by a router that knows which regime a day belongs
to, need not.

The trunk is notebook 04's iTransformer. Its single output head is replaced by
experts, and what decides the mix is the router, which comes in two kinds so
the paper's two N2 ablation rows can be measured against each other:

  meteorological  reads the six GATE_FEATURES -- blh_min, wind_speed, inv_idx,
                  fire_count_500km, blh_anomaly, is_stubble_season -- as they
                  stood on day t and averaged over the fortnight.
  learned         reads the trunk's own encoding: the target's token and the
                  mean of all of them. This is the "w/o Met Gating" row. It is
                  given more parameters and more inputs than the physics
                  router, so if physics still wins it is not for want of either.
  none            a single expert and no router. Not a novelty -- a control, so
                  that what an all-token head adds can be told apart from what
                  routing adds.

Two findings shaped this, both measured on a synthetic task whose answer flips
sign with a regime flag, so that it can only be solved by separating regimes:

  Experts must read every variate's token, not only the target's. Experts on
  the target token alone stalled at exactly the "average the regimes away"
  error -- the driver they needed only reaches that token if attention learns
  to carry it there, attention only learns that once routing rewards it, and
  routing only pays once the experts differ, so nothing ever starts. Experts on
  all tokens separated the regimes almost perfectly on every seed tried.

  The router must read meteorology's absolute level. The same all-token experts
  under the learned router did not separate the regimes at all: the trunk
  standardises every variate inside its window -- which is what lets one model
  serve Delhi and Chennai -- so how strong an inversion is never reaches its
  encoding. The physics router reads the level directly.

So the meteorological router's inputs are standardised once, with statistics
from the training windows only, and those statistics travel inside the model as
buffers: a saved Met-MoE needs nothing else to run, which matters for ONNX.

Two training objectives, after Jacobs, Jordan, Nowlan and Hinton (1991),
"Adaptive Mixtures of Local Experts":

  cooperative  the mixture's forecast is scored, ||y - sum_k g_k e_k||^2. The
               experts are free to become halves of one forecast that cancel
               each other's errors. In a first one-seed run on the real data
               each was worse alone than the even mix, and the router stayed
               at 50/50, picking the more accurate expert on 50.6% of windows.
               Kept as the ablation it is.
  competitive  each expert is scored on its own forecast, weighted by how much
               the router trusts it: -log sum_k g_k N(y; e_k, s^2), with the
               spread s learned alongside. An expert is pushed to forecast the
               windows it is given on its own, and the router to give each
               window to the expert that forecasts it best. The forecast is
               still the mixture's mean, so RMSE is measured the same way.
"""

import math

import numpy as np
import torch
from torch import nn

from src.config import GATE_FEATURES
from src.models.itransformer import ITransformer

ROUTERS = ('meteorological', 'learned', 'none')
OBJECTIVES = ('cooperative', 'competitive')
# Where the competitive objective's spread starts, in AQI points: about the size
# of a day-ahead forecast error (the backbone's validation RMSE is 31.6).
COMPETITIVE_SIGMA_INIT = 30.0
# Counts that run from zero to thousands: logged before standardising, or a single
# heavy fire day would set the router's whole scale.
LOG_SCALED_GATE_FEATURES = {'fire_count_500km'}


def gate_layout(variates, gate_features=None):
    """Indices of the gate features within the window, and which of them are logged."""
    gate_features = list(gate_features or GATE_FEATURES)
    missing = [f for f in gate_features if f not in variates]
    if missing:
        raise ValueError(f"gate features missing from the windows: {missing}")
    indices = [variates.index(f) for f in gate_features]
    logged = [f in LOG_SCALED_GATE_FEATURES for f in gate_features]
    return indices, logged


def gate_inputs(window, indices, logged):
    """The router's view of a window: each gate feature on day t and over the fortnight."""
    gate = window[..., indices]
    logged = torch.as_tensor(logged, dtype=torch.bool, device=gate.device)
    gate = torch.where(logged, torch.log1p(gate.clamp(min=0)), gate)
    return torch.cat([gate[:, -1, :], gate.mean(dim=1)], dim=-1)


def gate_statistics(X_train, indices, logged):
    """Centre and scale for the meteorological router, from training windows only."""
    with torch.no_grad():
        features = gate_inputs(torch.from_numpy(np.ascontiguousarray(X_train)), indices, logged)
    centre = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp(min=1e-6)
    return centre.numpy(), scale.numpy()


class MetMoE(nn.Module):
    def __init__(self, n_variates, lookback, horizon, target_index, gate_indices, gate_logged,
                 router='meteorological', n_experts=2, d_model=128, n_heads=8, n_layers=2,
                 dropout=0.3, router_hidden=32, gate_centre=None, gate_scale=None,
                 objective='cooperative'):
        super().__init__()
        if router not in ROUTERS:
            raise ValueError(f"router must be one of {ROUTERS}, not {router!r}")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, not {objective!r}")
        if router == 'none':
            if objective == 'competitive':
                raise ValueError("a competitive objective needs experts to compete and a router to choose")
            n_experts = 1
        elif n_experts < 2:
            raise ValueError("a mixture needs at least two experts")
        self.router_kind, self.n_experts, self.objective = router, n_experts, objective
        if objective == 'competitive':
            self.log_sigma = nn.Parameter(torch.tensor(math.log(COMPETITIVE_SIGMA_INIT)))
        self.gate_indices, self.gate_logged = list(gate_indices), list(gate_logged)

        self.trunk = ITransformer(n_variates, lookback, horizon, target_index,
                                  d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.trunk.project = nn.Identity()      # the experts replace notebook 04's single head
        self.expert_dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList([nn.Linear(n_variates * d_model, horizon) for _ in range(n_experts)])

        width = 2 * len(self.gate_indices)
        self.register_buffer('gate_centre', torch.zeros(width) if gate_centre is None
                             else torch.as_tensor(gate_centre, dtype=torch.float32))
        self.register_buffer('gate_scale', torch.ones(width) if gate_scale is None
                             else torch.as_tensor(gate_scale, dtype=torch.float32))
        if router == 'none':
            self.router = None
        else:
            router_in = width if router == 'meteorological' else 2 * d_model
            self.router = nn.Sequential(nn.Linear(router_in, router_hidden), nn.GELU(),
                                        nn.Linear(router_hidden, n_experts))
        self._mean_weights = None

    @property
    def target_index(self):
        return self.trunk.target_index

    @property
    def horizon(self):
        return self.trunk.horizon

    @property
    def uses_own_loss(self):
        """True when training must call training_loss instead of scoring forward()."""
        return self.objective == 'competitive'

    def _logits(self, window, encoded):
        if self.router_kind == 'meteorological':
            features = (gate_inputs(window, self.gate_indices, self.gate_logged)
                        - self.gate_centre) / self.gate_scale
        else:
            features = torch.cat([encoded[:, self.trunk.target_index, :], encoded.mean(dim=1)], dim=-1)
        return self.router(features)

    def route(self, window, encoded):
        """Mixture weights, (batch, experts), summing to one for every window."""
        if self.router is None:
            return torch.ones(window.shape[0], 1, device=window.device)
        return torch.softmax(self._logits(window, encoded), dim=-1)

    def forward(self, window):
        """window: (batch, lookback, variates) -> (batch, horizon) in AQI points."""
        encoded, centre, scale = self.trunk.encode(window)
        weights = self.route(window, encoded)
        flat = self.expert_dropout(encoded.flatten(1))
        heads = torch.stack([expert(flat) for expert in self.experts], dim=1)
        self._mean_weights = weights.mean(dim=0)
        mixed = (weights.unsqueeze(-1) * heads).sum(dim=1)
        return self.trunk.denormalise(mixed, centre, scale)

    def training_loss(self, window, answer):
        """The loss this model trains on, for a batch of windows and their answers.

        Cooperative: squared error of the mixture's forecast, as for any model.
        Competitive: -log sum_k g_k N(y; e_k, s^2 I) in AQI points, so that each
        expert answers for its own forecast on the windows the router gives it.
        It is never negative in practice -- that would need s below one AQI
        point -- which keeps the balance penalty's relative weighting meaningful.
        """
        if self.objective == 'cooperative':
            return nn.functional.mse_loss(self(window), answer)
        encoded, centre, scale = self.trunk.encode(window)
        log_weights = torch.log_softmax(self._logits(window, encoded), dim=-1)
        self._mean_weights = log_weights.exp().mean(dim=0)
        flat = self.expert_dropout(encoded.flatten(1))
        forecasts = torch.stack([self.trunk.denormalise(expert(flat), centre, scale)
                                 for expert in self.experts], dim=1)       # (batch, experts, horizon)
        squared = ((forecasts - answer.unsqueeze(1)) ** 2).sum(dim=-1)
        log_joint = log_weights - squared / (2 * torch.exp(2 * self.log_sigma))
        return (answer.shape[-1] * self.log_sigma - torch.logsumexp(log_joint, dim=-1)).mean()

    def auxiliary_loss(self):
        """Balance penalty: 0 when the experts share the load, K-1 when one takes it all.

        Without it a mixture can quietly become a one-expert model -- the router
        learns to send everything to whichever head got ahead early -- and the
        regime claim would be untestable because there would be one regime.
        """
        if self._mean_weights is None or self.n_experts < 2:
            return torch.zeros(())
        return self.n_experts * (self._mean_weights ** 2).sum() - 1.0

    def load_trunk(self, state_dict):
        """Start the trunk from a notebook 04 checkpoint; the experts and router start fresh."""
        trunk_state = {k: v for k, v in state_dict.items() if not k.startswith('project.')}
        return self.trunk.load_state_dict(trunk_state, strict=True)

    @torch.no_grad()
    def gate_weights(self, X, device=None, batch_size=512):
        """Mixture weights for every window, as a NumPy array (windows, experts)."""
        device = device or next(self.parameters()).device
        self.eval()
        out = []
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device)
            encoded, _, _ = self.trunk.encode(batch)
            out.append(self.route(batch, encoded).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, self.n_experts), dtype=np.float32)

    @torch.no_grad()
    def expert_forecasts(self, X, device=None, batch_size=512):
        """Every expert's forecast on its own, (windows, experts, horizon) in AQI points.

        Set beside gate_weights, it shows whether the router knows which expert
        to trust: on each window, did it lean towards the expert whose forecast
        turned out the more accurate? An expert's error alone cannot answer that
        -- cooperating halves are poor alone, and so are specialists scored on
        each other's windows.
        """
        device = device or next(self.parameters()).device
        self.eval()
        out = []
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device)
            encoded, centre, scale = self.trunk.encode(batch)
            flat = encoded.flatten(1)
            out.append(torch.stack([self.trunk.denormalise(expert(flat), centre, scale)
                                    for expert in self.experts], dim=1).cpu().numpy())
        return (np.concatenate(out) if out
                else np.empty((0, self.n_experts, self.horizon), dtype=np.float32))


def regime_report(weights, index, daily, threshold=200.0):
    """How the router divides days, set against the physics it is meant to have found.

    `weights` is (windows, experts) from gate_weights; `index` holds the windows'
    city and date. Reports, for the expert that hazardous days lean on most, how
    its weight tracks the inversion and the night boundary layer, how it moves
    through the seasons, and how much more of it hazardous days receive.
    """
    from src.evaluation.physics import IMD_SEASONS, SEASON_ORDER

    frame = index[['city', 'date']].copy()
    for k in range(weights.shape[1]):
        frame[f'w{k}'] = weights[:, k]
    frame = frame.merge(daily[['city', 'date', 'inv_idx', 'blh_min', 'AQI']],
                        on=['city', 'date'], how='left')
    hazardous = frame['AQI'] > threshold

    lean = [frame.loc[hazardous, f'w{k}'].mean() - frame.loc[~hazardous, f'w{k}'].mean()
            for k in range(weights.shape[1])]
    expert = int(np.nanargmax(lean))
    w = frame[f'w{expert}']
    season = frame['date'].dt.month.map(IMD_SEASONS)
    return {
        'stagnant_expert': expert,
        'weight_on_hazardous_days': float(w[hazardous].mean()),
        'weight_on_other_days': float(w[~hazardous].mean()),
        'corr_with_inv_idx': float(w.corr(frame['inv_idx'])),
        'corr_with_blh_min': float(w.corr(frame['blh_min'])),
        'by_season': {s: float(w[season == s].mean()) for s in SEASON_ORDER},
        'mean_weights': [float(weights[:, k].mean()) for k in range(weights.shape[1])],
    }

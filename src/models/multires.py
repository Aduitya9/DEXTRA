"""
The final model: daily and hourly series read together, and a forecast that chooses where to start.

Two changes to the all-token inverted transformer, each measured on its own in notebook 08.

Hourly tokens. Every model before this one saw each day as a single average, issued at
the end of that day. In the 200-300 band outside Delhi the last hour's PM2.5 separates
tomorrow's onsets far better than the day's AQI does (AUC 0.786 against 0.689, training
years only). An inverted transformer has a natural place for finer data: a token is a
variate's whole series, so a series of another length at another resolution is one more
token with its own embedding. Each hourly channel's last 72 hours becomes one token,
standardised inside its own window, so an unseen city still needs no statistics of its
own. Hourly gaps stay NaN, are left out of a channel's mean and spread, and enter the
embedding as zero, its mean after standardising.

An anchored forecast. Reversible normalisation restores every forecast from the
window's mean, which knows nothing of tonight, so a model has to climb back from a
fortnight's average to reach a sudden rise. Here the forecast starts from a choice of
three anchors, and the model weighs them window by window and horizon by horizon:

    the fortnight's mean   where reversible normalisation starts
    today's value          where persistence starts
    the nightly anchor     today's mean, moved towards what tonight's last hours imply for
                           that horizon, from the ratio that has held in this city over
                           the last two weeks (hourly.add_night_anchors)

    forecast(h) = sum_k weight_k(h) * anchor_k(h) + spread * correction(h)

The anchors all scale with the city and the weights come from standardised tokens, so
a city ten times dirtier still gets a forecast ten times higher: what makes reversible
normalisation transfer to unseen cities is kept. The nightly anchor needs no training
and no severe day from the city, only its own last fortnight, and on its own it already
beats persistence on next-day severe days. One extra token tells the attention where
the anchors sit relative to the window. The weights start close to the mean, so
training begins from ordinary reversible normalisation.

Two known methods are corners of this. All the weight on the mean is reversible
normalisation (Kim et al., ICLR 2022); all of it on today is the last-value start of
NLinear (Zeng et al., AAAI 2023). `mixture` restricts which anchors the forecast may
start from, so that what the nightly anchor adds to the mechanism can be told apart
from what a learned mix of those two already gives.

Inputs travel as one flat tensor per window when there are hourly tokens, because the
shared training loop passes a single input per batch. pack builds it and the model
splits it again. Without hourly tokens the input is the usual (batch, lookback, variates).
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from src.models.itransformer import CONSTANT_TOLERANCE, EPSILON, ITransformer, window_scale

HOURLY_CHANNELS = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'site_rel_humidity', 'site_air_temp', 'site_wind_speed']
HOURS = 72
LAST_HOUR = 23
ANCHORS = ('fortnight mean', 'today', 'nightly anchor')
# The same three, as `mixture` names them.
MIXTURE = ('mean', 'today', 'night')
# Where the anchor weights start: almost all on the mean, as in reversible normalisation.
MEAN_START_LOGIT = 3.0
# Standardised distances beyond this are clipped before they reach the anchor token.
ANCHOR_TOKEN_LIMIT = 10.0


def attach_hourly(index, table, channels=None, hours=HOURS):
    """The `hours` hours ending at 23:00 on each window's day: (windows, hours, channels) float32.

    `index` has city and date per window, as sequences.build_sequences returns it, and
    `table` has city, timestamp and the channels. Nothing after the day of issue can
    enter, because a window's series stops at that day's last hour. Windows whose whole
    span is not in the table stay NaN rather than being filled from another day.
    """
    channels = list(channels or HOURLY_CHANNELS)
    out = np.full((len(index), hours, len(channels)), np.nan, dtype=np.float32)
    cities = index['city'].to_numpy()
    dates = pd.to_datetime(index['date']).to_numpy()
    for city, part in table.groupby('city', sort=False):
        rows = np.where(cities == city)[0]
        if not len(rows):
            continue
        part = part.sort_values('timestamp')
        stamps = pd.to_datetime(part['timestamp']).to_numpy()
        values = part.reindex(columns=channels).to_numpy(dtype=np.float32)
        if len(values) < hours:
            continue
        strided = sliding_window_view(values, hours, axis=0).transpose(0, 2, 1)     # (n, hours, channels)
        last = dates[rows] + np.timedelta64(LAST_HOUR, 'h')
        position = np.searchsorted(stamps, last)
        inside = (position < len(stamps)) & (position >= hours - 1)
        inside[inside] &= stamps[position[inside]] == last[inside]
        out[rows[inside]] = strided[position[inside] - (hours - 1)]
    return out


def pack(X, H):
    """Daily windows (n, lookback, variates) and hourly ones (n, hours, channels) as one flat array."""
    return np.concatenate([X.reshape(len(X), -1), H.reshape(len(H), -1)], axis=1)


def standardise_hourly(hourly):
    """Each hourly channel standardised inside its window, ignoring gaps; gaps read as zero."""
    seen = torch.isfinite(hourly)
    filled = torch.nan_to_num(hourly) * seen
    count = seen.sum(dim=1, keepdim=True).clamp(min=1)
    mean = filled.sum(dim=1, keepdim=True) / count
    spread = torch.sqrt((((filled - mean) * seen) ** 2).sum(dim=1, keepdim=True) / count)
    moves = spread > CONSTANT_TOLERANCE
    return torch.where(moves, (filled - mean) / (spread + EPSILON), torch.zeros_like(filled)) * seen


class FinalModel(nn.Module):
    """All-token inverted transformer, with hourly tokens (n_channels > 0) and anchors (anchor_indices) optional.

    `anchor_indices` gives, for each horizon, the variate holding that horizon's nightly
    anchor; the value on the window's last day is the one used. `mixture` names the
    anchors the forecast may start from, out of MIXTURE: ('mean',) alone is reversible
    normalisation with the anchor token as an input, ('mean', 'today') a learned mix
    of the two known starts, and all three the anchored forecast.
    """

    def __init__(self, n_variates, lookback, horizon, target_index, anchor_indices=None, mixture=MIXTURE,
                 n_channels=len(HOURLY_CHANNELS), hours=HOURS, d_model=128, n_heads=8, n_layers=2, dropout=0.3):
        super().__init__()
        if anchor_indices is not None and len(anchor_indices) != horizon:
            raise ValueError(f"one anchor per horizon: {horizon} horizons, {len(anchor_indices)} anchors")
        unknown = [name for name in mixture if name not in MIXTURE]
        if unknown or 'mean' not in mixture:
            raise ValueError(f"mixture must include 'mean' and draw from {MIXTURE}, not {tuple(mixture)}")
        self.n_variates, self.lookback, self.horizon = n_variates, lookback, horizon
        self.n_channels, self.hours = n_channels, hours
        self.anchor_indices = None if anchor_indices is None else list(anchor_indices)
        self.mixture = [MIXTURE.index(name) for name in MIXTURE if name in mixture]
        self.trunk = ITransformer(n_variates, lookback, horizon, target_index,
                                  d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.trunk.project = nn.Identity()
        self.hourly_embed = nn.Linear(hours, d_model) if n_channels else None
        tokens = n_variates + n_channels
        if self.anchor_indices is not None:
            tokens += 1
            self.anchor_embed = nn.Linear(horizon + 1, d_model)
            self.mix = None
            if len(self.mixture) > 1:
                self.mix = nn.Linear(d_model, horizon * len(self.mixture))
                nn.init.zeros_(self.mix.weight)
                start = torch.zeros(horizon, len(self.mixture))
                start[:, 0] = MEAN_START_LOGIT
                self.mix.bias.data.copy_(start.flatten())
        self.drop = nn.Dropout(dropout)
        self.expert = nn.Linear(tokens * d_model, horizon)

    @property
    def target_index(self):
        return self.trunk.target_index

    def split(self, inputs):
        """(window, hourly), where hourly is None for a model without hourly tokens."""
        if not self.n_channels:
            return inputs, None
        cut = self.lookback * self.n_variates
        return (inputs[:, :cut].view(-1, self.lookback, self.n_variates),
                inputs[:, cut:].view(-1, self.hours, self.n_channels))

    def anchors(self, window):
        """The three anchors for every horizon, (batch, horizon, 3), in AQI points."""
        target = window[:, :, self.target_index]
        mean = target.mean(dim=1, keepdim=True).expand(-1, self.horizon)
        today = target[:, -1:].expand(-1, self.horizon)
        night = window[:, -1, self.anchor_indices]
        return torch.stack([mean, today, night], dim=-1)

    def _encode(self, inputs):
        window, hourly = self.split(inputs)
        centre, scale = window_scale(window)
        tokens = [self.trunk.embed(((window - centre) / scale).permute(0, 2, 1))]
        if hourly is not None:
            tokens.append(self.hourly_embed(standardise_hourly(hourly).permute(0, 2, 1)))
        anchors = None
        if self.anchor_indices is not None:
            anchors = self.anchors(window)
            spread = window[:, :, self.target_index].std(dim=1, keepdim=True, unbiased=False) + EPSILON
            mean = anchors[:, 0, 0].unsqueeze(-1)
            distances = torch.cat([(anchors[:, :, 2] - mean) / spread, (anchors[:, :1, 1] - mean) / spread], dim=1)
            distances = distances.clamp(-ANCHOR_TOKEN_LIMIT, ANCHOR_TOKEN_LIMIT)
            tokens.append(self.anchor_embed(distances).unsqueeze(1))
        encoded = self.trunk.norm(self.trunk.encoder(torch.cat(tokens, dim=1)))
        return encoded, centre, scale, anchors

    def _weights(self, encoded):
        """(batch, horizon, 3) weights over ANCHORS; an anchor outside the mixture gets none."""
        weights = encoded.new_zeros(encoded.shape[0], self.horizon, len(ANCHORS))
        if self.mix is None:
            weights[:, :, self.mixture[0]] = 1.0
        else:
            logits = self.mix(encoded.mean(dim=1)).view(-1, self.horizon, len(self.mixture))
            weights[:, :, self.mixture] = torch.softmax(logits, dim=-1)
        return weights

    def forward(self, inputs):
        """Inputs as described above -> (batch, horizon) forecasts in AQI points."""
        encoded, centre, scale, anchors = self._encode(inputs)
        correction = self.expert(self.drop(encoded.flatten(1)))
        if anchors is None:
            return self.trunk.denormalise(correction, centre, scale)
        start = (self._weights(encoded) * anchors).sum(dim=-1)
        return start + correction * scale[:, 0, self.target_index].unsqueeze(-1)

    @torch.no_grad()
    def anchor_weights(self, inputs, device=None, batch_size=512):
        """How much each anchor counts, (windows, horizon, 3), as a NumPy array."""
        device = device or next(self.parameters()).device
        self.eval()
        out = []
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size])).to(device)
            out.append(self._weights(self._encode(batch)[0]).cpu().numpy())
        return np.concatenate(out)

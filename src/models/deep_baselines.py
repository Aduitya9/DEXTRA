"""
The deep baselines a reviewer expects next to the final model.

Three standard architectures, sized and regularised like the rest of the project and
scored on the same windows. Each standardises per window and puts the target's mean
and spread back on the way out, as RevIN does and as every model here does, so none
is handicapped by the differences in scale between cities.

  LSTMForecaster  reads every variate, day by day: the usual model in the AQI literature.
  DLinear         the target's own series only, split into trend and remainder, one
                  linear map each (Zeng et al., AAAI 2023).
  PatchTST        the target's own series only, cut into overlapping patches for a
                  transformer across time (Nie et al., ICLR 2023). Fourteen days give
                  six patches of four, so this is a small version of it.
"""

import torch
from torch import nn

from src.models.itransformer import window_scale


def _denormalise(forecast, centre, scale, target_index):
    return forecast * scale[:, 0, target_index].unsqueeze(-1) + centre[:, 0, target_index].unsqueeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_variates, horizon, target_index, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.horizon, self.target_index = horizon, target_index
        self.lstm = nn.LSTM(n_variates, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(hidden, horizon)

    def forward(self, window):
        centre, scale = window_scale(window)
        output, _ = self.lstm((window - centre) / scale)
        return _denormalise(self.project(self.drop(output[:, -1])), centre, scale, self.target_index)


class DLinear(nn.Module):
    def __init__(self, lookback, horizon, target_index, kernel=5):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("the moving average needs an odd kernel, so it stays centred")
        self.horizon, self.target_index, self.kernel = horizon, target_index, kernel
        self.trend = nn.Linear(lookback, horizon)
        self.remainder = nn.Linear(lookback, horizon)

    def forward(self, window):
        centre, scale = window_scale(window)
        series = ((window - centre) / scale)[:, :, self.target_index]
        edge = self.kernel // 2
        padded = torch.cat([series[:, :1].repeat(1, edge), series, series[:, -1:].repeat(1, edge)], dim=1)
        trend = padded.unfold(1, self.kernel, 1).mean(dim=-1)
        return _denormalise(self.trend(trend) + self.remainder(series - trend), centre, scale, self.target_index)


class PatchTST(nn.Module):
    def __init__(self, lookback, horizon, target_index, patch=4, stride=2, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.3):
        super().__init__()
        self.horizon, self.target_index, self.patch, self.stride = horizon, target_index, patch, stride
        n_patches = (lookback - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(n_patches * d_model, horizon)

    def forward(self, window):
        centre, scale = window_scale(window)
        series = ((window - centre) / scale)[:, :, self.target_index]
        tokens = self.embed(series.unfold(1, self.patch, self.stride)) + self.position
        encoded = self.norm(self.encoder(tokens))
        return _denormalise(self.project(self.drop(encoded.flatten(1))), centre, scale, self.target_index)

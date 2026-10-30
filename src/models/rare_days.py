"""
Three cheap ways to make a forecaster take severe days seriously -- the bar
C-STDA has to clear.

Severe days (AQI above 300) are a few percent of the training targets. A model
fitted by squared error on that mix learns to hedge towards the common case,
and on a city it has never seen it hedges hardest: notebook 05's forecasts sat
72 AQI points below the truth on the new cities' severe days, against 32 on the
cities they trained on. Before generating synthetic episodes to fix that, the
obvious fixes have to be tried, because they are the first thing a reviewer
asks about:

  oversampled  draw the windows that hold a severe day more often, until they
               are a quarter of every epoch -- the tail share C-STDA targets
               (TARGET_TAIL_RATIO), made of real windows repeated rather than
               new ones generated
  weighted     keep the data as it is, but weigh severe targets so that they
               carry the same quarter of the loss
  quantile     train for the 90th percentile instead of the mean (pinball
               loss), so the forecast sits high exactly when the outlook is
               uncertain, which is when severe days arrive

All three leave the architecture alone -- notebook 05's all-token head -- so any
difference between them is the training, not the model.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import RARE_AQI_THRESHOLD, TARGET_TAIL_RATIO

QUANTILE = 0.9


def severe_windows(y, threshold=RARE_AQI_THRESHOLD):
    """Windows with at least one severe day among their targets."""
    return (np.asarray(y) > threshold).any(axis=1)


def oversampling_loader(X, y, batch_size, share=TARGET_TAIL_RATIO, threshold=RARE_AQI_THRESHOLD):
    """A training loader in which severe windows make up `share` of every epoch.

    Draws with replacement, len(y) windows an epoch, so an epoch is as long as
    before and only its make-up changes. The draws come from torch's global
    generator, which training.capture_rng saves, so a resumed run redraws the
    same epochs. With nothing to rebalance it is an ordinary shuffled loader.
    """
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                            torch.from_numpy(np.ascontiguousarray(y)))
    severe = severe_windows(y, threshold)
    if severe.all() or not severe.any():
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    weights = np.where(severe, share / severe.sum(), (1 - share) / (~severe).sum())
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(y), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def severe_weight(y, share=TARGET_TAIL_RATIO, threshold=RARE_AQI_THRESHOLD):
    """The weight that makes severe targets carry `share` of the total loss weight."""
    y = np.asarray(y)
    severe = int((y > threshold).sum())
    ordinary = y.size - severe
    if severe == 0 or ordinary == 0:
        return 1.0
    return float(share / (1 - share) * ordinary / severe)


class _Wrapped(nn.Module):
    """A forecaster with its own training loss; forecasting is the inner model's."""

    uses_own_loss = True

    def __init__(self, model):
        super().__init__()
        self.inner = model

    @property
    def horizon(self):
        return self.inner.horizon

    def forward(self, window):
        return self.inner(window)


class WeightedSquaredError(_Wrapped):
    """Squared error, with every severe target counted `weight` times."""

    def __init__(self, model, weight, threshold=RARE_AQI_THRESHOLD):
        super().__init__(model)
        self.weight, self.threshold = float(weight), float(threshold)

    def training_loss(self, window, answer):
        weights = torch.where(answer > self.threshold, torch.full_like(answer, self.weight),
                              torch.ones_like(answer))
        return (weights * (self.inner(window) - answer) ** 2).sum() / weights.sum()


def pinball(forecast, answer, quantile=QUANTILE):
    """Mean pinball loss: under-forecasts cost `quantile`, over-forecasts 1 - quantile."""
    error = answer - forecast
    return torch.maximum(quantile * error, (quantile - 1) * error).mean()


class QuantileForecaster(_Wrapped):
    """Trained for the `quantile`-th percentile of the outcome rather than its mean."""

    def __init__(self, model, quantile=QUANTILE):
        super().__init__(model)
        self.quantile = float(quantile)

    def training_loss(self, window, answer):
        return pinball(self.inner(window), answer, self.quantile)


def pinball_score(quantile=QUANTILE):
    """A validation score for training.train(score=...): mean pinball loss, lower is better."""

    def score(model, loader, device):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for window, answer in loader:
                answer = answer.to(device)
                total += float(pinball(model(window.to(device)), answer, quantile)) * answer.numel()
                count += answer.numel()
        return total / max(count, 1)

    score.__name__ = f"pinball_{quantile:g}"
    return score

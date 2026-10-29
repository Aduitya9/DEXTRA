"""
The iTransformer backbone: attention across variates rather than across time.

A conventional time-series transformer makes one token per time step, so its
attention asks which *days* matter to each other. iTransformer (Liu et al.,
ICLR 2024) inverts that: each variate's whole 14-day series becomes one token,
and attention asks which *variates* matter to each other. For DEXTRA that is the
question worth asking, because the thing being learned is how the boundary layer,
the inversion and the fires relate to the pollutants -- and notebook 02 showed
four of those relationships reverse by season, so they have to be read together
rather than separately.

Two consequences fall out of the inversion. There is no positional encoding,
because variates have no order to encode. And each token is a whole series, so a
14-day lookback costs no more attention than a 3-day one would.

Normalisation is per window and reversible. Each variate is standardised by its
own mean and spread inside that window, and the target's are put back on the way
out. Delhi's ordinary Tuesday and Chennai's emergency stop being the same number
without any global scaler being fitted, and a city the model has never seen needs
no statistics of its own.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPSILON = 1e-5
# Below this a variate is treated as constant inside the window, not standardised.
CONSTANT_TOLERANCE = 1e-6


def set_seed(seed=42):
    """Everything downstream of this is reproducible on the same machine."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer_gpu=True):
    return torch.device('cuda' if prefer_gpu and torch.cuda.is_available() else 'cpu')


class ITransformer(nn.Module):
    def __init__(self, n_variates, lookback, horizon, target_index,
                 d_model=128, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide by n_heads {n_heads}")
        self.n_variates, self.lookback, self.horizon = n_variates, lookback, horizon
        self.target_index = target_index

        self.embed = nn.Linear(lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, horizon)

    def forward(self, window):
        """window: (batch, lookback, variates) -> (batch, horizon) in AQI points."""
        mean = window.mean(dim=1, keepdim=True)
        spread = window.std(dim=1, keepdim=True, unbiased=False)

        # A variate that does not move inside the window is passed through at its
        # own value instead of being standardised. Subtracting its mean would
        # leave exactly zero, and three of the twenty-six -- archetype_id,
        # coastal_flag and industrial_index -- never move, so the features that
        # say which kind of city this is would reach the model as nothing at all.
        # It costs the calendar flags too: a fortnight wholly inside the stubble
        # season is otherwise indistinguishable from a fortnight wholly outside it.
        moves = spread > CONSTANT_TOLERANCE
        centre = torch.where(moves, mean, torch.zeros_like(mean))
        scale = torch.where(moves, spread + EPSILON, torch.ones_like(spread))
        standardised = (window - centre) / scale

        tokens = self.embed(standardised.permute(0, 2, 1))     # (batch, variates, d_model)
        encoded = self.norm(self.encoder(tokens))
        forecast = self.project(encoded)                       # (batch, variates, horizon)

        target = forecast[:, self.target_index, :]
        target_centre = centre[:, 0, self.target_index].unsqueeze(-1)
        target_scale = scale[:, 0, self.target_index].unsqueeze(-1)
        return target * target_scale + target_centre           # back into AQI points


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loader(X, y, batch_size=256, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                            torch.from_numpy(np.ascontiguousarray(y)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict(model, X, device=None, batch_size=512):
    """Forecasts for every window, in AQI points."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.horizon), dtype=np.float32)


def _rmse(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for window, answer in loader:
            error = model(window.to(device)) - answer.to(device)
            total += float((error ** 2).sum())
            count += answer.numel()
    return float(np.sqrt(total / max(count, 1)))


def train(model, train_loader, val_loader, device, epochs=60, lr=3e-4,
          weight_decay=1e-4, patience=8, clip=1.0, log=print):
    """Fit on the training split, stop on validation RMSE, keep the best weights.

    The loss is plain mean squared error on AQI points. Weighting it towards
    hazardous days would help the number this project cares about, and that is
    exactly why it is not done here: this notebook has to answer whether the
    architecture earns its place, and a loss change would answer a different
    question at the same time. Rare-event handling arrives with C-STDA, where it
    can be measured on its own.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()

    best = {'rmse': float('inf'), 'epoch': -1, 'state': copy.deepcopy(model.state_dict())}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for window, answer in train_loader:
            window, answer = window.to(device), answer.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(window), answer)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()
            running += float(loss.detach()) * len(answer)
            seen += len(answer)
        schedule.step()

        val_rmse = _rmse(model, val_loader, device)
        history.append({'epoch': epoch, 'train_rmse': float(np.sqrt(running / max(seen, 1))),
                        'val_rmse': val_rmse})
        if val_rmse < best['rmse'] - 1e-4:
            best = {'rmse': val_rmse, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
            marker = '  <- best'
        elif epoch - best['epoch'] >= patience:
            log(f"  epoch {epoch:>3}  val rmse {val_rmse:7.3f}   stopping: "
                f"{patience} epochs without improvement")
            break
        else:
            marker = ''
        if epoch % 5 == 0 or marker:
            log(f"  epoch {epoch:>3}  train rmse {history[-1]['train_rmse']:7.3f}  "
                f"val rmse {val_rmse:7.3f}{marker}")

    model.load_state_dict(best['state'])
    log(f"  best epoch {best['epoch']}, val rmse {best['rmse']:.3f}")
    return model, history

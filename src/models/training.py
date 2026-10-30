"""
A training loop that a dropped Colab session cannot undo.

The free tier gives no background execution: close the tab, or step away long
enough to look idle, and the runtime is reclaimed along with everything in its
memory. Colab's terms also rule out the scripts that fake activity to keep it
alive, so this does not try. It makes the work impossible to lose instead:

  Resumable. The full training state -- weights, optimiser, schedule, epoch,
  the best weights so far, the curve, and every random generator -- is saved
  to the Hugging Face store at intervals and at the end. Re-running the same
  run picks up from that state, and because the generators are restored too,
  a resumed run continues on the same path it would have taken. Only the same
  run: a checkpoint saved under other settings or other data is set aside.

  Bounded. A time budget stops a run cleanly, saved, before Colab's 12-hour
  ceiling can stop it mid-write.

  Guarded. A loss that turns NaN or infinite ends the run on the best weights
  it had, rather than training on and saving garbage.

  Early-stopped. Validation error that has not improved for `patience` epochs
  ends the run, and the best weights -- not the last -- are what come back.

Checkpoints are throttled to one every few minutes. Each is a commit to the
dataset repo, and a commit per epoch would be the rapid-fire write pattern the
Hub rate-limits; most runs here finish inside one interval and leave none.
"""

import copy
import hashlib
import io
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUTCOMES = ('early_stopped', 'completed', 'diverged', 'out_of_time')


def capture_rng():
    """Every generator a training step draws from, so a resumed run is the same run.

    NumPy's state is a tuple holding an array, and saving that array as-is would
    force checkpoints to be loaded with full unpickling, which can execute code.
    Stored as a tensor and plain numbers, the whole checkpoint loads with
    weights_only=True.
    """
    kind, keys, position, has_gauss, gauss = np.random.get_state()
    state = {'torch': torch.get_rng_state(),
             'numpy': {'kind': str(kind), 'keys': torch.from_numpy(keys.astype(np.int64)),
                       'position': int(position), 'has_gauss': int(has_gauss), 'gauss': float(gauss)}}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['kind'], numpy_state['keys'].numpy().astype(np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'], numpy_state['gauss']))
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class Checkpointer:
    """Save and restore one run's whole training state on a store, throttled."""

    def __init__(self, store, path, every_seconds=240, log=print):
        self.store, self.path, self.every_seconds, self.log = store, path, every_seconds, log
        self._last_save = time.monotonic()

    def load(self):
        if not self.store.exists(self.path):
            return None
        raw = self.store.read_bytes(self.path)
        # weights_only: tensors, numbers, strings and containers, nothing executable.
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)

    def due(self):
        return time.monotonic() - self._last_save >= self.every_seconds

    def save(self, state):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / Path(self.path).name
            torch.save(state, local)
            self.store.upload_file(local, self.path, f"Training checkpoint, epoch {state['epoch']}")
        self._last_save = time.monotonic()
        self.log(f"  checkpoint saved at epoch {state['epoch']}")


def run_signature(model, train_loader, val_loader, **settings):
    """What makes a run this run: its settings, the shape of its model, and its data.

    A checkpoint with another signature was saved by some other run -- before a
    setting was changed, or the windows rebuilt -- and resuming it would carry
    that run on under this one's name, because the optimiser restores its own
    learning rate and the schedule its own length.
    """
    signature = {name: float(value) for name, value in settings.items()}
    shapes = [(name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()]
    signature['model'] = hashlib.sha1(repr(shapes).encode()).hexdigest()
    for split, loader in (('train', train_loader), ('val', val_loader)):
        digest = hashlib.sha1(str(len(loader.dataset)).encode())
        for tensor in getattr(loader.dataset, 'tensors', ()):
            digest.update(tensor.detach().cpu().numpy().tobytes())
        signature[split] = digest.hexdigest()
    return signature


def _validation_rmse(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for window, answer in loader:
            error = model(window.to(device)) - answer.to(device)
            total += float((error ** 2).sum())
            count += answer.numel()
    return float(np.sqrt(total / max(count, 1)))


def train(model, train_loader, val_loader, device, epochs=30, lr=3e-4, weight_decay=1e-3,
          patience=10, clip=1.0, aux_weight=0.0, checkpointer=None, deadline=None,
          save_finished=True, log=print):
    """Fit on the training split, stop on validation RMSE, return the best weights.

    Returns (model, history, outcome), with outcome one of OUTCOMES.

    `aux_weight` scales a model's own auxiliary loss -- the Met-MoE's expert
    balance term -- relative to the forecast loss rather than in absolute terms.
    Mean squared error here is in the hundreds (AQI points squared) while a
    balance term lives between 0 and 1, so an absolute weight like the config's
    0.01 would make it vanish; scaled, 0.01 means it can be at most a hundredth
    of the forecast loss, which is the gentle nudge that weight was meant to be.

    `deadline` is a time.monotonic() value. Past it the run saves and returns
    'out_of_time' at the next epoch boundary, never mid-epoch.

    `save_finished` writes one last checkpoint when a run ends normally, so that
    re-running it returns at once. A caller that records finished runs itself --
    the notebooks, through RunRegistry -- turns it off: the run's result and
    weights already go up in one commit, and a finished checkpoint would be a
    second commit of several megabytes of optimiser state nobody reads.

    A model whose objective is not squared error on its forecast -- the
    Met-MoE's competitive mixture, which says so through `uses_own_loss` --
    trains on its own `training_loss`, and its curve records that loss as
    'train_loss'. Validation, early stopping and the weights handed back are
    judged exactly as for any other model: on the forecast's RMSE in AQI points.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()
    has_aux = aux_weight > 0 and callable(getattr(model, 'auxiliary_loss', None))
    own_loss = model.training_loss if getattr(model, 'uses_own_loss', False) else None
    train_label = 'train_loss' if own_loss else 'train_rmse'

    start_epoch, history = 1, []
    best = {'rmse': float('inf'), 'epoch': 0, 'state': copy.deepcopy(model.state_dict())}
    signature = run_signature(model, train_loader, val_loader, epochs=epochs, lr=lr,
                              weight_decay=weight_decay, patience=patience, clip=clip,
                              aux_weight=aux_weight)
    saved = checkpointer.load() if checkpointer else None
    if saved is not None and saved.get('signature') != signature:
        log("  a checkpoint exists for this run, but from other settings or data; starting afresh")
        saved = None
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimiser.load_state_dict(saved['optimiser'])
        schedule.load_state_dict(saved['schedule'])
        best, history = saved['best'], saved['history']
        start_epoch = saved['epoch'] + 1
        restore_rng(saved['rng'])
        log(f"  resuming from epoch {saved['epoch']} (best so far {best['rmse']:.3f} at epoch {best['epoch']})")
        if saved.get('finished'):
            model.load_state_dict(best['state'])
            return model, history, saved['outcome']

    def state(epoch, outcome=None):
        return {'model': model.state_dict(), 'optimiser': optimiser.state_dict(),
                'schedule': schedule.state_dict(), 'best': best, 'history': history,
                'epoch': epoch, 'rng': capture_rng(), 'outcome': outcome,
                'finished': outcome is not None, 'signature': signature}

    outcome = 'completed'
    for epoch in range(start_epoch, epochs + 1):
        if deadline is not None and time.monotonic() > deadline:
            outcome = 'out_of_time'
            log(f"  time budget reached before epoch {epoch}; saving and stopping")
            break

        model.train()
        running, seen, diverged = 0.0, 0, False
        for window, answer in train_loader:
            window, answer = window.to(device), answer.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = own_loss(window, answer) if own_loss else criterion(model(window), answer)
            if has_aux:
                # abs: a likelihood objective could in principle dip below zero, and a
                # negative scale would turn the balance penalty into a reward.
                loss = loss + aux_weight * loss.detach().abs() * model.auxiliary_loss()
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()
            running += float(loss.detach()) * len(answer)
            seen += len(answer)
        if diverged:
            outcome = 'diverged'
            log(f"  loss became non-finite in epoch {epoch}; keeping the best weights "
                f"(epoch {best['epoch']}) and stopping")
            break
        schedule.step()

        val_rmse = _validation_rmse(model, val_loader, device)
        mean_loss = running / max(seen, 1)
        history.append({'epoch': epoch, train_label: float(mean_loss if own_loss else np.sqrt(mean_loss)),
                        'val_rmse': val_rmse})
        improved = val_rmse < best['rmse'] - 1e-4
        if improved:
            best = {'rmse': val_rmse, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
        if improved or epoch % 5 == 0:
            log(f"  epoch {epoch:>3}  {train_label.replace('_', ' ')} {history[-1][train_label]:7.3f}  "
                f"val rmse {val_rmse:7.3f}" + ("  <- best" if improved else ""))
        if not improved and epoch - best['epoch'] >= patience:
            outcome = 'early_stopped'
            log(f"  epoch {epoch:>3}  stopping: {patience} epochs without improvement")
            break
        if checkpointer and checkpointer.due():
            checkpointer.save(state(epoch))

    last = history[-1]['epoch'] if history else start_epoch - 1
    if checkpointer and outcome == 'out_of_time':
        checkpointer.save(state(last))                       # unfinished: resume next session
    elif checkpointer and save_finished and last >= start_epoch:
        checkpointer.save(state(last, outcome))

    model.load_state_dict(best['state'])
    if math.isfinite(best['rmse']):
        log(f"  best epoch {best['epoch']}, val rmse {best['rmse']:.3f}  ({outcome})")
    return model, history, outcome


class RunRegistry:
    """Which runs have finished, kept on the store so a fresh session can skip them.

    One small JSON file per finished run rather than one shared table: two
    writes can never collide, and a session that dies mid-run leaves the
    finished ones untouched.
    """

    def __init__(self, store, prefix):
        self.store, self.prefix = store, prefix.rstrip('/')

    def path(self, name):
        return f"{self.prefix}/{name}.json"

    def done(self, name):
        return self.store.exists(self.path(name))

    def record(self, name, result, attachments=None):
        """Store a finished run's result, and any files that belong with it, in one commit.

        `attachments` maps a filename to a local path -- a run's weights, say.
        Going up together means a run is either recorded with its weights or not
        recorded at all: a session that dies between the two can never leave a
        result saying a run is done whose model no longer exists.
        """
        if not attachments:
            self.store.upload_json(result, self.path(name), f"Finished run {name}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, f"{name}.json").write_text(json.dumps(result, indent=1, sort_keys=True),
                                                 encoding='utf-8')
            for filename, source in attachments.items():
                shutil.copy2(source, Path(tmp, filename))
            self.store.upload_folder(tmp, self.prefix, f"Finished run {name}")

    def attachment(self, filename):
        """The store path of a file recorded alongside a run."""
        return f"{self.prefix}/{filename}"

    def load(self, name):
        return self.store.read_json(self.path(name))

    def finished(self):
        return sorted(p[len(self.prefix) + 1:-len('.json')]
                      for p in self.store.files(self.prefix + '/') if p.endswith('.json'))

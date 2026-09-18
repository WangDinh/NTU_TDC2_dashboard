"""Training for the Adapt-PCNN MLP.

Because build_pcnn_dataset returns a contiguous (dropna'd) frame, we can use
clean FIXED-length windows of `warm_start + horizon` steps sliding by
`overlapping_distance` — no NaN-jump handling or zero-padding (and hence no
padded-loss corruption) like the original variable-length sequence code needed.

Each window: `warm_start` steps seed the model on real RA_T, then `horizon`
steps are predicted autoregressively (RA_T fed back, drivers real) and scored
against the true RA_T with MSE. The tail `validation_percentage` of windows is
held out; the best-val checkpoint is returned.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .module import Adapt_PC_MLP, DEVICE


def make_windows(X: np.ndarray, cfg) -> np.ndarray:
    """Slice a scaled (n, n_feat) array into (n_windows, window_len, n_feat)."""
    L = cfg.window_len
    starts = range(0, len(X) - L + 1, cfg.overlapping_distance)
    if not starts:
        return np.empty((0, L, X.shape[1]), dtype=np.float32)
    return np.stack([X[s:s + L] for s in starts], axis=0).astype(np.float32)


def rollout_window(model, batch, cfg):
    """Forward a batch of windows (B, L, F) → (preds, targets) each (B, horizon).

    The model's output at input step t is `RA_T[t] + increments`, i.e. a
    prediction of temperature at t+1. So the prediction made at step t is scored
    against the real RA_T at t+1. The `horizon` scored predictions are those made
    at input steps warm_start-1 .. L-2 (the step warm_start-1 is the last warm-up
    step, still fed real RA_T — matching the original PCNN eval which collects
    from the final warm-up step onward). Their targets are RA_T[warm_start .. L-1].

    warm_start >= 1 is required (the model needs >=1 real step to seed last_D).
    Everything is in scaled space.
    """
    L = cfg.window_len
    tcol = cfg.temperature_col
    ws = cfg.warm_start
    model.last_D = None
    all_preds = []
    for t in range(L):
        p = model(batch[:, t, :], warm_start=t < ws)               # (B, 1) = pred of temp[t+1]
        all_preds.append(p)
    preds = torch.cat(all_preds[ws - 1: L - 1], dim=1)             # (B, horizon)
    targets = batch[:, ws:L, tcol]                                 # (B, horizon)
    return preds, targets


def train_pcnn(X_train_scaled: np.ndarray, cfg):
    """Train an Adapt_PC_MLP on scaled training rows. Returns the best-val model.

    Data stays on CPU; each mini-batch is moved to DEVICE just in time.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    windows = make_windows(X_train_scaled, cfg)
    if len(windows) == 0:
        raise ValueError(
            f'No training windows: need >= window_len={cfg.window_len} rows, '
            f'got {len(X_train_scaled)}.')

    # Time-ordered train/val split of the windows.
    split = int(len(windows) * (1 - cfg.validation_percentage))
    split = max(1, min(split, len(windows) - 1)) if len(windows) > 1 else 1
    train_w = torch.tensor(windows[:split])
    val_w   = torch.tensor(windows[split:]) if split < len(windows) else None

    model = Adapt_PC_MLP(cfg, device=DEVICE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float('inf'), None
    n = len(train_w)
    order = np.arange(n)

    pbar = tqdm(range(cfg.n_epochs), desc='  epochs', unit='ep',
                leave=False, disable=cfg.verbose == 0)
    for ep in pbar:
        model.train()
        np.random.shuffle(order)
        ep_loss, ep_seen = 0.0, 0
        for b in range(0, n, cfg.batch_size):
            idx = order[b:b + cfg.batch_size]
            batch = train_w[idx].to(DEVICE)
            opt.zero_grad()
            preds, targets = rollout_window(model, batch, cfg)
            loss = loss_fn(preds, targets)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
            ep_seen += len(idx)
        train_loss = ep_loss / max(1, ep_seen)

        # Validation.
        val_loss = train_loss
        if val_w is not None:
            model.eval()
            with torch.no_grad():
                vp, vt = rollout_window(model, val_w.to(DEVICE), cfg)
                val_loss = float(loss_fn(vp, vt))
        if cfg.verbose:
            pbar.set_postfix(train=f'{train_loss:.2e}', val=f'{val_loss:.2e}',
                             best=f'{best_val:.2e}')

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), best_val

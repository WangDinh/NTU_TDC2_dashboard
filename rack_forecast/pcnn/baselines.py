"""Black-box baselines for the RA_T forecast, for comparison against Adapt_PC_MLP.

Fair, paper-style ablation: each baseline gets the SAME information PCNN's rollout
uses — all exogenous drivers across the window plus the seed RA_T, with future
RA_T masked — and predicts the same `horizon` RA_T values on the SAME
non-overlapping test windows. The only thing that differs from PCNN is the model
(no physics structure), so a gap reflects the physics, not the inputs.

Formulation is DIRECT multi-step (predict all horizon steps at once), which is the
natural way to give a black-box model known future covariates without a feedback
loop. The DL architectures are the very ones from the power pipeline's model zoo
(rack_forecast.models.{lstm,cnn1d,transformer}) reused read-only with
`n_out=horizon`; xgboost/linear use their libraries directly. Nothing here trains
or evaluates through the power pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from ..models.lstm import LSTMModel
from ..models.cnn1d import CNN1DModel
from ..models.transformer import TransformerModel
from .module import DEVICE
from .evaluate import compute_metrics

BASELINE_MODELS = ['linear', 'xgboost', 'lstm', 'cnn1d', 'transformer']
_TABULAR = {'linear', 'xgboost'}


def make_baseline_windows(X_scaled: np.ndarray, cfg):
    """Non-overlapping windows matching PCNN's eval exactly.

    For each window starting at s (step = window_len):
        input : rows s .. s+L-2  (L-1 steps, all 6 features) with the future RA_T
                masked to 0 (RA_T kept only at the seed row) — same info PCNN has:
                real drivers across the horizon + the initial temperature.
        target: RA_T at rows s+warm_start .. s+L-1  (== PCNN's scored targets).

    Returns (seqs, y): seqs (n, L-1, 6), y (n, horizon).
    """
    L = cfg.window_len
    tcol = cfg.temperature_col
    seqs, ys = [], []
    for s in range(0, len(X_scaled) - L + 1, L):
        win = X_scaled[s:s + L]
        inp = win[:L - 1].copy()                 # rows 0..L-2
        inp[cfg.warm_start:, tcol] = 0.0         # mask future RA_T (keep seed rows)
        seqs.append(inp)
        ys.append(win[cfg.warm_start:L, tcol])   # horizon RA_T
    return np.asarray(seqs, np.float32), np.asarray(ys, np.float32)


def _train_dl(name, seqs, y, cfg):
    """Direct multi-step DL baseline: (n, L-1, 6) -> (n, horizon) RA_T."""
    torch.manual_seed(cfg.seed)
    n_feat = seqs.shape[2]
    seq_len = seqs.shape[1]
    arch = {
        'lstm': lambda: LSTMModel(n_feat, n_out=cfg.horizon),
        'cnn1d': lambda: CNN1DModel(n_feat, seq_len, n_out=cfg.horizon),
        'transformer': lambda: TransformerModel(n_feat, n_out=cfg.horizon),
    }[name]().to(DEVICE)

    # Time-ordered train/val split.
    split = max(1, int(len(seqs) * (1 - cfg.validation_percentage)))
    Xtr = torch.tensor(seqs[:split]); ytr = torch.tensor(y[:split])
    opt = torch.optim.Adam(arch.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    order = np.arange(len(Xtr))
    best, best_state = float('inf'), None

    for _ in range(cfg.n_epochs):
        arch.train(); np.random.shuffle(order)
        for b in range(0, len(order), cfg.batch_size):
            idx = order[b:b + cfg.batch_size]
            xb = Xtr[idx].to(DEVICE); yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(arch(xb), yb)
            loss.backward(); opt.step()
        # Track train loss as the checkpoint signal (small data; keep it simple).
        with torch.no_grad():
            arch.eval()
            vl = float(loss_fn(arch(Xtr.to(DEVICE)), ytr.to(DEVICE)))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in arch.state_dict().items()}
    if best_state:
        arch.load_state_dict(best_state)
    return arch.eval()


def train_baseline(name, seqs, y, cfg):
    """Fit one baseline. Returns (kind, model) where kind in {'tab','dl'}."""
    if name in _TABULAR:
        Xf = seqs.reshape(len(seqs), -1)
        if name == 'linear':
            model = Ridge(alpha=1.0).fit(Xf, y)
        else:
            # CPU: 15-output direct xgboost is fast, and CPU avoids the
            # cuda-booster/cpu-input device-mismatch warning at predict time.
            model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                 device='cpu', random_state=cfg.seed).fit(Xf, y)
        return 'tab', model
    return 'dl', _train_dl(name, seqs, y, cfg)


def evaluate_baseline(kind, model, seqs, y, cfg, normalizer):
    """Predict + score one baseline on the test windows (denormalized °C)."""
    if kind == 'tab':
        preds_s = np.asarray(model.predict(seqs.reshape(len(seqs), -1)), np.float32)
    else:
        with torch.no_grad():
            preds_s = model(torch.tensor(seqs).to(DEVICE)).cpu().numpy()
    preds_s = preds_s.reshape(len(seqs), cfg.horizon)

    preds   = normalizer.inverse_target(preds_s, 'RA_T')
    actuals = normalizer.inverse_target(y, 'RA_T')
    return {'preds': preds, 'actuals': actuals, 'metrics': compute_metrics(actuals, preds)}


def run_baselines(X_train_scaled, X_test_scaled, cfg, normalizer, models=None, verbose=True):
    """Train + evaluate every baseline; return {name: result_dict}."""
    models = models or BASELINE_MODELS
    seq_tr, y_tr = make_baseline_windows(X_train_scaled, cfg)
    seq_te, y_te = make_baseline_windows(X_test_scaled, cfg)
    out = {}
    for name in models:
        if verbose:
            print(f'[baseline:{name}] training...')
        kind, model = train_baseline(name, seq_tr, y_tr, cfg)
        out[name] = evaluate_baseline(kind, model, seq_te, y_te, cfg, normalizer)
        if verbose:
            m = out[name]['metrics']
            print(f"           MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
    return out

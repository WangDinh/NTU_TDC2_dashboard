"""Rollout evaluation + metrics for the Adapt-PCNN MLP.

Uses their design: warm-start autoregressive rollout where the exogenous drivers
(SA_T, SA_V, OA_T, OA_H, ITE_P) stay pinned to REAL data over the horizon and
only RA_T is fed back. This is a CONDITIONAL forecast — it assumes the future
driver trajectory is known — not open-loop forecasting.

Metrics are computed locally (not imported from rack_forecast.evaluate) so this
subpackage has zero import edge into the power pipeline.
"""

from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .module import DEVICE
from .train import make_windows, rollout_window


def compute_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / MAPE% / R2 on flat arrays (duplicated locally on purpose)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        'MAE':     mean_absolute_error(y_true, y_pred),
        'RMSE':    np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE(%)': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'R2':      r2_score(y_true, y_pred),
    }


def predict_window_trace(model, window_scaled: np.ndarray, cfg, normalizer):
    """Roll ONE window and return per-step trace for inspection/plotting.

    window_scaled: (window_len, n_feat) scaled array (one test window).
    Returns dict with, over the horizon steps (length cfg.horizon):
        pred     denormalized RA_T prediction (°C)
        actual   denormalized RA_T truth (°C)
        coeff_a  adaptive cooling coefficient per step
        coeff_b  adaptive heating coefficient per step
    """
    import torch.nn.functional as F

    dev = next(model.parameters()).device
    model.eval()
    x = torch.tensor(window_scaled[None, ...], dtype=torch.float32, device=dev)  # (1,L,F)
    L, ws, tcol = cfg.window_len, cfg.warm_start, cfg.temperature_col
    preds, ca, cb = [], [], []
    model.last_D = None
    with torch.no_grad():
        for t in range(L):
            # Recreate the coefficients for this step (same inputs the model uses).
            SA_T = x[:, t, cfg.supply_T_col:cfg.supply_T_col + 1]
            SA_V = x[:, t, cfg.supply_m_col:cfg.supply_m_col + 1]
            IT_P = x[:, t, cfg.power_col:cfg.power_col + 1]
            RA_T = (x[:, t, tcol:tcol + 1] if t < ws else model.last_D)
            coeff = F.softplus(model.fc2(F.softplus(model.fc1(
                torch.cat([SA_T, SA_V, IT_P, RA_T], dim=1)))))
            p = model(x[:, t, :], warm_start=t < ws)
            if ws - 1 <= t <= L - 2:
                preds.append(float(p[0, 0]))
                ca.append(float(coeff[0, 0]))
                cb.append(float(coeff[0, 1]))
    actual_s = window_scaled[ws:L, tcol]
    return {
        'pred':    normalizer.inverse_target(np.array(preds), 'RA_T'),
        'actual':  normalizer.inverse_target(actual_s, 'RA_T'),
        'coeff_a': np.array(ca),
        'coeff_b': np.array(cb),
    }


def rollout_evaluate(model, X_test_scaled: np.ndarray, cfg, normalizer):
    """Roll the model over non-overlapping test windows and score RA_T in °C.

    Returns:
        result: dict with
            preds     (n_windows, horizon)  denormalized °C
            actuals   (n_windows, horizon)  denormalized °C
            metrics   dict (MAE/RMSE/MAPE/R2 in °C)
            n_windows int
    """
    # Non-overlapping windows so each horizon step is scored once.
    windows = make_windows(X_test_scaled, replace(cfg, overlapping_distance=cfg.window_len))
    if len(windows) == 0:
        raise ValueError(
            f'No test windows: need >= window_len={cfg.window_len} rows, '
            f'got {len(X_test_scaled)}.')

    model = model.to(DEVICE).eval()
    with torch.no_grad():
        preds_s, targets_s = rollout_window(model, torch.tensor(windows).to(DEVICE), cfg)
    preds_s   = preds_s.cpu().numpy()        # (n_windows, horizon) scaled
    targets_s = targets_s.cpu().numpy()

    preds   = normalizer.inverse_target(preds_s,   'RA_T')
    actuals = normalizer.inverse_target(targets_s, 'RA_T')

    return {
        'preds':     preds,
        'actuals':   actuals,
        'metrics':   compute_metrics(actuals, preds),
        'n_windows': len(windows),
    }

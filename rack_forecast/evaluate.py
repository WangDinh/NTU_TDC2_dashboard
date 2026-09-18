"""Autoregressive multi-step evaluation and error metrics.

Prediction strategy: the test set is split into non-overlapping windows of
`horizon` steps. Each window is seeded from the last `lookback` GROUND-TRUTH
steps, then rolled out autoregressively — the model predicts the FULL next-step
feature vector (multi-output) and that whole vector is fed back in as the next
row, so no feature (target or otherwise) is frozen or given real future ground
truth within a window.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


class DivergenceError(RuntimeError):
    """A rollout blew up: predictions went non-finite before the horizon was reached.

    Long recursive rollouts feed each prediction back in as input, so error
    compounds multiplicatively. Models with unbounded outputs (linear, the DL
    models) can overflow float32 entirely — observed on cnn1d/lstm at
    horizon=2880. Raised instead of letting `inf` reach the scaler, where it
    surfaces as an opaque sklearn "Input contains infinity" ValueError.
    """


# ── shared inverse-scaling helper ───────────────────────────────────────────

def _inverse_target(values_scaled, scaler, target_col_idx, n_feat):
    """Inverse-transform just the target column via a scaler fit on all n_feat
    columns (pads the other columns with zeros, then slices the target back out)."""
    buf = np.zeros((len(values_scaled), n_feat), dtype=np.float32)
    buf[:, target_col_idx] = values_scaled
    return scaler.inverse_transform(buf)[:, target_col_idx]


# ── single-window rollout (used by the dashboard's live inference) ─────────

def autoregressive_predict(model, seed_window, horizon, target_col_idx):
    """Roll one seed window forward `horizon` steps.

    Args:
        model: object with .predict_step((1, lookback, n_feat)) → (n_feat,).
        seed_window: (lookback, n_feat) scaled array of real observations.
        horizon: number of steps to predict.
        target_col_idx: column index of the target, extracted for the return value.

    Returns:
        (horizon,) array of TARGET-ONLY predictions in SCALED space.
    """
    window = seed_window.copy()
    preds = []
    for _ in range(horizon):
        p = model.predict_step(window[np.newaxis])          # (n_feat,) full vector
        preds.append(p[target_col_idx])                      # scalar, for the return value
        window = np.vstack([window[1:], p[np.newaxis]])      # feed the WHOLE row back
    return np.array(preds, dtype=np.float32)


# ── batched evaluation over the whole test set ─────────────────────────────

def evaluate(model, X_test, y_test, lookback, horizon, target_col_idx, scaler, n_feat):
    """Roll out ALL test windows at once and return inverse-scaled TARGET results.

    Windows are advanced together — one model call per step instead of one per
    step per window — for a ~n_windows× speedup over a naive loop. Every step
    predicts the full feature vector and feeds it ALL back in (multi-output),
    so no feature is frozen or leaked from the future — only the target column
    is extracted for scoring/reporting.

    Returns:
        (preds, actuals): 1-D arrays in ORIGINAL units, length n_windows*horizon,
        laid out window-by-window so `.reshape(n_windows, horizon)` recovers them.
    """
    n_windows = (len(X_test) - lookback) // horizon

    # Seed each window with its preceding `lookback` ground-truth steps.
    windows = np.stack([
        X_test[lookback + w * horizon - lookback: lookback + w * horizon]
        for w in range(n_windows)
    ], axis=0).copy()                                        # (n_windows, lookback, n_feat)

    # Ground-truth targets for scoring.
    actuals_s = np.stack([
        y_test[lookback + w * horizon: lookback + w * horizon + horizon, target_col_idx]
        for w in range(n_windows)
    ], axis=0)                                               # (n_windows, horizon)

    all_preds_s = np.zeros_like(actuals_s)
    has_batch = hasattr(model, 'predict_batch')

    for step in tqdm(range(horizon), desc='  steps', unit='step', leave=False):
        if has_batch:
            preds = model.predict_batch(windows).astype(np.float32)   # (n_windows, n_feat)
        else:
            preds = np.array([model.predict_step(windows[i][np.newaxis])
                              for i in range(n_windows)], dtype=np.float32)

        # Catch a blown-up rollout here, at the step it happens, rather than
        # letting inf/NaN propagate into the scaler hundreds of steps later.
        if not np.isfinite(preds).all():
            raise DivergenceError(
                f'rollout diverged at step {step + 1}/{horizon}: predictions '
                f'went non-finite ({np.count_nonzero(~np.isfinite(preds))} of '
                f'{preds.size} values)'
            )

        all_preds_s[:, step] = preds[:, target_col_idx]

        # Slide every window forward one step, feeding the FULL predicted vector back.
        windows = np.concatenate([windows[:, 1:, :], preds[:, np.newaxis, :]], axis=1)

    # Back to original units for reporting (target column only).
    preds_orig   = _inverse_target(all_preds_s.ravel(), scaler, target_col_idx, n_feat)
    actuals_orig = _inverse_target(actuals_s.ravel(), scaler, target_col_idx, n_feat)
    return preds_orig, actuals_orig


# ── direct (multi_step) evaluation ─────────────────────────────────────────

def evaluate_direct(model, X_test, lookback, horizon, target_col_idx, scaler, n_feat):
    """Score a multi_step model: one forward pass per window, no rollout.

    Returns the SAME contract as `evaluate()` — (preds, actuals) as flat arrays
    in original units, laid out window-by-window so `.reshape(n_windows,
    horizon)` recovers them — so saving, plotting and the dashboard need no
    changes to read a multi_step run.

    There is no feedback loop here, so `DivergenceError` is structurally
    impossible: the model never consumes its own output.
    """
    from .windowing import make_direct_test_windows

    X_win, y_win, _ = make_direct_test_windows(X_test, target_col_idx, lookback, horizon)
    preds_s = np.asarray(model.predict_batch(X_win), dtype=np.float32)   # (n_windows, horizon)

    preds_orig   = _inverse_target(preds_s.ravel(), scaler, target_col_idx, n_feat)
    actuals_orig = _inverse_target(y_win.ravel(), scaler, target_col_idx, n_feat)
    return preds_orig, actuals_orig


# ── metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Standard regression metrics as a dict (MAE, RMSE, MAPE%, R2)."""
    return {
        'MAE':     mean_absolute_error(y_true, y_pred),
        'RMSE':    np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE(%)': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'R2':      r2_score(y_true, y_pred),
    }

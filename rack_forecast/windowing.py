"""Sliding-window construction for supervised sequence learning."""

import numpy as np


def make_supervised(X, y, lookback):
    """Turn flat (n, n_feat) arrays into overlapping windows.

    Args:
        X: scaled feature matrix, shape (n, n_feat).
        y: scaled target matrix, shape (n, n_feat) — the full next-step feature
           vector (multi-output: the model forecasts every column, not just
           the target, so the rollout never has to freeze anything).
        lookback: number of past steps per window.

    Returns:
        X_3d: shape (n - lookback, lookback, n_feat).
        y_2d: shape (n - lookback, n_feat) — the feature vector AFTER each window.
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── direct (multi_step) windowing ──────────────────────────────────────────
# The single_step pair above trains one-step-ahead and rolls out at evaluation.
# These two build the direct alternative: predict all `horizon` steps of the
# TARGET at once, so there is no rollout and nothing to feed back.

def make_direct_supervised(X, target_idx, lookback, horizon, stride=1):
    """Training windows for the multi_step strategy.

    X = the past `lookback` steps of ALL features; y = the next `horizon` steps
    of the TARGET column only (nothing is fed back, so the other columns never
    need forecasting).

    Args:
        X: scaled feature matrix, shape (n, n_feat).
        target_idx: column index of the target within X.
        lookback / horizon: window sizes in steps.
        stride: step between consecutive training windows. 1 = every overlapping
            window; larger strides trade training examples for memory, which
            matters because y alone costs n_windows*horizon float32s.

    Returns:
        X_3d: (n_windows, lookback, n_feat)
        y_2d: (n_windows, horizon) — target-only future
    """
    starts = range(lookback, len(X) - horizon + 1, stride)
    Xs = np.stack([X[i - lookback:i] for i in starts], axis=0).astype(np.float32)
    ys = np.stack([X[i:i + horizon, target_idx] for i in starts], axis=0).astype(np.float32)
    return Xs, ys


def make_direct_test_windows(X, target_idx, lookback, horizon):
    """Non-overlapping test windows for the multi_step strategy.

    Deliberately mirrors `evaluate.evaluate()`'s window definition exactly —
    same count, same offsets — so single_step and multi_step runs of the same
    config are scored on identical windows and their metrics are comparable.

    Returns:
        X_3d: (n_windows, lookback, n_feat) seed windows
        y_2d: (n_windows, horizon) target-only ground truth
        n_windows: int
    """
    n_windows = (len(X) - lookback) // horizon
    Xs = np.stack([X[w * horizon: w * horizon + lookback]
                   for w in range(n_windows)], axis=0).astype(np.float32)
    ys = np.stack([X[lookback + w * horizon: lookback + w * horizon + horizon, target_idx]
                   for w in range(n_windows)], axis=0).astype(np.float32)
    return Xs, ys, n_windows

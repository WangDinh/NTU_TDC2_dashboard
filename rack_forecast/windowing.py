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

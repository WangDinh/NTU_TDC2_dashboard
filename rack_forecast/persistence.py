"""Reading and writing run artifacts under results/<run_name>/.

Layout produced per run:
    <run>/config.json                 run config + derived info
    <run>/metrics.csv                 one row per model
    <run>/scalers.pkl                 {'scaler': ...} single StandardScaler (all columns)
    <run>/*.png                       run-level comparison figures
    <run>/<model>/predictions.npz     preds/actuals (n_windows, horizon) + timestamps
    <run>/<model>/*.png               per-model figures
    <run>/<model>/models/<model>.pt|.pkl   trained weights

torch and the model classes are imported lazily inside the functions that need
them, so lightweight readers (list_runs, load_predictions, the dashboard's
data pages) never pull in PyTorch.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .paths import RESULTS_ROOT

# Names that were saved as PyTorch state dicts (.pt); the rest are sklearn (.pkl).
DL_MODELS = {'lstm', 'cnn1d', 'transformer'}


# ── trained models ─────────────────────────────────────────────────────────

def save_model(name, model, out_dir):
    """Save a model into <out_dir>/models/ (.pt for DL, .pkl for sklearn/XGB)."""
    models_dir = Path(out_dir) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    if name in DL_MODELS:
        import torch
        torch.save(model.state_dict(), models_dir / f'{name}.pt')
    else:
        joblib.dump(model.model, models_dir / f'{name}.pkl')


def load_model(name, out_dir, n_feat=None, lookback=None):
    """Reconstruct a saved model. DL models need n_feat (and lookback for cnn1d)."""
    models_dir = Path(out_dir) / 'models'
    if name in DL_MODELS:
        import torch
        from .models.lstm import LSTMModel
        from .models.cnn1d import CNN1DModel
        from .models.transformer import TransformerModel
        arch = {'lstm': LSTMModel(n_feat),
                'cnn1d': CNN1DModel(n_feat, lookback),
                'transformer': TransformerModel(n_feat)}[name]
        arch.load_state_dict(torch.load(models_dir / f'{name}.pt', map_location='cpu'))
        arch.eval()
        return arch
    from .models import REGISTRY
    m = REGISTRY[name]()
    m.model = joblib.load(models_dir / f'{name}.pkl')
    return m


# ── per-window predictions ─────────────────────────────────────────────────

def save_predictions(run_folder, name, preds, actuals, window_times, horizon, window_racks=None):
    """Persist a model's full rollout as <run>/<model>/predictions.npz.

    preds/actuals are the flat arrays from `evaluate()`; they are reshaped to
    (n_windows, horizon). window_times holds one start timestamp per window.
    window_racks (optional) holds one rack name per window — used by
    shared-model runs (see pipeline.run_shared_experiment) where windows are
    pooled from several racks; omitted for ordinary single-rack runs.
    """
    model_dir = Path(run_folder) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    n_windows = len(preds) // horizon
    extra = {}
    if window_racks is not None:
        extra['window_racks'] = np.asarray(window_racks)
    np.savez(
        model_dir / 'predictions.npz',
        preds=np.asarray(preds).reshape(n_windows, horizon),
        actuals=np.asarray(actuals).reshape(n_windows, horizon),
        # store datetime64 values directly — round-trips cleanly regardless of the
        # pandas datetime unit (µs vs ns), unlike an int64 epoch representation.
        timestamps=pd.DatetimeIndex(window_times).values,
        **extra,
    )


def load_predictions(run_folder, name):
    """Load a model's predictions.npz → dict of preds/actuals/timestamps
    (plus window_racks if this was a shared-model run)."""
    d = np.load(Path(run_folder) / name / 'predictions.npz')
    out = {
        'preds': d['preds'],                                # (n_windows, horizon)
        'actuals': d['actuals'],                            # (n_windows, horizon)
        'timestamps': pd.to_datetime(d['timestamps']),      # (n_windows,)
    }
    if 'window_racks' in d:
        out['window_racks'] = d['window_racks']             # (n_windows,)
    return out


# ── scalers (needed to inverse-transform during live inference) ────────────

def save_scalers(run_folder, scaler):
    """Persist the fitted StandardScaler (fit across all columns) as <run>/scalers.pkl."""
    joblib.dump({'scaler': scaler}, Path(run_folder) / 'scalers.pkl')


def load_scalers(run_folder):
    """Return the single StandardScaler from <run>/scalers.pkl.

    Raises KeyError for runs saved by the old feat/target-pair format
    (pre multi-output) — callers should catch this and show a
    re-run-the-experiment warning instead of crashing.
    """
    d = joblib.load(Path(run_folder) / 'scalers.pkl')
    return d['scaler']


# ── config & metrics ───────────────────────────────────────────────────────

def save_config(cfg, run_folder, extra=None):
    """Write <run>/config.json from an ExperimentConfig plus optional `extra`."""
    data = cfg.to_dict()
    if extra:
        data.update(extra)
    data['timestamp'] = datetime.now().isoformat()
    with open(Path(run_folder) / 'config.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_config(run_folder):
    """Read <run>/config.json into a dict."""
    with open(Path(run_folder) / 'config.json') as f:
        return json.load(f)


def save_metrics(metrics_df, run_folder):
    """Write the per-model metrics table to <run>/metrics.csv."""
    metrics_df.to_csv(Path(run_folder) / 'metrics.csv')


def load_metrics(run_folder):
    """Read <run>/metrics.csv (model name as index)."""
    return pd.read_csv(Path(run_folder) / 'metrics.csv', index_col=0)


# ── run discovery (dashboard) ──────────────────────────────────────────────

def list_runs():
    """List every run folder that has a config.json, newest-looking last.

    Returns a list of dicts: {'run_name', 'path', **config}.
    """
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for d in sorted(RESULTS_ROOT.iterdir()):
        cfg_file = d / 'config.json'
        if d.is_dir() and cfg_file.exists():
            try:
                cfg = load_config(d)
            except Exception:
                cfg = {}
            runs.append({'run_name': d.name, 'path': d, **cfg})
    return runs


def available_models(run_folder):
    """Model names in a run that have a saved predictions.npz."""
    run_folder = Path(run_folder)
    return sorted(d.name for d in run_folder.iterdir()
                  if d.is_dir() and (d / 'predictions.npz').exists())

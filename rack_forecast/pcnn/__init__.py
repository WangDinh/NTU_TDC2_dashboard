"""Adapt-PCNN: physics-consistent return-air TEMPERATURE forecasting.

A self-contained subpackage, deliberately kept separate from the rack-power
forecasting pipeline (rack_forecast.pipeline etc.). It shares only the pure
raw-loading helpers in rack_forecast.data; it never imports the power pipeline's
config / trainer / evaluate / windowing / models, and writes to results/pcnn/.

Public API:
    PCNNConfig            run configuration
    build_pcnn_dataset    6-signal feature matrix from the TDC2.0 tree
    fit_normalizer        min-max [0,1] scaler (fit on train)
    month_split           chronological train/test split
    Adapt_PC_MLP          the model
    train_pcnn            training loop → best-val model
    rollout_evaluate      warm-start autoregressive rollout + metrics (°C)
    compute_metrics       MAE/RMSE/MAPE/R2 (local copy)
"""

from .config import PCNNConfig, PCNN_RESULTS_ROOT
from .data import build_pcnn_dataset, fit_normalizer, month_split, Normalizer, FEATURE_ORDER
from .module import Adapt_PC_MLP, DEVICE
from .train import train_pcnn, make_windows, rollout_window
from .evaluate import rollout_evaluate, compute_metrics, predict_window_trace
from .baselines import run_baselines, BASELINE_MODELS, make_baseline_windows

__all__ = [
    'PCNNConfig', 'PCNN_RESULTS_ROOT',
    'build_pcnn_dataset', 'fit_normalizer', 'month_split', 'Normalizer', 'FEATURE_ORDER',
    'Adapt_PC_MLP', 'DEVICE',
    'train_pcnn', 'make_windows', 'rollout_window',
    'rollout_evaluate', 'compute_metrics', 'predict_window_trace',
    'run_baselines', 'BASELINE_MODELS', 'make_baseline_windows',
]

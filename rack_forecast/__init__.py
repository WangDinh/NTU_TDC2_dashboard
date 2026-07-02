"""rack_forecast — TDC 2.0 rack power forecasting library.

Public entry points:
    from rack_forecast import ExperimentConfig
    from rack_forecast.pipeline import run_experiment

Submodules are imported lazily (import what you need) to keep light-weight
consumers — e.g. the dashboard's raw-data page — from pulling in torch.
"""

from .config import ExperimentConfig, DEFAULT_MODELS
from .paths import PROJECT_ROOT, DATA_ROOT, RESULTS_ROOT

__all__ = [
    'ExperimentConfig',
    'DEFAULT_MODELS',
    'PROJECT_ROOT',
    'DATA_ROOT',
    'RESULTS_ROOT',
]

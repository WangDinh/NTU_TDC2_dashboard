"""Data assembly + normalization for the Adapt-PCNN temperature pipeline.

Builds the fixed 6-signal feature matrix [SA_T, SA_V, OA_T, OA_H, ITE_P, RA_T]
from the TDC2.0 tree, using ONLY the pure raw-loading helpers from
`rack_forecast.data` (load_gw / load_agg_pm / _resampled). Nothing here depends
on the power pipeline.

Normalization is min-max to [0,1] (NOT StandardScaler): the physics `/100`
scalings in module.Adapt_PC_MLP assume [0,1]-ranged inputs, exactly as in the
original Adapt-PCNN, whose committed CSV was pre-normalized that way.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import load_gw, load_agg_pm, _resampled

# Fixed output column order — must match PCNNConfig.feature_names / index config.
FEATURE_ORDER = ['SA_T', 'SA_V', 'OA_T', 'OA_H', 'ITE_P', 'RA_T']


def build_pcnn_dataset(cfg) -> pd.DataFrame:
    """Assemble the 6 PCNN signals as a time-indexed DataFrame (raw units).

    Light by construction — loads only 4 GW streams + the target rack's kW, so
    it never triggers the power pipeline's 182-feature full-mode memory blow-up.
    Short gaps forward-filled (<=4 steps); remaining NaN rows dropped.
    """
    r = cfg.resample
    cols = {}

    # GW-1: supply air temperature + velocity.
    cols['SA_T'] = _resampled(load_gw(1, 'SAT1', ['timestamp', 'temp_C']), 'temp_C', r)
    cols['SA_V'] = _resampled(load_gw(1, 'SAAV1', ['timestamp', 'velocity_ms']), 'velocity_ms', r)

    # GW-2: outside air temp + humidity (one OATH read).
    oath = load_gw(2, 'OATH1', ['timestamp', 'temp_C', 'humidity_pct'])
    cols['OA_T'] = _resampled(oath, 'temp_C', r)
    cols['OA_H'] = _resampled(oath, 'humidity_pct', r)

    # IT-equipment power: the target rack's aggregate PDU kW (load proxy).
    cols['ITE_P'] = _resampled(load_agg_pm(cfg.target_rack), 'kW', r)

    # GW-2: return air temperature — the TARGET.
    cols['RA_T'] = _resampled(load_gw(2, 'RATH1', ['timestamp', 'temp_C', 'humidity_pct']), 'temp_C', r)

    data = pd.DataFrame(cols)[FEATURE_ORDER].sort_index().ffill(limit=4).dropna()
    return data


# ── min-max normalization (fit on train only) ──────────────────────────────

@dataclass
class Normalizer:
    """Per-column min-max scaler to [0,1], fit on training rows only.
    Build via `fit_normalizer(train_df)`."""
    min_: pd.Series
    max_: pd.Series

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return ((df - self.min_) / (self.max_ - self.min_)).to_numpy(dtype=np.float32)

    def inverse_target(self, values, target_name: str = 'RA_T') -> np.ndarray:
        """Scale a target-only array back to physical units (°C for RA_T)."""
        lo, hi = self.min_[target_name], self.max_[target_name]
        return np.asarray(values, dtype=np.float32) * (hi - lo) + lo


def fit_normalizer(train_df: pd.DataFrame) -> Normalizer:
    """Fit a min-max Normalizer on the training frame (exact min/max per column)."""
    mn = train_df.min()
    mx = train_df.max()
    # Constant column guard: make max strictly > min so the divide is safe.
    mx = mx.where(mx - mn > 1e-12, mn + 1.0)
    return Normalizer(min_=mn, max_=mx)


# ── chronological split (train = all but last month, test = last month) ─────

def month_split(data: pd.DataFrame):
    """Split by calendar month: last month = test, the rest = train.

    Mirrors the power pipeline's convention (rack_forecast.pipeline.prepare_data)
    for consistency, rather than the original PCNN percentage split.
    """
    months = sorted(data.index.to_period('M').unique())
    train_df = data[data.index.to_period('M').isin(months[:-1])]
    test_df  = data[data.index.to_period('M').isin(months[-1:])]
    return train_df, test_df, months

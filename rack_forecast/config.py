"""Experiment configuration.

Every tunable knob lives in one `ExperimentConfig` dataclass. Notebooks, the CLI
script, and the dashboard all build one of these and pass it around, so there is
a single, self-documenting definition of "what a run is".
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .paths import RESULTS_ROOT

# Models available in the registry (rack_forecast/models/__init__.py).
# `svr` is intentionally excluded — too slow on CPU at this data scale.
DEFAULT_MODELS = ['linear', 'rf', 'xgboost', 'lstm', 'cnn1d', 'transformer']

# Forecasting strategies (see ExperimentConfig.strategy).
STRATEGIES = ('single_step', 'multi_step')


@dataclass
class ExperimentConfig:
    """One prediction run. 1 step = the RESAMPLE interval (default 30 s)."""

    # ── what to predict ──────────────────────────────────────────────────
    target_rack: str = 'R0605-PA'      # rack whose aggregate kW is the target
    resample:    str = '30s'           # resampling grid for all signals

    # ── windowing ────────────────────────────────────────────────────────
    lookback: int = 20                 # past steps fed as context
    horizon:  int = 60                 # steps predicted per window (60 = 30 min)

    # ── forecasting strategy ─────────────────────────────────────────────
    # 'single_step': train to predict the next step's FULL feature vector, then
    #   roll out `horizon` times feeding each prediction back in. Errors compound
    #   (observed: divergence to inf at horizon=20160).
    # 'multi_step':  train to predict all `horizon` steps of the TARGET in one
    #   forward pass. No feedback loop, so nothing can compound or diverge.
    strategy: str = 'single_step'

    # multi_step only: stride between overlapping TRAINING windows. The direct
    # target is (n_windows, horizon), so stride=1 costs n_rows*horizon float32s
    # on the full train set: ~53MB at horizon=30, ~211MB at 120, ~5.0GB at 2880,
    # ~34GB at 20160. Larger strides trade training examples for memory;
    # stride=horizon makes windows non-overlapping (~n_rows floats, flat in
    # horizon). None → auto (see resolved_direct_stride).
    direct_stride: int | None = None

    # ── models & training ────────────────────────────────────────────────
    models:    list = field(default_factory=lambda: list(DEFAULT_MODELS))
    dl_epochs: int  = 30               # max epochs for the DL models

    # ── data scope (smaller = faster iteration / demo) ───────────────────
    fast_mode:    bool = True          # True → load target rack only (no GW sensors)
    train_days:   int | None = None    # None = full 5 training months; N = first N days
    predict_days: int | None = None    # None = full test month;        N = first N days

    # ── bookkeeping ──────────────────────────────────────────────────────
    run_id: str = 'run_001'            # names the output folder

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f'strategy must be one of {STRATEGIES}, got {self.strategy!r}')

    # ── derived values (no separate storage needed) ──────────────────────
    @property
    def is_multi_step(self) -> bool:
        return self.strategy == 'multi_step'

    def resolved_direct_stride(self, n_train_rows: int,
                               budget_bytes: int = 2 * 1024 ** 3) -> int:
        """Training-window stride actually used by the multi_step path.

        An explicit `direct_stride` is honoured as-is. Otherwise pick the
        smallest stride whose direct target array fits in `budget_bytes`:
        stride=1 keeps every overlapping window (best for short horizons),
        and long horizons fall back toward stride=horizon automatically
        rather than trying to allocate terabytes.
        """
        if self.direct_stride is not None:
            return max(1, self.direct_stride)
        n_windows = max(1, n_train_rows - self.lookback - self.horizon + 1)
        bytes_per_window = self.horizon * 4          # float32 target row
        stride = int(np.ceil(n_windows * bytes_per_window / budget_bytes))
        return max(1, min(stride, self.horizon))

    @property
    def target_col(self) -> str:
        """Column name of the target in the merged dataframe."""
        return f'{self.target_rack}__kW'

    @property
    def rack_tag(self) -> str:
        """Folder tag: the rack name in fast mode, else 'all_racks'."""
        return self.target_rack if self.fast_mode else 'all_racks'

    @property
    def run_name(self) -> str:
        """e.g. 'run_001_R0605-PA_L20_H60' (single_step) or the same name with a
        '_multistep' suffix. single_step names are left unchanged so existing
        result folders keep resolving."""
        base = f'{self.run_id}_{self.rack_tag}_L{self.lookback}_H{self.horizon}'
        return f'{base}_multistep' if self.is_multi_step else base

    @property
    def run_folder(self) -> Path:
        """Absolute path of this run's output folder under results/."""
        return RESULTS_ROOT / self.run_name

    def to_dict(self) -> dict:
        """Plain dict for JSON serialisation (config.json)."""
        return asdict(self)

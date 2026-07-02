"""Experiment configuration.

Every tunable knob lives in one `ExperimentConfig` dataclass. Notebooks, the CLI
script, and the dashboard all build one of these and pass it around, so there is
a single, self-documenting definition of "what a run is".
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

from .paths import RESULTS_ROOT

# Models available in the registry (rack_forecast/models/__init__.py).
# `svr` is intentionally excluded — too slow on CPU at this data scale.
DEFAULT_MODELS = ['linear', 'rf', 'xgboost', 'lstm', 'cnn1d', 'transformer']


@dataclass
class ExperimentConfig:
    """One prediction run. 1 step = the RESAMPLE interval (default 30 s)."""

    # ── what to predict ──────────────────────────────────────────────────
    target_rack: str = 'R0605-PA'      # rack whose aggregate kW is the target
    resample:    str = '30s'           # resampling grid for all signals

    # ── windowing ────────────────────────────────────────────────────────
    lookback: int = 20                 # past steps fed as context
    horizon:  int = 60                 # steps predicted per window (60 = 30 min)

    # ── models & training ────────────────────────────────────────────────
    models:    list = field(default_factory=lambda: list(DEFAULT_MODELS))
    dl_epochs: int  = 30               # max epochs for the DL models

    # ── data scope (smaller = faster iteration / demo) ───────────────────
    fast_mode:    bool = True          # True → load target rack only (no GW sensors)
    train_days:   int | None = None    # None = full 5 training months; N = first N days
    predict_days: int | None = None    # None = full test month;        N = first N days

    # ── bookkeeping ──────────────────────────────────────────────────────
    run_id: str = 'run_001'            # names the output folder

    # ── derived values (no separate storage needed) ──────────────────────
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
        """e.g. 'run_001_R0605-PA_L20_H60'."""
        return f'{self.run_id}_{self.rack_tag}_L{self.lookback}_H{self.horizon}'

    @property
    def run_folder(self) -> Path:
        """Absolute path of this run's output folder under results/."""
        return RESULTS_ROOT / self.run_name

    def to_dict(self) -> dict:
        """Plain dict for JSON serialisation (config.json)."""
        return asdict(self)

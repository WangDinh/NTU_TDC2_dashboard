"""Configuration for the Adapt-PCNN temperature-forecast experiments.

Deliberately SEPARATE from `rack_forecast.config.ExperimentConfig` (the power
pipeline). This subpackage forecasts return-air *temperature* with a
physics-consistent MLP and shares nothing with the power code except pure
raw-loading helpers in `rack_forecast.data`.

1 step = the resample interval (default 1 min, matching the Adapt-PCNN MLP setup).
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..paths import RESULTS_ROOT

# PCNN runs live under results/pcnn/ so they never show up in the power
# dashboard's list_runs() (which scans results/*/config.json directly).
PCNN_RESULTS_ROOT = RESULTS_ROOT / 'pcnn'


@dataclass
class PCNNConfig:
    """One Adapt-PCNN run. Feature order is fixed at
    [SA_T, SA_V, OA_T, OA_H, ITE_P, RA_T]; RA_T (index -1) is the target."""

    # ── what to model ────────────────────────────────────────────────────
    target_rack: str = 'R0605-PA'      # rack whose PDU kW is the IT-load driver
    resample:    str = '1min'          # resampling grid (1 step = this interval)

    # ── rollout horizon ──────────────────────────────────────────────────
    warm_start: int = 1                # real RA_T steps fed in to seed last_D
    horizon:    int = 15               # steps predicted per window (15 = 15 min @1min)

    # ── training-window construction ─────────────────────────────────────
    overlapping_distance: int = 60     # stride between consecutive train windows
    validation_percentage: float = 0.2 # tail fraction of train windows held out

    # ── MLP (the `D` term) ───────────────────────────────────────────────
    mlp_hidden_size: int = 128
    mlp_num_layers:  int = 3
    activation:      int = 1           # 0=ReLU, 1=Sigmoid, 2=Tanh (paper MLP=1)
    division_factor: float = 10.0      # scales D so the physics term dominates

    # ── optimisation ─────────────────────────────────────────────────────
    n_epochs:   int = 100
    batch_size: int = 256
    lr:         float = 0.005
    seed:       int = 0
    verbose:    int = 1

    # ── bookkeeping ──────────────────────────────────────────────────────
    run_id: str = 'pcnn_001'

    # Fixed feature layout — see module.Adapt_PC_MLP. Indices into the
    # [SA_T, SA_V, OA_T, OA_H, ITE_P, RA_T] column order.
    feature_names: list = field(
        default_factory=lambda: ['SA_T', 'SA_V', 'OA_T', 'OA_H', 'ITE_P', 'RA_T'])
    supply_T_col:  int = 0             # SA_T
    supply_m_col:  int = 1             # SA_V
    power_col:     int = 4             # ITE_P
    temperature_col: int = 5           # RA_T (the target, also fed back)
    inputs_D:      list = field(default_factory=lambda: [2, 3, 5])  # OA_T, OA_H, RA_T

    @property
    def window_len(self) -> int:
        """Total length of one train/eval window."""
        return self.warm_start + self.horizon

    @property
    def run_name(self) -> str:
        """e.g. 'pcnn_001_R0605-PA_H15'."""
        return f'{self.run_id}_{self.target_rack}_H{self.horizon}'

    @property
    def run_folder(self) -> Path:
        return PCNN_RESULTS_ROOT / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)

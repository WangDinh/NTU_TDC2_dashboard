"""Dataset loading and assembly.

This is the SINGLE source of truth for turning the raw TDC 2.0 CSV tree into a
clean, resampled, feature-by-column dataframe. The notebook and the CLI script
both call `build_dataset(cfg)` — no copy-pasted loading logic anywhere else.

All raw files are tab-separated, header-less, ~30 s sampling. Column layouts
come from `data/TDC2.0 Dataset/data_csv_file_structures.pdf`.
"""

import glob
import pandas as pd

from .paths import DATA_ROOT

# ── column layouts for the two most common file types ──────────────────────
PM_COLS = ['timestamp', 'kW', 'V', 'A', 'PF', 'Hz']       # PDU aggregate power
TH_COLS = ['timestamp', 'temp_C', 'humidity_pct']          # temp/humidity sensor


# ── low-level loaders ──────────────────────────────────────────────────────

def load_files(pattern, cols):
    """Read every CSV matching `pattern` (relative to DATA_ROOT) and concat them.

    Returns an empty DataFrame if nothing matches or all reads fail.
    """
    files = sorted(glob.glob(str(DATA_ROOT / pattern)))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t', header=None, names=cols, parse_dates=['timestamp'])
            frames.append(df)
        except Exception:
            # Skip malformed / partial files rather than aborting the whole load.
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_agg_pm(rack):
    """Aggregate PDU power for a rack, e.g. 'R0605-PA' → all PM-<rack>-*.csv."""
    return load_files(f'{rack}/*/PM-{rack}-*.csv', PM_COLS)


def load_th(rack, prefix):
    """Temperature/humidity sensor `prefix` (e.g. 'THFM') under a rack folder."""
    return load_files(f'{rack}/*/{prefix}-*.csv', TH_COLS)


def load_gw(gw_num, sensor, cols):
    """SensorGW reading, e.g. load_gw(1, 'SAT1', ['timestamp','temp_C']).

    Treats '-' as NaN (the GW files use it for missing samples).
    """
    pat = str(DATA_ROOT / f'SensorGW-{gw_num}' / '*' / f'{sensor}-*.csv')
    files = sorted(glob.glob(pat))
    frames = [pd.read_csv(f, sep='\t', header=None, names=cols,
                          parse_dates=['timestamp'], na_values=['-']) for f in files]
    return pd.concat(frames, ignore_index=True).sort_values('timestamp') if frames else pd.DataFrame()


# ── helper: resample one series onto the config grid ───────────────────────

def _resampled(df, col, resample):
    """Set timestamp index, resample `col` to the grid, return the mean series."""
    df = df.sort_values('timestamp').set_index('timestamp')
    return df[col].resample(resample).mean()


# ── high-level assembly ────────────────────────────────────────────────────

def build_dataset(cfg):
    """Assemble the full feature matrix for a run as a time-indexed DataFrame.

    Fast mode: target rack PM (kW/V/A/PF/Hz) + its 6 TH sensors only.
    Full mode: adds every other rack's PM plus all SensorGW-1..5 signals.

    Columns are named '<source>__<field>' (e.g. 'R0605-PA__kW', 'THFM__temp_C').
    Short gaps are forward-filled (<=4 steps); remaining NaN rows are dropped.
    """
    frames = {}

    # ── which racks contribute PM signals ──────────────────────────────
    if cfg.fast_mode:
        racks = [cfg.target_rack]
    else:
        racks = sorted(d.name for d in DATA_ROOT.iterdir()
                       if d.is_dir() and d.name.startswith('R'))

    for rack in racks:
        df = load_agg_pm(rack)
        for col in ['kW', 'V', 'A', 'PF', 'Hz']:
            frames[f'{rack}__{col}'] = _resampled(df, col, cfg.resample)

    # ── target rack TH sensors (front via PA, back via PB) ─────────────
    # Derived from the base rack name, not assumed from target_rack's own
    # suffix — target_rack may itself be either the PA or the PB side.
    base_rack = cfg.target_rack.rsplit('-', 1)[0]
    pa_rack, pb_rack = f'{base_rack}-PA', f'{base_rack}-PB'
    for s in ['THFT', 'THFM', 'THFB', 'THBB']:
        df = load_th(pa_rack, s)
        if not df.empty:
            frames[f'{s}__temp_C']       = _resampled(df, 'temp_C', cfg.resample)
            frames[f'{s}__humidity_pct'] = _resampled(df, 'humidity_pct', cfg.resample)
    for s in ['THBM', 'THBT']:
        df = load_th(pb_rack, s)
        if not df.empty:
            frames[f'{s}__temp_C']       = _resampled(df, 'temp_C', cfg.resample)
            frames[f'{s}__humidity_pct'] = _resampled(df, 'humidity_pct', cfg.resample)

    # ── full mode only: SensorGW-1..5 (room / cooling-loop / PCU) ──────
    if not cfg.fast_mode:
        _add_sensorgw(frames, cfg.resample)

    # ── merge, fill short gaps, drop remaining NaNs ────────────────────
    data = pd.DataFrame(frames).sort_index().ffill(limit=4).dropna()
    return data


def _add_sensorgw(frames, resample):
    """Load all SensorGW signals into `frames` (full-mode features)."""
    # GW-1: supply air velocity + temperature
    for s in ['SAAV1', 'SAAV2']:
        df = load_gw(1, s, ['timestamp', 'velocity_ms'])
        if not df.empty:
            frames[f'{s}__velocity_ms'] = _resampled(df, 'velocity_ms', resample)
    df = load_gw(1, 'SAT1', ['timestamp', 'temp_C'])
    if not df.empty:
        frames['SAT1__temp_C'] = _resampled(df, 'temp_C', resample)

    # GW-2: room temp/humidity + differential pressure
    for s in ['OATH1', 'OATH2', 'RATH1', 'RATH2', 'SATH1', 'SATH2']:
        df = load_gw(2, s, ['timestamp', 'temp_C', 'humidity_pct'])
        if not df.empty:
            frames[f'{s}__temp_C']       = _resampled(df, 'temp_C', resample)
            frames[f'{s}__humidity_pct'] = _resampled(df, 'humidity_pct', resample)
    for s in ['DPS1', 'DPS2', 'DPS3']:
        df = load_gw(2, s, ['timestamp', 'pressure'])
        if not df.empty:
            frames[f'{s}__pressure'] = _resampled(df, 'pressure', resample)

    # GW-3 & GW-4: cooling-loop temperatures and pressures
    for gw in [3, 4]:
        for s in ['T-DL-1', 'T-DL-2', 'T-EL-1', 'T-EL-2', 'T-LL-1', 'T-SL-1', 'T-SL-2']:
            df = load_gw(gw, s, ['timestamp', 'temp_C'])
            if not df.empty:
                frames[f'GW{gw}_{s}__temp_C'] = _resampled(df, 'temp_C', resample)
        for s in ['PS-DL-1', 'PS-DL-2', 'PS-EL-1', 'PS-LL-1', 'PS-SL-1']:
            df = load_gw(gw, s, ['timestamp', 'pressure'])
            if not df.empty:
                frames[f'GW{gw}_{s}__pressure'] = _resampled(df, 'pressure', resample)

    # GW-5: PCU power (single-phase CF; three-phase CM/CP/EF)
    sp_cols = ['timestamp', 'W', 'V', 'A', 'PF', 'kWh']
    tp_cols = ['timestamp', 'W', 'V_L1', 'V_L2', 'V_L3', 'kWh']
    for s in ['PCU-4-P-CF', 'PCU-5-P-CF']:
        df = load_gw(5, s, sp_cols)
        if not df.empty:
            frames[f'{s}__W'] = _resampled(df, 'W', resample)
    for s in ['PCU-4-P-CM', 'PCU-4-P-CP1', 'PCU-4-P-CP2',
              'PCU-5-P-CM', 'PCU-5-P-CP1', 'PCU-5-P-CP2', 'PCU-4-P-EF', 'PCU-5-P-EF']:
        df = load_gw(5, s, tp_cols)
        if not df.empty:
            frames[f'{s}__W'] = _resampled(df, 'W', resample)

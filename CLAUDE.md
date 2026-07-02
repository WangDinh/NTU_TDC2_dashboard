# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python work uses the **`ntu_cooling`** conda environment. The core code is an
editable-installed package (`pip install -e .`), so `import rack_forecast` works
from any directory. First-time env setup (fresh clone) is in `README.md` —
`pip install -r requirements.txt` for pinned deps, `torch` installed separately
from its CUDA-matched index.

```bash
conda run -n ntu_cooling python scripts/run_prediction.py
conda run -n ntu_cooling streamlit run dashboard/app.py
conda run -n ntu_cooling jupyter nbconvert --to notebook --execute --inplace notebooks/prediction.ipynb \
  --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=ntu_cooling
```

**Important:** `conda run -n ntu_cooling python -c "..."` does **not** support multiline `-c` scripts. Always write code to a `.py` file and run it via `conda run`. After changing package structure, re-run `pip install -e .`.

## Dataset

**Location:** `data/TDC2.0 Dataset/`  
**Documentation:** `data/TDC2.0 Dataset/data_csv_file_structures.pdf`

All files: tab-separated, no header row, ~30s sampling interval.

### File structure hierarchy

```
TDC 2.0 Dataset/
│
├── Rack PDU data  (24 folders: R0501–R0506, R0601–R0606, each × PA/PB)
│   └── <RACK>-PA or PB/
│       └── YYYY-MM/  (6 months: 2022-09 → 2023-02)
│           ├── PM-<RACK>-YYYY-MM-DD.csv       PDU aggregate power  (kW, V, A, PF, kWh)
│           ├── PM01..PM24-YYYY-MM-DD.csv      Individual outlets   (kW, V, A, PF, kWh)
│           └── TH sensors                     (°C, humidity %)
│               PA: THFT, THFM, THFB, THBB    front-top/mid/bot, back-bot
│               PB: THBM, THBT                 back-mid, back-top
│
├── SensorGW-1/   Supply air
│   ├── SAAV1, SAAV2    Air velocity (m/s)
│   └── SAT1            Supply air temperature (°C)
│
├── SensorGW-2/   Room-level environment
│   ├── OATH1/2         Outside air temp+humidity
│   ├── RATH1/2         Return air temp+humidity
│   ├── SATH1/2         Supply air temp+humidity
│   └── DPS1/2/3        Differential pressure
│
├── SensorGW-3/   Cooling loop #1 (liquid side)
│   ├── T-DL/EL/LL/SL-*    Loop temperatures (°C)
│   └── PS-DL/EL/LL/SL-*   Loop pressures (PSI)
│
├── SensorGW-4/   Cooling loop #2 (liquid side, same sensor set as GW-3)
│   ├── T-DL/EL/LL/SL-*    Loop temperatures (°C)
│   └── PS-DL/EL/LL/SL-*   Loop pressures (PSI)
│
└── SensorGW-5/   Cooling unit power (PCU-4 and PCU-5)
    ├── PCU-x-P-CF          Cooling fan    single-phase (W, V, A, PF, kWh)
    ├── PCU-x-P-CM          Compressor     three-phase  (W, V_L1, V_L2, V_L3, kWh)
    ├── PCU-x-P-CP1/2       Coolant pumps  three-phase  (W, V_L1, V_L2, V_L3, kWh)
    └── PCU-x-P-EF          Exhaust fan    three-phase  (W, V_L1, V_L2, V_L3, kWh)
```

### Key numbers
- ~521k records per rack per 6-month period (aggregate PM)
- 24 outlet sockets per PDU → 48 monitored outlets per physical rack
- 5 TH sensors per physical rack (3 via PA, 2 via PB)
- PA = front-side PDU, PB = back-side PDU of the same physical rack

### Physical cooling chain
```
PCU-4/5 (compressor + pumps + fans)  [GW-5]
        ↓ chilled air
    SAT1 — supply air temperature     [GW-1]
    SAAV1/2 — supply air velocity     [GW-1]
        ↓
  Server rack front intake
    THFT/THFM/THFB                    [rack PA]
        ↓ heated exhaust out back
    THBB/THBM/THBT                    [rack PB]
        ↓ return air / chilled water loop
    GW-2 (RATH, DPS), GW-3/4 (T/PS loop sensors)
```

## Project layout

```
rack_forecast/     core library (editable-installed package)
  paths.py         CWD-independent DATA_ROOT / RESULTS_ROOT (resolved from __file__)
  config.py        ExperimentConfig dataclass — every knob; .run_folder / .target_col
  data.py          load_agg_pm/load_th/load_gw + build_dataset(cfg)  ← single loader source
  windowing.py     make_supervised()
  trainer.py       DEVICE, train_dl(), build_and_train()
  evaluate.py      evaluate() (batched rollout), autoregressive_predict(), compute_metrics()
  persistence.py   save/load model, predictions(.npz), scalers(.pkl), config, metrics; list_runs()
  plots.py         figure-returning helpers (notebook displays, pipeline saves)
  pipeline.py      prepare_data() (split+scale) + run_experiment() + save_results()
  models/          linear rf xgboost lstm cnn1d transformer  (svr.py kept, not in REGISTRY)
notebooks/         eda.ipynb, prediction.ipynb  (playgrounds; import from rack_forecast)
scripts/           run_prediction.py  (thin CLI: build cfg → run_experiment)
dashboard/         app.py + views/{raw_data,runs,results,inference}.py  (Streamlit)
results/           per-run artifacts
```

**Key rule:** data-loading logic lives ONLY in `rack_forecast/data.py`; notebooks and
script both call `build_dataset(cfg)`. `prepare_data()` in `pipeline.py` is the single
split/scale implementation. `build_dataset()` derives TH sensor racks from
`target_rack`'s base name (`rsplit('-', 1)[0]`), not from an assumed `-PA` suffix —
it must work correctly whether `target_rack` is itself the `-PA` or `-PB` side.

`trainer.train_dl()` keeps the full train/val tensors on CPU and moves only one
mini-batch at a time to `DEVICE` (train and validation loops both use a
`DataLoader`) — GPU memory use must not scale with dataset size.

## EDA notebook

**`notebooks/eda.ipynb`** — 10 sections: rack PM time-series, TH sensors (PA+PB), cross-rack
comparison, cell-level PM, anomaly detection, monthly energy, SensorGW plots. Focus rack `R0605-PA/PB`.

## Prediction pipeline

Build an `ExperimentConfig`, then `run_experiment(cfg)` (script/dashboard) or run
`notebooks/prediction.ipynb` step by step. Config fields: `target_rack`, `lookback`,
`horizon` (steps; 1 step = 30 s), `models`, `dl_epochs`, `fast_mode` (target rack only),
`train_days`/`predict_days` (None = full; N = first N days for fast iteration/demo), `run_id`.

### Models (`linear`, `rf`, `xgboost`, `lstm`, `cnn1d`, `transformer`)
- `svr` excluded by default — too slow on CPU at scale (cuML is Linux-only).
- `xgboost` + DL models use GPU (RTX 4060, CUDA 12.6); `linear`/`rf` on CPU.
- Each `models/<name>.py` exposes `.fit()`, `.predict_step()`, `.predict_batch()`.
- `evaluate()` rolls all windows in a batch (horizon model calls total, not per-window).

### Results structure (per run)
```
results/{run_id}_{rack}_L{lookback}_H{horizon}/
  config.json  metrics.csv  scalers.pkl  metrics_bar.png  all_models_vs_actual.png
  {model}/  predictions.npz  actual_vs_pred_1window.png  residuals.png  models/{model}.pt|.pkl
```
`predictions.npz` = preds/actuals `(n_windows, horizon)` + per-window `timestamps` (datetime64).
`scalers.pkl` + `predictions.npz` are what the dashboard reads (no retraining).

### Sweeping racks / shared models

`scripts/run_prediction.py` sets `TARGET` to either a single rack (e.g. `'R0605-PA'`)
or a phase (`'PA'`/`'PB'`) to sweep every rack on that side, via `racks_for(target)`.

- `SHARE_MODEL = False` (default): each rack in the sweep gets its own independently
  trained model via `run_experiment(cfg)`, in its own `results/` folder.
- `SHARE_MODEL = True` (only used when sweeping >1 rack): trains ONE model per
  requested type, **fine-tuned rack-by-rack** (not pooled) via
  `pipeline.run_shared_experiment(cfg, racks)` → `_train_rack_by_rack()`:
  - `canonicalize_own_rack(data, rack)` renames a rack's own PM columns
    (`<rack>__kW/V/A/PF/Hz`) to rack-agnostic `OWN__<field>` names — TH sensor
    columns are already rack-agnostic and left as-is — so every rack's data
    shares one column layout.
  - `_prepare_pooled()` computes `feature_cols` as the intersection of columns
    present in every rack, and fits **one shared `StandardScaler`** on the
    pooled training rows across all racks (only the scaler is pooled).
  - Training itself is sequential, not pooled: the model trains on rack 1's
    windows, then keeps training the SAME weights on rack 2's windows, and so
    on — only one rack's windowed data is ever materialized at once, so memory
    doesn't scale with rack count (an earlier pooled-concatenation approach
    OOM'd on a 12-rack × 5-month sweep — 16GB RAM vs a >20GB pooled array).
  - Per model type, "continue training" means: DL models (`lstm`/`cnn1d`/
    `transformer`) keep the same weight tensors and call `train_dl()` again per
    rack; `xgboost` continues boosting via `xgb_model=` (grows the same
    ensemble); `linear` uses `_fit_ridge_accumulated()` — Ridge's closed-form
    solution only needs pooled sufficient statistics (`sum(X)`, `sum(y)`, `XᵀX`,
    `Xᵀy`, `n`), which accumulate additively per rack, so the result is
    mathematically identical to fitting Ridge on every rack's data concatenated
    together — exact, no learning rate to tune, no rack-order dependence (unlike
    the DL/XGBoost paths, which are order-sensitive). `coef_`/`intercept_` are
    set directly on a fresh `Ridge()` instance rather than calling `.fit()` —
    `rf` isn't supported here (no incremental-fit or closed-form path
    implemented) and will raise if included in a `SHARE_MODEL` sweep.
  - Evaluation is still per-rack, but metrics are pooled into a single table —
    there is no per-rack metrics breakdown.
  - Saved to one folder: `results/{run_id}_shared_{phase}_L{lookback}_H{horizon}/`,
    same artifact layout as above, plus `config.json['shared_model'] = True` and
    `config.json['racks']` (the pooled rack list). `predictions.npz` additionally
    stores a `window_racks` array (which rack each window came from).

## Dashboard

`conda run -n ntu_cooling streamlit run dashboard/app.py` — 4 pages: Raw Data (sensor
viewer, with a Side PA/PB dropdown — never overlays both), Training Runs (browse
results/), Prediction Results (all windows from `predictions.npz`), Live Inference
(load a model + scalers, pick a rack to infer on, forecast from a chosen day/hour/minute).
`dashboard/app.py` adds its own dir to `sys.path` so `views` resolves under any launcher.

Live Inference's "Rack to infer on" branches on `config.json['shared_model']`:
- Shared runs: dropdown lists exactly `cfg['racks']`; the freshly built dataset is
  canonicalized the same way training did (`canonicalize_own_rack`), so column names
  already line up — no substitution needed.
- Non-shared runs: dropdown lists every rack sharing the run's own PA/PB side
  (`_racks_same_side`); picking a different rack substring-replaces the run's rack
  name in `feature_cols`/`target_col` so the model sees the same feature layout,
  sourced from the newly picked rack's data.
- Either way, `scaler.transform(data[feature_cols].to_numpy())` is used (not a
  DataFrame) — sklearn's scaler validates feature *names* on a DataFrame input,
  which breaks as soon as the inferred rack's column names differ from the
  scaler's fit-time names, even when values line up positionally.

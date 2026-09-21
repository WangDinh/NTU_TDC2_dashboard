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
conda run -n ntu_cooling jupyter nbconvert --to notebook --execute --inplace notebooks/02_prediction.ipynb \
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
  windowing.py     make_supervised() + make_direct_supervised()/make_direct_test_windows()
  trainer.py       DEVICE, train_dl(), build_and_train()
  evaluate.py      evaluate() (batched rollout), evaluate_direct(), autoregressive_predict(),
                   compute_metrics(), DivergenceError
  persistence.py   save/load model, predictions(.npz), scalers(.pkl), config, metrics; list_runs()
  plots.py         figure-returning helpers (notebook displays, pipeline saves)
  pipeline.py      prepare_data() (split+scale) + run_experiment() + save_results()
  models/          linear rf xgboost lstm cnn1d transformer  (svr.py kept, not in REGISTRY)
  pcnn/            Adapt-PCNN — separate return-air-temp model, see "PCNN" section below
notebooks/         numbered in workflow order:
                   01_eda.ipynb, 02_prediction.ipynb  (main pipeline playgrounds)
                   03_feature_eda.ipynb, 04_direct_forecast_eda.ipynb  (preprocessing-variant
                   and single_step/multi_step exploration — see below)
                   05_pcnn_mlp.ipynb  (PCNN entry point — see "PCNN" section)
                   06_spic_data_export.ipynb  (exports data_for_spic/, unrelated utility)
scripts/           run_prediction.py  (thin CLI: build cfg → run_experiment)
dashboard/         app.py + views/{raw_data,runs,results,inference}.py  (Streamlit)
results/           per-run artifacts (results/pcnn/ is separate, see "PCNN" section)
data_for_spic/     generated exports for an external Single-Phase Immersion Cooling test
                   rig — not part of the forecasting pipeline, see "PCNN" section
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

**`notebooks/01_eda.ipynb`** — 10 sections: rack PM time-series, TH sensors (PA+PB), cross-rack
comparison, cell-level PM, anomaly detection, monthly energy, SensorGW plots. Focus rack `R0605-PA/PB`.

## Prediction pipeline

Build an `ExperimentConfig`, then `run_experiment(cfg)` (script/dashboard) or run
`notebooks/02_prediction.ipynb` step by step. Config fields: `target_rack`, `lookback`,
`horizon` (steps; 1 step = 30 s), `models`, `dl_epochs`, `fast_mode` (target rack only),
`train_days`/`predict_days` (None = full; N = first N days for fast iteration/demo), `run_id`,
`strategy` (`'single_step'`/`'multi_step'`, see below), `direct_stride` (multi_step only).

**Per-model failure isolation:** `run_experiment()` wraps each model's train+evaluate in
try/except — one model failing (e.g. a `DivergenceError`, see below) no longer aborts the
whole run; other models still complete, and the failure is recorded to
`<run_folder>/failed_models.txt`. Only raises if *every* model in the run fails.

**Memory risk with `fast_mode=False`:** full mode loads 182 features (all 24 racks'
PM + target rack TH + all 5 SensorGW streams) vs. 17 for `fast_mode=True`.
`make_supervised()` ([windowing.py](rack_forecast/windowing.py)) materializes the
entire `(n_windows, lookback, n_feat)` array in one eager `np.array()` call — no
chunking exists anywhere downstream. At the default `lookback=60` with unclipped
`train_days=None` (~438K training rows), that array is ~19GB, which exceeds this
machine's ~16.8GB RAM (same failure mode as the earlier 12-rack pooled-training
OOM, just triggered by feature-count instead of rack-count). The rack-by-rack
shared-model path only bounds memory *across racks* — it does nothing for a single
rack's own array size. **Use `train_days`/`predict_days` clipping whenever running
`fast_mode=False`** to stay within a safe array size (`fast_mode=True` at full data
is ~1.8GB — safe by comparison).

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

### Forecasting strategy: single_step vs multi_step

`ExperimentConfig.strategy` (`'single_step'`, default, or `'multi_step'`) is a real
pipeline option now, promoted from an earlier exploratory notebook
(`notebooks/04_direct_forecast_eda.ipynb`, still present, no longer the only way to
run this).

- **`single_step`** (autoregressive/recursive): `evaluate()` predicts one step,
  feeds the full predicted feature vector back in as if real, repeats `horizon`
  times — errors compound. This is what motivated `multi_step`'s existence: at
  very long horizons (observed at `horizon=20160` = 1 week at the default 30s
  resample) the rollout can diverge to `inf`. `evaluate()` now raises `DivergenceError`
  (`evaluate.py`) as soon as predictions go non-finite, reporting the step it
  happened at — instead of letting `inf`/`NaN` silently propagate into the scaler
  and surface as an opaque sklearn error.
- **`multi_step`** (direct): predicts all `horizon` steps of the target in a
  single forward pass via `evaluate_direct()` — no feedback loop, so
  `DivergenceError` is structurally impossible. Since nothing needs to be fed
  back in, it only predicts the target column, not the full multi-output vector
  `single_step` requires. Training windows come from
  `windowing.make_direct_supervised()` (target-only future window, optionally
  strided); test windows from `make_direct_test_windows()`, deliberately built to
  mirror `evaluate()`'s window definition so `single_step`/`multi_step` runs
  score on comparable windows. `models/{lstm,cnn1d,transformer}.py` each take an
  `n_out` param (defaults to `n_feat`, settable to `horizon`) so only the final
  layer's output width changes; `trainer.build_and_train()` and
  `persistence.load_model()` both thread `n_out` through so a multi_step
  checkpoint reloads with the matching layer shape. `linear`/`xgboost`/`rf` need
  no changes — they size output from whatever `y` they're fit on.
- **`direct_stride`** (multi_step only, default `None` = auto): stride between
  overlapping training windows. `cfg.resolved_direct_stride(n_train_rows,
  budget_bytes=2GiB)` auto-picks the smallest stride whose resulting
  `(n_windows, horizon)` float32 target array fits a memory budget — same
  category of fix as the earlier rack-pooling/`fast_mode=False` OOMs, applied to
  multi_step's own target array this time.
- **Not supported for shared/pooled runs**: `run_shared_experiment()` raises
  `NotImplementedError` if `cfg.is_multi_step` — the rack-by-rack fine-tuning
  path still assumes a single_step target layout. Use `SHARE_MODEL=False` (or
  `strategy='single_step'`) for shared runs.
- `run_name` appends a `_multistep` suffix when `cfg.is_multi_step`, so old
  single_step result folders keep resolving identically.

Earlier single-rack/XGBoost-only result from the original exploratory notebook
(`R0605-PA`, lookback=60/horizon=30, `train_days=60`): `multi_step` won by a small
margin (RMSE 0.01582 vs 0.01594, R2 0.323 vs 0.313), ~4.5x faster inference but
~50% longer training — kept as a reference data point, not a general verdict.

### Horizon-sweep runs in `results/` (what's actually there)

All `R0605-PA`, `lookback=60`, `resample=30s`, `train_days=None`. **These are NOT
one consistent method across horizons** — check `config.json['strategy']` (and the
`_multistep` folder suffix) before quoting any of these side by side:

| Horizon | single_step folder | multi_step folder |
|---|---|---|
| 15 min (H=30) | `run_03_..._H30` (all 5 models) | — none |
| 1 hour (H=120) | `run_1h_..._H120` | `run_1h_..._H120_multistep` (all 5) |
| 1 day (H=2880) | `run_1d_..._H2880` | `run_1d_..._H2880_multistep` (**no xgboost**) |
| 1 week (H=20160) | `run_1w_..._H20160` (**no cnn1d** — diverged) | `run_1w_..._H20160_multistep` (**no xgboost**) |

Two gaps to know about when assembling comparison tables:
- **xgboost is absent from the 1d/1w multistep runs** — it was dropped from
  `MODELS` in `scripts/run_prediction.py` before those ran. Filling those cells
  from the single_step folder silently mixes strategies within a row.
- **cnn1d has no 1-week single_step result** — `DivergenceError: rollout diverged
  at step 1220/20160` (recorded in that run's `failed_models.txt`). Only the
  multistep value exists for that cell.

### Feature-engineering comparison (exploratory)

`notebooks/03_feature_eda.ipynb` — temporary, standalone (no changes to
`rack_forecast/`), tests preprocessing variants against a raw-sensor baseline on
rack R0605-PA via a shared `run_variant()` harness. Note: `run_variant()` trains
whatever model is in `cfg.models[0]` (not hardcoded XGBoost) — check that field
before trusting results. Gains from preprocessing were marginal overall;
`delta_current_freq_remove` (5 features) trades a little accuracy for a large
speed win. Exploratory only, not integrated into the production pipeline.

## PCNN (Adapt-PCNN) — return-air temperature forecasting

`rack_forecast/pcnn/` is a **separate, self-contained subpackage** — a different
model family for a **different target** (return-air temperature `RA_T`, not rack
power), deliberately isolated from the main pipeline: it only reuses raw loaders
(`load_gw`/`load_agg_pm`/`_resampled`) from `rack_forecast.data`, never imports
`config`/`trainer`/`evaluate`/`windowing`/pipeline logic, and writes to
`results/pcnn/` (`PCNN_RESULTS_ROOT`) specifically so its runs don't show up in
the power dashboard's `list_runs()`. It has its own local `DEVICE`,
`compute_metrics`, and min-max `Normalizer` — duplicated on purpose rather than
sharing the main pipeline's.

"PCNN" = **Adapt-PCNN** (physics-consistent neural network), ported from the
reference implementation at https://doi.org/10.1016/j.apenergy.2024.124637.
`pcnn/module.py`'s `Adapt_PC_MLP` models one-step RA_T update as
`RA_T + D/division_factor - cool_effect + heat_effect`, where `D` comes from an
MLP and `coeff_a`/`coeff_b` (cooling/heating coefficients) from a softplus head —
autoregressive state is only `self.last_D` (no RNN hidden state), which is the
stated reason it stays stable over long rollouts, unlike `single_step`'s
divergence issue above.

**No CLI script exists for it** — `notebooks/05_pcnn_mlp.ipynb` is the only runnable
entry point: build `PCNNConfig(...)` → `build_pcnn_dataset(cfg)` → `month_split` +
`fit_normalizer` → `train_pcnn(...)` → `rollout_evaluate(...)` for PCNN's own
metrics, plus `run_baselines(...)` which gives 5 black-box models
(`linear`/`xgboost`/`lstm`/`cnn1d`/`transformer`, reusing the main package's DL
model classes with `n_out=horizon`) the *same* information PCNN's rollout gets,
scored on the same windows, as a fairness-matched comparison. No
persistence/model-saving step exists yet (unlike the main pipeline).

**Known gap:** `pyproject.toml`'s `[tool.setuptools] packages` list was not
updated to include `rack_forecast.pcnn` — fine for the current in-place
`pip install -e .` dev checkout, but worth fixing before this is ever installed
as a built package elsewhere.

`data_for_spic/` is unrelated to PCNN's modeling — it's a generated-data export
for an external **Single-Phase Immersion Cooling** test rig (`notebooks/
06_spic_data_export.ipynb`): rescales one rack/day's real power profile into a
range a physical heater/coupon can dissipate (`power_load.csv`), pairs it with
that day's outside temp/humidity as the boundary condition (`weather.csv`), and
optionally exports a `.mat` window for MATLAB/Simulink. Not part of the
forecasting pipeline; safe to ignore unless working on that export.

## Dashboard

`conda run -n ntu_cooling streamlit run dashboard/app.py` — 4 pages: Raw Data (sensor
viewer, with a Side PA/PB dropdown — never overlays both), Training Runs (browse
results/), Prediction Results (all windows from `predictions.npz`), Live Inference
(load a model + scalers, pick a rack to infer on, forecast from a chosen day/hour/minute).
`dashboard/app.py` adds its own dir to `sys.path` so `views` resolves under any launcher.

Prediction Results also branches on shared vs. non-shared runs: if `predictions.npz`
has a `window_racks` array (shared runs only), a **Rack** dropdown appears and filters
every window/plot/metric down to the selected rack before anything is drawn — without
it, a shared run's windows from every rack in the sweep get concatenated into one
time series, which is unreadable.

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

# TDC 2.0 Rack Power Forecasting

Predict a data-center rack's aggregate power (kW) from its own and neighbouring
sensor signals, and explore everything through a web dashboard.

## Layout

```
rack_forecast/        core library (installable package)
  paths.py            CWD-independent DATA_ROOT / RESULTS_ROOT
  config.py           ExperimentConfig — every knob in one dataclass
  data.py             loaders + build_dataset() (single source of truth)
  windowing.py        make_supervised()
  trainer.py          train_dl(), build_and_train(), DEVICE
  evaluate.py         evaluate(), autoregressive_predict(), compute_metrics()
  persistence.py      save/load models, predictions, scalers, config, metrics
  plots.py            figure-returning plot helpers
  pipeline.py         prepare_data() + run_experiment() orchestration
  models/             linear, rf, xgboost, lstm, cnn1d, transformer
notebooks/
  eda.ipynb           dataset exploration playground
  prediction.ipynb    step-by-step forecasting playground
scripts/
  run_prediction.py   CLI: edit config, run one experiment
dashboard/
  app.py + views/     Streamlit app (raw data, runs, results, live inference)
results/              per-run artifacts (config, metrics, models, predictions, figures)
data/                 raw TDC 2.0 dataset
```

## Dataset

TDC 2.0 (NTU air-cooled tropical data-center testbed) — 24 rack PDUs (PA/PB), 5
SensorGW streams, 6 months (2022-09 → 2023-02), 30s sampling.

Source: https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/BLBQ2T

Download the dataset from the link above and extract it so the folder layout is:

```
data/TDC2.0 Dataset/
  R0501-PA/  R0501-PB/  ...  R0606-PA/  R0606-PB/
  SensorGW-1/  SensorGW-2/  SensorGW-3/  SensorGW-4/  SensorGW-5/
```

`data/` is gitignored and not included in this repo — `rack_forecast/paths.py` resolves
`DATA_ROOT` to `data/TDC2.0 Dataset/` relative to the project root, so this is the only
required layout.

## Setup (one time)

```bash
conda create -n ntu_cooling python=3.11 -y
conda activate ntu_cooling
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or /cpu for CPU-only
pip install -e .                                    # editable install of rack_forecast
```

`pip install -e .` makes `import rack_forecast` work from notebooks, scripts, and
the dashboard regardless of the working directory. `torch` is installed separately
because the CUDA-matched build isn't on plain PyPI — swap the index URL for your
platform (see `requirements.txt` for the CPU-only alternative).

## Run an experiment

Edit the config at the top of `scripts/run_prediction.py`, then:

```bash
python scripts/run_prediction.py
```

or open `notebooks/prediction.ipynb` to run it step by step. Either way, artifacts
land in `results/<run_id>_<rack>_L<lookback>_H<horizon>/`:

```
config.json  metrics.csv  scalers.pkl  metrics_bar.png  all_models_vs_actual.png
<model>/  predictions.npz  actual_vs_pred_1window.png  residuals.png  models/<model>.pt|.pkl
```

Key config knobs: `models`, `lookback`, `horizon`, `fast_mode` (target rack only),
`train_days` / `predict_days` (clip for fast iteration or demos).

## Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Pages: **Raw Data** (sensor viewer), **Training Runs** (browse results), **Prediction
Results** (all windows for a model), **Live Inference** (load a model, forecast from a
chosen time).

## Notes
- GPU (CUDA) is used automatically for `xgboost` and the DL models; `linear`/`rf`
  run on CPU (cuML is Linux-only).
- `svr` is available in `models/` but excluded by default — too slow on CPU at scale.

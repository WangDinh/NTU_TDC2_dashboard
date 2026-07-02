"""Page 4 — Live inference with a saved model.

Loads a trained model + its scalers, lets you pick a start time in the data,
seeds from the preceding `lookback` real observations, rolls the model forward
`horizon` steps autoregressively, and plots the forecast against the actuals.
"""

from datetime import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from rack_forecast import ExperimentConfig
from rack_forecast.data import build_dataset
from rack_forecast.evaluate import autoregressive_predict, _inverse_target
from rack_forecast.paths import DATA_ROOT
from rack_forecast.pipeline import canonicalize_own_rack
from rack_forecast.persistence import (list_runs, available_models,
                                       load_config, load_scalers, load_model)


@st.cache_data(show_spinner='Building dataset…')
def _data(target_rack, fast_mode, resample):
    """Cached dataset build keyed by the fields that affect its contents."""
    cfg = ExperimentConfig(target_rack=target_rack, fast_mode=fast_mode, resample=resample)
    return build_dataset(cfg)


def _racks_same_side(rack):
    """All racks sharing `rack`'s PA/PB side (a model's feature layout only
    lines up across racks of the same side, per data.py's column scheme)."""
    side = rack.rsplit('-', 1)[1]
    return sorted(d.name for d in DATA_ROOT.iterdir()
                  if d.is_dir() and d.name.endswith(f'-{side}'))


def render():
    st.header('Live Inference')

    runs = list_runs()
    if not runs:
        st.info('No runs found in results/.')
        return

    sel = st.selectbox('Run', [r['run_name'] for r in runs][::-1])
    run = next(r for r in runs if r['run_name'] == sel)
    path = run['path']
    cfg = load_config(path)

    # Live inference needs the saved scaler (added in the current pipeline).
    if not (path / 'scalers.pkl').exists():
        st.warning('This run has no scalers.pkl — re-run the experiment with the '
                   'current pipeline to enable live inference.')
        return
    try:
        scaler = load_scalers(path)
    except KeyError:
        st.warning('This run has an old-format scalers.pkl (pre multi-output) — '
                   're-run the experiment with the current pipeline to enable live inference.')
        return

    models = available_models(path) or cfg.get('models', [])
    if not models:
        st.warning('No saved models found in this run.')
        return
    model_name = st.selectbox('Model', models)

    lookback = int(cfg['lookback'])
    horizon  = int(cfg['horizon'])
    run_rack = cfg['target_rack']
    target_col = cfg.get('target_col', f'{run_rack}__kW')
    feature_cols = cfg.get('feature_cols')
    shared = cfg.get('shared_model', False)

    if shared:
        # A shared-model run's columns are already rack-agnostic ('OWN__kW'
        # etc) — any rack it was trained on can be picked with no renaming.
        racks_list = cfg.get('racks') or [run_rack]
        infer_rack = st.selectbox('Rack to infer on', racks_list, index=0)
        data = _data(infer_rack, cfg['fast_mode'], cfg['resample'])
        data = canonicalize_own_rack(data, infer_rack)
        feature_cols = feature_cols or list(data.columns)
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            st.warning(f'{infer_rack} is missing columns this shared model needs: '
                       f'{missing}. Pick a different rack.')
            return
    else:
        # ── rack to infer on (defaults to the run's own rack; same PA/PB
        #    side only — a model's feature layout is tied to that side's
        #    column set) ──
        side_racks = _racks_same_side(run_rack)
        infer_rack = st.selectbox('Rack to infer on', side_racks,
                                  index=side_racks.index(run_rack) if run_rack in side_racks else 0)

        data = _data(infer_rack, cfg['fast_mode'], cfg['resample'])
        feature_cols = feature_cols or list(data.columns)
        if infer_rack != run_rack:
            # Column names are '<rack>__<field>' for that rack's own PM signals;
            # substitute the rack so the model sees the same feature layout it
            # was trained on, just sourced from the newly picked rack's data.
            feature_cols = [c.replace(run_rack, infer_rack) for c in feature_cols]
            target_col = target_col.replace(run_rack, infer_rack)
            missing = [c for c in feature_cols if c not in data.columns]
            if missing:
                st.warning(f'{infer_rack} is missing columns this model needs: {missing}. '
                           'Pick a different rack.')
                return

    # models are saved under <run>/<model>/models/<model>.pt|.pkl
    model = load_model(model_name, path / model_name,
                       n_feat=len(feature_cols), lookback=lookback)

    # .to_numpy() drops column names — needed when inferring on a different
    # rack than the one the scaler was fit on (values line up positionally).
    X_all = scaler.transform(data[feature_cols].to_numpy()).astype('float32')
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

    # ── choose a start time within the test month ──────────────────────
    test_month = cfg.get('test_month')
    idx = data.index
    if test_month:
        idx = idx[idx.to_period('M').astype(str) == test_month]
    # need `lookback` rows before and `horizon` rows after the chosen start
    lo = data.index.get_loc(idx[0]) + lookback
    hi = data.index.get_loc(idx[-1]) - horizon
    if hi <= lo:
        st.warning('Not enough data in the test month for this lookback/horizon.')
        return

    tmin = data.index[lo].to_pydatetime()
    tmax = data.index[hi].to_pydatetime()

    d1, d2, d3 = st.columns(3)
    day = d1.date_input('Day', min_value=tmin.date(), max_value=tmax.date(), value=tmin.date())
    hour = d2.selectbox('Hour', list(range(24)), index=tmin.hour)
    minute = d3.selectbox('Minute', list(range(60)), index=tmin.minute)

    picked = pd.Timestamp.combine(day, time(hour=hour, minute=minute)).to_pydatetime()
    picked = min(max(picked, tmin), tmax)
    pos = data.index.get_indexer([pd.Timestamp(picked)], method='nearest')[0]

    # ── autoregressive rollout ─────────────────────────────────────────
    seed = X_all[pos - lookback: pos]                       # (lookback, n_feat)
    preds_s = autoregressive_predict(model, seed, horizon, target_idx)
    preds = _inverse_target(preds_s, scaler, target_idx, len(feature_cols))

    fut = data.index[pos: pos + horizon]
    actual = data[target_col].iloc[pos: pos + horizon].values

    # ── plot (with the seed context for reference) ─────────────────────
    ctx = data.index[pos - lookback: pos]
    ctx_y = data[target_col].iloc[pos - lookback: pos].values

    fig = go.Figure()
    fig.add_scatter(x=ctx, y=ctx_y, name='Seed (actual)', line=dict(color='#999999', width=1))
    fig.add_scatter(x=fut, y=actual, name='Actual', line=dict(color='#dddddd', width=1.6))
    fig.add_scatter(x=fut, y=preds, name=f'{model_name} forecast',
                    line=dict(color='crimson', width=1.6, dash='dot'))
    fig.add_vline(x=picked, line=dict(color='orange', width=1, dash='dash'))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                      yaxis_title='Power (kW)',
                      title=f'{model_name} — {horizon}-step forecast from {picked:%b-%d %H:%M}')
    st.plotly_chart(fig, use_container_width=True)

    mae = np.abs(actual - preds).mean()
    rmse = np.sqrt(((actual - preds) ** 2).mean())
    c1, c2 = st.columns(2)
    c1.metric('MAE (this window)', f'{mae:.4f}')
    c2.metric('RMSE (this window)', f'{rmse:.4f}')

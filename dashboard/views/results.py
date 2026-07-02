"""Page 3 — Visualize a model's full prediction rollout.

Reads <run>/<model>/predictions.npz (all windows, no model load needed),
reconstructs a continuous time axis, and shows actual-vs-predicted across every
window with a slider to focus on one window, plus per-window error and residuals.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from rack_forecast.persistence import (list_runs, available_models,
                                       load_predictions, load_config)


def _step_times(window_starts, horizon, resample):
    """Expand per-window start times into a per-step DatetimeIndex."""
    freq = pd.Timedelta(resample)
    blocks = [pd.DatetimeIndex(window_starts[w] + np.arange(horizon) * freq)
              for w in range(len(window_starts))]
    return pd.DatetimeIndex(np.concatenate([b.values for b in blocks]))


def render():
    st.header('Prediction Results')

    runs = list_runs()
    if not runs:
        st.info('No runs found in results/.')
        return

    sel = st.selectbox('Run', [r['run_name'] for r in runs][::-1])
    run = next(r for r in runs if r['run_name'] == sel)

    models = available_models(run['path'])
    if not models:
        st.warning('This run has no saved predictions.npz.')
        return
    model = st.selectbox('Model', models)

    cfg = load_config(run['path'])
    d = load_predictions(run['path'], model)
    preds, actuals, starts = d['preds'], d['actuals'], d['timestamps']

    # Shared-model runs pool windows from every rack in one predictions.npz —
    # plotting them all together interleaves unrelated racks' time series, so
    # let the user pick one rack to view at a time.
    if 'window_racks' in d:
        window_racks = d['window_racks']
        racks = sorted(set(window_racks))
        rack_sel = st.selectbox('Rack', racks)
        mask = window_racks == rack_sel
        preds, actuals, starts = preds[mask], actuals[mask], starts[mask]

    n_windows, horizon = preds.shape
    times = _step_times(starts, horizon, cfg.get('resample', '30s'))

    st.caption(f'{n_windows} windows × {horizon} steps '
               f'({horizon} steps = {horizon // 2} min per window)')

    # ── all-windows time series ────────────────────────────────────────
    flat_pred = preds.ravel()
    flat_actual = actuals.ravel()
    fig = go.Figure()
    fig.add_scatter(x=times, y=flat_actual, name='Actual',
                    line=dict(color='#dddddd', width=1.3))
    fig.add_scatter(x=times, y=flat_pred, name=model,
                    line=dict(color='steelblue', width=1))

    # Highlight one window with a shaded band.
    w = st.slider('Highlight window', 0, n_windows - 1, 0)
    w0, w1 = times[w * horizon], times[w * horizon + horizon - 1]
    fig.add_vrect(x0=w0, x1=w1, fillcolor='orange', opacity=0.2, line_width=0)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                      yaxis_title='Power (kW)',
                      title=f'{model} — actual vs predicted (all windows)')
    st.plotly_chart(fig, use_container_width=True)

    # ── selected-window zoom ───────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        wt = times[w * horizon: w * horizon + horizon]
        zf = go.Figure()
        zf.add_scatter(x=wt, y=actuals[w], name='Actual', line=dict(color='#dddddd', width=1.6))
        zf.add_scatter(x=wt, y=preds[w], name=model, line=dict(color='steelblue', width=1.4))
        zf.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                         yaxis_title='kW', title=f'Window {w} — starts {starts[w]}')
        st.plotly_chart(zf, use_container_width=True)

    # ── per-window error + residuals ───────────────────────────────────
    per_win = pd.DataFrame({
        'window': range(n_windows),
        'start': starts,
        'MAE': np.abs(preds - actuals).mean(axis=1),
        'RMSE': np.sqrt(((preds - actuals) ** 2).mean(axis=1)),
    })
    with c2:
        st.metric('Overall MAE', f'{np.abs(flat_pred - flat_actual).mean():.4f}')
        st.metric('Overall RMSE', f'{np.sqrt(((flat_pred - flat_actual) ** 2).mean()):.4f}')

    st.subheader('Per-window error')
    st.dataframe(per_win.round(4), use_container_width=True, hide_index=True, height=220)

    st.subheader('Residual distribution')
    resid = flat_actual - flat_pred
    hf = go.Figure(go.Histogram(x=resid, nbinsx=60, marker_color='steelblue'))
    hf.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                     xaxis_title='Error (kW)')
    st.plotly_chart(hf, use_container_width=True)

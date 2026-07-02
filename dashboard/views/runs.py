"""Page 2 — Browse saved training runs.

Lists every run folder under results/, then shows the selected run's config,
metrics table, and the run-level comparison figures.
"""

import streamlit as st
import pandas as pd

from rack_forecast.persistence import list_runs, load_config, load_metrics


def render():
    st.header('Training Runs')

    runs = list_runs()
    if not runs:
        st.info('No runs found in results/. Run an experiment first.')
        return

    # ── overview table ─────────────────────────────────────────────────
    cols = ['run_name', 'target_rack', 'lookback', 'horizon',
            'test_month', 'n_features', 'timestamp']
    table = pd.DataFrame([{c: r.get(c) for c in cols} for r in runs])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ── inspect one run ────────────────────────────────────────────────
    sel = st.selectbox('Inspect run', [r['run_name'] for r in runs][::-1])
    run = next(r for r in runs if r['run_name'] == sel)
    path = run['path']

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader('config.json')
        st.json(load_config(path), expanded=False)
    with c2:
        st.subheader('metrics.csv')
        try:
            st.dataframe(load_metrics(path).sort_values('RMSE'), use_container_width=True)
        except Exception as e:
            st.warning(f'No metrics.csv ({e})')

    # ── run-level figures ──────────────────────────────────────────────
    st.subheader('Comparison figures')
    for img in ['metrics_bar.png', 'all_models_vs_actual.png']:
        p = path / img
        if p.exists():
            st.image(str(p), caption=img, use_container_width=True)

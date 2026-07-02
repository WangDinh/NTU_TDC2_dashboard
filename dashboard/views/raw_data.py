"""Page 1 — Raw sensor data viewer.

Pick a rack and date range, then inspect three groups of signals: rack power,
the rack's temperature/humidity sensors, and the shared SensorGW readings.
All loads are cached so re-filtering is instant.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rack_forecast.paths import DATA_ROOT
from rack_forecast.data import load_agg_pm, load_th, load_gw


# ── cached loaders (raw files are large; load once per rack) ───────────────

@st.cache_data(show_spinner='Loading power…')
def _pm(rack, freq):
    df = load_agg_pm(rack).sort_values('timestamp').set_index('timestamp')
    return df[['kW', 'V', 'A', 'PF', 'Hz']].resample(freq).mean()


@st.cache_data(show_spinner='Loading temp/humidity…')
def _th(rack, prefix, freq):
    df = load_th(rack, prefix)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('timestamp').set_index('timestamp')
    return df[['temp_C', 'humidity_pct']].resample(freq).mean()


@st.cache_data(show_spinner='Loading SensorGW…')
def _gw(gw, sensor, cols, field, freq):
    df = load_gw(gw, sensor, cols)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index('timestamp')[field].resample(freq).mean()


def _rack_list():
    return sorted({d.name.rsplit('-', 1)[0] for d in DATA_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith('R')})


def _clip(df, start, end):
    """Restrict a time-indexed frame/series to [start, end]."""
    return df.loc[str(start):str(end)]


# ── page ───────────────────────────────────────────────────────────────────

def render():
    st.header('Raw Data Viewer')

    racks = _rack_list()
    c1, c2, c3 = st.columns([1, 1, 1])
    base_rack = c1.selectbox('Rack', racks)
    side = c2.selectbox('Side', ['PA', 'PB'])
    freq = c3.selectbox('Resolution', ['1h', '30min', '15min', '30s'], index=0)
    rack = f'{base_rack}-{side}'

    pm = _pm(rack, freq)
    if pm.empty:
        st.warning(f'No PM data for {rack}.'); return

    # Date-range picker bounded by the available data.
    dmin, dmax = pm.index.min().date(), pm.index.max().date()
    start, end = st.slider('Date range', min_value=dmin, max_value=dmax,
                           value=(dmin, dmax), format='YYYY-MM-DD')

    # ── Power ──────────────────────────────────────────────────────────
    with st.expander('Power (kW / V / A / PF / Hz)', expanded=True):
        p = _clip(pm, start, end)
        show_vapfhz = st.checkbox('Show V / A / PF / Hz', value=False)

        fig = go.Figure()
        fig.add_scatter(x=p.index, y=p['kW'], name=f'{rack} kW', line=dict(width=1))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0),
                          yaxis_title='Power (kW)', title=f'{rack} — aggregate power')
        st.plotly_chart(fig, use_container_width=True)

        # V, A, PF, Hz each get their own subplot (different units/scales).
        if show_vapfhz:
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Voltage (V)', 'Current (A)',
                                                 'Power Factor', 'Frequency (Hz)'))
            for row, col in enumerate(['V', 'A', 'PF', 'Hz'], start=1):
                fig.add_scatter(x=p.index, y=p[col], name=f'{rack} {col}',
                                line=dict(width=1), row=row, col=1)
            fig.update_layout(height=760, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Temperature & Humidity ─────────────────────────────────────────
    with st.expander('Temperature & Humidity (rack TH sensors)'):
        sensors = [(f'{base_rack}-PA', s) for s in ['THFT', 'THFM', 'THFB', 'THBB']] + \
                  [(f'{base_rack}-PB', s) for s in ['THBM', 'THBT']]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Temperature (°C)', 'Humidity (%)'))
        for src_rack, s in sensors:
            th = _th(src_rack, s, freq)
            if th.empty:
                continue
            th = _clip(th, start, end)
            fig.add_scatter(x=th.index, y=th['temp_C'], name=s, legendgroup=s, row=1, col=1)
            fig.add_scatter(x=th.index, y=th['humidity_pct'], name=s, legendgroup=s,
                            showlegend=False, row=2, col=1)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── SensorGW (shared room / cooling-loop / PCU signals) ────────────
    with st.expander('SensorGW (supply air, cooling loop, PCU power)'):
        st.caption('These signals are facility-wide (not per rack).')

        st.markdown('**GW-1 — Supply air**')
        sat = _gw(1, 'SAT1', ['timestamp', 'temp_C'], 'temp_C', freq)
        v1 = _gw(1, 'SAAV1', ['timestamp', 'velocity_ms'], 'velocity_ms', freq)
        v2 = _gw(1, 'SAAV2', ['timestamp', 'velocity_ms'], 'velocity_ms', freq)
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Supply air temp (°C)',
                                                            'Air velocity (m/s)'))
        if not sat.empty:
            s = _clip(sat, start, end); fig.add_scatter(x=s.index, y=s.values, name='SAT1', row=1, col=1)
        for nm, v in [('SAAV1', v1), ('SAAV2', v2)]:
            if not v.empty:
                s = _clip(v, start, end); fig.add_scatter(x=s.index, y=s.values, name=nm, row=1, col=2)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('**GW-3/4 — Cooling loop temperatures**')
        fig = go.Figure()
        for gw in (3, 4):
            for s in ['T-DL-1', 'T-EL-1', 'T-LL-1', 'T-SL-1']:
                ser = _gw(gw, s, ['timestamp', 'temp_C'], 'temp_C', freq)
                if not ser.empty:
                    ser = _clip(ser, start, end)
                    fig.add_scatter(x=ser.index, y=ser.values, name=f'GW{gw} {s}', line=dict(width=1))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title='°C')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('**GW-5 — PCU power (kW)**')
        fig = go.Figure()
        tp_cols = ['timestamp', 'W', 'V_L1', 'V_L2', 'V_L3', 'kWh']
        for s in ['PCU-4-P-CM', 'PCU-5-P-CM', 'PCU-4-P-CP1', 'PCU-5-P-CP1']:
            ser = _gw(5, s, tp_cols, 'W', freq)
            if not ser.empty:
                ser = _clip(ser, start, end)
                fig.add_scatter(x=ser.index, y=ser.values / 1000, name=s, line=dict(width=1))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title='kW')
        st.plotly_chart(fig, use_container_width=True)

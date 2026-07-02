"""Rack Forecast dashboard — Streamlit entry point.

Run with:
    conda run -n ntu_cooling streamlit run dashboard/app.py

The sidebar radio routes to one of four page modules in `views/`, each of which
exposes a `render()` function. All data access goes through the installed
`rack_forecast` package, so this app works regardless of the launch directory.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure this file's folder is importable so `views` resolves no matter how the
# app is launched (`streamlit run`, AppTest, a different CWD, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from views import raw_data, runs, results, inference

st.set_page_config(page_title='Rack Forecast', page_icon='⚡', layout='wide')

st.sidebar.title('⚡ Rack Forecast')
st.sidebar.caption('TDC 2.0 power prediction')

# Page registry: label → module. Add a page by dropping a module in views/.
PAGES = {
    'Raw Data':            raw_data,
    'Training Runs':       runs,
    'Prediction Results':  results,
    'Live Inference':      inference,
}

choice = st.sidebar.radio('Page', list(PAGES))
PAGES[choice].render()

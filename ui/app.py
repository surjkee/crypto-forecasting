import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# -------------------------

import streamlit as st
import plotly.express as px
import pandas as pd

from features.transform import build_feature_frame
from models.baseline import naive_constant_forecast

from ui.tabs.debugging_tab import render_debugging_tab
from ui.tabs.data_tab import render_data_tab
from ui.tabs.forecast_tab import render_forecast_tab
from ui.tabs.features_tab import render_features_tab

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from jobs.fetch_history import fetch_and_store_history

# --- Загальні налаштування сторінки ---
st.set_page_config(
    page_title="Crypto Forecasting - Data",
    layout="wide",
)

def main():
    tab_data, tab_features, tab_forecast, tab_debug = st.tabs(["📊 Data", "🧩 Features", "🔮 Forecast", "🛠 Debugging"])

    with tab_data:
        render_data_tab()

    with tab_features:
        render_features_tab()

    with tab_forecast:
        render_forecast_tab()

    with tab_debug:
        render_debugging_tab()



if __name__ == "__main__":
    main()

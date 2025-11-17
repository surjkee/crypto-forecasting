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
    tab_labels = ["Data", "Features", "Debugging"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_data_tab()

    with tabs[1]:
        render_features_tab()

    with tabs[2]:
        render_debugging_tab()



if __name__ == "__main__":
    main()

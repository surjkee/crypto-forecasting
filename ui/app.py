import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# -------------------------

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.tabs.debugging_tab import render_debugging_tab
from ui.tabs.data_tab import render_data_tab
from ui.tabs.forecast_tab import render_forecast_tab
from ui.tabs.features_tab import render_features_tab
from ui.components.footer import render_footer


def main():
    # --- Загальні налаштування сторінки ---
    st.set_page_config(
        page_title="Crypto Forecasting - Data",
        layout="wide",
    )
    if not st.session_state.get("developer_mode", True):
        st.markdown(
            """
            <style>
            [data-testid="stToolbar"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
    <style>
        .stAppHeader { 
            background: rgba(14, 12, 23, 0.0) !important;  /* темне скло */
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # Глобальний сайдбар (монета + час + автооновлення)
    render_sidebar()
    
    active_tab = st.session_state.get("active_tab", "data")

    if active_tab == "data":
        render_data_tab()
    elif active_tab == "features":
        render_features_tab()
    elif active_tab == "forecast":
        render_forecast_tab()
    elif active_tab == "debug":
        render_debugging_tab()

    render_footer()

if __name__ == "__main__":
    main()

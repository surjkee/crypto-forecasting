# ui/tabs/data_tab.py

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from data.db import load_ohlcv_hourly

@st.cache_data(show_spinner=False)
def load_market_data(coin_id: str, vs_currency: str) -> pd.DataFrame:
    """
    Завантажує ринкові дані з DuckDB для обраної монети та валюти.
    Кешується Streamlit-ом.
    """
    df = load_ohlcv_hourly(coin_id, vs_currency)
    return df


def render_data_tab():
    settings = get_settings()

    # --- Вибір валюти (поки лише одна, але можна розширити) ---
    vs_currency = settings.default_vs_currency

    # --- Вибір монети ---
    selected_coin_id = st.session_state.get("selected_coin_id", "bitcoin")
    selected_label = st.session_state.get("selected_coin_label", "Bitcoin (BTC)")

    st.markdown("""
<style>
        /* Remove blank space at top and bottom */ 
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
</style>
""", unsafe_allow_html=True)

    st.markdown(
    """
    <h1 style="text-align: center; margin-top: 0;">
        📊 Data
    </h1>
    """,
    unsafe_allow_html=True
)

    # --- Завантажуємо дані з DuckDB ---
    df = load_market_data(selected_coin_id, vs_currency)

    if df.empty:
        st.warning(
            "У базі немає даних для цієї монети. "
            "Спочатку запусти job для завантаження історії:\n\n"
            "`python -m jobs.fetch_history`"
        )
        return

    # Обмежуємо діапазон дат згідно з глобальним налаштуванням
    time_range_hours = st.session_state.get("time_range_hours")

    if time_range_hours is not None and not df.empty:
        cutoff_ts = df["ts"].max() - pd.Timedelta(hours=time_range_hours)
        df = df[df["ts"] >= cutoff_ts]

    # --- Базовий графік ціни ---
    fig_price = px.line(
        df,
        x="ts",
        y="price",
        title=f"Price {selected_label} - {vs_currency.upper()}",
    )
    fig_price.update_layout(
        xaxis_title="Time",
        yaxis_title=f"Price ({vs_currency.upper()})",
        height=500,
    )

    st.plotly_chart(fig_price, width="stretch")

    # --- Опціонально: додаткова інформація / таблиця ---
    with st.expander(f"Table Data for {selected_label}"):
        st.dataframe(
            df.sort_values("ts", ascending=False),
            width="stretch",
            height=400,
        )

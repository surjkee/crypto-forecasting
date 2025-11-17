# ui/tabs/data_tab.py

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from ui.constants import TRACKED_COINS


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

    st.sidebar.header("Налаштування даних")

    # --- Вибір валюти (поки лише одна, але можна розширити) ---
    vs_currency = settings.default_vs_currency
    st.sidebar.markdown(f"**Валюта відображення:** `{vs_currency}`")

    # --- Вибір монети ---
    st.sidebar.subheader("Монета")

    with st.sidebar:
        labels = [label for label, _ in TRACKED_COINS]
        ids = [cid for _, cid in TRACKED_COINS]

        default_index = ids.index("bitcoin") if "bitcoin" in ids else 0

        selected_label = st.selectbox(
            "Оберіть монету:",
            options=labels,
            index=default_index,
        )

        selected_coin_id = ids[labels.index(selected_label)]

    st.title("📊 Дані ринку криптовалют")

    st.caption(
        f"Поточні налаштування: `vs_currency={vs_currency}`, "
        f"`history_days_default={settings.history_days_default}`, "
        f"`history_interval={settings.history_interval}`"
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

    # --- Базовий графік ціни ---
    st.subheader("Графік ціни")

    fig_price = px.line(
        df,
        x="ts",
        y="price",
        title=f"Ціна {selected_label} ({vs_currency.upper()})",
    )
    fig_price.update_layout(
        xaxis_title="Час",
        yaxis_title=f"Ціна ({vs_currency.upper()})",
        height=500,
    )

    st.plotly_chart(fig_price, width="stretch")

    # --- Опціонально: додаткова інформація / таблиця ---
    with st.expander("Показати сирі дані (таблиця)"):
        st.dataframe(
            df.sort_values("ts", ascending=False),
            width="stretch",
            height=400,
        )

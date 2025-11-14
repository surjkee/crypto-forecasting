import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# -------------------------

TRACKED_COINS = [
    ("Bitcoin (BTC)", "bitcoin"),
    ("Ethereum (ETH)", "ethereum"),
    ("Solana (SOL)", "solana"),
    ("Binance Coin (BNB)", "binancecoin"),
    ("Ripple (XRP)", "ripple"),
]

import streamlit as st
import plotly.express as px
import pandas as pd

from features.transform import build_feature_frame

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from jobs.fetch_history import fetch_and_store_history

# --- Загальні налаштування сторінки ---
st.set_page_config(
    page_title="Crypto Forecasting - Data",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def load_market_data(
    coin_id: str,
    vs_currency: str,
) -> pd.DataFrame:
    """
    Читаємо дані з DuckDB.
    """
    df = load_ohlcv_hourly(coin_id=coin_id, vs_currency=vs_currency)
    return df

@st.cache_data(show_spinner=False)
def compute_features(
    coin_id: str,
    vs_currency: str,
) -> pd.DataFrame:
    """
    Обчислює технічні фічі поверх історичних даних.
    """
    df = load_market_data(coin_id, vs_currency)
    if df.empty:
        return df

    feat_df = build_feature_frame(df)
    return feat_df


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
            "Натисни кнопку **«Оновити історію з CoinGecko»** для завантаження."
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

    st.plotly_chart(fig_price, use_container_width=True)

    # --- Опціонально: додаткова інформація / таблиця ---
    with st.expander("Показати сирі дані (таблиця)"):
        st.dataframe(
            df.sort_values("ts", ascending=False),
            use_container_width=True,
            height=400,
        )

def render_features_tab():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    st.title("🧩 Features (технічні показники)")

    # Для узгодженості з вкладкою Data переоберемо монету так само:
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]

    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected_label = st.selectbox(
            "Оберіть монету:",
            options=labels,
            index=default_index,
            key="features_coin_select",
        )
        selected_coin_id = ids[labels.index(selected_label)]


    with col_info:
        st.caption(
            "Тут ми дивимося на похідні ознаки (features), "
            "які потім підуть у моделі LSTM / CNN-LSTM / Attention."
        )

    # Завантажуємо дані + фічі
    df_raw = load_market_data(selected_coin_id, vs_currency)
    if df_raw.empty:
        st.warning(
            "У базі немає даних для цієї монети. "
            "Перейдіть на вкладку **Data** і натисніть "
            "«Оновити історію з CoinGecko»."
        )
        return

    df_feat = compute_features(selected_coin_id, vs_currency)

    # --- 2x2 grid з графіками ---
    st.subheader(f"Графіки для {selected_label}")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1) Ціна + SMA
    with row1_col1:
        st.markdown("**Ціна + SMA**")
        fig_price = px.line(
            df_feat,
            x="ts",
            y=["price", "sma_5", "sma_20", "sma_50"],
            labels={"value": f"Ціна / SMA ({vs_currency.upper()})", "ts": "Час"},
        )
        fig_price.update_layout(height=350, legend_title_text="Серія")
        st.plotly_chart(fig_price, use_container_width=True)

    # 2) Returns
    with row1_col2:
        st.markdown("**Добові зміни (returns)**")
        fig_ret = px.line(
            df_feat,
            x="ts",
            y="return",
            labels={"return": "Return", "ts": "Час"},
        )
        fig_ret.update_layout(height=350)
        st.plotly_chart(fig_ret, use_container_width=True)

    # 3) Волатильність
    vol_col = f"volatility_24"
    with row2_col1:
        st.markdown("**Ковзна волатильність (24 точки)**")
        if vol_col in df_feat.columns:
            fig_vol = px.line(
                df_feat,
                x="ts",
                y=vol_col,
                labels={vol_col: "Volatility (σ)", "ts": "Час"},
            )
            fig_vol.update_layout(height=350)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Колонка волатильності ще не розрахована.")

    # 4) Обʼєм
    with row2_col2:
        st.markdown("**Обʼєм торгів (volume)**")
        fig_volm = px.line(
            df_feat,
            x="ts",
            y="volume",
            labels={"volume": "Обʼєм", "ts": "Час"},
        )
        fig_volm.update_layout(height=350)
        st.plotly_chart(fig_volm, use_container_width=True)

    with st.expander("Показати таблицю з фічами"):
        st.dataframe(
            df_feat.sort_values("ts", ascending=False),
            use_container_width=True,
            height=400,
        )


def main():
    tab_labels = ["Data", "Features"]  # далі додамо: "Models", "Monitoring"
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_data_tab()

    with tabs[1]:
        render_features_tab()


if __name__ == "__main__":
    main()

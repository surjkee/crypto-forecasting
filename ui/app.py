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

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from jobs.fetch_history import fetch_and_store_history
from ingest.coingecko_client import CoinGeckoClient


# --- Загальні налаштування сторінки ---
st.set_page_config(
    page_title="Crypto Forecasting - Data",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_top_coins_options(vs_currency: str) -> list[tuple[str, str]]:
    """
    Завантажуємо топові монети для селектора.
    Повертаємо список кортежів (label, coin_id).
    """
    client = CoinGeckoClient()
    coins = client.get_top_coins(vs_currency=vs_currency, per_page=25)

    options: list[tuple[str, str]] = []
    for c in coins:
        label = f"{c['name']} ({c['symbol'].upper()})"
        options.append((label, c["id"]))

    return options


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


def render_data_tab():
    settings = get_settings()

    st.sidebar.header("Налаштування даних")

    # --- Вибір валюти (поки лише одна, але можна розширити) ---
    vs_currency = settings.default_vs_currency
    st.sidebar.markdown(f"**Валюта відображення:** `{vs_currency}`")

    # --- Вибір монети ---
    st.sidebar.subheader("Монета")

    with st.sidebar:
        try:
            top_coins = get_top_coins_options(vs_currency)
        except Exception as e:
            st.error(f"Не вдалося завантажити список монет: {e}")
            top_coins = [("Bitcoin (BTC)", "bitcoin")]

        labels = [label for label, _ in top_coins]
        ids = [cid for _, cid in top_coins]

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

    # --- Кнопка оновлення історії з CoinGecko ---
    col_left, col_right = st.columns([1, 3])

    with col_left:
        if st.button("🔄 Оновити історію з CoinGecko"):
            with st.spinner("Завантажуємо дані з CoinGecko та зберігаємо в DuckDB..."):
                try:
                    fetch_and_store_history(
                        coin_id=selected_coin_id,
                        vs_currency=vs_currency,
                        days=settings.history_days_default,
                        interval=settings.history_interval,  # type: ignore[arg-type]
                    )
                    st.success("Готово! Дані оновлено.")
                    # Очищаємо кеш, щоб перезавантажити дані
                    load_market_data.clear()
                except Exception as e:
                    st.error(f"Помилка при оновленні даних: {e}")

    with col_right:
        st.info(
            "Ця вкладка працює як **шар даних**: "
            "CoinGecko → DuckDB → візуалізація. "
            "Пізніше тут з’являться додаткові фільтри по датах, інтервалах, тощо."
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


def main():
    # Тепер у нас буде структура з вкладками (поки тільки Data)
    tab_labels = ["Data"]  # пізніше додамо: "Features", "Models", "Monitoring"
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_data_tab()


if __name__ == "__main__":
    main()

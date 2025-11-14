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
from models.baseline import naive_constant_forecast

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

def render_debugging_tab():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    st.title("🛠 Debugging (backtest на 'вчора')")

    # Вибір монети (той самий TRACKED_COINS)
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]
    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected_label = st.selectbox(
            "Оберіть монету:",
            options=labels,
            index=default_index,
            key="debug_coin_select",
        )
        selected_coin_id = ids[labels.index(selected_label)]

    with col_info:
        st.caption(
            "Тут ми тестуємо просту baseline-модель (naive constant forecast), "
            "ніби ми знаходимося у 'вчорашній день', і дивимося, як вона "
            "прогнозує наступні 24 години."
        )

    # Завантажуємо дані з DuckDB
    df_raw = load_ohlcv_hourly(selected_coin_id, vs_currency)

    if df_raw.empty:
        st.warning(
            "У базі немає даних для цієї монети. "
            "Спочатку запусти job для завантаження історії:\n\n"
            "`python -m jobs.fetch_history`"
        )
        return

    # Нормалізуємо timestamps до цілої години
    df_raw = df_raw.copy()
    df_raw["ts_hour"] = df_raw["ts"].dt.floor("H")

    # Робимо 1 запис на годину (на випадок, якщо є дублікати)
    df_hourly = (
        df_raw.sort_values("ts_hour")
        .drop_duplicates(subset=["ts_hour"], keep="last")
        .reset_index(drop=True)
    )

    # Перевіряємо, що даних достатньо
    if len(df_hourly) < 24 * 3:
        st.warning(
            "Замало даних для адекватного backtest'у (потрібно хоча б 3 дні "
            "з погодинними даними). Спробуй завантажити більший інтервал історії."
        )
        return

    max_hour = df_hourly["ts_hour"].max()
    anchor_hour = max_hour - pd.Timedelta(hours=24)

    # Історія, доступна моделі до 'anchor_hour'
    df_history = df_hourly[df_hourly["ts_hour"] <= anchor_hour].copy()

    # Фактичні ціни на 24 години після 'anchor_hour'
    df_future_true = df_hourly[
        (df_hourly["ts_hour"] > anchor_hour)
        & (df_hourly["ts_hour"] <= anchor_hour + pd.Timedelta(hours=24))
    ].copy()

    if len(df_future_true) < 1:
        st.warning(
            "Не вдалося знайти дані після 'вчора' для побудови backtest'у. "
            "Можливо, історія ще не повна."
        )
        return

    # Готуємо історію для baseline-моделі:
    # ставимо ts = ts_hour, щоб timestamps були чітко погодинні
    hist_for_model = df_history.sort_values("ts_hour").copy()
    hist_for_model["ts"] = hist_for_model["ts_hour"]

    # Робимо baseline-прогноз на стільки, скільки маємо фактів (звичайно 24)
    try:
        df_forecast, _ = naive_constant_forecast(
            history=hist_for_model,
            horizon_hours=len(df_future_true),
        )
    except Exception as e:
        st.error(f"Помилка під час побудови прогнозу: {e}")
        return

    # Нормалізуємо час у прогнозі та обʼєднуємо по ts_hour
    df_forecast = df_forecast.copy()
    df_forecast["ts_hour"] = df_forecast["ts"].dt.floor("H")

    df_merged = pd.merge(
        df_future_true[["ts_hour", "price"]],
        df_forecast[["ts_hour", "y_pred"]],
        on="ts_hour",
        how="inner",
    )

    if df_merged.empty:
        st.warning(
            "Не вдалося зіставити фактичні та прогнозні значення по годинах. "
            "Перевір, чи дані мають погодинну частоту."
        )
        return

    # Рахуємо метрики
    y_true = df_merged["price"]
    y_pred = df_merged["y_pred"]

    mae = (y_true - y_pred).abs().mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5

    st.subheader("Метрики якості (baseline на 'вчорашній' добі)")
    st.write(
        f"**MAE:** {mae:.4f} {vs_currency.upper()}  \n"
        f"**RMSE:** {rmse:.4f} {vs_currency.upper()}"
    )

    # Готуємо дані для графіка (тільки погодинні ts_hour)
    ctx_hours = 24  # скільки годин історії показати перед anchor
    ts_min_plot = anchor_hour - pd.Timedelta(hours=ctx_hours)

    df_plot_hist = df_hourly[
        (df_hourly["ts_hour"] >= ts_min_plot) & (df_hourly["ts_hour"] <= anchor_hour)
    ].copy()
    df_plot_hist["series"] = "Історія (факт)"
    df_plot_hist["ts_plot"] = df_plot_hist["ts_hour"]

    df_plot_future = df_future_true.copy()
    df_plot_future["series"] = "Майбутнє (факт)"
    df_plot_future["ts_plot"] = df_plot_future["ts_hour"]

    df_plot_forecast = df_forecast.copy()
    df_plot_forecast["series"] = "Прогноз (baseline)"
    df_plot_forecast["ts_plot"] = df_plot_forecast["ts_hour"]
    df_plot_forecast = df_plot_forecast.rename(columns={"y_pred": "price"})

    df_plot_actual = pd.concat(
        [
            df_plot_hist[["ts_plot", "price", "series"]],
            df_plot_future[["ts_plot", "price", "series"]],
        ],
        ignore_index=True,
    )

    df_plot_all = pd.concat(
        [
            df_plot_actual,
            df_plot_forecast[["ts_plot", "price", "series"]],
        ],
        ignore_index=True,
    )

    st.subheader("Графік: історія, майбутнє та прогноз (baseline)")

    fig = px.line(
        df_plot_all,
        x="ts_plot",
        y="price",
        color="series",
        labels={
            "ts_plot": "Час (погодинно)",
            "price": f"Ціна ({vs_currency.upper()})",
            "series": "Серія",
        },
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Таблиця фактичних та прогнозованих значень (24 години після 'вчора')"):
        st.dataframe(
            df_merged.sort_values("ts_hour"),
            use_container_width=True,
            height=400,
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

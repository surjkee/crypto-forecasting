import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from data.db import load_ohlcv_hourly, load_hourly_forecasts
from ui.constants import TRACKED_COINS


def render_forecast_tab():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    st.title("🔮 Forecast (t+1)")

    # --- Вибір монети ---
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]
    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected_label = st.selectbox(
            "Оберіть монету:",
            options=labels,
            index=default_index,
            key="forecast_coin_select",
        )
        selected_coin_id = ids[labels.index(selected_label)]

    with col_info:
        st.caption(
            "Тут відображаються останні фактичні значення ціни та "
            "збережені в DuckDB прогнози LSTM на наступну годину (t+1). "
            "Прогнози генеруються job'ом `python -m jobs.run_forecast`."
        )

    # --- Завантажуємо фактичні дані ---
    df_price = load_ohlcv_hourly(selected_coin_id, vs_currency)

    if df_price.empty:
        st.warning(
            "У базі немає фактичних даних для цієї монети. "
            "Спочатку запусти:\n\n"
            "`python -m jobs.fetch_history`"
        )
        return

    # --- Завантажуємо прогнози ---
    df_fc = load_hourly_forecasts(
        selected_coin_id,
        vs_currency,
        limit=200,
        model="lstm_v0.4",
    )

    if df_fc.empty:
        st.warning(
            "У таблиці forecast_hourly немає прогнозів для цієї монети.\n\n"
            "Спочатку натренуй модель та запусти:\n\n"
            "`python -m jobs.train_lstm_all`\n"
            "`python -m jobs.run_forecast`"
        )
        return

    # --- Останній факт та останній прогноз ---
    last_fact = df_price.iloc[-1]
    last_fc = df_fc.iloc[-1]

    last_price = float(last_fact["price"])
    last_price_ts = last_fact["ts"]

    y_pred = float(last_fc["y_pred"])
    ts_forecast = last_fc["ts_forecast"]
    ts_anchor = last_fc["ts_anchor"]

    delta_abs = y_pred - last_price
    delta_pct = (delta_abs / last_price) * 100 if last_price != 0 else 0.0

    col_fact, col_fc = st.columns(2)

    with col_fact:
        st.subheader("Останнє фактичне значення")
        st.write(f"**Час:** {last_price_ts}")
        st.write(f"**Ціна:** {last_price:,.2f} {vs_currency.upper()}")

    with col_fc:
        st.subheader("Останній прогноз t+1 (LSTM)")
        st.write(f"**ts_anchor (останній відомий факт):** {ts_anchor}")
        st.write(f"**ts_forecast (t+1):** {ts_forecast}")
        st.write(f"**Прогноз:** {y_pred:,.2f} {vs_currency.upper()}")
        st.write(
            f"**Δ до останнього факту:** "
            f"{delta_abs:+.2f} {vs_currency.upper()} "
            f"({delta_pct:+.2f}%)"
        )

    st.markdown("---")

    # --- Графік: фактична ціна + прогнози ---
    st.subheader("Графік: факт та прогнози t+1")

    # беремо фактичні дані за останні 72 години
    window_hours = 72
    ts_max = df_price["ts"].max()
    ts_min = ts_max - pd.Timedelta(hours=window_hours)

    df_price_plot = df_price[df_price["ts"] >= ts_min].copy()
    df_price_plot["series"] = "Факт"
    df_price_plot["ts_plot"] = df_price_plot["ts"]

    # фільтруємо прогнози у тому ж вікні (або ширше, якщо хочеш)
    df_fc_plot = df_fc[df_fc["ts_forecast"] >= ts_min].copy()
    if not df_fc_plot.empty:
        df_fc_plot = df_fc_plot.copy()
        df_fc_plot["series"] = "Прогноз t+1"
        df_fc_plot["ts_plot"] = df_fc_plot["ts_forecast"]
        df_fc_plot = df_fc_plot.rename(columns={"y_pred": "price"})

        df_plot_all = pd.concat(
            [
                df_fc_plot[["ts_plot", "price", "series"]],
                df_price_plot[["ts_plot", "price", "series"]],
            ],
            ignore_index=True,
        )
    else:
        df_plot_all = df_price_plot[["ts_plot", "price", "series"]]

    fig = px.line(
        df_plot_all,
        x="ts_plot",
        y="price",
        color="series",
        labels={
            "ts_plot": "Час",
            "price": f"Ціна ({vs_currency.upper()})",
            "series": "Серія",
        },
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")

    # --- Таблиця прогнозів ---
    with st.expander("Показати таблицю останніх прогнозів"):
        st.dataframe(
            df_fc.sort_values("ts_forecast", ascending=False),
            width="stretch",
            height=400,
        )

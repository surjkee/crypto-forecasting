import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from data.db import load_ohlcv_hourly, load_hourly_forecasts
from ui.constants import TRACKED_COINS




def render_forecast_tab():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

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
        🔮 Forecast
    </h1>
    """,
    unsafe_allow_html=True
)

    # --- Використовуємо глобально обрану монету ---
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]

    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0
    default_label = labels[default_index]
    default_id = ids[default_index]

    selected_coin_id = st.session_state.get("selected_coin_id", default_id)
    selected_label = st.session_state.get("selected_coin_label", default_label)

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
        model="lstm_v1.0",
    )

    df_fc_gru = load_hourly_forecasts(
        selected_coin_id,
        vs_currency,
        limit=200,
        model="gru_v1.0",
    )


    if df_fc.empty:
        st.warning(
            "У таблиці forecast_hourly немає прогнозів для цієї монети.\n\n"
            "Спочатку натренуй модель та запусти:\n\n"
            "`python -m jobs.train_lstm_all`\n"
            "`python -m jobs.run_forecast`"
        )
        return

    # --- Обмеження по часовому діапазону ---
    time_range_hours = st.session_state.get("time_range_hours")

    if time_range_hours is not None:
        # Для цін
        if not df_price.empty:
            cutoff_price = df_price["ts"].max() - pd.Timedelta(hours=time_range_hours)
            df_price = df_price[df_price["ts"] >= cutoff_price]

        # Для LSTM-прогнозів
        if not df_fc.empty:
            cutoff_fc = df_fc["ts_forecast"].max() - pd.Timedelta(hours=time_range_hours)
            df_fc = df_fc[df_fc["ts_forecast"] >= cutoff_fc]

        # Для GRU-прогнозів (якщо є)
        if df_fc_gru is not None and not df_fc_gru.empty:
            cutoff_fc_gru = df_fc_gru["ts_forecast"].max() - pd.Timedelta(hours=time_range_hours)
            df_fc_gru = df_fc_gru[df_fc_gru["ts_forecast"] >= cutoff_fc_gru]


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



    # --- Графік: фактична ціна + прогнози t+1 ---
    # Беремо глобальний часовий діапазон із sidebar
    time_range_hours = st.session_state.get("time_range_hours")

    if df_price.empty:
        st.info("Немає цінових даних для побудови графіка")
        return

    ts_max = df_price["ts"].max()

    if time_range_hours is None:
        # "Увесь період" — показуємо всі доступні дані
        ts_min = df_price["ts"].min()
    else:
        ts_min = ts_max - pd.Timedelta(hours=time_range_hours)

    # --- Фактична ціна ---
    df_price_plot = df_price[df_price["ts"] >= ts_min].copy()
    df_price_plot["series"] = "Real"
    df_price_plot["ts_plot"] = df_price_plot["ts"]

    df_plot_all = df_price_plot[["ts_plot", "price", "series"]].copy()

    # LSTM
    df_fc_plot = df_fc[df_fc["ts_forecast"] >= ts_min].copy()
    if not df_fc_plot.empty:
        df_fc_plot["series"] = "LSTM"
        df_fc_plot["ts_plot"] = df_fc_plot["ts_forecast"]
        df_fc_plot = df_fc_plot.rename(columns={"y_pred": "price"})
        df_plot_all = pd.concat(
            [df_fc_plot[["ts_plot", "price", "series"]], df_plot_all],
            ignore_index=True,
        )

    # GRU
    df_fc_gru_plot = df_fc_gru[df_fc_gru["ts_forecast"] >= ts_min].copy()
    if not df_fc_gru_plot.empty:
        df_fc_gru_plot["series"] = "GRU"
        df_fc_gru_plot["ts_plot"] = df_fc_gru_plot["ts_forecast"]
        df_fc_gru_plot = df_fc_gru_plot.rename(columns={"y_pred": "price"})
        df_plot_all = pd.concat(
            [df_fc_gru_plot[["ts_plot", "price", "series"]], df_plot_all],
            ignore_index=True,
        )

    # Малюємо
    fig = px.line(
        df_plot_all,
        x="ts_plot",
        y="price",
        color="series",
        title=f"Price {selected_label} - {vs_currency.upper()}",
        labels={
            "ts_plot": "Time",
            "price": f"Price ({vs_currency.upper()})",
            "series": "Series",
        },
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")
    
    with st.expander("Show extended metrics"):
        st.subheader("Last Real Price")
        st.write(f"**Time:** {last_price_ts}")
        st.write(f"**Price:** {last_price:,.2f} {vs_currency.upper()}")

        st.subheader("Last Prediction LSTM")
        st.write(f"**ts_anchor (Last knows Price):** {ts_anchor}")
        st.write(f"**ts_forecast (t+1):** {ts_forecast}")
        st.write(f"**Prediction:** {y_pred:,.2f} {vs_currency.upper()}")
        st.write(
            f"**Δ to last price:** "
            f"{delta_abs:+.2f} {vs_currency.upper()} "
            f"({delta_pct:+.2f}%)"
            )
        if not df_fc_gru.empty:
            last_gru = df_fc_gru.iloc[-1]
            y_pred_gru = float(last_gru["y_pred"])
            ts_forecast_gru = last_gru["ts_forecast"]

            st.subheader("Last Prediction GRU")
            st.write(f"**ts_forecast (t+1):** {ts_forecast_gru}")
            st.write(f"**Forecast:** {y_pred_gru:,.2f} {vs_currency.upper()}")

            st.markdown("---")

    # --- Таблиця прогнозів ---
    with st.expander("Forecast Logs LSTM"):
        if df_fc.empty:
            st.info("No LSTM Forecasts for this coin")
        else:
            st.dataframe(
                df_fc.sort_values("ts_forecast", ascending=False),
                width="stretch",
                height=400,
            )
    with st.expander("Forecast Logs GRU"):
        if df_fc_gru.empty:
            st.info("No GRU Forecasts for this coin")
        else:
            st.dataframe(
                df_fc_gru.sort_values("ts_forecast", ascending=False),
                width="stretch",
                height=400,
            )


# ui/tabs/debugging.py

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from models.baseline import naive_constant_forecast
from ui.constants import TRACKED_COINS


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
    df_raw["ts_hour"] = df_raw["ts"].dt.floor("h")

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
    df_forecast["ts_hour"] = df_forecast["ts"].dt.floor("h")

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

    st.plotly_chart(fig, width="stretch")

    with st.expander("Таблиця фактичних та прогнозованих значень (24 години після 'вчора')"):
        st.dataframe(
            df_merged.sort_values("ts_hour"),
            width="stretch",
            height=400,
        )

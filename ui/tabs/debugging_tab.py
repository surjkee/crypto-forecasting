# ui/tabs/debugging.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from features.transform import build_feature_frame

from models.baseline import naive_constant_forecast
from models.lstm.inference import load_lstm_checkpoint
from models.lstm.train import _inverse_scale_target
from models.gru.inference import load_gru_checkpoint
from models.gru.config import GRUConfig


from ui.constants import TRACKED_COINS




def render_debugging_tab():
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
        🛠 Debugging
    </h1>
    """,
    unsafe_allow_html=True
)
    # Використовуємо глобально обрану монету + локальний вибір моделі
    labels = [label for label, _ in TRACKED_COINS]
    ids = [cid for _, cid in TRACKED_COINS]
    default_index = ids.index("bitcoin") if "bitcoin" in ids else 0
    default_label = labels[default_index]
    default_id = ids[default_index]

    selected_coin_id = st.session_state.get("selected_coin_id", default_id)
    selected_label = st.session_state.get("selected_coin_label", default_label)


    model_choice = st.radio(
        "Модель:",
        options=["Baseline", "LSTM", "GRU"],
        horizontal=True,
        key="debug_model_choice",
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

    # Один запис на годину
    df_hourly = (
        df_raw.sort_values("ts_hour")
        .drop_duplicates(subset=["ts_hour"], keep="last")
        .reset_index(drop=True)
    )

    if len(df_hourly) < 24 * 3:
        st.warning(
            "Замало даних для адекватного backtest'у (потрібно хоча б 3 дні "
            "з погодинними даними). Спробуй завантажити більший інтервал історії."
        )
        return

    max_hour = df_hourly["ts_hour"].max()
    anchor_hour = max_hour - pd.Timedelta(hours=24)

    # Історія до anchor_hour (включно)
    df_history = df_hourly[df_hourly["ts_hour"] <= anchor_hour].copy()

    # Факт на 24 години після anchor_hour
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

    # ---------- BASELINE ВАРІАНТ (як був) ----------
    if model_choice == "Baseline":
        hist_for_model = df_history.sort_values("ts_hour").copy()
        hist_for_model["ts"] = hist_for_model["ts_hour"]

        try:
            df_forecast, _ = naive_constant_forecast(
                history=hist_for_model,
                horizon_hours=len(df_future_true),
            )
        except Exception as e:
            st.error(f"Помилка під час побудови baseline-прогнозу: {e}")
            return

        model_name = "Baseline (naive constant)"

    # ---------- LSTM: teacher forcing 1-step backtest на 'вчора' ----------
    elif model_choice == "LSTM":
        try:
            # вантажимо модель + scaler + список фіч
            model, scaler, feature_cols, target_col_idx, cfg = load_lstm_checkpoint(
                selected_coin_id, vs_currency
            )
        except FileNotFoundError:
            st.error(
                "Не знайдено збережену LSTM-модель для цієї монети.\n\n"
                "Спочатку натренуй її командою:\n\n"
                f"`python -m jobs.train_lstm --coin_id {selected_coin_id}`"
            )
            return
        except Exception as e:
            st.error(f"Помилка при завантаженні LSTM-моделі: {e}")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Для фіч: використовуємо ts_hour як еталонний час
        df_feat_input = df_hourly.copy()
        df_feat_input["ts"] = df_feat_input["ts_hour"]

        df_feat = build_feature_frame(df_feat_input)

        # df_model: тільки ts + ті фічі, на яких навчалась модель
        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            st.error(
                "В поточному фреймі фіч не вистачає колонок, "
                "на яких навчалась модель:\n\n"
                + ", ".join(missing)
            )
            return

        df_model = df_feat[["ts"] + feature_cols].copy()
        df_model = df_model.dropna(subset=feature_cols).reset_index(drop=True)

        if len(df_model) <= cfg.window_size + 1:
            st.error(
                f"Замало даних для побудови вікон: {len(df_model)} рядків після dropna, "
                f"потрібно хоча б window_size={cfg.window_size}."
            )
            return

        # Масштабуємо всі фічі тим самим scaler'ом, що був на train
        values = df_model[feature_cols].values.astype(np.float32)
        scaled_all = scaler.transform(values)

        # Створюємо мапу ts -> індекс у df_model
        ts_series = df_model["ts"]
        ts_to_idx = {ts: idx for idx, ts in enumerate(ts_series)}

        # Цільові точки прогнозу: (anchor, anchor + 24h], з кроком 1 година
        ts_start = anchor_hour + pd.Timedelta(hours=1)   # anchor+1
        ts_end = anchor_hour + pd.Timedelta(hours=len(df_future_true))

        target_ts_list = []
        target_indices = []

        for ts in ts_series:
            if ts_start <= ts <= ts_end:
                idx = ts_to_idx[ts]
                if idx >= cfg.window_size:
                    target_ts_list.append(ts)
                    target_indices.append(idx)

        if not target_indices:
            st.error(
                "Не вдалося знайти достатньо точок для побудови вікон LSTM "
                "на 'вчорашній' добі. Можливо, замало даних після dropna."
            )
            return

        # One-step ahead прогнози з teacher forcing:
        # для кожного t_pred беремо реальне вікно [t_pred-window_size .. t_pred-1]
        preds_scaled = []

        with torch.no_grad():
            for idx in target_indices:
                window_scaled = scaled_all[idx - cfg.window_size : idx, :]  # (W, F)
                x = torch.tensor(
                    window_scaled[None, :, :],
                    dtype=torch.float32,
                    device=device,
                )
                y_scaled = model(x).cpu().numpy()[0, 0]
                preds_scaled.append(y_scaled)

        preds_scaled_arr = np.array(preds_scaled, dtype=np.float32)

        # Інверсія масштабу для таргета
        y_pred = _inverse_scale_target(
            scaler,
            feature_cols,
            target_col_idx,
            preds_scaled_arr,
        )

        # Реальні ціни (таргет) на ці самі моменти часу
        y_true = (
            df_model.loc[target_indices, cfg.target_col]
            .to_numpy(dtype=float)
        )

        df_forecast = pd.DataFrame(
            {
                "ts": target_ts_list,
                "y_pred": y_pred,
            }
        )

        # Для коректного мерджу з df_future_true працюємо через ts_hour
        df_forecast["ts_hour"] = df_forecast["ts"].dt.floor("h")

        model_name = "LSTM"

    # ---------- GRU: teacher forcing 1-step backtest ----------
    elif model_choice == "GRU":
        try:
            model, scaler, feature_cols, target_col_idx, cfg = load_gru_checkpoint(
                selected_coin_id, vs_currency
            )
        except FileNotFoundError:
            st.error(
                "Не знайдено збережену GRU-модель для цієї монети.\n\n"
                "Спочатку натренуй її командою:\n\n"
                f"`python -m jobs.train_gru --coin_id {selected_coin_id}`"
            )
            return
        except Exception as e:
            st.error(f"Помилка при завантаженні GRU-моделі: {e}")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        df_feat_input = df_hourly.copy()
        df_feat_input["ts"] = df_feat_input["ts_hour"]
        df_feat = build_feature_frame(df_feat_input)

        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            st.error(
                "В поточному фреймі фіч не вистачає колонок, "
                "на яких навчалась GRU:\n\n" + ", ".join(missing)
            )
            return

        df_model = df_feat[["ts"] + feature_cols].dropna().reset_index(drop=True)

        values = df_model[feature_cols].values.astype(np.float32)
        scaled_all = scaler.transform(values)

        ts_series = df_model["ts"]
        ts_to_idx = {ts: idx for idx, ts in enumerate(ts_series)}

        ts_start = anchor_hour + pd.Timedelta(hours=1)
        ts_end = anchor_hour + pd.Timedelta(hours=len(df_future_true))

        target_ts_list = []
        target_indices = []

        for ts in ts_series:
            if ts_start <= ts <= ts_end:
                idx = ts_to_idx[ts]
                if idx >= cfg.window_size:
                    target_ts_list.append(ts)
                    target_indices.append(idx)

        preds_scaled = []

        with torch.no_grad():
            for idx in target_indices:
                window_scaled = scaled_all[idx - cfg.window_size : idx, :]
                x = torch.tensor(
                    window_scaled[None, :, :], dtype=torch.float32, device=device
                )
                y_scaled = model(x).cpu().numpy()[0, 0]
                preds_scaled.append(y_scaled)

        preds_scaled_arr = np.array(preds_scaled, dtype=np.float32)

        y_pred = _inverse_scale_target(
            scaler, feature_cols, target_col_idx, preds_scaled_arr
        )

        y_true = df_model.loc[target_indices, cfg.target_col].to_numpy(float)

        df_forecast = pd.DataFrame(
            {"ts": target_ts_list, "y_pred": y_pred}
        )
        df_forecast["ts_hour"] = df_forecast["ts"].dt.floor("h")

        model_name = "GRU"


    # ---------- Спільна частина: метрики, графік, таблиця ----------

    # Нормалізуємо час у прогнозі та обʼєднуємо по ts_hour
    if model_choice == "Baseline":
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

    # Метрики
    y_true_merge = df_merged["price"]
    y_pred_merge = df_merged["y_pred"]

    mae = (y_true_merge - y_pred_merge).abs().mean()
    rmse = ((y_true_merge - y_pred_merge) ** 2).mean() ** 0.5

    st.subheader(f"Metrics for {model_name} on {selected_label}")
    st.write(
        f"**MAE:** {mae:.4f} {vs_currency.upper()}  \n"
        f"**RMSE:** {rmse:.4f} {vs_currency.upper()}"
    )

    # Графік
    ctx_hours = 24
    ts_min_plot = anchor_hour - pd.Timedelta(hours=ctx_hours)

    df_plot_hist = df_hourly[
        (df_hourly["ts_hour"] >= ts_min_plot) & (df_hourly["ts_hour"] <= anchor_hour)
    ].copy()
    df_plot_hist["series"] = "History (Real)"
    df_plot_hist["ts_plot"] = df_plot_hist["ts_hour"]

    df_plot_future = df_future_true.copy()
    df_plot_future["series"] = "Future (Real)"
    df_plot_future["ts_plot"] = df_plot_future["ts_hour"]

    df_plot_forecast = df_forecast.copy()
    df_plot_forecast["series"] = f"Forecast ({model_name})"
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

    st.subheader(f"History and yesterday's forecast ({model_name})")

    fig = px.line(
        df_plot_all,
        x="ts_plot",
        y="price",
        color="series",
        labels={
            "ts_plot": "Time",
            "price": f"Price ({vs_currency.upper()})",
            "series": "Series",
        },
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")

    with st.expander("Yesterday's forecast table"):
        st.dataframe(
            df_merged.sort_values("ts_hour"),
            width="stretch",
            height=400,
        )



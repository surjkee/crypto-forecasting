# models/lstm/inference.py

from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from config.settings import get_settings

from data.db import load_ohlcv_hourly
from data.db import store_hourly_forecast
from data.db import ensure_forecast_continuity

from features.transform import build_feature_frame

from models.lstm.config import LSTMConfig
from models.lstm.model import LSTMForecastModel
from models.lstm.train import _inverse_scale_target


def _to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lstm_checkpoint(coin_id: str, vs_currency: str | None = None):
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency
    cfg = LSTMConfig()
    path = cfg.artifact_path(coin_id, vs_currency)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    feature_cols = checkpoint["feature_cols"]
    target_col_idx = checkpoint["target_col_idx"]
    scaler = checkpoint["scaler"]
    cfg_loaded = cfg  # можна пізніше зчитати з checkpoint["config"]

    model = LSTMForecastModel(
        input_size=len(feature_cols),
        hidden_size=cfg_loaded.hidden_size,
        num_layers=cfg_loaded.num_layers,
        dropout=cfg_loaded.dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, scaler, feature_cols, target_col_idx, cfg_loaded

def forecast_from_history(
    coin_id: str,
    history_df: pd.DataFrame,
    vs_currency: str | None = None,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Рекурсивний прогноз LSTM на horizon_hours кроків вперед,
    використовуючи довільну історію (history_df) як "світ до тепер".

    Використовується в Debugging-режимі, коли ми "відрізаємо" дані на рівні 'вчора'.
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency

    model, scaler, feature_cols, target_col_idx, cfg = load_lstm_checkpoint(
        coin_id, vs_currency
    )
    device = _to_device()
    model.to(device)

    # Будуємо фічі так само, як при тренуванні
    df_feat = build_feature_frame(history_df)
    df_model = df_feat[["ts"] + feature_cols].copy()

    # Чистимо NaN як у train-пайплайні
    df_model = df_model.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df_model) < cfg.window_size:
        raise RuntimeError(
            f"Замало даних в history_df після dropna: {len(df_model)} рядків, "
            f"потрібно хоча б window_size={cfg.window_size}."
        )

    values = df_model[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(values)

    window = scaled[-cfg.window_size :, :]  # останнє вікно
    last_ts = df_model["ts"].max()

    preds_scaled = []
    ts_future = []

    with torch.no_grad():
        for step in range(horizon_hours):
            x = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
            y_scaled = model(x).cpu().numpy()[0, 0]

            preds_scaled.append(y_scaled)

            # оновлюємо вікно: зсуваємо і додаємо новий рядок
            new_row = np.zeros((window.shape[1],), dtype=np.float32)
            new_row[target_col_idx] = y_scaled

            window = np.vstack([window[1:], new_row])
            ts_future.append(last_ts + timedelta(hours=step + 1))

    preds_scaled_arr = np.array(preds_scaled, dtype=np.float32)

    from models.lstm.train import _inverse_scale_target

    y_pred = _inverse_scale_target(
        scaler, feature_cols, target_col_idx, preds_scaled_arr
    )

    df_forecast = pd.DataFrame(
        {
            "ts": ts_future,
            "y_pred": y_pred,
        }
    )
    return df_forecast


def forecast_next_t1_and_store(
    coin_id: str,
    vs_currency: str | None = None,
    model_name: str = "lstm_v1.0",
):
    """
    1) Завантажує останні дані (OHLCV)
    2) Будує фічі
    3) Беремо останнє історичне вікно (window_size)
    4) Проганяємо через LSTM -> отримуємо y_pred(t+1)
    5) Зберігаємо результат у forecast_hourly
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency

    ensure_forecast_continuity(coin_id, vs_currency, model_name)

    # 1) Завантажуємо сирі дані
    df_raw = load_ohlcv_hourly(coin_id, vs_currency)
    if df_raw.empty:
        raise RuntimeError(
            f"Немає OHLCV для {coin_id} ({vs_currency}). Спершу запусти jobs.fetch_history."
        )

    # 2) Готуємо фічі
    df_feat = build_feature_frame(df_raw)
    cfg = LSTMConfig()

    # Колонки фіч — усі, крім ts
    feature_cols = [c for c in df_feat.columns if c not in ("ts",)]
    if cfg.target_col not in feature_cols:
        raise RuntimeError("В df_feat немає 'price' як таргета!")

    df_model = df_feat[["ts"] + feature_cols].dropna(subset=feature_cols).reset_index(drop=True)
    if df_model.empty:
        raise RuntimeError("Після dropna df_model порожній.")

    # 3) Завантажуємо модель + scaler + параметри
    try:
        model, scaler, trained_feature_cols, target_col_idx, cfg_loaded = load_lstm_checkpoint(
            coin_id, vs_currency
        )
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити LSTM-модель: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Перевіряємо фічі (в тренованій моделі порядок може бути іншим)
    missing = [c for c in trained_feature_cols if c not in df_model.columns]
    if missing:
        raise RuntimeError(
            "У df_model не вистачає фіч, на яких тренувалась модель:\n" + ", ".join(missing)
        )

    # Перебудовуємо df_model відповідно до порядку фіч моделі
    df_model = df_model[["ts"] + trained_feature_cols]

    # Масштабуємо всі фічі (але без fit!)
    values = df_model[trained_feature_cols].values.astype(np.float32)
    scaled = scaler.transform(values)

    # 4) Витягуємо останнє вікно
    if len(df_model) < cfg.window_size:
        raise RuntimeError(
            f"Замало даних: {len(df_model)} рядків, потрібно {cfg.window_size}."
        )

    window_scaled = scaled[-cfg.window_size :, :]  # (W, F)
    x = torch.tensor(window_scaled[None, :, :], dtype=torch.float32, device=device)

    # Прогноз (в scaled-space)
    with torch.no_grad():
        y_scaled = model(x).cpu().numpy()[0, 0]

    # 5) Інверсія масштабу
    y_pred = _inverse_scale_target(
        scaler,
        trained_feature_cols,
        target_col_idx,
        np.array([y_scaled]),
    )[0]

    # Часові мітки
    ts_anchor = df_model["ts"].max()
    ts_forecast = ts_anchor + timedelta(hours=1)

    # 6) Зберігаємо в БД
    store_hourly_forecast(
        coin_id=coin_id,
        vs_currency=vs_currency,
        ts_anchor=ts_anchor,
        ts_forecast=ts_forecast,
        model=model_name,
        y_pred=float(y_pred),
        is_backfilled=False,
    )

    return {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "ts_anchor": ts_anchor,
        "ts_forecast": ts_forecast,
        "y_pred": y_pred,
    }
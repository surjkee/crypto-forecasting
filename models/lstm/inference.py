# models/lstm/inference.py

from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from features.transform import build_feature_frame
from models.lstm.config import LSTMConfig
from models.lstm.model import LSTMForecastModel


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


def forecast_next_horizon(
    coin_id: str,
    vs_currency: str | None = None,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Рекурсивний прогноз на horizon_hours кроків вперед від останньої точки.
    Поки простий варіант без складного перерахунку фіч — baseline для UI.
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency

    model, scaler, feature_cols, target_col_idx, cfg = load_lstm_checkpoint(
        coin_id, vs_currency
    )
    device = _to_device()
    model.to(device)

    # Беремо останні дані з DuckDB
    df_raw = load_ohlcv_hourly(coin_id, vs_currency)
    if df_raw.empty:
        raise RuntimeError("Немає даних для прогнозу.")

    df_feat = build_feature_frame(df_raw)
    df_model = df_feat[["ts"] + feature_cols].copy()

    values = df_model[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(values)

    if len(scaled) < cfg.window_size:
        raise RuntimeError("Замало даних для побудови вікна заданого розміру.")

    window = scaled[-cfg.window_size :, :]  # (window_size, num_features)

    last_ts = df_model["ts"].max()

    preds_scaled = []
    ts_future = []

    with torch.no_grad():
        for step in range(horizon_hours):
            x = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
            y_scaled = model(x).cpu().numpy()[0, 0]  # scalar

            preds_scaled.append(y_scaled)

            # оновлюємо вікно: зсуваємо і вставляємо новий "рядок"
            # тут ми вставляємо тільки таргет у свою колонку, решту 0 (бо поки не рахуємо фічі)
            new_row = np.zeros((window.shape[1],), dtype=np.float32)
            new_row[target_col_idx] = y_scaled

            window = np.vstack([window[1:], new_row])

            ts_future.append(last_ts + timedelta(hours=step + 1))

    preds_scaled_arr = np.array(preds_scaled, dtype=np.float32)

    # інверсія масштабу лише для ціни
    from models.lstm.train import _inverse_scale_target  # невеликий reuse

    y_pred = _inverse_scale_target(scaler, feature_cols, target_col_idx, preds_scaled_arr)

    df_forecast = pd.DataFrame(
        {
            "ts": ts_future,
            "y_pred": y_pred,
        }
    )
    return df_forecast

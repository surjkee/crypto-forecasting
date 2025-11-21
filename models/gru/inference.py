# models/gru/inference.py

import os
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from config.settings import get_settings
from data.db import load_ohlcv_hourly, store_hourly_forecast, ensure_forecast_continuity
from features.transform import build_feature_frame
from models.gru.config import GRUConfig
from models.gru.model import GRUForecastModel


def load_gru_checkpoint(
    coin_id: str,
    vs_currency: str,
    device: torch.device | None = None,
):
    """
    Завантажує збережену GRU-модель + scaler + фічі з чекпоінта.
    Повертає:
    - model (на потрібному device)
    - scaler
    - feature_cols (list[str])
    - target_col_idx (int)
    - cfg (GRUConfig)
    """
    device = device or torch.device("cpu")

    cfg = GRUConfig()
    artifact_path: Path = cfg.get_artifact_path(coin_id, vs_currency)

    checkpoint = torch.load(artifact_path, map_location=device, weights_only=False)

    # Витягуємо конфіг, якщо він є в чекпоінті
    cfg_dict = checkpoint.get("config", {})
    if cfg_dict:
        cfg = GRUConfig(**cfg_dict)

    feature_cols = checkpoint["feature_cols"]
    target_col_idx = int(checkpoint["target_col_idx"])
    scaler = checkpoint["scaler"]

    model = GRUForecastModel(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, scaler, feature_cols, target_col_idx, cfg

def _inverse_scale_target(
    scaler: MinMaxScaler,
    feature_cols: list[str],
    target_col_idx: int,
    y_scaled: np.ndarray,
) -> np.ndarray:
    """
    Той самий трюк, що й у LSTM:
    кладемо y_scaled у колонку target_col, інші фічі = 0,
    ганяємо через inverse_transform і дістаємо лише target.
    """
    dummy = np.zeros((len(y_scaled), len(feature_cols)), dtype=np.float32)
    dummy[:, target_col_idx] = y_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_idx]


def forecast_next_t1_and_store(
    coin_id: str,
    vs_currency: str | None = None,
    model_name: str = "gru_v1.0",
) -> Dict[str, Any]:
    """
    t+1-прогноз для GRU з записом у forecast_hourly.

    1) тягнемо останні OHLCV з DuckDB
    2) рахуємо фічі як у train_gru
    3) будуємо останнє history-вікно, ганяємо через GRU
    4) inverse-scale і запис у forecast_hourly з model = model_name
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency

    ensure_forecast_continuity(coin_id, vs_currency, model_name)

    # 1. Сирі дані
    df_raw = load_ohlcv_hourly(coin_id, vs_currency)
    if df_raw.empty:
        raise RuntimeError(
            f"Немає даних OHLCV для {coin_id} ({vs_currency}) у DuckDB. "
            "Спочатку запусти jobs.fetch_history."
        )

    # 2. Фічі як у тренуванні
    df_feat = build_feature_frame(df_raw)
    cfg = GRUConfig()

    # беремо тільки числові колонки (price, volume, sma, volatility, returns, etc.)
    num_cols = df_feat.select_dtypes(include=["number"]).columns.tolist()
    # ts нам для моделі не потрібен як фіча
    feature_cols = [c for c in num_cols if c != "ts"]

    if cfg.target_col not in feature_cols:
        raise RuntimeError(
            f"У build_feature_frame немає числової колонки '{cfg.target_col}' як таргета."
        )

    target_col_idx = feature_cols.index(cfg.target_col)

    df_model = (
        df_feat[["ts"] + feature_cols]
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
    )
    if len(df_model) <= cfg.window_size:
        raise RuntimeError(
            f"Замало даних для прогнозу: {len(df_model)} рядків, "
            f"window_size={cfg.window_size}"
        )

    ts_anchor = df_model["ts"].max()
    ts_forecast = ts_anchor + timedelta(hours=1)

    # 3. Скейлінг по всій історії (як у LSTM-інференсі)
    values = df_model[feature_cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    window = values_scaled[-cfg.window_size :]  # (window, n_features)
    x = torch.from_numpy(window).unsqueeze(0)   # (1, window, n_features)

    # 4. Завантажуємо модель GRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUForecastModel(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    # припускаю, що в GRUConfig є такий самий метод, як у LSTMConfig
    artifact_path = cfg.get_artifact_path(coin_id, vs_currency)

    state = torch.load(artifact_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        # випадок, коли збережено чистий state_dict без обгортки
        state_dict = state

    model.load_state_dict(state_dict)
    model.eval()


    with torch.no_grad():
        y_scaled = model(x.to(device)).cpu().numpy().reshape(-1)[-1]

    # 5. inverse-scale тільки таргет
    y_pred = float(
        _inverse_scale_target(
            scaler,
            feature_cols,
            target_col_idx,
            np.array([y_scaled], dtype=np.float32),
        )[0]
    )

    # 6. Запис у DuckDB
    store_hourly_forecast(
        coin_id=coin_id,
        vs_currency=vs_currency,
        ts_anchor=ts_anchor,
        ts_forecast=ts_forecast,
        model=model_name,
        y_pred=y_pred,
    )

    return {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "ts_anchor": ts_anchor,
        "ts_forecast": ts_forecast,
        "y_pred": y_pred,
    }